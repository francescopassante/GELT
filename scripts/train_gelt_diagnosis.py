import math
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt.blocks import GELT
from gelt.lattice import rectangular_wilson_loop
from gelt.sampler import mcmc_ensemble


def _param_norm(p):
    return p.detach().abs().pow(2).mean().sqrt().item()


def _grad_norm(g):
    if g is None:
        return float("nan")
    return g.detach().abs().pow(2).mean().sqrt().item()


def report_attention_state(model, label="", out_stream=None):
    """Dump everything load-bearing for the 'is attention actually working?'
    question, per GEMHSA layer.

    Reads the intermediates stashed by ``GEMHSA.attend`` / ``GEMHSA.forward``
    on the most recent forward pass (``_last_score``, ``_last_alpha``, and
    activation/residual norms). Also reads current parameter values and
    parameter gradients (the gradient of the LAST backward call — the caller
    is responsible for ensuring a backward has run if grad readout is wanted).

    What to look at:
      • score σ_across_off → if ~0, softmax is uniform per site → no axis
        selection happening, regardless of overall score magnitude.
      • softmax entropy / log(n_off) → 1.0 = perfectly uniform, 0 = one-hot.
        > 0.99 for many epochs means the attention is dead.
      • α̅[n] per offset → if all ≈ 1/n_off, attention isn't picking a
        direction. If concentrated on one offset, attention IS working.
      • |W_act| / |W_in| → sublayer contribution to the residual stream.
        Tiny ratio = the block is effectively the identity.
      • |∇Q|, |∇K| → is gradient even reaching the Q/K projections? If
        they are < 1e-10 while |∇V|, |∇mix| are healthy, the score channel
        is detached from the loss in practice.
      • gate mean/std → is the L-Act gate killing the sublayer output?
    """
    write = (out_stream.write if out_stream is not None else print)
    if not hasattr(model, "gemhsa_models"):
        write("[report_attention_state] model has no .gemhsa_models — skipping\n")
        return
    first = model.gemhsa_models[0]
    n_off = first.n_offsets
    H = first.H
    offsets = first.offsets
    log_n_off = math.log(n_off)

    lines = []
    lines.append(f"\n========= ATTENTION DIAG {label} =========")
    lines.append(f"  n_offsets={n_off}  n_heads={H}  d_qkv={first.d_qkv}")
    lines.append(f"  offsets = {offsets}")
    if hasattr(model, "lift"):
        lift_w = model.lift.weight
        lines.append(
            f"  lift: shape={tuple(lift_w.shape)}  |W|={_param_norm(lift_w):.3e}  "
            f"|∇W|={_grad_norm(lift_w.grad):.2e}"
        )
    if hasattr(model, "mlp"):
        lines.append(
            f"  mlp.fc1: |W|={_param_norm(model.mlp.fc1.weight):.3e}  "
            f"|∇W|={_grad_norm(model.mlp.fc1.weight.grad):.2e}  "
            f"|b|={_param_norm(model.mlp.fc1.bias):.3e}"
        )
        lines.append(
            f"  mlp.fc2: |W|={_param_norm(model.mlp.fc2.weight):.3e}  "
            f"|∇W|={_grad_norm(model.mlp.fc2.weight.grad):.2e}  "
            f"|b|={_param_norm(model.mlp.fc2.bias):.3e}"
        )

    for i, layer in enumerate(model.gemhsa_models):
        w = layer.w_QKV
        w_d = w.detach()
        wQ_n = w_d[0].abs().pow(2).mean().sqrt().item()
        wK_n = w_d[1].abs().pow(2).mean().sqrt().item()
        wV_n = w_d[2].abs().pow(2).mean().sqrt().item()
        wmix_n = _param_norm(layer.w_mix)
        bh = layer.b_h.detach()
        bh_max = bh.abs().max().item()
        bh_std = bh.std(unbiased=False).item() if bh.numel() > 1 else 0.0
        rezero_alpha = layer.alpha.item()

        # Grads (last backward). Split on the fused w_QKV by Q/K/V slice.
        if w.grad is not None:
            g = w.grad
            gQ = g[0].abs().pow(2).mean().sqrt().item()
            gK = g[1].abs().pow(2).mean().sqrt().item()
            gV = g[2].abs().pow(2).mean().sqrt().item()
        else:
            gQ = gK = gV = float("nan")
        gmix = _grad_norm(layer.w_mix.grad)
        gbh = _grad_norm(layer.b_h.grad)
        galpha = _grad_norm(layer.alpha.grad)

        # Activations from last forward (set by attend / forward).
        Q_n = getattr(layer, "_last_Q_norm", float("nan"))
        Kt_n = getattr(layer, "_last_K_tilde_norm", float("nan"))
        Vt_n = getattr(layer, "_last_V_tilde_norm", float("nan"))
        bilin_n = getattr(layer, "_last_bilin_norm", float("nan"))
        W_in_n = getattr(layer, "_last_W_in_norm", float("nan"))
        W_mix_n = getattr(layer, "_last_W_mix_norm", float("nan"))
        W_act_n = getattr(layer, "_last_W_act_norm", float("nan"))
        gate_mu = getattr(layer, "_last_gate_mean", float("nan"))
        gate_sigma = getattr(layer, "_last_gate_std", float("nan"))
        ratio = W_act_n / W_in_n if (W_in_n and W_in_n > 0) else float("nan")

        # Score / softmax stats. The σ_across_off line is the KEY one:
        # if it's near zero, the softmax is uniform per site no matter how
        # large the overall score is.
        score = getattr(layer, "_last_score", None)
        alpha_t = getattr(layer, "_last_alpha", None)

        lines.append(f"\n-- Layer {i} --")
        lines.append(
            f"  params:  |w_Q|={wQ_n:.3e}  |w_K|={wK_n:.3e}  |w_V|={wV_n:.3e}  "
            f"|w_mix|={wmix_n:.3e}  α_rezero={rezero_alpha:+.4f}  "
            f"|b_h|max={bh_max:.2e}  b_h_σ={bh_std:.2e}"
        )
        lines.append(
            f"  grads :  |∇w_Q|={gQ:.2e}  |∇w_K|={gK:.2e}  |∇w_V|={gV:.2e}  "
            f"|∇w_mix|={gmix:.2e}  |∇b_h|={gbh:.2e}  |∇α|={galpha:.2e}"
        )
        lines.append(
            f"  acts  :  |Q|={Q_n:.3e}  |K̃|={Kt_n:.3e}  |Ṽ|={Vt_n:.3e}  "
            f"|bilin|={bilin_n:.3e}"
        )
        lines.append(
            f"  resid :  |W_in|={W_in_n:.3e}  |W_mix|={W_mix_n:.3e}  "
            f"|W_act|={W_act_n:.3e}  |W_act|/|W_in|={ratio:.3e}"
        )
        lines.append(
            f"  gate  :  μ={gate_mu:+.3e}  σ={gate_sigma:.3e}"
        )

        if score is not None:
            s = score.float()
            score_mu = s.mean().item()
            score_sigma = s.std(unbiased=False).item()
            # std *across the offset axis* at each (B, H, *Λ), then averaged.
            score_sigma_off = s.std(dim=2, unbiased=False).mean().item()
            lines.append(
                f"  score :  μ={score_mu:+.3e}  σ_global={score_sigma:.3e}  "
                f"σ_across_off={score_sigma_off:.3e}  "
                f"[{s.min().item():+.2e}, {s.max().item():+.2e}]"
            )
        if alpha_t is not None:
            a = alpha_t.float()
            entropy = -(a * a.clamp_min(1e-30).log()).sum(dim=2)
            ent_norm = (entropy / log_n_off).mean().item()
            # Average attention per offset (over B, H, *Λ).
            reduce_dims = (0, 1) + tuple(range(3, a.ndim))
            alpha_per_off = a.mean(dim=reduce_dims).tolist()
            per_off_str = "  ".join(
                f"{off}:{a_v:.3f}" for off, a_v in zip(offsets, alpha_per_off)
            )
            argmax_off = int(torch.tensor(alpha_per_off).argmax().item())
            lines.append(
                f"  softmx:  entropy/log(n)={ent_norm:.4f}  "
                f"argmax_off={offsets[argmax_off]}"
            )
            lines.append(f"  α̅[n] :  {per_off_str}")
            if H > 1:
                # Per-head: max attention offset for each head (interesting
                # to see whether heads specialize differently).
                a_per_head = a.mean(dim=(0,) + tuple(range(3, a.ndim)))  # (H, n_off)
                head_str = []
                for h in range(H):
                    a_h = a_per_head[h].tolist()
                    a_h_max = max(range(n_off), key=lambda k: a_h[k])
                    head_str.append(
                        f"H{h}→{offsets[a_h_max]}({a_h[a_h_max]:.2f})"
                    )
                lines.append(f"  heads :  {'  '.join(head_str)}")

    lines.append("==========================================\n")
    write("\n".join(lines) + "\n")


def diag_forward_backward(model, batch, criterion, device):
    """Single forward+backward pass solely to populate gradients and the
    GEMHSA intermediates, without touching the optimizer state. Returns the
    loss for convenience.
    """
    X, T, y = batch
    X, T, y = X.to(device), T.to(device), y.to(device)
    model.zero_grad(set_to_none=False)
    outputs = model(X, T)
    loss = criterion(outputs, y)
    loss.backward()
    return loss.item(), outputs.detach()


def averaged_wilson_loop(config, gaugegroup, R, T, mu, nu):
    """Lattice-averaged Re Tr W(R, T) / nc — one scalar per config.

    Wraps ``rectangular_wilson_loop`` (per-site, shape ``(B, *Λ)``) and reduces
    over all spatial axes. Pairs with ``reduction="mean"`` on GELT so the model
    output and target are both shape ``(B,)``.
    """
    y = rectangular_wilson_loop(config, gaugegroup, R=R, T=T, mu=mu, nu=nu)
    return y.mean(dim=tuple(range(1, y.ndim)))

"""
========================================================================================
 This is a minimal training script used for lookup on how to train the architecture
========================================================================================
"""


def evaluate(model, test_loader, criterion, device, save_outputs=False, progress=True):
    model.eval()

    test_loss = 0.0
    test_count = 0
    if save_outputs:
        all_targets = []
        all_outputs = []
    iterator = tqdm(test_loader) if progress else test_loader
    with torch.no_grad():
        for X, T, y in iterator:
            X, T, y = X.to(device), T.to(device), y.to(device)
            outputs = model(X, T)
            loss = criterion(outputs, y)
            batch_size = y.shape[0]
            test_loss += loss.item() * batch_size
            test_count += batch_size
            if save_outputs:
                all_targets.append(y.cpu())
                all_outputs.append(outputs.cpu())

    test_loss /= test_count
    if save_outputs:
        all_targets = torch.cat(all_targets)
        all_outputs = torch.cat(all_outputs)
        return test_loss, all_targets, all_outputs
    return test_loss


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    patience=5,
    checkpoint_path: str = "best_model.pth",
    diag_every: int = 25,
    diag_log_path: str | None = None,
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    # Open a log file for the verbose per-layer dumps so the tqdm bar in
    # stdout stays readable. The compact per-epoch attention signals
    # (entropy, σ_across_off, max-α offset, |W_act|/|W_in|) still print
    # alongside the loss in the bar's stream.
    diag_log = open(diag_log_path, "w") if diag_log_path else None
    if diag_log is not None:
        diag_log.write(f"diag log — diag_every={diag_every}\n")
        diag_log.flush()

    # Initial diagnostic dump: one forward+backward at epoch 0 (untrained
    # weights) to capture the state we're starting from. Especially useful
    # for checking whether the score σ_across_off has the O(1) magnitude
    # the qk_init_scale knob was supposed to give it.
    init_batch = next(iter(train_loader))
    init_loss, _ = diag_forward_backward(model, init_batch, criterion, device)
    report_attention_state(
        model, label=f"epoch -1 (init, loss={init_loss:.4f})",
        out_stream=diag_log,
    )
    if diag_log is None:
        # Already printed to stdout above; nothing more to do.
        pass
    else:
        # Mirror a one-line summary to stdout so the user knows it ran.
        first = model.gemhsa_models[0]
        s = first._last_score.float()
        a = first._last_alpha.float()
        log_n_off = math.log(first.n_offsets)
        ent = -(a * a.clamp_min(1e-30).log()).sum(dim=2)
        print(
            f"[init] L0 score σ_across_off={s.std(dim=2, unbiased=False).mean().item():.3e}  "
            f"softmax entropy/log(n)={(ent / log_n_off).mean().item():.4f}  "
            f"loss={init_loss:.4f}"
        )

    epoch_bar = tqdm(range(epochs))
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_count = 0
        # Dead-signal diagnostics, accumulated over the epoch:
        #   out_*  — does the prediction actually move, or has it collapsed
        #            to a constant (std → 0) at the standardized mean (= 0)?
        #   grad_norm — is any gradient reaching the parameters at all?
        # First batch only: dump target stats to confirm the loader isn't
        # serving zeros / a degenerate label distribution.
        out_sum = 0.0
        out_sq_sum = 0.0
        out_count = 0
        grad_norm_sum = 0.0
        grad_norm_batches = 0
        first_batch = True
        for X, T, y in train_loader:
            X, T, y = X.to(device), T.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, T)
            loss = criterion(outputs, y)
            loss.backward()

            with torch.no_grad():
                o = outputs.detach()
                out_sum += o.sum().item()
                out_sq_sum += (o * o).sum().item()
                out_count += o.numel()
                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().pow(2).sum().item()
                grad_norm_sum += total_sq**0.5
                grad_norm_batches += 1
                if first_batch and epoch < 3:
                    epoch_bar.write(
                        f"  ep {epoch:>3d} [first batch]  "
                        f"y μ={y.mean().item():+.4f} σ={y.std(unbiased=False).item():.4f}  "
                        f"out μ={o.mean().item():+.4f} σ={o.std(unbiased=False).item():.4f}  "
                        f"loss={loss.item():.4f}"
                    )
                    first_batch = False

            # Clip *after* the diagnostic grad-norm measurement above so the
            # reported ‖grad‖ still shows pre-clip spikes. Prevents a single
            # bad batch from poisoning Adam's second-moment buffer.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_size = y.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)
        scheduler.step()

        out_mean = out_sum / out_count
        out_var = max(out_sq_sum / out_count - out_mean * out_mean, 0.0)
        out_std = out_var**0.5
        avg_grad_norm = grad_norm_sum / max(grad_norm_batches, 1)

        val_loss = evaluate(model, val_loader, criterion, device, progress=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        # Unfreeze-cascade diagnostic. With ReZero α=0 and zero-init mlp.fc2,
        # only the MLP receives gradient on step 0; α and the GEMHSA params
        # follow once fc2 grows. If both α and ‖fc2‖ stay near 0 while the loss
        # is flat, training is stuck at the identity branch — bump LR, warm-
        # start α, or drop the mlp.fc2 zero-init. See notes/architecture.html §3.8.
        # Dead-signal readout every epoch: output σ ≈ 0 with grad-norm ≈ 0 means
        # the model has collapsed to a constant and no learning signal is
        # flowing through any parameter.
        if hasattr(model, "gemhsa_models"):
            alphas = [f"{layer.alpha.item():+.3f}" for layer in model.gemhsa_models]
            fc2_std = model.mlp.fc2.weight.detach().abs().mean().item()
            # Compact attention signals per layer for the per-epoch bar:
            #   ent  = softmax entropy / log(n_off): 1.0 = uniform, 0 = one-hot.
            #          The single best "is attention picking a direction?"
            #          number — independent of overall score scale.
            #   σoff = std of score across the offset axis at fixed (B, H, x).
            #          If ent ≈ 1.0 because σoff is tiny, the score channel is
            #          the bottleneck (Q/K init / gradient flow). If σoff is
            #          large but ent is still near 1, softmax temperature is
            #          the issue.
            #   wact = |W_act| / |W_in|: how much the sublayer contributes
            #          to the residual stream.
            ent_str = []
            soff_str = []
            wact_str = []
            for layer in model.gemhsa_models:
                a = layer._last_alpha.float()
                s = layer._last_score.float()
                log_n_off = math.log(layer.n_offsets)
                ent = -(a * a.clamp_min(1e-30).log()).sum(dim=2)
                ent_str.append(f"{(ent / log_n_off).mean().item():.3f}")
                soff_str.append(
                    f"{s.std(dim=2, unbiased=False).mean().item():.2e}"
                )
                w_in = layer._last_W_in_norm
                w_act = layer._last_W_act_norm
                wact_str.append(
                    f"{(w_act / w_in if w_in > 0 else float('nan')):.2e}"
                )
            epoch_bar.write(
                f"  ep {epoch + 1:>3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"out μ={out_mean:+.4f} σ={out_std:.4f}  "
                f"‖grad‖={avg_grad_norm:.2e}  "
                f"α=[{', '.join(alphas)}]  |fc2|̄={fc2_std:.4f}\n"
                f"      ent/log(n)=[{', '.join(ent_str)}]  "
                f"σ_off=[{', '.join(soff_str)}]  "
                f"|W_act|/|W_in|=[{', '.join(wact_str)}]"
            )

            # Periodic full dump (parameters, grads, activations, per-offset
            # attention vectors) to the diag log.
            if diag_every > 0 and (epoch + 1) % diag_every == 0:
                report_attention_state(
                    model, label=f"epoch {epoch + 1}", out_stream=diag_log,
                )
                if diag_log is not None:
                    diag_log.flush()

        if epochs_no_improve >= patience:
            epoch_bar.write(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Final full dump (post-training weights, with last-batch grads).
    report_attention_state(
        model, label=f"epoch {epoch + 1} (FINAL)", out_stream=diag_log,
    )
    if diag_log is not None:
        diag_log.close()

    if best_val_loss == float("inf"):
        raise RuntimeError(
            f"No checkpoint written to {checkpoint_path}: val_loss never improved "
            f"over inf. Likely NaN/Inf losses — check the training output."
        )

    return train_losses, val_losses, epoch + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    from gelt import SU, Z2, build_plaquette_datasets, load_plaquette_datasets

    D = 2
    L = 8
    gaugegroup = SU(2)
    R = 1
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    # 2D SU(2) is exactly solvable (no phase transition); β≈2 sits in the
    # weak-coupling regime where Wilson loops have a clean, smoothly varying
    # signal — appropriate working point for the diagnostic run.
    beta = 2.0
    # Lattice-averaged Wilson loop target: y has shape (B,). Paired with
    # ``reduction="mean"`` on GELT, the model averages its per-site readout
    # over the spatial axes and is supervised against the configuration-level
    # mean ⟨Re Tr W⟩. This is a strictly weaker constraint than per-site
    # supervision — errors at different anchors can cancel — and matches the
    # translation-invariant quantity most physical observables actually need.
    loop_R, loop_T, mu, nu = 3, 3, 0, 1
    dataset_parameters = {
        "N": 1000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": True,
        "prefix": f"su2_plaquette_L{L}_D{D}_N1000_beta{beta}_R{R}_wloop{loop_R}x{loop_T}_singlepath",
        "structured": True,
        # SU(2) Metropolis via _SWEEP_FN[SU] = su2_metropolis_sweep.
        "sampler": mcmc_ensemble,
        "beta": beta,
        "target": partial(averaged_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.complex64,
        # Single canonical shortest path per offset (rotation symmetry
        # broken); A/B variant against the default "average" mode for the
        # path-averaging diagnostic — see notes/architecture.html §3.3.
        "transport_mode": "single",
    }

    train_parameters = {
        # ReZero α and zero-init mlp.fc2 mean the gradient-flow unfreezing
        # cascade (fc2 → fc1 → α → Q/K/V/mix) is slow at lr=1e-3 — pushing the
        # LR up gets training past the identity-branch stall in a few epochs.
        "lr": 3e-3,
        "batch_size": 64,
        "epochs": 3000,
        "patience": 3000,
        "checkpoint_path": "best_gelt.pth",
    }

    # Debug-capacity GELT for the lattice-averaged Wilson loop target. A 3×3
    # loop has plaquette degree 9 in Z₂, so the model still needs depth and
    # head count to compose multi-site products — averaging only relaxes the
    # per-anchor reconstruction requirement, not the underlying loop algebra.
    # The earlier matched-capacity config (nhead=2, d_qkv=8, gemhsa_layers=2,
    # mlp_hidden=16) was sized for the linear-in-P action target and is too
    # small for Wilson loops — leave the matched shootout for after the
    # averaged path is validated.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 1,
        "gemhsa_layers": 4,
        "d_qkv": 16,
        "gate": "softplus",
        # Z2 can run as a real model. SU(N) must stay complex; otherwise
        # GELT.forward would cast complex plaquettes/transports down to real.
        "dtype": model_dtype,
        "mlp_hidden": 32,
        "mlp_out": 1,
        # Lattice-averaged target → "mean" spatial reduction on the GELT
        # readout. Use "none" for per-site supervision, "sum" for the
        # extensive Wilson action.
        "reduction": "mean",
        # Warm-start the ReZero α and drop the MLP fc2 zero-init: the default
        # α=0 + fc2=0 combo traps training at the constant-mean predictor on
        # the 2×2 Wilson loop target. α=0.5 puts the multiplicative path at
        # ~half the residual stream (α=0.05 left it at ~4% and the MLP just
        # kept reading the raw plaquette at site x, never using the multi-
        # site contribution). init_scale now controls σ_V only (value path —
        # kept small so the residual stream is near-identity at init);
        # qk_init_scale controls σ_QK (score channel) and is decoupled so the
        # softmax can have real dynamic range from epoch 0 — with the old
        # tied init, score variance was O(σ⁴) ~ 1e-8 and softmax was uniform,
        # so no per-offset gradient ever reached Q/K and bias was the only
        # path to axis selection.
        "alpha_init": 0.5,
        "init_scale": 10.0,
        "qk_init_scale": 1.0,
        "mlp_zero_init": False,
        # Widen the residual-stream beyond the small plaquette channel count
        # D(D-1)/2 ∈ {1, 3, 6} via the front-end ChannelLift. Decouples the
        # GEMHSA working width from the input dimensionality so intermediate
        # layers don't collapse to 1–6 channels.
        "d_model": 16,
    }

    # Reuse a previously saved dataset if one exists under this prefix;
    # otherwise generate it now (save=True writes it for next time).
    from pathlib import Path

    _prefix = dataset_parameters["prefix"]
    if all(
        Path(f"datasets/{s}_dataset_{_prefix}.pt").exists()
        for s in ("train", "val", "test")
    ):
        train_dataset, val_dataset, test_dataset = load_plaquette_datasets(_prefix)
    else:
        train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
            **dataset_parameters
        )

    # Standardize the target (notes/architecture.html §6.1). Compute (μ_y, σ_y)
    # on the train split, then mutate the shared full-y tensor in place — all
    # three subsets are Subsets of the same TensorDataset, so val/test see the
    # normalized labels too. Paired with the zero-init of the MLP's last layer
    # (gelt/blocks.py), the untrained model is the constant predictor at the
    # normalized mean (= 0), giving R² = 0 — the trivial mean-baseline.
    # Predictions and saved targets are denormalized at the end for plotting.

    y_train = train_dataset.dataset.tensors[-1][train_dataset.indices]
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    train_dataset.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)
    print(f"target scaler fit: μ_y = {mu_y.item():.4f} | σ_y = {sigma_y.item():.4f}")

    model = GELT(**model_parameters)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TensorDataset lives entirely in RAM (no decoding / disk I/O), so worker
    # processes only add overhead and the staleness footgun above. pin_memory
    # still helps the host→GPU copy on CUDA.
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_parameters["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )

    X, T, y = next(iter(train_loader))
    n_params = sum(p.numel() for p in model.parameters())
    n_real_dofs = sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )
    print(
        f"GELT | params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
    )

    # Dataset-level sanity: are the inputs/targets even informative? If
    # |X| is degenerate or y is near-constant we'll spin forever chasing an
    # attention bug that is actually a data bug.
    with torch.no_grad():
        Xn = X.float() if not X.is_complex() else X.abs()
        print(
            f"input X   : |X|={Xn.abs().mean().item():.3e}  "
            f"σ(|X|)={Xn.abs().std(unbiased=False).item():.3e}  "
            f"Re(Tr X)/nc mean per-chan={X.diagonal(dim1=-2, dim2=-1).sum(-1).real.mean().item():+.4f}"
        )
        Tn = T.float() if not T.is_complex() else T.abs()
        print(
            f"transport T: |T|={Tn.abs().mean().item():.3e}  shape={tuple(T.shape)}"
        )
        print(
            f"target y  : μ={y.mean().item():+.4f}  σ={y.std(unbiased=False).item():.4f}  "
            f"min={y.min().item():+.4f}  max={y.max().item():+.4f}  "
            f"shape={tuple(y.shape)}"
        )
        # Per-channel target-input correlation: even one linear-in-X channel
        # gives a non-trivial signal floor. Useful to see whether a trivial
        # site-sum predictor would already fit the target.
        if y.ndim == 1:
            X_re = X.real if X.is_complex() else X
            x_summed = X_re.diagonal(dim1=-2, dim2=-1).sum(-1)
            # Sum/mean over spatial axes to align shape with y.
            spatial = tuple(range(2, x_summed.ndim))
            x_per_chan = x_summed.mean(dim=spatial)  # (B, C_in)
            for c in range(x_per_chan.shape[1]):
                xc = x_per_chan[:, c]
                if xc.std() > 0:
                    corr = (
                        (xc - xc.mean()) * (y - y.mean())
                    ).mean() / (xc.std(unbiased=False) * y.std(unbiased=False) + 1e-30)
                    print(f"  corr(Re Tr ⟨X_{c}⟩, y) = {corr.item():+.4f}")

    model = model.to(device)

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=train_parameters["epochs"],
        patience=train_parameters["patience"],
        checkpoint_path=train_parameters["checkpoint_path"],
        diag_every=25,
        diag_log_path="gelt_diag.log",
    )

    # Load best model to evaluate on test set
    model.load_state_dict(
        torch.load(
            train_parameters["checkpoint_path"], map_location=device, weights_only=True
        )
    )

    test_loss, all_targets, all_outputs = evaluate(
        model, test_loader, criterion, device, save_outputs=True
    )

    # Plots and visualizations

    # ``test_loss`` and the saved arrays are in normalized space (y was
    # standardized in place above). R² is invariant under linear label
    # transforms, so we can compute it either way. Denormalize to show the
    # scatter plot in physical Wilson-action units.
    all_targets = all_targets * sigma_y + mu_y
    all_outputs = all_outputs * sigma_y + mu_y
    test_label_var = all_targets.var(unbiased=False).item()
    test_mse_physical = ((all_outputs - all_targets) ** 2).mean().item()
    test_r2 = (
        1.0 - test_mse_physical / test_label_var if test_label_var > 0 else float("nan")
    )

    print(
        f"Test Loss (norm): {test_loss:.4f} | "
        f"Test MSE (physical): {test_mse_physical:.4f} | "
        f"Var(y): {test_label_var:.4f} | R²: {test_r2:.4f}"
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("gelt_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Flatten targets/predictions for the scatter; with the lattice-averaged
    # target this is one point per test config (≈ 150 at the default split),
    # but the subsample guard is kept for parity with the per-site path.
    t_flat = all_targets.reshape(-1).numpy()
    o_flat = all_outputs.reshape(-1).numpy()
    if t_flat.size > 20000:
        rng = torch.Generator().manual_seed(0)
        idx = torch.randperm(t_flat.size, generator=rng)[:20000].numpy()
        t_flat, o_flat = t_flat[idx], o_flat[idx]
    plt.figure(figsize=(8, 8))
    plt.scatter(t_flat, o_flat, alpha=0.5, s=4)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values (Test Set)")
    plt.grid(True)
    plt.savefig("gelt_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
