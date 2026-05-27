from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import haar_ensemble
from gelt.blocks_routed import GELT
from gelt.lattice import rectangular_wilson_loop

"""
========================================================================================
 Diagnostic training script for the ROUTED three-branch GELT
 (notes/codex_architecture.md §5).

 Mirrors train_gelt_diagnosis_new.py — same loop, same dead-signal probes, same Wilson-
 loop target — but trains the new ``gelt.blocks_routed.GELT`` (Branch A: L-Conv
 algebraic transport, Branch B: soft attention routing, Branch C: local L-Bilin) instead
 of the fused GEMHSA or the split GEAttention+GEFFN. The per-layer α diagnostic is
 extended to print all four ReZero scalars (outer α plus per-branch α_A/α_B/α_C) so you
 can see whether the algebraic basis, the attention route, or the local multiplication
 is engaging fastest under the chosen warm-start schedule.

 Expected qualitative result vs. the fused-GEMHSA and split baselines (per §6 of the
 codex notes): the wall at the 3×3 Wilson loop should move further, because additive
 parallel branches break the multiplicative coupling chicken-and-egg at random init —
 the algebraic transport branch (constructively initialised) already provides an L-CNN
 basis of transported features that the L-Bilin branch can multiply at depth, while
 attention sits on top as the adaptive component.
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
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

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
        # only the MLP receives gradient on step 0; α and the routed-block
        # params follow once fc2 grows. If outer α and ‖fc2‖ stay near 0
        # while the loss is flat, training is stuck at the identity branch —
        # bump LR, warm-start α, or drop the mlp.fc2 zero-init. See
        # notes/architecture.html §3.8 and notes/codex_architecture.md §6.
        #
        # Routed-block extension: in addition to the outer ReZero α, each
        # block has three per-branch ReZeros α_A (L-Conv), α_B (attention),
        # α_C (L-Bilin). The codex notes argue that the algebraic-basis
        # branches (A, C) should engage before the attention branch (B);
        # if α_B leaves zero before α_A or α_C the schedule is wrong.
        if hasattr(model, "blocks"):
            outer = [f"{layer.alpha.item():+.3f}" for layer in model.blocks]
            aA = [f"{layer.alpha_A.item():+.3f}" for layer in model.blocks]
            aB = [f"{layer.alpha_B.item():+.3f}" for layer in model.blocks]
            aC = [f"{layer.alpha_C.item():+.3f}" for layer in model.blocks]
            fc2_std = model.mlp.fc2.weight.detach().abs().mean().item()
            epoch_bar.write(
                f"  ep {epoch + 1:>3d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                f"out μ={out_mean:+.4f} σ={out_std:.4f}  "
                f"‖grad‖={avg_grad_norm:.2e}  "
                f"α=[{', '.join(outer)}]  "
                f"α_A=[{', '.join(aA)}]  α_B=[{', '.join(aB)}]  α_C=[{', '.join(aC)}]  "
                f"|fc2|̄={fc2_std:.4f}"
            )

        if epochs_no_improve >= patience:
            epoch_bar.write(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_val_loss == float("inf"):
        raise RuntimeError(
            f"No checkpoint written to {checkpoint_path}: val_loss never improved "
            f"over inf. Likely NaN/Inf losses — check the training output."
        )

    return train_losses, val_losses, epoch + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    from gelt import SU, Z2, build_plaquette_datasets

    D = 2
    L = 8
    gaugegroup = Z2()
    R = 1
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    beta = 1
    # Per-site Wilson loop target: y has shape (B, *Λ). Paired with
    # ``reduction="none"`` on GELT, the model's per-site readout is supervised
    # directly — every site contributes a sample, and the equivariant trace
    # head outputs the locally gauge-invariant quantity at x.
    # The 3×3 target is the one that walled both the fused GEMHSA and the
    # split GEAttention+GEFFN baselines (codex_architecture.md §§2-3) — the
    # canonical benchmark for the routed three-branch architecture.
    loop_R, loop_T, mu, nu = 3, 3, 0, 1
    dataset_parameters = {
        "N": 1000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "prefix": f"z2_plaquette_L{L}_D{D}_N1000_beta{beta}_R{R}_wloop{loop_R}x{loop_T}",
        "structured": True,
        "sampler": haar_ensemble,
        "beta": beta,
        "target": partial(rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.complex64,
    }

    train_parameters = {
        # ReZero α and zero-init mlp.fc2 mean the gradient-flow unfreezing
        # cascade (fc2 → fc1 → α → Q/K/V/mix) is slow at lr=1e-3 — pushing the
        # LR up gets training past the identity-branch stall in a few epochs.
        "lr": 3e-3,
        "batch_size": 64,
        "epochs": 3000,
        "patience": 3000,
        # Distinct checkpoint name so the routed run doesn't clobber the
        # fused-GEMHSA or split-block checkpoints.
        "checkpoint_path": "best_gelt_routed.pth",
    }

    # Three-branch routed GELT for the 3×3 per-site Wilson loop target.
    # Each RoutedBlock runs all three branches in parallel and combines them
    # via the residual update
    #     W ← W + α · ( g(W + α_A·A + α_B·B + α_C·C) · (W + α_A·A + α_B·B + α_C·C) − W ).
    # Per codex §5, the algebraic-basis branches A (L-Conv, constructively
    # initialised on a basis of single-offset routes) and C (local L-Bilin)
    # are warm-started ABOVE the attention branch B — they form the
    # reliable backbone of transported features that the multiplicative
    # path can grow into Wilson loops, while attention adds the adaptive
    # routing component on top.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 1,
        # Same loop-doubling reach as the 4-layer fused GEMHSA stack: every
        # routed block exposes its own L-Conv + attention + L-Bilin step,
        # so 4 blocks = 4 attention + 4 algebraic-transport + 4 L-Bilin
        # passes with decoupled gradients.
        "n_layers": 4,
        "d_qkv": 16,
        "gate": "softplus",
        # Z2 can run as a real model. SU(N) must stay complex; otherwise
        # GELT.forward would cast complex plaquettes/transports down to real.
        "dtype": model_dtype,
        "mlp_hidden": 32,
        "mlp_out": 1,
        # Per-site target → no spatial reduction. Use "sum" for the Wilson
        # action, "mean" for the average ⟨W⟩.
        "reduction": "none",
        # ReZero schedule from codex §5/§6: algebraic backbone (A, C)
        # warm-started large so the L-Conv basis is active from epoch 0
        # and L-Bilin can multiply it; attention starts smaller and grows
        # as it learns offset selectivity on top of an already non-trivial
        # transported feature bank. Outer α=0.5 engages the block from
        # epoch 0 (same logic as train_gelt_diagnosis.py).
        "alpha_init": 0.5,
        "alpha_A_init": 1.0,
        "alpha_B_init": 0.3,
        "alpha_C_init": 1.0,
        # init_scale=10 lifts attention score magnitudes off the near-uniform
        # softmax floor so Branch B has per-offset signal to learn from on
        # epoch 0.
        "init_scale": 10.0,
        "mlp_zero_init": False,
        # Widen the residual stream beyond D(D-1)/2 ∈ {1, 3, 6} via the
        # front-end ChannelLift so the routed branches have a meaningful
        # working width.
        "d_model": 16,
        # Constructive L-Conv init: each output channel of Branch A starts
        # as a single (offset, augmented-channel) basis route. With
        # α_A=1.0 above, this gives an L-CNN-like algebraic basis from
        # epoch 0 instead of asking attention to discover transport routes
        # from random init (codex §5, last bullet).
        "constructive_A": True,
        # Degree-2 (bilinear) Branch C by default. Flip to True for the
        # degree-3 product Q† · V · V2 — useful when the target needs
        # faster operator-degree growth than one L-Bilin step per block
        # (codex §5, "Consider local multilinear FFNs").
        "trilinear": False,
    }

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
        f"GELT (routed) | n_layers={len(model.blocks)} "
        f"trilinear={model_parameters['trilinear']} "
        f"constructive_A={model_parameters['constructive_A']} | "
        f"params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
    )

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
    plt.title("Training and Validation Loss (routed three-branch GELT)")
    plt.legend()
    plt.grid(True)
    # Distinct plot names so this run doesn't overwrite the fused-GEMHSA or
    # split-block diagnostic outputs.
    plt.savefig("gelt_routed_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Flatten per-site targets/predictions for the scatter; subsample if dense
    # so matplotlib stays responsive (per-site targets give |Λ| points per
    # config, e.g. 8²·N_test ≈ 9.6k points at L=8, D=2).
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
    plt.title("True vs Predicted Values (routed three-branch GELT, Test Set)")
    plt.grid(True)
    plt.savefig("gelt_routed_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "trilinear": model_parameters["trilinear"],
        "constructive_A": model_parameters["constructive_A"],
    }
