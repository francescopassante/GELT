from functools import partial
import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import haar_ensemble
from gelt.blocks import GELT
from gelt.lattice import rectangular_wilson_loop

"""
========================================================================================
 This is a minimal training script used for lookup on how to train the architecture
========================================================================================
"""


def cuda_elapsed_ms(fn):
    """Run ``fn`` once and return ``(result, elapsed_ms)`` using CUDA events."""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def print_gpu_profile_header(model, train_loader, device):
    print("\n=== GELT GPU profile context ===")
    print(f"torch: {torch.__version__}")
    print(f"device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"gpu: {props.name} | cc {props.major}.{props.minor} | "
            f"memory {props.total_memory / 1024**3:.2f} GiB"
        )
        print(
            "cuda flags: "
            f"allow_tf32_matmul={torch.backends.cuda.matmul.allow_tf32} | "
            f"allow_tf32_cudnn={torch.backends.cudnn.allow_tf32}"
        )
    print(
        f"dataloader: batch_size={train_loader.batch_size} | "
        f"num_workers={train_loader.num_workers} | pin_memory={train_loader.pin_memory}"
    )
    first_layer = model.gemhsa_models[0]
    print(
        "model: "
        f"layers={len(model.gemhsa_models)} | H={first_layer.H} | "
        f"d_qkv={first_layer.d_qkv} | C={first_layer.C} | "
        f"C_prime={first_layer.C_prime} | n_offsets={first_layer.n_offsets} | "
        f"gate={first_layer.gate} | reduction={model.reduction}"
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_real_dofs = sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )
    print(f"params: {n_params:,} tensors ({n_real_dofs:,} real DOFs)")


def profile_training_step(model, batch, criterion, optimizer, device):
    """Print one full training-step breakdown on a real batch."""
    if device.type != "cuda":
        print("[profile] CUDA is unavailable; skipping CUDA-event timings.")
        return

    X_cpu, T_cpu, y_cpu = batch
    torch.cuda.reset_peak_memory_stats(device)

    (X, T, y), copy_ms = cuda_elapsed_ms(
        lambda: (
            X_cpu.to(device, non_blocking=True),
            T_cpu.to(device, non_blocking=True),
            y_cpu.to(device, non_blocking=True),
        )
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs, fwd_ms = cuda_elapsed_ms(lambda: model(X, T))
    loss, loss_ms = cuda_elapsed_ms(lambda: criterion(outputs, y))
    _, bwd_ms = cuda_elapsed_ms(lambda: loss.backward())
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    _, step_ms = cuda_elapsed_ms(lambda: optimizer.step())
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    print("\n=== one training step timing ===")
    print(
        f"batch: X={tuple(X.shape)} {X.dtype} | T={tuple(T.shape)} {T.dtype} | "
        f"y={tuple(y.shape)} {y.dtype} | out={tuple(outputs.shape)}"
    )
    print(
        f"copy={copy_ms:.3f} ms | forward={fwd_ms:.3f} ms | "
        f"loss={loss_ms:.3f} ms | backward={bwd_ms:.3f} ms | "
        f"optim_step={step_ms:.3f} ms | total={copy_ms + fwd_ms + loss_ms + bwd_ms + step_ms:.3f} ms"
    )
    print(
        f"loss={loss.detach().item():.6g} | "
        f"peak_cuda_alloc={peak_mb:.1f} MiB"
    )
    optimizer.zero_grad(set_to_none=True)


def profile_gemhsa_sections(model, batch, device):
    """Print forward-only section timings for each GEMHSA layer on one batch."""
    if device.type != "cuda":
        return

    X_cpu, T_cpu, _ = batch
    X = X_cpu.to(device, non_blocking=True)
    T = T_cpu.to(device, non_blocking=True)
    first_layer = model.gemhsa_models[0]
    w_dtype = first_layer.w_QKV.dtype
    if X.dtype != w_dtype:
        X = X.to(w_dtype)
    if T.dtype != w_dtype:
        T = T.to(w_dtype)
    T_dag = first_layer.gaugegroup.dagger(T)

    was_training = model.training
    model.eval()
    W = X

    print("\n=== GEMHSA forward section timings ===")
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.gemhsa_models):
            nc = W.shape[-1]
            trailing = W.shape[2:]
            B = W.shape[0]
            HD = layer.H * layer.d_qkv

            def f_qkv():
                W_aug = layer.augment(W)
                W_aug_flat = W_aug.view(B, layer.C_prime, -1)
                w_QKV_flat = layer.w_QKV.view(3 * HD, layer.C_prime)
                QKV = torch.matmul(w_QKV_flat, W_aug_flat)
                QKV = QKV.view(B, 3, layer.H, layer.d_qkv, *trailing)
                return QKV.unbind(dim=1)

            (Q, K, V), qkv_ms = cuda_elapsed_ms(f_qkv)

            idx = tuple(layer._nbr_idx[k] for k in range(layer.D))
            nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))

            def f_gather():
                return K[nb_indexer], V[nb_indexer]

            (K_nb, V_nb), gather_ms = cuda_elapsed_ms(f_gather)

            def f_transport():
                KV_nb = torch.cat((K_nb, V_nb), dim=2)
                KV_tilde = layer.transport(KV_nb, T, T_dag)
                return KV_tilde.split(layer.d_qkv, dim=2)

            (K_tilde, V_tilde), transport_ms = cuda_elapsed_ms(f_transport)

            def f_score_value():
                score = torch.einsum(
                    "bhd...ij,bhdn...ij->bhn...", Q.conj(), K_tilde
                ).real
                score = score / (layer.d_qkv * nc) ** 0.5
                bias = layer.b_h.real if layer.b_h.is_complex() else layer.b_h
                score = score + bias.view(layer._bias_view_shape)
                alpha = torch.softmax(score, dim=2)
                alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
                V_weighted = (alpha_b * V_tilde).sum(dim=3)
                Q_dag = layer.gaugegroup.dagger(Q)
                return torch.matmul(Q_dag, V_weighted)

            out, score_value_ms = cuda_elapsed_ms(f_score_value)

            def f_mix_gate():
                out_flat = out.reshape(B, HD, -1)
                w_mix_flat = layer.w_mix.view(layer.C, HD)
                W_mix = torch.matmul(w_mix_flat, out_flat).view(B, layer.C, *trailing)
                W_res = W + W_mix
                trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
                if layer.gate == "relu":
                    g = torch.relu(trace_per_chan)
                else:
                    g = torch.nn.functional.softplus(trace_per_chan)
                W_act = g.unsqueeze(-1).unsqueeze(-1) * W_res
                return W + layer.alpha * (W_act - W)

            W, mix_gate_ms = cuda_elapsed_ms(f_mix_gate)
            total_ms = qkv_ms + gather_ms + transport_ms + score_value_ms + mix_gate_ms
            print(
                f"layer {layer_idx}: "
                f"qkv={qkv_ms:.3f} ms | gather={gather_ms:.3f} ms | "
                f"transport={transport_ms:.3f} ms | score_value={score_value_ms:.3f} ms | "
                f"mix_gate={mix_gate_ms:.3f} ms | sum={total_ms:.3f} ms"
            )

        _, end_to_end_ms = cuda_elapsed_ms(lambda: model(X, T))
        print(f"model forward end-to-end: {end_to_end_ms:.3f} ms")

    if was_training:
        model.train()


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
    profile_every: int = 0,
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
        epoch_t0 = time.perf_counter()
        data_wait = 0.0
        compute_time = 0.0
        last_batch_end = time.perf_counter()
        for batch_idx, (X, T, y) in enumerate(train_loader):
            batch_t0 = time.perf_counter()
            data_wait += batch_t0 - last_batch_end
            X, T, y = (
                X.to(device, non_blocking=True),
                T.to(device, non_blocking=True),
                y.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(X, T)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            batch_t1 = time.perf_counter()
            compute_time += batch_t1 - batch_t0
            last_batch_end = batch_t1
            batch_size = y.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device, progress=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_s = time.perf_counter() - epoch_t0
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}", val=f"{val_loss:.4f}", sec=f"{epoch_s:.1f}"
        )
        if profile_every > 0 and (epoch + 1) % profile_every == 0:
            epoch_bar.write(
                f"  ep {epoch + 1:>3d} timing: epoch={epoch_s:.2f}s | "
                f"train_compute={compute_time:.2f}s | data_wait={data_wait:.2f}s | "
                f"batches={batch_idx + 1}"
            )

        # Unfreeze-cascade diagnostic. With ReZero α=0 and zero-init mlp.fc2,
        # only the MLP receives gradient on step 0; α and the GEMHSA params
        # follow once fc2 grows. If both α and ‖fc2‖ stay near 0 while the loss
        # is flat, training is stuck at the identity branch — bump LR, warm-
        # start α, or drop the mlp.fc2 zero-init. See notes/architecture.html §3.8.
        if (epoch + 1) % 10 == 0 and hasattr(model, "gemhsa_models"):
            alphas = [f"{layer.alpha.item():+.3f}" for layer in model.gemhsa_models]
            fc2_std = model.mlp.fc2.weight.detach().abs().mean().item()
            epoch_bar.write(
                f"  ep {epoch + 1:>3d}  α=[{', '.join(alphas)}]  |fc2|̄={fc2_std:.4f}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print GPU timing and memory diagnostics before training.",
    )
    parser.add_argument(
        "--profile-every",
        type=int,
        default=10,
        help="When --profile is set, print per-epoch timing every N epochs.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    from gelt import SU, Z2, build_plaquette_datasets

    D = 3
    L = 8
    gaugegroup = Z2()
    R = 2

    beta = 1
    # Per-site Wilson loop target: y has shape (B, *Λ). Paired with
    # ``reduction="none"`` on GELT, the model's per-site readout is supervised
    # directly — every site contributes a sample, and the equivariant trace
    # head outputs the locally gauge-invariant quantity at x.
    loop_R, loop_T, mu, nu = 2, 2, 0, 1
    dataset_parameters = {
        "N": 2000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "prefix": f"z2_plaquette_L{L}_D{D}_N2000_beta{beta}_R{R}_wloop{loop_R}x{loop_T}",
        "structured": True,
        "sampler": haar_ensemble,
        "beta": beta,
        "target": partial(rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    train_parameters = {
        # ReZero α and zero-init mlp.fc2 mean the gradient-flow unfreezing
        # cascade (fc2 → fc1 → α → Q/K/V/mix) is slow at lr=1e-3 — pushing the
        # LR up gets training past the identity-branch stall in a few epochs.
        "lr": 1e-2,
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "checkpoint_path": "best_gelt.pth",
    }

    # Debug-capacity GELT for the per-site Wilson loop target. The 2×2 loop is
    # quartic in plaquettes (W = P·P·P·P in Z₂), so the model needs depth and
    # head count to compose multi-site products. The earlier matched-capacity
    # config (nhead=2, d_qkv=8, gemhsa_layers=2, mlp_hidden=16) was sized for
    # the linear-in-P action target and is too small for Wilson loops — leave
    # the matched shootout for after the per-site path is validated.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 4,
        "gemhsa_layers": 3,
        "d_qkv": 16,
        "gate": "softplus",
        "dtype": torch.complex64,
        "mlp_hidden": 32,
        "mlp_out": 1,
        # Per-site target → no spatial reduction. Use "sum" for the Wilson
        # action, "mean" for the average ⟨W⟩.
        "reduction": "none",
        # Warm-start the ReZero α and drop the MLP fc2 zero-init: the default
        # α=0 + fc2=0 combo traps training at the constant-mean predictor on
        # the 2×2 Wilson loop target. α=0.5 puts the multiplicative path at
        # ~half the residual stream (α=0.05 left it at ~4% and the MLP just
        # kept reading the raw plaquette at site x, never using the multi-
        # site contribution). init_scale=10 lifts score magnitudes off the
        # near-uniform softmax floor so attention has per-offset signal to
        # learn from on epoch 0.
        "alpha_init": 0.5,
        "init_scale": 10.0,
        "mlp_zero_init": False,
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
        f"GELT | params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} | T {tuple(T.shape)} {T.dtype} | "
        f"out {tuple(model(X, T).shape)}"
    )

    model = model.to(device)

    if args.profile:
        print_gpu_profile_header(model, train_loader, device)
        profile_batch = next(iter(train_loader))
        profile_training_step(model, profile_batch, criterion, optimizer, device)
        profile_gemhsa_sections(model, profile_batch, device)

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
        profile_every=args.profile_every if args.profile else 0,
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

    # Flatten per-site targets/predictions for the scatter; subsample if dense
    # so matplotlib stays responsive (per-site targets give |Λ| points per
    # config, e.g. 8³·N_test ≈ 150 k points at L=8, D=3).
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
