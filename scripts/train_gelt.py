import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import functools

from gelt import haar_ensemble
from gelt.blocks import GELT
from gelt.lattice import rectangular_wilson_loop

# Output artifacts are grouped by study under results/; create the dirs the
# first time this runs in a fresh clone (they hold generated files only).
for _d in ("results/sampler", "results/glueball", "results/attention",
          "results/wilson_regression", "datasets"):
    os.makedirs(_d, exist_ok=True)


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
        for X, T, y in train_loader:
            X, T, y = X.to(device), T.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X, T)
            loss = criterion(outputs, y)
            loss.backward()
            # Clip before stepping: prevents a single bad batch from poisoning
            # Adam's second-moment buffer (a ~1e10 gradient permanently zeros
            # the effective LR via lr / (√v + ε)).
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        # Unfreeze-cascade diagnostic. With zero-init mlp.fc2, only the MLP
        # receives gradient on step 0; the GEMHSA params follow once fc2 grows.
        # If ‖fc2‖ stays near 0 while the loss is flat, training is stuck —
        # bump LR or drop the mlp.fc2 zero-init.
        if (epoch + 1) % 10 == 0 and hasattr(model, "gemhsa_models"):
            fc2_std = model.mlp.fc2.weight.detach().abs().mean().item()
            epoch_bar.write(f"  ep {epoch + 1:>3d}  |fc2|̄={fc2_std:.4f}")

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
    from gelt import Z2, build_plaquette_datasets, load_plaquette_datasets

    # ── The comparison of main.tex § "Validation and tests" ──────────────────
    # Per-site rectangular Wilson-loop regression, GELT against the CNN of
    # scripts/train_cnn.py. Every setting in this block that has a twin over
    # there (D, L, group, β, sampler, loop size, splits) is deliberately the
    # same value: the point of the figure is that the two architectures see the
    # *identical* regression problem and only the inductive bias differs.
    #
    # The 1×2 loop is the published one. It is the smallest target the CNN
    # cannot reach by summing its inputs — a plaquette is linear in the input
    # channels, a 1×2 loop is a product of links at two sites — and it is
    # exactly what GELT's bilinear value path is built to compose.
    D = 3
    L = 8
    gaugegroup = Z2()
    R = 2  # transport radius: the L1-ball the attention may reach into
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    beta = 1
    # Haar links (β is ignored by haar_ensemble): the regression is a statement
    # about representational reach, not about the ensemble — every plaquette is
    # an independent ±1, so there is no correlation for either model to exploit.
    loop_R, loop_T, mu, nu = 1, 2, 0, 1
    # Per-site Wilson-loop target: y has shape (B, *Λ). Paired with
    # ``reduction="none"`` on GELT the per-site readout is supervised directly —
    # every site is a sample, and the equivariant trace head outputs the locally
    # gauge-invariant quantity at x.
    N = 1000
    dataset_parameters = {
        "N": N,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": True,
        "prefix": f"{gaugegroup}_L{L}_D{D}_N{N}_beta{beta}_R{R}_wloop{loop_R}x{loop_T}",
        "structured": True,
        "sampler": haar_ensemble,
        "beta": beta,
        "target": functools.partial(
            rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu
        ),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64,
    }

    train_parameters = {
        # The zero-init of mlp.fc2 makes the gradient-flow unfreezing cascade
        # (fc2 → fc1 → Q/K/V/mix) slow at lr=1e-3 — pushing the LR up gets
        # training past the identity-branch stall in a few epochs.
        "lr": 3e-3,
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "checkpoint_path": "results/wilson_regression/best_gelt.pth",
    }

    # Small GELT for the per-site Wilson-loop target: ~1k parameters against the
    # CNN's ~500k, which is the point of the figure. The 1×2 loop is a product
    # of plaquettes at neighbouring sites, so the model needs at least one
    # transport + one bilinear value path; the depth here is slack for the
    # softmax self-selection and the L-Act gate, not an algebraic requirement.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        # One head, four layers: depth is what composes the loop (each block
        # roughly doubles the reachable loop length through the bilinear value
        # path), head count is not. This lands at ~1.5k parameters.
        "nhead": 1,
        "gemhsa_layers": 4,
        # d_qkv ≥ 2D is REQUIRED: RoPE assigns pair p to axis p % D, so anything
        # below 2D leaves whole axes on the identity rotation (README caveat).
        "d_qkv": 6,
        "gate": "softplus",
        # Z2 can run as a real model. SU(N) must stay complex; otherwise
        # GELT.forward would cast complex plaquettes/transports down to real.
        "dtype": model_dtype,
        "mlp_hidden": 4,
        "mlp_out": 1,
        # Per-site target → no spatial reduction. Use "sum" for the Wilson
        # action, "mean" for the average ⟨W⟩.
        "reduction": "none",
        # init_scale controls σ_V (value path — kept small so the residual
        # stream is near-identity at init); qk_init_scale controls σ_QK (score
        # channel) and is decoupled so the softmax can have real dynamic range
        # from epoch 0 without inflating the value path.
        "init_scale": 10,
        "qk_init_scale": 1.0,
        "mlp_zero_init": True,
        # Widen the residual stream beyond the plaquette channel count
        # D(D-1)/2 ∈ {1, 3, 6} via the front-end ChannelLift, so intermediate
        # layers don't collapse to 1–6 channels.
        "d_model": 6,
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
    # scatter plot in physical (un-standardized) Wilson-loop units.
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
    plt.savefig("results/wilson_regression/gelt_loss.png", dpi=150, bbox_inches="tight")
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
    plt.savefig("results/wilson_regression/gelt_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
