import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import haar_ensemble, mcmc_ensemble
from gelt.blocks_rope import GELT
from gelt.lattice import topological_charge_density

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
    from gelt import SU, Z2, build_plaquette_datasets, load_plaquette_datasets

    # Topological charge density q_x is defined only in D=4 (the ε_μνρσ needs
    # four directions) and is non-trivial only for non-abelian SU(N≥2); SU(2)
    # is the minimal physically meaningful case. L=4 keeps the 4D ensemble
    # tractable on a laptop (4⁴ = 256 sites/config).
    D = 4
    L = 4
    gaugegroup = SU(2)
    R = 1
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    beta = 1
    # Per-site topological charge density target: q_x has shape (B, *Λ). Paired
    # with ``reduction="none"`` on GELT, the model's per-site readout is
    # supervised directly — every site contributes a sample, and the equivariant
    # trace head outputs the locally gauge-invariant quantity at x.
    N = 1000
    dataset_parameters = {
        "N": N,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": R,
        "splits": [0.7, 0.15, 0.15],
        "save": True,
        "prefix": f"{gaugegroup}_L{L}_D{D}_N{N}_beta{beta}_R{R}_topo",
        "structured": True,
        "sampler": mcmc_ensemble,
        "beta": beta,
        "target": topological_charge_density,
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
        "checkpoint_path": "best_gelt.pth",
    }

    # Debug-capacity GELT for the per-site topological charge density target.
    # q_x is *quadratic* in the on-site plaquettes (a single matrix bilinear
    # Tr[F_μν F_ρσ], F = (P−P†)/2i), so in principle one GEMHSA value path
    # suffices — the depth/head count here is generous slack for the softmax
    # self-selection and the L-Act gate, not an algebraic requirement. Note
    # H·d_qkv ≥ 3 is needed to hold the three dual-plane products.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "R": R,
        "nhead": 1,
        "gemhsa_layers": 4,
        "d_qkv": 4,
        "gate": "softplus",
        # Z2 can run as a real model. SU(N) must stay complex; otherwise
        # GELT.forward would cast complex plaquettes/transports down to real.
        "dtype": model_dtype,
        "mlp_hidden": 3,
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
        # Widen the residual-stream beyond the small plaquette channel count
        # D(D-1)/2 ∈ {1, 3, 6} via the front-end ChannelLift. Decouples the
        # GEMHSA working width from the input dimensionality so intermediate
        # layers don't collapse to 1–6 channels. In D=4 the plaquette input is
        # already 6 channels, so d_model must be ≥ 6.
        "d_model": 8,
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
    # scatter plot in physical (un-standardized) topological-charge-density units.
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
