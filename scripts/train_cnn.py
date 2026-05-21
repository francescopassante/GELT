from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import LatticeCNN, haar_ensemble
from gelt.lattice import rectangular_wilson_loop

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
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            batch_size = targets.shape[0]
            test_loss += loss.item() * batch_size
            test_count += batch_size
            if save_outputs:
                all_targets.append(targets.cpu())
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_size = targets.shape[0]
            train_loss += loss.item() * batch_size
            train_count += batch_size

        train_loss /= train_count
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device, progress=False)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_bar.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if epochs_no_improve >= patience:
            epoch_bar.write(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses, epoch + 1


if __name__ == "__main__":
    torch.manual_seed(0)
    from gelt import SU, Z2, build_plaquette_datasets

    D = 3
    L = 8
    gaugegroup = Z2()

    beta = 1
    # Same per-site Wilson loop target as scripts/train_gelt.py, so the CNN
    # baseline and GELT are trained against the identical regression problem.
    loop_R, loop_T, mu, nu = 2, 2, 0, 1
    dataset_parameters = {
        "N": 1000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "R": None,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "prefix": f"z2_plaquette_L{L}_D{D}_N2000_beta{beta}_wloop{loop_R}x{loop_T}",
        "structured": False,
        "sampler": haar_ensemble,
        "beta": beta,
        "target": partial(rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    train_parameters = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 300,
        "patience": 30,
        "checkpoint_path": "best_model.pth",
    }

    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )

    # Derive in_channels from the data so it stays in sync with structured/dtype/group:
    # for SU(N) with structured=False, flatten_color yields 2 · n_pairs · nc² channels.
    in_channels = train_dataset[0][0].shape[0]
    # NOTE: matched-capacity numbers below were calibrated for the global FC
    # head (per-config target). With reduction="none" the head becomes two 1×1
    # convs whose parameter count is independent of L^D, so the CNN's total
    # numel is far smaller than the original 1678. Re-tune hidden_channels /
    # fc_hidden once the per-site comparison is the canonical benchmark.
    model = LatticeCNN(
        L,
        D,
        in_channels=in_channels,
        hidden_channels=[16, 32],
        kernel_size=3,
        fc_hidden=2,
        reduction="none",
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_parameters["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )

    # Standardize the target (notes/architecture.html §6.1). Same recipe as
    # scripts/train_gelt.py so the two models train against an identically
    # scaled regression problem and R² is directly comparable. Mutates the
    # shared full-y tensor in place — all three subsets share it.
    y_train = torch.cat([batch[-1] for batch in train_loader])
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    train_dataset.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)
    print(f"target scaler fit: μ_y = {mu_y.item():.4f} | σ_y = {sigma_y.item():.4f}")

    X, y = next(iter(train_loader))
    n_params = sum(p.numel() for p in model.parameters())
    n_real_dofs = sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )
    print(
        f"CNN | params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
        f"X {tuple(X.shape)} {X.dtype} "
        f"out {model(X).shape}"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
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
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnn_loss.png", dpi=150, bbox_inches="tight")
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
    plt.savefig("cnn_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
