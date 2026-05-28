from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, random_split
from tqdm import tqdm

from gelt import SU, Z2, haar_ensemble
from gelt.lattice import plaquette_tensor, rectangular_wilson_loop
from gelt.lcnn import LCNN, build_axis_transports

"""
========================================================================================
 Minimal training script for the Favoni et al L-CNN.
 Mirrors scripts/train_gelt.py: same evaluate / train_model loop, same dataset
 split, same target standardisation, same plotting. The only architectural
 difference is the transport input — L-CNN expects axis-aligned link
 products ``U^(k)_μ(x)`` rather than the L1-ball shortest-path-averaged T.
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
            # Adam's second-moment buffer.
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

        if epochs_no_improve >= patience:
            epoch_bar.write(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_val_loss == float("inf"):
        raise RuntimeError(
            f"No checkpoint written to {checkpoint_path}: val_loss never improved "
            f"over inf. Likely NaN/Inf losses — check the training output."
        )

    return train_losses, val_losses, epoch + 1


def build_lcnn_datasets(
    N: int,
    D: int,
    L: int,
    gaugegroup,
    K: int,
    target,
    sampler,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    splits=(0.7, 0.15, 0.15),
    dtype: torch.dtype = torch.complex64,
):
    """Sample configs, build (plaquettes, axis-transports, target) triples,
    and return train/val/test splits in the same shape as
    :func:`gelt.data.build_plaquette_datasets`.

    Axis-aligned transports replace the L1-ball shortest-path average used by
    GELT — they are the right primitive for L-Conv's single-direction
    convolution kernel.
    """
    configs, _ = sampler(
        L, D, gaugegroup, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    plaq = plaquette_tensor(configs, gaugegroup)
    transports = build_axis_transports(configs, K, gaugegroup)
    y = target(configs, gaugegroup)

    if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Expected three split fractions summing to 1.0, got {splits}.")
    lengths = [int(s * N) for s in splits]
    for i in range(N - sum(lengths)):
        lengths[i % 3] += 1

    full = TensorDataset(plaq, transports, y)
    return random_split(full, lengths)


if __name__ == "__main__":
    torch.manual_seed(0)

    D = 2
    L = 8
    gaugegroup = SU(2)
    K = 2  # L-Conv kernel half-size (positive shifts k = 0 … K)
    model_dtype = torch.float32 if isinstance(gaugegroup, Z2) else torch.complex64

    beta = 1
    # Per-site Wilson loop target: y has shape (B, *Λ). Paired with
    # ``reduction="none"`` on LCNN, the model's per-site readout is supervised
    # directly — matches train_gelt.py.
    loop_R, loop_T, mu, nu = 3, 3, 0, 1

    dataset_parameters = {
        "N": 1000,
        "D": D,
        "L": L,
        "gaugegroup": gaugegroup,
        "K": K,
        "sampler": haar_ensemble,
        "beta": beta,
        "target": partial(rectangular_wilson_loop, R=loop_R, T=loop_T, mu=mu, nu=nu),
        "splits": [0.7, 0.15, 0.15],
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.complex64,
    }

    train_parameters = {
        "lr": 3e-3,
        "batch_size": 64,
        "epochs": 3000,
        "patience": 3000,
        "checkpoint_path": "best_lcnn.pth",
    }

    # Debug-capacity L-CNN — matches the depth/width used in train_gelt.py
    # for a fair side-by-side comparison on the same Wilson-loop target.
    model_parameters = {
        "gaugegroup": gaugegroup,
        "L": L,
        "D": D,
        "K": K,
        "c_hidden": 8,
        "n_layers": 4,
        "dtype": model_dtype,
        "mlp_hidden": 32,
        "mlp_out": 1,
        "reduction": "none",
        "use_l_act": True,
        "gate": "softplus",
    }

    train_dataset, val_dataset, test_dataset = build_lcnn_datasets(
        **dataset_parameters
    )

    # Standardise the target (notes/architecture.html §6.1) — same protocol as
    # train_gelt.py: μ/σ fit on the train split, mutate the shared full-y
    # tensor in place; val/test inherit the normalisation through the Subset
    # views. Denormalise at the end for plotting.
    y_train = train_dataset.dataset.tensors[-1][train_dataset.indices]
    mu_y = y_train.mean()
    sigma_y = y_train.std(unbiased=False).clamp_min(1e-12)
    train_dataset.dataset.tensors[-1].sub_(mu_y).div_(sigma_y)
    print(f"target scaler fit: μ_y = {mu_y.item():.4f} | σ_y = {sigma_y.item():.4f}")

    model = LCNN(**model_parameters)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_parameters["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        f"LCNN | params: {n_params:,} ({n_real_dofs:,} real DOFs) | "
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

    model.load_state_dict(
        torch.load(
            train_parameters["checkpoint_path"], map_location=device, weights_only=True
        )
    )

    test_loss, all_targets, all_outputs = evaluate(
        model, test_loader, criterion, device, save_outputs=True
    )

    # Plots and visualisations — same protocol as train_gelt.py.
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
    plt.title("L-CNN Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("lcnn_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

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
    plt.title("L-CNN: True vs Predicted Values (Test Set)")
    plt.grid(True)
    plt.savefig("lcnn_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()

    results = {
        "test_loss": test_loss,
        "test_label_var": test_label_var,
        "test_r2": test_r2,
        "epochs": full_epochs,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
