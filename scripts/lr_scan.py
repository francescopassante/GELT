"""Learning-rate sweep for the CNN baseline at fixed L on 2D Z₂ data."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from gelt import LatticeCNN, Z2, build_plaquette_datasets, mcmc_ensemble
from gelt.train import evaluate, train_model

if __name__ == "__main__":
    seed = 0
    D = 2
    L = 8
    hidden_channels = [16, 32]
    lrs = np.logspace(-2, -5, 7)  # 1e-2 … 1e-5, seven points

    dataset_parameters = {
        "N": 1000,
        "D": D,
        "L": L,
        "gaugegroup": Z2(),
        "R": None,
        "splits": [0.7, 0.15, 0.15],
        "save": False,
        "structured": False,
        "sampler": mcmc_ensemble,
        "beta": 1.0,
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    train_parameters = {
        "epochs": 2000,
        "patience": 20,
        "batch_size": 32,
    }

    # The dataset doesn't depend on lr — build it once.
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )
    in_channels = train_dataset[0][0].shape[0]

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_parameters["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=train_parameters["batch_size"], shuffle=False
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    criterion = nn.MSELoss()

    test_losses = np.zeros(len(lrs))
    test_label_vars = np.zeros(len(lrs))
    test_r2s = np.zeros(len(lrs))
    train_epochs = np.zeros(len(lrs))
    train_losses_all = []
    val_losses_all = []

    for i, lr in enumerate(tqdm(lrs)):
        torch.manual_seed(seed)
        model = LatticeCNN(
            L, D, in_channels=in_channels, hidden_channels=hidden_channels, kernel_size=3
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=float(lr))
        checkpoint_path = f"best_model_lr{lr:.0e}.pth"

        train_losses, val_losses, full_epochs = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=train_parameters["epochs"],
            patience=train_parameters["patience"],
            checkpoint_path=checkpoint_path,
            verbose=False,
        )

        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        test_loss, all_targets, all_outputs = evaluate(
            model, test_loader, criterion, device, save_outputs=True
        )
        test_label_var = all_targets.var(unbiased=False).item()
        test_r2 = (
            1.0 - test_loss / test_label_var if test_label_var > 0 else float("nan")
        )

        test_losses[i] = test_loss
        test_label_vars[i] = test_label_var
        test_r2s[i] = test_r2
        train_epochs[i] = full_epochs
        train_losses_all.append(np.array(train_losses))
        val_losses_all.append(np.array(val_losses))

    print("lrs:          ", lrs)
    print("test_loss:    ", test_losses)
    print("var(y):       ", test_label_vars)
    print("R²:           ", test_r2s)
    print("epochs:       ", train_epochs)

    def _save(fig_name):
        plt.tight_layout()
        plt.savefig(fig_name)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.semilogx(lrs, test_r2s, marker="o")
    plt.axhline(1.0, color="g", ls=":", label="R² = 1 (perfect)")
    plt.xlabel("Learning rate")
    plt.ylabel("R² = 1 − MSE / Var(y)")
    plt.title("LR scan: test R² (CNN, plaquettes, L=8)")
    plt.grid(True, ls=":")
    plt.legend()
    _save("LR scan R2.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lr, train_l, val_l in zip(lrs, train_losses_all, val_losses_all):
        label = f"lr={lr:.0e}"
        axes[0].plot(train_l, label=label, alpha=0.8)
        axes[1].plot(val_l, label=label, alpha=0.8)

    for ax, title in zip(axes, ("Train loss", "Val loss")):
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.grid(True, ls=":")
        ax.legend(fontsize=8)

    plt.suptitle("LR scan: convergence curves (CNN, plaquettes, L=8)")
    _save("LR scan curves.png")
