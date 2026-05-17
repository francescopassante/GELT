"""Single-(L, β) run of the CNN baseline on 2D Z₂ Metropolis data."""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from gelt import LatticeCNN, Z2, build_plaquette_datasets, mcmc_ensemble
from gelt.train import evaluate, train_model

if __name__ == "__main__":
    torch.manual_seed(0)

    D = 2
    L = 32

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
        "beta": 0.8,
        "n_therm": 200,
        "n_skip": 5,
        "dtype": torch.float32,
    }

    train_parameters = {
        "lr": 1e-4,
        "epochs": 400,
        "patience": 10,
        "checkpoint_path": f"best_model_L{L}.pth",
        "batch_size": 32,
    }

    train_dataset, val_dataset, test_dataset = build_plaquette_datasets(
        **dataset_parameters
    )

    in_channels = train_dataset[0][0].shape[0]
    model = LatticeCNN(
        L, D, in_channels=in_channels, hidden_channels=[16, 32], kernel_size=3
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

    # Population variance of the labels — the natural scale to normalise MSE by.
    test_label_var = all_targets.var(unbiased=False).item()
    test_r2 = 1.0 - test_loss / test_label_var if test_label_var > 0 else float("nan")

    print("L:            ", L)
    print("test_loss:    ", test_loss)
    print("var(y):       ", test_label_var)
    print("R²:           ", test_r2)
    print("epochs:       ", full_epochs)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.scatter(all_targets.numpy(), all_outputs.numpy(), alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predictions")
    plt.title("True vs Predicted Values (Test Set)")
    plt.grid(True)
    plt.show()
