import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import LatticeCNN


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    verbose=True,
    patience=5,
):
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        wrap = tqdm if verbose else lambda x: x
        for inputs, targets in wrap(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in wrap(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, val_losses, epoch + 1


def full_pipeline(
    L,
    D,
    N,
    channel_dimensions,
    splits=[0.7, 0.15, 0.15],
    lr=1e-3,
    epochs=100,
    patience=10,
    plots=False,
    verbose=True,
):

    from data import build_datasets

    train_dataset, val_dataset, test_dataset = build_datasets(N, D, L, splits)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    model = LatticeCNN(L, channel_dimensions)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, full_epochs = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        patience=patience,
        verbose=verbose,
    )

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        wrap = tqdm if verbose else lambda x: x
        for inputs, targets in wrap(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())

    test_loss /= len(test_loader)
    if verbose:
        print(f"Test Loss: {test_loss:.4f}")

    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs).numpy()

    # To plot the losses, make sure to capture the output of train_model:
    # train_losses, val_losses = train_model(...)

    if plots:
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
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_outputs, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        plt.title("True vs Predicted Values (Test Set)")
        plt.grid(True)
        plt.show()

    return test_loss, full_epochs


if __name__ == "__main__":
    import numpy as np

    D = 2
    N = 1000
    in_channels = D * (D - 1) // 2
    channel_dimensions = [in_channels, 16, 32]
    Ls = np.arange(4, 33, 4, dtype=np.int64)
    print(Ls)
    test_losses = np.zeros(len(Ls))
    train_epochs = np.zeros(len(Ls))
    for i, L in tqdm(enumerate((Ls))):
        test_loss, train_epoch = full_pipeline(
            L,
            D,
            N,
            channel_dimensions,
            splits=[0.7, 0.15, 0.15],
            lr=1e-3,
            epochs=300,
            patience=10,
            plots=False,
            verbose=True,
        )
        test_losses[i] = test_loss
        train_epochs[i] = train_epoch
    print(Ls)
    print(test_losses)
    print(train_epochs)
