from model import LatticeCNN
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm



def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
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
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

if __name__ == "__main__":
    # Example usage (assuming train_loader and val_loader are defined):
    L = 30
    D = 2
    N = 1000
    epochs = 100

    from data import build_datasets
    train_dataset, val_dataset, test_dataset = build_datasets(N, D, L, [0.7, 0.15, 0.15])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    channels = D*(D-1)//2
    model = LatticeCNN(L, [channels, 16, 32])
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs)

    # Load the best model
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    test_loss = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    all_targets = torch.cat(all_targets).numpy()
    all_outputs = torch.cat(all_outputs).numpy()

    # To plot the losses, make sure to capture the output of train_model:
    # train_losses, val_losses = train_model(...)

    import matplotlib.pyplot as plt
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets, all_outputs, alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('True vs Predicted Values (Test Set)')
        plt.grid(True)
        plt.show()
    except NameError:
        print("Note: To plot losses, assign the output of train_model to train_losses and val_losses.")
