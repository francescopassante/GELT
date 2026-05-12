from lattice import Z2
from model import LatticeCNN
from train import full_pipeline

if __name__ == "__main__":
    D = 2
    N = 1000
    L = 32
    hidden_channels = [16, 32]

    model = LatticeCNN(int(L), D, in_channels=1, hidden_channels=hidden_channels)
    result = full_pipeline(
        L=L,
        D=D,
        N=N,
        model=model,
        group=Z2(),
        splits=(0.7, 0.15, 0.15),
        lr=1e-5,
        epochs=300,
        patience=10,
        plots=True,
        verbose=True,
        input="plaquettes",
        seed=0,
        checkpoint_path=f"best_model_L{int(L)}.pth",
    )
    test_loss = result["test_loss"]
    test_label_var = result["test_label_var"]
    test_r2 = result["test_r2"]
    train_epoch = result["epochs"]

    print("L:            ", L)
    print("test_loss:    ", test_loss)
    print("var(y):       ", test_label_var)
    print("R^2:          ", test_r2)
    print("epochs:       ", train_epoch)
