import numpy as np
from tqdm import tqdm

from train import full_pipeline

if __name__ == "__main__":
    D = 2
    N = 1000
    in_channels = D
    channel_dimensions = [in_channels, 16, 32]
    Ls = np.arange(4, 33, 4, dtype=np.int64)
    print(Ls)
    test_losses = np.zeros(len(Ls))
    train_epochs = np.zeros(len(Ls))
    for i, L in enumerate(tqdm(Ls)):
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
            input="links",
        )
        test_losses[i] = test_loss
        train_epochs[i] = train_epoch
    print(Ls)
    print(test_losses)
    print(train_epochs)
