import torch
from torch.utils.data import TensorDataset, random_split

from lattice import Z2, Lattice


def build_datasets(N, D, L, split_percent = [0.7,0.15,0.15],save=False):
    channels = D*(D-1)//2 # number of plaquettes at each site
    X = torch.zeros((N,channels) + (L,) * D)
    y = torch.zeros(N)

    for i in range(N):
        lat = Lattice(L=L, D=D, gaugegroup=Z2()).initialize_random_links()
        X[i] = lat.plaquette_tensor()
        y[i] = lat.action()

    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, split_percent)

    # save datasets to file
    if save:
        torch.save(train_dataset, 'datasets/train_dataset.pt')
        torch.save(val_dataset, 'datasets/val_dataset.pt')
        torch.save(test_dataset, 'datasets/test_dataset.pt')

    return train_dataset, val_dataset, test_dataset
