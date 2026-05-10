import torch
from torch.utils.data import TensorDataset, random_split

from lattice import Z2, Lattice


def build_plaquette_datasets(N, D, L, split_percent=[0.7, 0.15, 0.15], save=False):
    """Build train, val, test datasets with X = [N, D*(D-1)/2, L, .., L] plaquette configuration and y = [N] action"""
    n_pairs = D * (D - 1) // 2
    X = torch.zeros((N, n_pairs) + (L,) * D)
    y = torch.zeros(N)

    for i in range(N):
        lat = Lattice(L=L, D=D, gaugegroup=Z2()).initialize_random_links()
        plaq = lat.plaquette_tensor()
        X[i] = plaq
        y[i] = lat.action(plaq)

    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, split_percent)

    if save:
        torch.save(train_dataset, "datasets/train_dataset_plaquette.pt")
        torch.save(val_dataset, "datasets/val_dataset_plaquette.pt")
        torch.save(test_dataset, "datasets/test_dataset_plaquette.pt")

    return train_dataset, val_dataset, test_dataset


def build_link_datasets(N, D, L, split_percent=[0.7, 0.15, 0.15], save=False):
    """Build train, val, test datasets with X = [N, D, L, .., L] link configuration and y = [N] action"""
    n_links = D
    X = torch.zeros((N, n_links) + (L,) * D)
    y = torch.zeros(N)

    for i in range(N):
        lat = Lattice(L=L, D=D, gaugegroup=Z2()).initialize_random_links()
        X[i] = lat.link_tensor()
        y[i] = lat.action()

    full_dataset = TensorDataset(X, y)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, split_percent)

    if save:
        torch.save(train_dataset, "datasets/train_dataset_link.pt")
        torch.save(val_dataset, "datasets/val_dataset_link.pt")
        torch.save(test_dataset, "datasets/test_dataset_link.pt")

    return train_dataset, val_dataset, test_dataset
