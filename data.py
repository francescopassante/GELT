from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import TensorDataset, random_split

from lattice import (
    GaugeGroup,
    action,
    as_ml_input,
    as_ml_plaquettes,
    plaquette_tensor,
)
from sampler import mcmc_ensemble


def build_link_datasets(
    N: int,
    D: int,
    L: int,
    group: GaugeGroup,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    sampler=None,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    dtype: torch.dtype = torch.float32,
    structured: bool = True,
):
    """Dataset of (link config, action).

    ``sampler`` : ensemble-generator callable with the same signature as
    ``mcmc_ensemble``.  Defaults to ``mcmc_ensemble`` (Metropolis MC).
    Pass ``sampler=haar_ensemble`` for Haar-uniform configurations.

    ``structured=True`` (default): X shape ``(N, D, *Λ, nc, nc)`` — full matrix layout, for G-GAT.
    ``structured=False``         : X shape ``(N, D · nc², *Λ)``    — flattened color axes, for CNN.
    """
    if sampler is None:
        sampler = mcmc_ensemble
    configs, _, _ = sampler(
        L, D, group, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    X = configs if structured else torch.stack([as_ml_input(c) for c in configs])
    y = torch.stack([action(c, group, beta=beta) for c in configs])

    prefix = _dataset_prefix(
        group.name.lower(), "link", L, D, N, beta, dtype, structured
    )
    return _split(X, y, splits, save, prefix=prefix)


def build_plaquette_datasets(
    N: int,
    D: int,
    L: int,
    group: GaugeGroup,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    sampler=None,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    dtype: torch.dtype = torch.float32,
    structured: bool = False,
):
    """Dataset of (plaquette config, action).

    ``sampler`` : ensemble-generator callable with the same signature as
    ``mcmc_ensemble``.  Defaults to ``mcmc_ensemble`` (Metropolis MC).
    Pass ``sampler=haar_ensemble`` for Haar-uniform configurations.

    ``structured=False`` (default): X shape ``(N, n_pairs · nc², *Λ)`` — flattened color axes, for CNN.
    ``structured=True``            : X shape ``(N, n_pairs, *Λ, nc, nc)`` — full matrix layout, for G-GAT.
    """
    if sampler is None:
        sampler = mcmc_ensemble
    configs, _, _ = sampler(
        L, D, group, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    Ps = torch.stack([plaquette_tensor(c, group) for c in configs])
    X = Ps if structured else torch.stack([as_ml_plaquettes(p) for p in Ps])
    y = torch.stack(
        [action(configs[i], group, beta=beta, plaquettes=Ps[i]) for i in range(N)]
    )

    prefix = _dataset_prefix(
        group.name.lower(), "plaquette", L, D, N, beta, dtype, structured
    )
    return _split(X, y, splits, save, prefix=prefix)


def _dataset_prefix(
    group_name: str,
    kind: str,
    L: int,
    D: int,
    N: int,
    beta: float,
    dtype: torch.dtype,
    structured: bool,
) -> str:
    dtype_tag = str(dtype).replace("torch.", "")
    layout = "structured" if structured else "flat"
    return f"{group_name}_{kind}_L{L}_D{D}_N{N}_beta{beta}_dtype{dtype_tag}_{layout}"


def _split(X, y, splits, save, prefix):
    if len(splits) != 3 or any(s <= 0 for s in splits):
        raise ValueError(f"Expected three positive split fractions, got {splits}.")

    full = TensorDataset(X, y)
    train, val, test = random_split(full, list(splits))

    if save:
        out_dir = Path("datasets")
        out_dir.mkdir(exist_ok=True)

        train_ds = TensorDataset(X[train.indices], y[train.indices])
        val_ds = TensorDataset(X[val.indices], y[val.indices])
        test_ds = TensorDataset(X[test.indices], y[test.indices])

        torch.save(train_ds, out_dir / f"train_dataset_{prefix}.pt")
        torch.save(val_ds, out_dir / f"val_dataset_{prefix}.pt")
        torch.save(test_ds, out_dir / f"test_dataset_{prefix}.pt")

    return train, val, test
