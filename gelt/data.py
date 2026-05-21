from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import TensorDataset, random_split

from gelt.lattice import (
    GaugeGroup,
    action,
    build_transport_average,
    plaquette_tensor,
)


def flatten_color(U: torch.Tensor) -> torch.Tensor:
    """Flatten color dimensions of a batched tensor ``(B, D, *Λ, nc, nc)`` into ``(B, C, *Λ)``.

    Used for non-equivariant models only (breaks group structure).

    Real groups: ``C = D · nc²``.
    Complex groups: ``C = 2 · D · nc²`` (real and imaginary parts as separate channels).
    """
    B = U.shape[0]
    D = U.shape[1]
    spatial = U.shape[2:-2]
    nc = U.shape[-1]
    ndim_s = len(spatial)
    if torch.is_complex(U):
        re_im = torch.stack([U.real, U.imag], dim=2)  # (B, D, 2, *Λ, nc, nc)
        # Color axes sit after spatial; permute to (B, D, 2, nc, nc, *Λ) before
        # reshaping so each output channel is a pure (pair, re/im, row, col)
        # tuple and the spatial axes remain contiguous and un-mixed.
        perm = (0, 1, 2) + (ndim_s + 3, ndim_s + 4) + tuple(range(3, 3 + ndim_s))
        return re_im.permute(*perm).contiguous().reshape(B, D * 2 * nc * nc, *spatial)
    # Same fix for real tensors: (B, D, *Λ, nc, nc) → (B, D, nc, nc, *Λ) → (B, D·nc², *Λ).
    perm = (0, 1) + (ndim_s + 2, ndim_s + 3) + tuple(range(2, 2 + ndim_s))
    return U.permute(*perm).contiguous().reshape(B, D * nc * nc, *spatial)


def build_plaquette_datasets(
    N: int,
    D: int,
    L: int,
    gaugegroup: GaugeGroup,
    target: Callable,
    beta: float = 1.0,
    n_therm: int = 200,
    n_skip: int = 5,
    sampler=None,
    splits: Sequence[float] = (0.7, 0.15, 0.15),
    save: bool = False,
    dtype: torch.dtype = torch.float32,
    structured: bool = True,
    R: Optional[int] = None,
):
    """Dataset of (plaquette config, target), optionally with precomputed transports.

    ``target`` : callable with signature ``target(configs, gaugegroup) -> Tensor``.
        Use ``functools.partial`` to pre-bind any extra arguments, e.g.::

            from functools import partial
            from gelt.lattice import action, rectangular_wilson_loop
            target = partial(action, beta=1.5)
            target = partial(rectangular_wilson_loop, R=2, T=3, mu=0, nu=1)

    ``sampler`` : ensemble-generator callable.

    ``structured=False``: X shape ``(N, n_pairs · nc², *Λ)`` — flattened color axes, for CNN.
    ``structured=True`` : X shape ``(N, n_pairs, *Λ, nc, nc)`` — full matrix layout, for GELT.

    ``R`` : if given, the shortest-path-averaged transport tensor is computed
    once per link config (from which the plaquettes were derived) and stored
    alongside ``X`` and ``y``.
    """
    configs, _ = sampler(
        L, D, gaugegroup, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    Ps = plaquette_tensor(configs, gaugegroup)
    X = Ps if structured else flatten_color(Ps)
    y = target(configs, gaugegroup)
    T = (
        build_transport_average(configs, R=R, gaugegroup=gaugegroup)
        if R is not None
        else None
    )

    prefix = dataset_prefix(
        gaugegroup.name.lower(), "plaquette", L, D, N, beta, dtype, structured, R
    )
    return split(X, y, splits, save, prefix=prefix, T=T)


def dataset_prefix(
    group_name: str,
    kind: str,
    L: int,
    D: int,
    N: int,
    beta: float,
    dtype: torch.dtype,
    structured: bool,
    R: Optional[int] = None,
) -> str:
    dtype_tag = str(dtype).replace("torch.", "")
    layout = "structured" if structured else "flat"
    base = f"{group_name}_{kind}_L{L}_D{D}_N{N}_beta{beta}_dtype{dtype_tag}_{layout}"
    return base if R is None else f"{base}_R{R}"


def split(X, y, splits, save, prefix, T: Optional[torch.Tensor] = None):
    if len(splits) != 3 or any(s <= 0 for s in splits):
        raise ValueError(f"Expected three positive split fractions, got {splits}.")
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same length, got {len(X)} and {len(y)}."
        )
    if T is not None and len(T) != len(y):
        raise ValueError(
            f"T and y must have the same length, got {len(T)} and {len(y)}."
        )
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {splits}.")

    n_samples = len(y)
    lengths = [int(split * n_samples) for split in splits]
    for i in range(n_samples - sum(lengths)):
        lengths[i % len(lengths)] += 1
    if any(length == 0 for length in lengths):
        raise ValueError(
            f"Dataset with N={n_samples} is too small for non-empty splits {splits}; "
            f"computed split lengths {tuple(lengths)}."
        )

    tensors = (X, y) if T is None else (X, T, y)
    full = TensorDataset(*tensors)
    train, val, test = random_split(full, lengths)

    if save:
        out_dir = Path("datasets")
        out_dir.mkdir(exist_ok=True)

        def _subset(idxs):
            return TensorDataset(*(t[idxs] for t in tensors))

        torch.save(_subset(train.indices), out_dir / f"train_dataset_{prefix}.pt")
        torch.save(_subset(val.indices), out_dir / f"val_dataset_{prefix}.pt")
        torch.save(_subset(test.indices), out_dir / f"test_dataset_{prefix}.pt")

    return train, val, test


if __name__ == "__main__":
    from gelt import SU, haar_ensemble

    from functools import partial

    beta = 1.0
    train, val, test = build_plaquette_datasets(
        N=100,
        D=3,
        L=5,
        gaugegroup=SU(3),
        beta=beta,
        target=partial(action, beta=beta),
        structured=True,
        sampler=haar_ensemble,
        R=3,
    )
    print(train[0])
