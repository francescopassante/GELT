from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import Subset, TensorDataset, random_split

from gelt.lattice import (
    GaugeGroup,
    build_transport_average,
    plaquette_tensor,
)


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
    prefix: Optional[str] = None,
    transport_mode: str = "average",
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

    ``R`` : if given, the transport tensor is computed once per link config
    (from which the plaquettes were derived) and stored alongside ``X`` and ``y``.

    ``transport_mode`` : passed through to :func:`build_transport_average`.
        ``"average"`` (default) builds the shortest-path-averaged transport
        (rotation-symmetric, the architecture's default). ``"single"`` builds
        the single-canonical-path variant (rotation symmetry broken; useful
        for A/B testing whether path averaging dilutes a specific-path target
        like a rectangular Wilson loop).
    """
    if save and prefix is None:
        raise ValueError("prefix must be provided when save=True.")
    configs, _ = sampler(
        L, D, gaugegroup, beta, N, n_therm=n_therm, n_skip=n_skip, dtype=dtype
    )
    Ps = plaquette_tensor(configs, gaugegroup)
    X = Ps if structured else flatten_color(Ps)
    y = target(configs, gaugegroup)
    T = (
        build_transport_average(configs, R=R, gaugegroup=gaugegroup, mode=transport_mode)
        if R is not None
        else None
    )
    return split(X, y, splits, save, prefix=prefix, T=T)


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


def load_plaquette_datasets(prefix: str, datasets_dir: str = "datasets"):
    """Reload a ``(train, val, test)`` triple previously written by ``save=True``.

    Mirrors the return structure of :func:`build_plaquette_datasets`: the three
    splits are :class:`~torch.utils.data.Subset` views over a *single* shared
    ``TensorDataset``, so the in-place target standardization the train scripts
    do on the train split (``train.dataset.tensors[-1][train.indices]``)
    propagates to val/test exactly as it does on a freshly built dataset.
    """
    out_dir = Path(datasets_dir)
    order = ("train", "val", "test")
    splits = {}
    for name in order:
        path = out_dir / f"{name}_dataset_{prefix}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No saved dataset at {path}.")
        # Saved objects are TensorDataset pickles, not plain tensors.
        splits[name] = torch.load(path, weights_only=False)

    # Concatenate the splits back into one TensorDataset and re-derive contiguous
    # index ranges, restoring the shared-underlying-tensor property that
    # random_split gives a freshly built dataset.
    n_tensors = len(splits["train"].tensors)
    combined = [
        torch.cat([splits[name].tensors[i] for name in order]) for i in range(n_tensors)
    ]
    full = TensorDataset(*combined)

    n_train, n_val = len(splits["train"]), len(splits["val"])
    n_test = len(splits["test"])
    train = Subset(full, list(range(0, n_train)))
    val = Subset(full, list(range(n_train, n_train + n_val)))
    test = Subset(full, list(range(n_train + n_val, n_train + n_val + n_test)))
    return train, val, test


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
