"""Classical 0⁺⁺ glueball spectroscopy on an SU(N) ensemble.

This is the validation baseline of notes/glueball_spectroscopy.md §6.1: build a
spatial, gauge-invariant scalar operator, project it to zero momentum on each
timeslice, form the connected (vacuum-subtracted) temporal correlator, and read
off the effective mass plateau. The whole point is to learn — before any
network is involved — whether the ensemble can resolve a mass at all.

Spatial APE smearing (§7) is included because thin-link operators overlap too
poorly onto the ground state for the plateau to be reachable at small Δ where
the signal-to-noise still lives.

Conventions
-----------
- Ensembles are batched links ``(B, D, *Λ, nc, nc)`` (as elsewhere).
- Time is lattice axis 0; directions 1..D-1 are spatial. The operator is built
  only from spatial-plane Wilson loops, and smearing touches spatial links
  only — never time, or the transfer-matrix interpretation breaks.
"""

from typing import Tuple

import torch
from tqdm import tqdm

from .lattice import GaugeGroup, rectangular_wilson_loop
from .sampler import staple_sum


def ape_smear(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    alpha: float = 0.5,
    n_steps: int = 1,
    progress: bool = False,
) -> torch.Tensor:
    """Spatial APE smearing of an ensemble (time = axis 0).

    Each spatial link is replaced by the group projection of
        V_μ(x) = (1 − α) U_μ(x) + (α / n_staples) Σ spatial staples,
    iterated ``n_steps`` times. Time links are left untouched, and only spatial
    staples enter, so the result is gauge covariant and the time axis is clean.

    ``staple_sum`` returns the *action* staple A_μ(x), which runs from x+μ̂ back
    to x (it transforms as Ω(x+μ̂)·A·Ω†(x)). APE adds the open path that runs
    from x to x+μ̂ — i.e. its dagger, which transforms like the link itself —
    so that V is gauge covariant.

    Parameters
    ----------
    U : ``(B, D, *Λ, nc, nc)`` batched links.
    alpha : staple weight (0 = no smearing).
    n_steps : number of smearing iterations.
    progress : show a tqdm bar over the (n_steps × B) per-config smear updates
        — the serial batch loop is slow, so a bar is useful on large ensembles.

    Returns
    -------
    Smeared links of the same shape.
    """
    D = U.shape[1]
    spatial = list(range(1, D))  # time is axis 0
    if len(spatial) < 2:
        raise ValueError("Spatial smearing needs at least two spatial directions.")
    n_staples = 2 * (len(spatial) - 1)

    out = U.clone()
    with tqdm(
        total=n_steps * out.shape[0],
        desc="APE smearing",
        disable=not progress,
        leave=False,
    ) as bar:
        for _ in range(n_steps):
            new = out.clone()
            for b in range(out.shape[0]):
                for mu in spatial:
                    staples = gaugegroup.dagger(
                        staple_sum(out[b], mu, gaugegroup, nu_dirs=spatial)
                    )
                    V = (1 - alpha) * out[b, mu] + (alpha / n_staples) * staples
                    new[b, mu] = gaugegroup.project(V)
                bar.update(1)
            out = new
    return out


def glueball_operator(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    R: int = 1,
    T: int = 1,
) -> torch.Tensor:
    """Per-site scalar 0⁺⁺ operator: sum of spatial R×T Wilson loops (time = axis 0).

    Summing over the spatial planes (μ < ν, both spatial) makes the operator a
    rotational scalar; at R = T = 1 each term is a spatial plaquette.

    Parameters
    ----------
    U : ``(B, D, *Λ, nc, nc)`` batched links.

    Returns
    -------
    ``(B, *Λ)`` real field O(x).
    """
    D = U.shape[1]
    spatial = list(range(1, D))  # time is axis 0
    # Start from a scalar 0; the first Wilson loop broadcasts it to (B, *Λ).
    O = 0.0
    for i, mu in enumerate(spatial):
        for nu in spatial[i + 1 :]:
            O = O + rectangular_wilson_loop(U, gaugegroup, R, T, mu, nu)
    return O


def zero_momentum(O: torch.Tensor) -> torch.Tensor:
    """Project the operator field to zero momentum: sum over spatial sites.

    ``O`` has shape ``(B, *Λ)`` with time = lattice axis 0, i.e. tensor index 1
    (index 0 is the batch). The spatial sites are the remaining indices 2…D.

    Returns
    -------
    ``(B, Nt)`` timeslice operator Ō(t) per config.
    """
    spatial_axes = tuple(range(2, O.dim()))
    return O.sum(dim=spatial_axes)


def connected_correlator(Obar: torch.Tensor) -> torch.Tensor:
    """Connected, vacuum-subtracted temporal correlator C(Δ).

    ``Obar`` has shape ``(B, Nt)``. The vacuum ⟨Ō⟩ is subtracted using the mean
    over all configs and timeslices (0⁺⁺ has a nonzero VEV, so this is
    essential), and C(Δ) is averaged over the batch and over every time origin
    (time-translation invariance, periodic in time).

    Returns
    -------
    ``(Nt,)`` with ``C[Δ] = ⟨ d(t+Δ) d(t) ⟩``, d = Ō − ⟨Ō⟩.
    """
    d = Obar - Obar.mean()
    Nt = d.shape[1]
    return torch.stack([(d.roll(-dt, dims=1) * d).mean() for dt in range(Nt)])


def effective_mass(C: torch.Tensor) -> torch.Tensor:
    """Effective mass m_eff(Δ) = log[C(Δ) / C(Δ+1)] → m_G as Δ grows."""
    return torch.log(C[:-1] / C[1:])


def jackknife_effective_mass(
    Obar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jackknife mean and error of m_eff(Δ), resampling over configs.

    ``Obar`` has shape ``(B, Nt)``. Returns ``(mean, err)``, each of shape
    ``(Nt − 1,)``, with the standard leave-one-out jackknife error.
    """
    B = Obar.shape[0]
    samples = torch.stack(
        [
            effective_mass(connected_correlator(Obar[torch.arange(B) != i]))
            for i in range(B)
        ]
    )
    mean = samples.mean(dim=0)
    err = ((B - 1) / B * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean, err
