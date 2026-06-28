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


# ── Multi-operator variational analysis (GEVP) ────────────────────────────────
# A single smeared operator overlaps the 0⁺⁺ ground state too poorly to plateau
# before the signal drowns (notes/glueball_spectroscopy.md §5, §7). The standard
# fix (Morningstar–Peardon) is a *variational basis* of operators at several
# smearing levels: form the correlator matrix C_ij(Δ) and solve the generalized
# eigenvalue problem (GEVP), which isolates each state and pulls the ground-state
# plateau to small Δ where the signal-to-noise still lives.


def smearing_operator_basis(
    configs: torch.Tensor,
    gaugegroup: GaugeGroup,
    levels,
    alpha: float = 0.5,
    R: int = 1,
    T: int = 1,
    progress: bool = False,
) -> torch.Tensor:
    """Zero-momentum 0⁺⁺ operators at several APE smearing levels (a GEVP basis).

    ``levels`` is an iterable of *cumulative* smearing-step counts, e.g.
    ``[0, 2, 4, 6]`` (0 = thin links). Smearing is applied incrementally, so
    each level reuses the previous level's links rather than re-smearing from
    scratch. Each level contributes one zero-momentum operator row.

    Parameters
    ----------
    configs : ``(B, D, *Λ, nc, nc)`` batched links.
    levels : iterable of cumulative APE step counts (sorted internally).

    Returns
    -------
    ``(n_levels, B, Nt)`` stack of zero-momentum operators, one row per level.
    """
    levels = sorted(levels)
    obars = []
    U = configs
    done = 0
    for lvl in levels:
        if lvl > done:
            U = ape_smear(
                U, gaugegroup, alpha=alpha, n_steps=lvl - done, progress=progress
            )
            done = lvl
        obars.append(zero_momentum(glueball_operator(U, gaugegroup, R=R, T=T)))
    return torch.stack(obars)


def connected_correlator_matrix(Obar: torch.Tensor) -> torch.Tensor:
    """Connected, vacuum-subtracted correlator matrix C_ij(Δ) for a basis.

    ``Obar`` has shape ``(n_ops, B, Nt)``. Each operator's VEV is subtracted
    separately (0⁺⁺ has a nonzero VEV per operator), and C is averaged over the
    batch and every time origin (periodic in time):

        C_ij(Δ) = ⟨ d_i(t+Δ) d_j(t) ⟩,   d_i = Ō_i − ⟨Ō_i⟩.

    The raw estimator is not symmetric in (i, j) for Δ ≠ 0; the true matrix is,
    so callers symmetrise (the GEVP solver does). Returns ``(Nt, n_ops, n_ops)``.
    """
    d = Obar - Obar.mean(dim=(1, 2), keepdim=True)
    n_ops, B, Nt = d.shape
    C = torch.empty(Nt, n_ops, n_ops, dtype=d.dtype)
    for dt in range(Nt):
        C[dt] = torch.einsum("ibt,jbt->ij", d.roll(-dt, dims=2), d) / (B * Nt)
    return C


def gevp_eigenvalues(C: torch.Tensor, t0: int = 1, eps: float = 1e-12) -> torch.Tensor:
    """Generalized eigenvalues λ_n(Δ) of the GEVP  C(Δ) v = λ C(t0) v.

    ``C`` has shape ``(Nt, n_ops, n_ops)``. The reference matrix C(t0) is a
    covariance at the reference time and is *ideally* positive definite, but
    low statistics or near-collinear basis operators can make the noisy
    estimate indefinite. We therefore whiten with its symmetric eigendecomposition
    C(t0) = Q diag(s) Qᵀ, flooring the eigenvalues at ``eps · s_max`` (so a
    Cholesky on a marginally non-PD matrix does not crash), and diagonalise the
    symmetric W⁻¹ C(Δ) W⁻ᵀ with W = Q diag(√s). Eigenvalues are returned
    **descending** (column 0 = largest = ground state) of shape ``(Nt, n_ops)``.

    λ_n(Δ) ≈ e^{−m_n (Δ − t0)}, so the t0 offset cancels in the effective mass.
    The per-Δ descending sort only tracks states consistently for Δ ≥ t0 (at
    Δ < t0 the ordering inverts); read masses off the Δ ≥ t0 region.
    """
    Nt = C.shape[0]
    C = 0.5 * (C + C.transpose(-1, -2))  # symmetrise the noisy estimator
    s, Q = torch.linalg.eigh(C[t0])  # ascending eigenvalues; C[t0] = Q diag(s) Qᵀ
    s = s.clamp_min(eps * s[-1].clamp_min(eps))  # floor near-zero / negative modes
    W = Q * s.rsqrt()  # columns scaled: Wᵀ C[t0] W = I
    lams = []
    for dt in range(Nt):
        M = W.transpose(-1, -2) @ C[dt] @ W
        M = 0.5 * (M + M.transpose(-1, -2))
        lams.append(torch.linalg.eigvalsh(M).flip(-1))  # eigvalsh ascending → flip
    return torch.stack(lams)


def gevp_effective_mass(lams: torch.Tensor) -> torch.Tensor:
    """Per-state effective mass m_n(Δ) = log[λ_n(Δ)/λ_n(Δ+1)] from GEVP eigenvalues.

    ``lams`` is ``(Nt, n_ops)`` (descending, column 0 = ground state); returns
    ``(Nt − 1, n_ops)``. Meaningful for Δ ≥ t0 (see :func:`gevp_eigenvalues`).
    """
    return torch.log(lams[:-1] / lams[1:])


def jackknife_gevp_effective_mass(
    Obar: torch.Tensor, t0: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Jackknife mean/err of the GEVP effective masses, resampling over configs.

    ``Obar`` has shape ``(n_ops, B, Nt)``. Returns ``(mean, err)`` each of shape
    ``(Nt − 1, n_ops)`` (column 0 = ground state). Noise can push high-Δ
    eigenvalues non-positive (→ nan in the log); callers should mask non-finite
    entries when plotting / fitting.
    """
    B = Obar.shape[1]
    samples = torch.stack(
        [
            gevp_effective_mass(
                gevp_eigenvalues(
                    connected_correlator_matrix(Obar[:, torch.arange(B) != i]), t0=t0
                )
            )
            for i in range(B)
        ]
    )
    mean = samples.mean(dim=0)
    err = ((B - 1) / B * ((samples - mean) ** 2).sum(dim=0)).sqrt()
    return mean, err
