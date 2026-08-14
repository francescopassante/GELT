"""3D Ising model — the Kramers–Wannier–Wegner dual of 3D Z₂ gauge theory.

Why an Ising model lives in a lattice-gauge codebase
----------------------------------------------------
`notes/attention_as_operator.md` §6.1.2 leaves one question open, and the
headline turns on it: the trained attention field reads a correlation length
~21% *above* the classical smeared-plaquette operator, and that can only be
called a *candidate* for the true mass gap because "the classical operator is
not ground truth — it is one more contaminated operator". The note names the
price of settling it: an L = 32 gauge run, new ensembles *and* new checkpoints
at all five β, ≈40 h wall clock.

There is a cheaper and stronger reference, and it is exact. Wegner's duality
maps 3D Z₂ gauge theory onto the 3D Ising model with

    β* = −½ ln tanh β         (equivalently  sinh 2β · sinh 2β* = 1),

carries plaquettes of Λ onto bonds of the dual lattice Λ*, and sends the gauge
theory's **confined** phase (β < β_c) to the Ising **broken** phase
(β* > β*_c). Three things follow, and each is load-bearing:

1. *The mass gap is the same number.* The 0⁺⁺ glueball is the lightest state
   created by a Z₂-even, parity-even operator; in the dual that is the lightest
   Ising excitation. So ξ measured in the Ising model at β* **is** the ξ the
   gauge measurement is trying to estimate — not a proxy for it.
2. *The dual has an operator the gauge theory does not.* In the broken phase
   ⟨σ⟩ ≠ 0, so the order parameter itself interpolates the one-particle state
   with near-unit overlap. σ is the disorder ('t Hooft) operator on the gauge
   side: **non-local in the links, and therefore not a member of any
   smeared-loop variational basis.** The reference is genuinely from outside
   the gauge side's toolbox, which is exactly what "uncontaminated" has to mean
   here.
3. *It is free.* An Ising sweep is two `torch.roll` stencils; a gauge sweep is
   a staple sum over matrix-valued links. Statistics and volumes that the gauge
   ensembles cannot reach are minutes of GPU here — including the L = 48 run
   that answers §7's finite-volume question without a single new gauge sweep.

Conventions match the rest of the package: time is lattice axis 0, spatial
directions are 1..D−1, periodic BCs via `torch.roll`, `(R, *Λ)` batched over
independent replicas so a sweep is one vectorised kernel.

Operator dictionary (D = 3, time = axis 0)
------------------------------------------
A plaquette of Λ in the (μ,ν) plane pierces exactly one bond of Λ*, the one in
the perpendicular direction ρ. The glueball operator sums **spatial** plaquettes
— plane (1,2) — so its literal dual is the sum of **temporal** dual bonds
s(t,x)·s(t+1,x). `wall_observables` returns that as ``e_t``, alongside the
spatial-bond energy ``e_s`` and the wall magnetisation ``m``.

(A temporal dual bond sits at half-integer time. For a zero-momentum correlator
in t that is a constant shift of the time argument and cannot move the fitted
mass, so we label it by its lower endpoint and say no more about it.)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "dual_beta",
    "gauge_beta",
    "predicted_plaquette",
    "ISING_BETA_C",
    "GAUGE_BETA_C",
    "random_spins",
    "neighbour_sum",
    "heatbath_sweep",
    "bond_field",
    "mean_bond_energy",
    "wall_observables",
    "ising_measure",
]


# Ferrenberg, Xu & Landau, Phys. Rev. E 97, 043301 (2018): the 3D Ising
# critical coupling on the simple cubic lattice, β*_c = 0.221654626(5). Mapped
# through the duality below it gives the Z₂ gauge β_c to nine digits — the
# scripts' hardcoded 0.7614 is this number truncated.
ISING_BETA_C = 0.221654626


def dual_beta(beta: float) -> float:
    """Ising coupling dual to gauge coupling ``beta``: β* = −½ ln tanh β.

    Equivalent to the symmetric form sinh 2β · sinh 2β* = 1. Monotonically
    decreasing, so the gauge theory's weak-coupling (deconfined) side maps to
    the Ising high-temperature (symmetric) side and vice versa.
    """
    if beta <= 0.0:
        raise ValueError("duality is defined for β > 0")
    return -0.5 * math.log(math.tanh(beta))


def gauge_beta(beta_star: float) -> float:
    """Inverse of :func:`dual_beta` — the duality is an involution."""
    if beta_star <= 0.0:
        raise ValueError("duality is defined for β* > 0")
    return -0.5 * math.log(math.tanh(beta_star))


GAUGE_BETA_C = gauge_beta(ISING_BETA_C)  # 0.761413292…


def predicted_plaquette(beta: float, bond_energy: float) -> float:
    """Exact duality prediction for the gauge mean plaquette ⟨P⟩(β).

    Differentiating the duality relation between the two partition functions

        Z_gauge(β) = 2^{N_l−1} (cosh β)^{N_p} (tanh β)^{N_p/2} · Z_Ising(β*)

    with respect to β (using dβ*/dβ = −1/sinh 2β) gives, with N_p = N_bonds,

        ⟨P⟩(β) = tanh β + [1 − ⟨s_i s_j⟩(β*)] / sinh 2β.

    This is a **parameter-free prediction with no fitted quantity**, so it is
    the sharpest available end-to-end check on this module: the Ising sweep,
    the duality map and the derivation all have to be right simultaneously for
    it to reproduce a gauge ⟨P⟩ that was measured by completely separate code.

    A useful corollary, and a second check: expanding at small β with
    1 − ⟨ss⟩ ≈ 4 t⁶ (t = tanh β, the one-spin-flip term of the low-temperature
    series) reproduces the gauge strong-coupling series ⟨P⟩ ≈ t + 2t⁵, whose
    coefficient 2 counts the cubes containing a given plaquette in 3D.

    Parameters
    ----------
    beta : gauge coupling.
    bond_energy : ⟨s_i s_j⟩ per bond of the dual Ising model at β* = dual_beta(β).
    """
    return math.tanh(beta) + (1.0 - bond_energy) / math.sinh(2.0 * beta)


def _site_parity(shape: Sequence[int], device: torch.device) -> torch.Tensor:
    """Checkerboard parity (0 or 1) for each site. Shape: ``(*shape)``.

    Mirrors ``sampler._site_parity``; kept local so the Ising code has no
    dependency on the gauge sampler's internals.
    """
    coords = torch.meshgrid(
        *[torch.arange(s, device=device) for s in shape], indexing="ij"
    )
    return sum(coords) % 2


def random_spins(
    shape: Sequence[int],
    n_replicas: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    ordered: bool = False,
) -> torch.Tensor:
    """``(n_replicas, *shape)`` spins in {−1, +1}.

    ``ordered=True`` starts every replica at all-(+1) — the right cold start in
    the broken phase, where a hot start would have to tunnel into a
    magnetisation sector and waste thermalisation on it.
    """
    if ordered:
        return torch.ones((n_replicas, *shape), device=device, dtype=dtype)
    r = torch.randint(
        0, 2, (n_replicas, *shape), device=device, generator=generator,
        dtype=torch.int64,
    )
    return (2 * r - 1).to(dtype)


def neighbour_sum(s: torch.Tensor) -> torch.Tensor:
    """h(x) = Σ_μ [s(x+μ̂) + s(x−μ̂)] over all D lattice axes. Shape as ``s``.

    ``s`` is ``(R, *Λ)`` — axis 0 is the replica index and is never rolled.
    """
    h = torch.zeros_like(s)
    for ax in range(1, s.dim()):
        h = h + s.roll(-1, dims=ax) + s.roll(1, dims=ax)
    return h


def heatbath_sweep(
    s: torch.Tensor,
    beta: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One exact checkerboard heat-bath sweep. Returns a new tensor.

    The local weight is a two-point distribution, so the heat-bath is exact and
    rejection-free: with h the neighbour sum, the spin-dependent part of the
    energy is −β·s·h, hence

        P(s = +1) = e^{βh} / (e^{βh} + e^{−βh}) = σ(2βh),

    the same one-liner as :func:`gelt.sampler.z2_heatbath_sweep` for the gauge
    links. Same-parity sites share no bond, so their updates commute and each
    half-sweep needs one neighbour-sum evaluation.

    Rejection-free matters for the same reason it did on the gauge side: the
    Metropolis flip proposal's acceptance collapses as the system orders.

    **Cost of the slow mode — read this before choosing n_therm/n_skip.** The
    first version of this docstring argued that clustering was unnecessary
    because critical slowing down (τ ~ ξ^z, z ≈ 2) is only ~ξ² sweeps and
    ``n_skip`` plus many replicas would absorb it. That reasoning is wrong in
    the **broken** phase, and the 2026-08-14 production run paid for it. The
    slowest mode there is not the critical one, it is **tunnelling between the
    two magnetisation sectors**, whose barrier gives a τ that is exponential in
    the interface area rather than power-law in ξ. Measured on 48×24² at
    β = 0.760: ξ² ≈ 76 sweeps but τ_int(M) = **1575 sweeps**, a factor 20, with
    3.4% of consecutive measurements straddling a sign flip. Under-thermalising
    against *that* leaves the chain too ordered, which biases ⟨ss⟩ high and is
    directly visible as a failure of the duality check.

    So: **fix n_therm and n_skip from τ_int of the magnetisation, never from
    ξ^z or from τ_int of the energy** (the energy is fast — 3.4 vs 31.5
    measurements at the same point). `scripts/dual_ground_truth.py` reports
    both and gates on the magnetisation. The tunnelling barrier grows with
    volume, so the *larger* lattice is the better-behaved one here, which
    inverts the usual intuition.
    """
    shape = tuple(s.shape[1:])
    if any(L % 2 for L in shape):
        raise ValueError(
            f"checkerboard update needs even extents in every direction, got {shape}"
        )
    parity = _site_parity(shape, s.device)  # (*Λ)
    s = s.clone()
    for par in (0, 1):
        h = neighbour_sum(s)
        p_plus = torch.sigmoid(2.0 * beta * h)
        u = torch.rand(s.shape, device=s.device, dtype=s.dtype, generator=generator)
        drawn = torch.where(u < p_plus, 1.0, -1.0).to(s.dtype)
        s = torch.where(parity == par, drawn, s)
    return s


def bond_field(s: torch.Tensor, axis: int) -> torch.Tensor:
    """s(x)·s(x+ê_axis) for every site. ``axis`` is a lattice axis (1-based)."""
    return s * s.roll(-1, dims=axis)


def mean_bond_energy(s: torch.Tensor) -> torch.Tensor:
    """⟨s_i s_j⟩ averaged over all D·V bonds, per replica. Shape ``(R,)``.

    This is the quantity the duality relates to the gauge ⟨P⟩; see
    :func:`predicted_plaquette`.
    """
    D = s.dim() - 1
    tot = sum(bond_field(s, ax).flatten(1).mean(dim=1) for ax in range(1, D + 1))
    return tot / D


def wall_observables(s: torch.Tensor, time_axis: int = 0) -> Dict[str, torch.Tensor]:
    """Zero-momentum (summed over the slice) observables per timeslice.

    ``s`` is ``(R, *Λ)``; ``time_axis`` indexes Λ, so lattice axis
    ``time_axis + 1`` of the tensor. Returns ``(R, Nt)`` tensors:

    ``m``   wall magnetisation Σ_x s(t,x). **Sign-fixed per replica** by the
            global magnetisation, which is the standard restricted-ensemble
            treatment of the broken phase: the two sectors are exchanged by a
            global flip, so choosing one is exact up to tunnelling, and
            tunnelling is reported as a diagnostic rather than assumed absent.
            Sign-fixing is applied to the *wall array*, not to the spins.
    ``e_t`` temporal-bond energy Σ_x s(t,x)s(t+1,x) — the literal dual of the
            spatial-plaquette glueball operator (see module docstring).
    ``e_s`` spatial-bond energy, summed over the D−1 spatial directions.

    ``m`` and ``e`` couple to the same lightest state but with very different
    overlaps (σ is the order parameter; ε is a two-σ composite that inherits its
    one-particle piece from ⟨σ⟩ ≠ 0). Quoting both is the internal consistency
    check: they must give the same mass, and they do so from very different
    signal-to-noise.
    """
    D = s.dim() - 1
    t_ax = time_axis + 1
    spatial_axes = [a for a in range(1, D + 1) if a != t_ax]

    def wall(field: torch.Tensor) -> torch.Tensor:
        return field.sum(dim=spatial_axes)

    m = wall(s)
    sign = torch.sign(m.sum(dim=1, keepdim=True))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)

    e_t = wall(bond_field(s, t_ax))
    e_s = sum(wall(bond_field(s, ax)) for ax in spatial_axes)
    return {"m": m * sign, "e_t": e_t, "e_s": e_s}


def ising_measure(
    beta_star: float,
    shape: Sequence[int],
    n_replicas: int = 16,
    n_measure: int = 200,
    n_therm: int = 500,
    n_skip: int = 10,
    time_axis: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
    ordered_start: bool = True,
    progress: bool = False,
) -> Dict[str, torch.Tensor]:
    """Thermalise ``n_replicas`` chains, then collect wall observables.

    Returns a dict of ``(n_replicas * n_measure, Nt)`` observable arrays —
    already in the ``(B, Nt)`` layout that :func:`gelt.glueball.connected_correlator`
    and :func:`gelt.glueball.fit_cosh_correlator` expect, so the dual analysis
    runs through *exactly* the gauge-side code path — plus:

    ``bond_energy``   ``(B,)`` per-measurement ⟨s_i s_j⟩, for the duality check;
    ``magnetisation`` ``(B,)`` per-measurement mean spin **before** sign fixing,
                      so its sign changes count tunnelling events;
    ``chain``         ``(B,)`` replica index, so a blocked jackknife can block
                      within a chain rather than across chains.

    Measurements are ordered chain-major (all of replica 0, then replica 1, …)
    so that a contiguous block of the returned arrays is a contiguous stretch of
    one Markov chain.
    """
    gen = None
    if seed is not None:
        # A device generator keeps the stream independent of global RNG state,
        # but MPS has no generator of its own; fall back to the global seed
        # there rather than silently drawing on the wrong device.
        try:
            gen = torch.Generator(device=device or "cpu")
            gen.manual_seed(seed)
            torch.rand(1, device=device, generator=gen)
        except (RuntimeError, TypeError):
            gen = None
            torch.manual_seed(seed)

    s = random_spins(
        shape, n_replicas, device=device, dtype=dtype,
        generator=gen, ordered=ordered_start,
    )
    for _ in range(n_therm):
        s = heatbath_sweep(s, beta_star, generator=gen)

    walls: Dict[str, list] = {}
    energies, mags = [], []
    it = range(n_measure)
    if progress:
        try:
            from tqdm import tqdm

            it = tqdm(it, desc=f"β*={beta_star:.6f}", leave=False)
        except ImportError:
            pass
    for _ in it:
        for _ in range(n_skip):
            s = heatbath_sweep(s, beta_star, generator=gen)
        obs = wall_observables(s, time_axis=time_axis)
        for k, v in obs.items():
            # .cpu() before .double(): MPS has no float64.
            walls.setdefault(k, []).append(v.cpu().double())
        energies.append(mean_bond_energy(s).cpu().double())
        mags.append(s.flatten(1).mean(dim=1).cpu().double())

    # stack → (n_measure, R, …) → transpose to chain-major → flatten
    out: Dict[str, torch.Tensor] = {}
    for k, v in walls.items():
        out[k] = torch.stack(v).transpose(0, 1).reshape(-1, v[0].shape[-1])
    out["bond_energy"] = torch.stack(energies).transpose(0, 1).reshape(-1)
    out["magnetisation"] = torch.stack(mags).transpose(0, 1).reshape(-1)
    out["chain"] = (
        torch.arange(n_replicas).repeat_interleave(n_measure).to(torch.int64)
    )
    return out
