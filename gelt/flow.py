"""Wilson (gradient) flow for SU(N) links.

The flow is the deterministic smoother
``V̇_t(x,μ) = −g₀² ∂_{x,μ} S_W[V_t] · V_t(x,μ)``, ``V_0 = U``, whose effect at
flow time ``t`` is to average the field over a radius ``r_sm = √(8t)``.  On the
lattice the drift is read straight off the staple: with ``A_μ(x)`` the staple sum
in this repository's convention (so that the local action is
``S_local = −(β/nc) Re Tr[U_μ(x) A_μ(x)]``, see :func:`gelt.sampler.staple_sum`),
one integrates

    U_μ(x)  ←  exp(ε Z_μ(x)) · U_μ(x),      Z_μ(x) = −[ U_μ(x) A_μ(x) ]_TA

with ``[M]_TA = (M − M†)/2 − Tr(M − M†)·𝟙/(2N)`` the traceless anti-Hermitian
projection.  There is no β in the drift: the coupling cancels against the
``g₀²`` in the flow equation, and this normalisation is the one that makes the
linearised flow the *unit-coefficient* heat equation — for links
``U_μ = exp(iθ_μ)`` in the small-field limit the drift reduces to ``θ̇_μ = Δ²θ_μ``
up to a gauge term, whose 4D kernel has ``⟨x²⟩ = 8t``.  Flow time is therefore in
lattice units ``t/a²`` throughout, and ``r_sm/a = √(8 t/a²)``.

Both the projection and the exponential are closed-form at ``nc = 2``
(``exp(iθ n̂·σ⃗) = cos θ + i sin θ n̂·σ⃗``), in keeping with the SU(2) fast paths in
:mod:`gelt.lattice`; ``nc ≥ 3`` falls back to ``torch.linalg.matrix_exp``.

Conventions follow the rest of the library: links are **batched**,
``(B, D, *Λ, nc, nc)``, periodic BCs via ``torch.roll``, time on lattice axis 0.
The flow is isotropic — it is a smoother, not dynamics, so like APE smearing it
carries no anisotropy weight (``xi = 1``).
"""

import math
from typing import List, Optional, Sequence

import torch
from tqdm import tqdm

from gelt.lattice import GaugeGroup
from gelt.sampler import staple_sum


def traceless_antihermitian(M: torch.Tensor, nc: Optional[int] = None) -> torch.Tensor:
    """``[M]_TA = (M − M†)/2 − Tr(M − M†)·𝟙/(2N)`` — the su(N) part of ``M``.

    Acts on a batch of matrices ``(*batch, nc, nc)``.  The result is exactly
    anti-Hermitian and exactly traceless (up to rounding), i.e. an element of the
    Lie algebra, which is what makes ``exp`` of it land on the group.
    """
    A = (M - M.conj().transpose(-1, -2)) / 2
    n = A.shape[-1] if nc is None else nc
    tr = A.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (*batch,)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    return A - (tr / n).unsqueeze(-1).unsqueeze(-1) * eye


def group_exp(X: torch.Tensor, nc: Optional[int] = None) -> torch.Tensor:
    """Matrix exponential of a traceless anti-Hermitian ``X``: an SU(N) element.

    At ``nc = 2`` this is closed form.  A traceless 2×2 matrix satisfies
    ``X² = −det(X)·𝟙`` by Cayley–Hamilton, and ``det X = a² + b² + c² ≥ 0`` is
    real for the anti-Hermitian case (``X = i(a σ₃ + b σ₂ + c σ₁)``-like), so with
    ``θ = √det X``

        exp(X) = cos θ · 𝟙 + (sin θ / θ) · X ,

    which needs one real square root and no ``linalg`` kernel — the same
    motivation as :func:`gelt.lattice._polar_factor_2x2`, and the same
    consequence: the flow runs on MPS, where complex ``linalg`` does not.
    ``sin θ / θ`` is taken as ``sinc`` so that ``θ → 0`` (a link already at a
    stationary point) is not a division by zero.

    ``nc ≥ 3`` goes through ``torch.linalg.matrix_exp``, which is exact for any
    matrix and needs no special casing (SU(3) would want Cabibbo–Marinari-style
    hand-rolling only if this were ever the hot path).
    """
    n = X.shape[-1] if nc is None else nc
    if n != 2:
        return torch.linalg.matrix_exp(X)
    det = X[..., 0, 0] * X[..., 1, 1] - X[..., 0, 1] * X[..., 1, 0]
    theta = det.real.clamp_min(0).sqrt()
    eye = torch.eye(2, dtype=X.dtype, device=X.device)
    cos = torch.cos(theta).to(X.dtype).unsqueeze(-1).unsqueeze(-1)
    # torch.sinc(z) = sin(πz)/(πz), so sin θ/θ = sinc(θ/π) — finite at θ = 0.
    sinc = torch.sinc(theta / math.pi).to(X.dtype).unsqueeze(-1).unsqueeze(-1)
    return cos * eye + sinc * X


def flow_drift(U: torch.Tensor, gaugegroup: GaugeGroup) -> torch.Tensor:
    """``Z_μ(x) = −[U_μ(x) A_μ(x)]_TA`` for every link — the Wilson-flow drift.

    Parameters
    ----------
    U
        Batched links ``(B, D, *Λ, nc, nc)``.
    gaugegroup
        Gauge group (used for the dagger inside the staple).

    Returns
    -------
    ``(B, D, *Λ, nc, nc)``, anti-Hermitian and traceless in the colour axes.
    """
    D = U.shape[1]
    Z = []
    for mu in range(D):
        A = staple_sum(U, mu, gaugegroup, batched=True)
        Z.append(-traceless_antihermitian(U[:, mu] @ A, gaugegroup.nc))
    return torch.stack(Z, dim=1)


def _apply(X: torch.Tensor, U: torch.Tensor, nc: int) -> torch.Tensor:
    """``exp(X) · U`` link by link."""
    return group_exp(X, nc) @ U


def _euler_step(U, gaugegroup, h):
    """One explicit-Euler step of size ``h``. First order; the cross-check."""
    return _apply(h * flow_drift(U, gaugegroup), U, gaugegroup.nc)


def _rk3_step(U, gaugegroup, h):
    """One step of Lüscher's third-order Runge–Kutta (arXiv:1006.4518, §appendix).

    With ``Z_i = h · Z(W_i)``::

        W₁ = exp(¼ Z₀) W₀
        W₂ = exp(  8/9 Z₁ − 17/36 Z₀) W₁
        V' = exp(  ¾  Z₂ −  8/9  Z₁ + 17/36 Z₀) W₂

    Three drift evaluations per step for a local error ``O(h⁴)``, so at the step
    sizes used in production it is far cheaper than Euler at matched accuracy.
    """
    nc = gaugegroup.nc
    Z0 = h * flow_drift(U, gaugegroup)
    W1 = _apply(Z0 / 4, U, nc)
    Z1 = h * flow_drift(W1, gaugegroup)
    W2 = _apply(Z1 * (8 / 9) - Z0 * (17 / 36), W1, nc)
    Z2 = h * flow_drift(W2, gaugegroup)
    return _apply(Z2 * (3 / 4) - Z1 * (8 / 9) + Z0 * (17 / 36), W2, nc)


_INTEGRATORS = {"rk3": _rk3_step, "euler": _euler_step}


def _check_flowable(U: torch.Tensor, gaugegroup: GaugeGroup) -> None:
    if not U.is_complex():
        raise ValueError(
            f"Wilson flow needs a continuous gauge group, got real links "
            f"({gaugegroup}). The traceless anti-Hermitian part of a real 1×1 "
            f"matrix is identically zero, so the flow would silently be the "
            f"identity map rather than a smoother."
        )
    if U.dim() < 4:
        raise ValueError(
            f"Expected batched links (B, D, *Λ, nc, nc), got shape {tuple(U.shape)}."
        )


def wilson_flow(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    t: float,
    eps: float = 0.02,
    integrator: str = "rk3",
    reunitarize_every: int = 0,
    progress: bool = False,
) -> torch.Tensor:
    """Flow ``U`` to lattice-unit flow time ``t`` (i.e. ``t/a²``).

    Parameters
    ----------
    U
        Batched links ``(B, D, *Λ, nc, nc)``, complex (SU(N)).
    gaugegroup
        Gauge group.
    t
        Target flow time in lattice units.  ``t = 0`` returns a copy of ``U``.
    eps
        Maximum step size.  The actual step is ``t / ceil(t / eps)`` so the
        trajectory lands *exactly* on ``t`` rather than overshooting or leaving a
        ragged remainder — which matters because the target ladder is defined at
        specific flow times and the rungs must be comparable across β.
    integrator
        ``"rk3"`` (default, Lüscher's third-order Runge–Kutta) or ``"euler"``.
        Euler at small ``eps`` is the fallback and the cross-check.
    reunitarize_every
        If > 0, project back onto the group every this many steps.  Off by
        default: ``exp`` of a Lie-algebra element is analytically *on* the group,
        so this only fights float32 rounding accumulated over a long trajectory
        (``t/a² = 16`` at ``eps = 0.02`` is 800 steps).
    progress
        Show a tqdm bar over the steps.

    Returns
    -------
    Flowed links, same shape and dtype as ``U``.
    """
    _check_flowable(U, gaugegroup)
    if integrator not in _INTEGRATORS:
        raise ValueError(
            f"integrator must be one of {sorted(_INTEGRATORS)}, got {integrator!r}."
        )
    if t < 0:
        raise ValueError(f"Flow time must be non-negative, got {t}.")
    if t == 0:
        return U.clone()
    if eps <= 0:
        raise ValueError(f"Step size must be positive, got {eps}.")

    n_steps = max(1, math.ceil(t / eps - 1e-12))
    h = t / n_steps
    step = _INTEGRATORS[integrator]
    V = U
    it = range(n_steps)
    if progress:
        it = tqdm(it, desc=f"flow t={t:g}", leave=False)
    for i in it:
        V = step(V, gaugegroup, h)
        if reunitarize_every and (i + 1) % reunitarize_every == 0:
            V = gaugegroup.project(V)
    return V


def flow_trajectory(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    times: Sequence[float],
    eps: float = 0.02,
    integrator: str = "rk3",
    reunitarize_every: int = 0,
    progress: bool = False,
) -> List[torch.Tensor]:
    """Flow ``U`` once and return the configuration at each requested time.

    The target ladder of ``notes/flow_free_topology.md`` is five flow times on
    *one* trajectory, so this integrates through the sorted ``times`` and keeps a
    snapshot at each: the cost is that of the longest rung alone, not the sum.
    Each rung is reached exactly, by restarting the step-size arithmetic of
    :func:`wilson_flow` on every segment, so a snapshot here is bit-comparable to
    a standalone ``wilson_flow(U, ..., t)`` call with the same ``eps`` **when the
    segment boundaries fall on the same grid** — the tests pin the case that
    matters (a ladder whose rungs are multiples of the step size).

    Parameters
    ----------
    times
        Flow times in lattice units; need not be sorted, and the returned list
        follows the order given.  Duplicates are fine.

    Returns
    -------
    List of flowed link tensors, one per entry of ``times``, in the order given.
    Memory is ``len(set(times))`` copies of the link field — for the five-rung
    production ladder at L=16 that is deliberate (one trajectory, five targets),
    but it is why the caller chunks over configurations.
    """
    _check_flowable(U, gaugegroup)
    for t in times:
        if t < 0:
            raise ValueError(f"Flow times must be non-negative, got {t}.")

    order = sorted(set(times))
    snapshots = {}
    V = U
    reached = 0.0
    for t in order:
        segment = t - reached
        if segment > 0:
            V = wilson_flow(
                V,
                gaugegroup,
                segment,
                eps=eps,
                integrator=integrator,
                reunitarize_every=reunitarize_every,
                progress=progress,
            )
            reached = t
        snapshots[t] = V if segment > 0 else V.clone()
    return [snapshots[t] for t in times]
