"""The Yang–Mills vacuum wavefunctional from Euclidean data, by force matching.

The identity
------------
On a lattice with time = axis 0, the spatial links U_t of one timeslice are the
Markov state of the transfer matrix. Their marginal distribution is

    p(U_t) = ⟨U_t| T^{Nt} |U_t⟩ / Tr T^{Nt}  =  ψ₀(U_t)² · (1 + O(e^{−m Nt})),

with ψ₀ the (real, positive, gauge-invariant) vacuum wavefunctional of the
transfer matrix. Split the Wilson action into the spatial plaquettes of slice t,
S_s(U_t), and the rest. Differentiating p(U_t) = ∫ e^{−S} under the integral
(Lie derivatives on a compact group, Haar measure invariant, no boundary terms):

    ∇ log ψ₀²(U_t) = −∇S_s(U_t) + E[ F_temp | U_t ],

where F_temp = −∇S_temp is the force that the temporal plaquettes at t−1 and t
exert on the spatial links of slice t. Its conditional expectation is the
non-trivial part of the vacuum's score; S_s is known in closed form.

So with log ψ₀² = −S_s + Φ, any scalar gauge-invariant Φ_θ(U) trained to
minimise  E ‖∇Φ_θ(U_t) − F_temp‖²  converges to the true Φ (the minimiser of a
squared error over functions of U_t is the conditional expectation, and the
conditional expectation here *is* a gradient field). This is force matching:
a first-derivative objective, unbiased, needing nothing but the ensemble.

What this module provides
-------------------------
- ``lie_generators``, ``lie_grad``: the Lie-algebra gradient of a scalar
  function of links, ∂_a f = d/dε f(e^{iεT^a}U)|₀, through autograd
  (``create_graph=True`` for training a network on it).
- ``temporal_force`` / ``spatial_force``: F^a = −∂_a S per link, from staples.
- ``loop_basis`` / ``bilinear_basis``: the classical comparators — Wilson-loop
  sums (shapes × APE levels; {1×1} alone is the rescaled-coupling,
  "dimensional reduction" ansatz) and transport-dressed B·B shell sums (a
  Greensite–Olejník-type bilinear with a free radial kernel).
- ``frozen_slice_mask``, ``slice_temporal_staples``: the conditional chain with
  one slice frozen. It gives two things nothing else does: (i) the signal
  ‖E[F|U]‖² and the noise floor, hence an *absolute* error ε for any model, and
  (ii) exact relative weights ψ₀²(U')/ψ₀²(U) by reweighting, the ground truth of
  scripts/vacuum_relative_weights.py.

Conventions follow the rest of the package: links ``(B, D, *Λ, nc, nc)``,
time = lattice axis 0, plaquette Re Tr P = Re Tr(U·A) with ``staple_sum``'s A.
Lie components are taken with Tr(T^a T^b) = δ_ab/2 and left multiplication,
U → e^{iεT^a} U, so a force component on U_μ(x) rotates in the adjoint of Ω(x)
under a gauge transformation and its per-link norm is invariant.
"""

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .glueball import ape_smear
from .lattice import (
    GaugeGroup,
    build_transport_average,
    l1_ball_offsets,
    plaquette_tensor,
    rectangular_wilson_loop,
)
from .sampler import staple_sum

_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}


def lie_generators(nc: int, dtype=torch.complex128, device=None) -> torch.Tensor:
    """Hermitian traceless generators of su(nc), Tr(T^a T^b) = δ_ab/2.

    The generalised Gell-Mann basis divided by two: off-diagonal symmetric and
    antisymmetric pairs, then the nc − 1 diagonal ones. At nc = 2 this is σ^a/2
    in the order (σ₁, σ₂, σ₃). Returns ``(nc²−1, nc, nc)``.
    """
    gens = []
    for j in range(nc):
        for k in range(j + 1, nc):
            S = torch.zeros(nc, nc, dtype=dtype, device=device)
            S[j, k] = S[k, j] = 0.5
            A = torch.zeros(nc, nc, dtype=dtype, device=device)
            A[j, k], A[k, j] = -0.5j, 0.5j
            gens += [S, A]
    for l in range(1, nc):
        Dg = torch.zeros(nc, nc, dtype=dtype, device=device)
        for m in range(l):
            Dg[m, m] = 1.0
        Dg[l, l] = -float(l)
        gens.append(Dg / math.sqrt(2 * l * (l + 1)))
    return torch.stack(gens)


def casimir_fundamental(nc: int) -> float:
    """C_F = Σ_a T^a T^a = (nc² − 1)/(2 nc) (3/4 for SU(2))."""
    return (nc * nc - 1) / (2 * nc)


def spatial_slices(U4: torch.Tensor) -> torch.Tensor:
    """``(B, D, Lt, *Ls, nc, nc)`` → the 3D slices ``(B·Lt, D−1, *Ls, nc, nc)``.

    Spatial directions 1..D−1 at each fixed time, the time axis folded into the
    batch config-major (index b·Lt + t) — the same layout as
    ``train_glueball.config_inputs``.
    """
    B, D, Lt = U4.shape[:3]
    return U4[:, 1:].movedim(2, 1).reshape(B * Lt, D - 1, *U4.shape[3:])


def _staple_force(U, mu, gaugegroup, nu_dirs, coupling, gens):
    """F^a on every link U_μ of a batch for S = coupling Σ_planes (1 − Re Tr P/nc).

    With the local action −(coupling/nc) Re Tr(U·A), A = ``staple_sum`` over
    ``nu_dirs``: ∂_a Re Tr(e^{iεT^a} U A)|₀ = Re Tr(iT^a U A) = −Im Tr(T^a U A),
    so F^a = −∂_a S = −(coupling/nc) Im Tr(T^a U A). Returns ``(B, *Λ, n_gen)``.
    """
    nc = U.shape[-1]
    A = staple_sum(U, mu, gaugegroup, nu_dirs=nu_dirs, batched=True)
    UA = U[:, mu] @ A
    tr = torch.einsum("aij,...ji->...a", gens.to(UA.dtype), UA)  # Tr(T^a U A)
    return -(coupling / nc) * tr.imag


def temporal_force(U4: torch.Tensor, gaugegroup: GaugeGroup, beta: float,
                   xi: float = 1.0) -> torch.Tensor:
    """Force of the temporal plaquettes on the spatial links of every slice.

    ``U4`` : ``(B, D, Lt, *Ls, nc, nc)``. The temporal plaquettes (0, i) at t−1
    and at t are exactly the two staples of U_i(x, t) in direction 0, with the
    temporal coupling β_t = β·ξ (``xi = 1``: β). Returns ``(B·Lt, D−1, *Ls,
    n_gen)`` in :func:`spatial_slices`' layout.
    """
    B, D, Lt = U4.shape[:3]
    gens = lie_generators(U4.shape[-1], device=U4.device)
    beta_t = beta * xi
    F = torch.stack(
        [_staple_force(U4, mu, gaugegroup, [0], beta_t, gens) for mu in range(1, D)],
        dim=1,
    )  # (B, D−1, Lt, *Ls, n_gen)
    return F.movedim(2, 1).reshape(B * Lt, D - 1, *F.shape[3:])


def spatial_force(U3: torch.Tensor, gaugegroup: GaugeGroup, beta_s: float) -> torch.Tensor:
    """F^a = −∂_a S_s for 3D slices ``(B, d, *Λ, nc, nc)``; S_s = β_s Σ_p (1 − Re Tr P/nc)."""
    d = U3.shape[1]
    gens = lie_generators(U3.shape[-1], device=U3.device)
    return torch.stack(
        [_staple_force(U3, mu, gaugegroup, None, beta_s, gens) for mu in range(d)], dim=1
    )


def lie_grad(
    fn: Callable[[torch.Tensor], torch.Tensor],
    U: torch.Tensor,
    create_graph: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """∂_a f for every link: d/dε f((1 + iΣ ε^a T^a)·U) at ε = 0.

    The first-order perturbation has the same first derivative as e^{iεT}U, which
    is all force matching needs. ``fn(U) -> (B,)``; returns ``(grad, value)``
    with ``grad`` of shape ``U.shape[:-2] + (n_gen,)``. With ``create_graph``
    the gradient stays differentiable in the parameters of ``fn`` (training).
    """
    gens = lie_generators(U.shape[-1], dtype=U.dtype, device=U.device)
    eps = torch.zeros(U.shape[:-2] + (gens.shape[0],), dtype=_REAL[U.dtype],
                      device=U.device, requires_grad=True)
    M = 1j * torch.einsum("...a,aij->...ij", eps.to(U.dtype), gens)
    val = fn(U + M @ U)
    (g,) = torch.autograd.grad(val.sum(), eps, create_graph=create_graph)
    return g, val


# ── Classical comparators ─────────────────────────────────────────────────────


def loop_basis(
    U3: torch.Tensor,
    gaugegroup: GaugeGroup,
    shapes: Sequence[Tuple[int, int]] = ((1, 1),),
    levels: Sequence[int] = (0,),
    alpha: float = 0.5,
) -> Tuple[torch.Tensor, List[str]]:
    """Σ_x Re Tr W/nc of R×T loops on every plane of a 3D slice, per APE level.

    Both orientations of a rectangle are summed, so every operator is a cubic
    scalar. Smearing is the 3D APE of the slice itself (all its directions —
    there is no time axis inside a slice). ``U3`` : ``(B, d, *Λ, nc, nc)``.
    Returns ``(B, K)`` and the labels ``n<level>·<R>x<T>``.
    """
    d = U3.shape[1]
    feats, labels = [], []
    Us, done = U3, 0
    for lvl in sorted(levels):
        if lvl > done:
            Us = ape_smear(Us, gaugegroup, alpha=alpha, n_steps=lvl - done,
                           directions=range(d))
            done = lvl
        for r, t in shapes:
            O = 0.0
            for mu in range(d):
                for nu in range(mu + 1, d):
                    O = O + rectangular_wilson_loop(Us, gaugegroup, r, t, mu, nu)
                    if r != t:
                        O = O + rectangular_wilson_loop(Us, gaugegroup, t, r, mu, nu)
            feats.append(O.flatten(1).sum(1))
            labels.append(f"n{lvl}·{r}x{t}")
    return torch.stack(feats, dim=1), labels


def magnetic_field(U3: torch.Tensor, gaugegroup: GaugeGroup) -> torch.Tensor:
    """F_p = traceless anti-Hermitian part of each plaquette, ``(B, n_planes, *Λ, nc, nc)``.

    (P − P†)/2 minus its trace part: at small fields i·a²·B^a T^a per plane, the
    lattice magnetic field. Transforms in the adjoint at its base point.
    """
    P = plaquette_tensor(U3, gaugegroup)
    F = 0.5 * (P - gaugegroup.dagger(P))
    nc = P.shape[-1]
    tr = F.diagonal(dim1=-2, dim2=-1).sum(-1) / nc
    return F - tr[..., None, None] * torch.eye(nc, dtype=P.dtype, device=P.device)


def bilinear_basis(
    U3: torch.Tensor, gaugegroup: GaugeGroup, R: int = 3
) -> Tuple[torch.Tensor, List[str]]:
    """Shell sums of the transported B·B correlator, r = 0..R.

        O_r = Σ_x Σ_{|Δ|₁ = r} Σ_p Re Tr[ F_p(x)† · T_Δ(x) F_p(x+Δ) T_Δ(x)† ]

    with T the shortest-path-averaged transport (``build_transport_average``),
    O_0 = Σ_x Σ_p ‖F_p(x)‖². A linear combination Σ_r K_r O_r is the
    Greensite–Olejník form −½ Σ B·K·B with a free, radially symmetric,
    field-independent kernel truncated at |Δ|₁ ≤ R — the classical counterpart of
    a GELT layer with frozen offset weights. Returns ``(B, R+1)`` and labels.
    """
    d = U3.shape[1]
    F = magnetic_field(U3, gaugegroup)  # (B, np, *Λ, nc, nc)
    Fd = gaugegroup.dagger(F)
    sp = tuple(range(2, 2 + d))
    shells = [(Fd @ F).diagonal(dim1=-2, dim2=-1).sum(-1).real.flatten(1).sum(1)]
    if R >= 1:
        T = build_transport_average(U3, R, gaugegroup)  # (B, n_off, *Λ, nc, nc)
        Td = gaugegroup.dagger(T)
        acc = [0.0] * R
        for i, off in enumerate(l1_ball_offsets(d, R)):
            r = sum(abs(o) for o in off)
            Fs = torch.roll(F, shifts=tuple(-o for o in off), dims=sp)  # F(x + Δ)
            Ft = T[:, i, None] @ Fs @ Td[:, i, None]
            acc[r - 1] = acc[r - 1] + (Fd @ Ft).diagonal(dim1=-2, dim2=-1).sum(-1).real.flatten(1).sum(1)
        shells += acc
    return torch.stack(shells, dim=1), [f"BB·r{r}" for r in range(R + 1)]


def basis_gradients(
    U3: torch.Tensor, ops_fn: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Lie gradient of every operator of a linear basis: ``(K, B, d, *Λ, n_gen)``.

    ``ops_fn(U3) -> (B, K)``. One forward, K backward passes through the same
    graph — the design matrix of the force-matching least squares.
    """
    gens = lie_generators(U3.shape[-1], dtype=U3.dtype, device=U3.device)
    eps = torch.zeros(U3.shape[:-2] + (gens.shape[0],), dtype=_REAL[U3.dtype],
                      device=U3.device, requires_grad=True)
    M = 1j * torch.einsum("...a,aij->...ij", eps.to(U3.dtype), gens)
    ops = ops_fn(U3 + M @ U3)  # (B, K)
    K = ops.shape[1]
    out = []
    for k in range(K):
        (g,) = torch.autograd.grad(ops[:, k].sum(), eps, retain_graph=k < K - 1)
        out.append(g)
    return torch.stack(out)


# ── The conditional chain: one slice frozen ───────────────────────────────────


def frozen_slice_mask(shape_D_Lambda: Tuple[int, ...], t0: int, device=None) -> torch.Tensor:
    """``(D, Lt, *Ls)`` bool, True on the spatial links (μ ≥ 1) of slice ``t0``.

    Passed as ``frozen=`` to the SU(2) sweeps, it makes the chain sample every
    other link conditionally on that slice: the multilevel construction.
    """
    m = torch.zeros(shape_D_Lambda, dtype=torch.bool, device=device)
    m[1:, t0] = True
    return m


def slice_temporal_staples(U4: torch.Tensor, gaugegroup: GaugeGroup, t0: int) -> torch.Tensor:
    """Temporal staples of the spatial links of slice ``t0``: ``(B, D−1, *Ls, nc, nc)``.

    They depend only on the *other* links (U_0 at t0−1 and t0, the spatial links
    at t0 ± 1), so for a frozen slice the temporal action that touches it is
    β_t Σ_{x,i} [2 − Re Tr(U_i(x,t0) A_i(x,t0))/nc] — and a *change* of the
    slice, U → U', changes it by −(β_t/nc) Σ Re Tr((U' − U)·A), computed from the
    difference directly (no cancellation between two O(10⁴) totals).
    """
    D = U4.shape[1]
    return torch.stack(
        [staple_sum(U4, mu, gaugegroup, nu_dirs=[0], batched=True)[:, t0]
         for mu in range(1, D)], dim=1)


def delta_temporal_action(dU3: torch.Tensor, A: torch.Tensor, beta_t: float) -> torch.Tensor:
    """−(β_t/nc) Σ Re Tr(dU·A) per config, in float64. ``dU3 = U'_slice − U_slice``."""
    nc = dU3.shape[-1]
    tr = (dU3.to(torch.complex128) @ A.to(torch.complex128)).diagonal(dim1=-2, dim2=-1).sum(-1).real
    return -(beta_t / nc) * tr.flatten(1).sum(1)


def slice_spatial_action(U3: torch.Tensor, gaugegroup: GaugeGroup, beta_s: float) -> torch.Tensor:
    """S_s = β_s Σ_p (1 − Re Tr P/nc) of 3D slices, in float64. ``(B,)``."""
    P = plaquette_tensor(U3.to(torch.complex128), gaugegroup)
    nc = P.shape[-1]
    re = P.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
    return beta_s * (1.0 - re).flatten(1).sum(1)


def su2_log(W: torch.Tensor) -> torch.Tensor:
    """X with su2_exp(X) = W for W ∈ SU(2): the rotation vector θ·â, |X| = θ ∈ [0, 2π).

    W = a₀ + i a·σ with a₀ = Re W₀₀, a = (Im W₀₁, Re W₀₁, Im W₀₀); θ = 2 atan2(|a|, a₀).
    Returns ``(..., 3)`` float64.
    """
    W = W.to(torch.complex128)
    a0 = W[..., 0, 0].real
    a = torch.stack([W[..., 0, 1].imag, W[..., 0, 1].real, W[..., 0, 0].imag], dim=-1)
    n = a.norm(dim=-1, keepdim=True)
    theta = 2 * torch.atan2(n, a0[..., None])
    return torch.where(n > 0, theta * a / n.clamp_min(1e-300), torch.zeros_like(a))


def su2_exp(X: torch.Tensor, dtype=torch.complex128) -> torch.Tensor:
    """exp(i Σ_a X^a σ^a/2) in closed form: cos(|X|/2) + i sin(|X|/2) X̂·σ.

    ``X`` : ``(..., 3)`` real. Exact on the group (used to build the
    relative-weight ladders, never inside a gradient).
    """
    X = X.to(torch.float64)
    n = X.norm(dim=-1, keepdim=True)
    c = torch.cos(n / 2)
    s = torch.where(n > 0, torch.sin(n / 2) / n.clamp_min(1e-300), torch.full_like(n, 0.5))
    a = s * X  # (…, 3)
    out = torch.empty(X.shape[:-1] + (2, 2), dtype=dtype, device=X.device)
    out[..., 0, 0] = torch.complex(c[..., 0], a[..., 2])
    out[..., 1, 1] = torch.complex(c[..., 0], -a[..., 2])
    out[..., 0, 1] = torch.complex(a[..., 1], a[..., 0])
    out[..., 1, 0] = torch.complex(-a[..., 1], a[..., 0])
    return out
