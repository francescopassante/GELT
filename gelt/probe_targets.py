"""
=========================================================================
Probe targets — per-site scalar fields for the M1 ablation.
=========================================================================

The design record is ``notes/m1_probe.md``; the question it serves is
``notes/where_attention_can_win.md`` §6. All three targets are built from one
gauge-invariant scalar, the **local action density**

    f(x) = 1 − Re Tr P̄(x) / nc ,    P̄ = mean over the D(D−1)/2 planes at x,

and differ only in how a bounded neighbourhood of f is reduced to one number:

======  ==========================  ==============================================
 name    reduction                   what it demands of the offset weights
======  ==========================  ==============================================
 T0      Σ_Δ c_{|Δ|₁} f(x+Δ)         nothing — it *is* a convolution. Calibration.
 T1      max_{|Δ|₁≤R} f(x+Δ)         the winning offset varies per site (position).
 T2      the θ-enclosing radius      a scale-free ratio test at variable radius
         of ``f**SCALE_POWER``       (position *and* scale, amplitude-invariant).
======  ==========================  ==============================================

Two properties are deliberate and are pinned in ``tests/test_probe_targets.py``:

* **Every target is exactly computable from the L1-ball of radius R around x**,
  which the network reaches. There is no unbounded classical method that does
  better, so ``notes/where_attention_can_win.md`` §5's L2 cannot bite — and no
  reading here is ever of the form "the network beats classical X". They are
  all architecture-vs-architecture differences at matched budget.
* **T2 is invariant under f → λf** and T0/T1 are homogeneous of degree 1. That
  is the amplitude-invariant identity criterion 2 asks for, isolated.

The flip side, stated plainly because the note demands it: these are *not*
physics targets. This module builds a mechanism assay, not an observable.
"""

import torch

from gelt.lattice import l1_ball_offsets, plaquette_tensor

# The targets' neighbourhood radius. Must stay well inside the architecture's
# own reach — GELT at R = 2 over 4 layers reaches Manhattan 8, the L-CNN at
# K = 2 over 4 layers likewise — so that neither arm is handicapped by
# geometry rather than by its offset weights (criterion 3).
BALL_RADIUS = 4

# The θ of T2: the radius enclosing half of the neighbourhood's action mass.
SCALE_THETA = 0.5

# T2 encloses the mass of ``f**SCALE_POWER``, not of ``f``. **This is a
# pre-flight finding, not a decoration** (``notes/m1_probe.md`` §3.1). At p = 1
# the outer shell holds 66 sites, so the cumulative profile concentrates, the
# "min{r}" selection never actually selects — ``r_first`` is 4 at 82% of sites —
# and what is left is the smooth ratio
#   r* = 4 + (θS₄ − S₃)/(S₄ − S₃),
# a ratio of two linear functionals which linearises about its mean. Measured:
# the best linear filter over the same ball reaches **R² = 0.978**, i.e. 2%
# headroom — precisely the "competing in a smoother's residual" failure
# ``notes/where_attention_can_win.md`` §5 L1 is about.
#
# Raising f to a power sparsifies the mass onto the largest few sites in the
# ball, so the shell the threshold lands in genuinely varies site to site. The
# power is chosen by a rule fixed in advance — **the smallest p whose linear
# ceiling is below 0.80**, smallest because polynomial degree is a cost the
# architectures pay for something that is not the mechanism under test:
#
#   p        1      2      3      4      5      6      8
#   R²_lin  .978   .939   .864   .774   .679   .588   .429
#   r_first  .18/.82  .31/.69  .37/.63  .40/.60  .42/.58  .43/.57  .45/.55
#
# p = 4 it is: 22 points of headroom against a ΔR² resolution of ~0.01, a
# near-even shell split, and degree 4 — two of the four bilinear layers — so
# both architectures can build it and still have two layers left to aggregate
# with. f**p is still exactly gauge invariant, still supported on the ball, and
# still **scale-free**: f → λf scales the profile and the threshold alike.
SCALE_POWER = 4

# T0's radial kernel decay. The alternating sign makes it a band-pass rather
# than a smoother, so the calibration target is not simply "the local mean".
MEAN_DECAY = 2.0

TARGETS = ("T0", "T1", "T2")


def _check_ball_fits(f, R):
    """Refuse a ball that wraps the periodic lattice onto itself.

    With ``L < 2R+1`` two distinct offsets reach the same site, so ``f(x+Δ)``
    is double-counted and the shell structure T2 reads is not the geometry it
    claims to be. The production lattice is L = 12 against R = 4, which fits;
    this exists so a smoke test on a small lattice fails loudly instead of
    quietly measuring something else.
    """
    L = min(f.shape[1:])
    if L < 2 * R + 1:
        raise ValueError(
            f"L1-ball of radius {R} does not fit in a periodic lattice of "
            f"extent {L}: it needs L >= {2 * R + 1}, or the ball aliases onto "
            f"itself and the shells double-count sites."
        )


def action_density(U, gaugegroup, plaquettes=None):
    """``f(x) = 1 − Re Tr P̄(x)/nc`` for a batch of link configurations.

    ``U`` : ``(B, D, *Λ, nc, nc)``. Returns ``(B, *Λ)`` real. ``plaquettes``
    short-circuits the ``plaquette_tensor`` call when the caller already has
    ``(B, n_pairs, *Λ, nc, nc)`` — the training script builds them once as the
    network's input channels and reuses them here.
    """
    P = plaquette_tensor(U, gaugegroup) if plaquettes is None else plaquettes
    nc = P.shape[-1]
    tr = P.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, n_pairs, *Λ)
    return 1.0 - tr.real.mean(dim=1) / nc


def ball_reduce(f, R=BALL_RADIUS):
    """Shell sums and the running max over the L1-ball of radius ``R``.

    ``f`` : ``(B, *Λ)``. Returns ``(S, M)`` with ``S`` of shape
    ``(R+1, B, *Λ)`` the *cumulative* sums ``S[r](x) = Σ_{|Δ|₁≤r} f(x+Δ)`` and
    ``M`` of shape ``(B, *Λ)`` the max over the same ball.

    One pass over the ball: the offsets are visited once and accumulated into
    both reductions, so the three targets cost one traversal between them.
    ``torch.roll`` with shift ``−Δ`` is ``f(x+Δ)`` and carries the periodic
    boundary conditions (never manual modulo arithmetic — see CLAUDE.md).
    """
    D = f.ndim - 1
    _check_ball_fits(f, R)
    dims = tuple(range(1, D + 1))
    shells = [f.clone()] + [torch.zeros_like(f) for _ in range(R)]
    M = f.clone()
    for dx in l1_ball_offsets(D, R):
        rolled = torch.roll(f, shifts=tuple(-d for d in dx), dims=dims)
        shells[sum(abs(d) for d in dx)] += rolled
        M = torch.maximum(M, rolled)
    return torch.cumsum(torch.stack(shells, dim=0), dim=0), M


def target_mean(S, decay=MEAN_DECAY):
    """T0 — ``Σ_Δ c_{|Δ|₁} f(x+Δ)`` with ``c_r = (−1)^r e^{−r/decay}``.

    Exactly a convolution, so a linear filter over the same ball reproduces it
    to machine precision. That is the point: T0 is the calibration arm, and
    reading R-A on it is how a capacity difference between the arms would
    announce itself before any of the other readings are believed.

    Takes the cumulative sums from :func:`ball_reduce` and differences them
    back into shells, so the kernel is applied without a second traversal.
    """
    shells = torch.cat([S[:1], S[1:] - S[:-1]], dim=0)  # (R+1, B, *Λ)
    r = torch.arange(shells.shape[0], dtype=S.dtype, device=S.device)
    c = torch.where(r % 2 == 0, 1.0, -1.0) * torch.exp(-r / decay)
    return (c.view(-1, *([1] * (shells.ndim - 1))) * shells).sum(dim=0)


def target_max(M):
    """T1 — ``max_{|Δ|₁≤R} f(x+Δ)``, straight from :func:`ball_reduce`."""
    return M


def target_scale(S, theta=SCALE_THETA, eps=1e-12):
    """T2 — the θ-enclosing radius of the local action mass.

    ``r*(x)`` is where the piecewise-linear interpolant of the cumulative
    profile ``S(r)`` first reaches ``θ·S(R)``, under the convention
    ``S(−1) = 0`` and reported shifted by +1 so it lands in ``[0, R+1]``:

        r*(x) = r_first + (θ·S_R − S_{r_first−1}) / (S_{r_first} − S_{r_first−1})

    ``f ≥ 0`` for every gauge group (``Re Tr P/nc ≤ 1``), so ``S`` is
    non-decreasing in r and ``r_first`` is well defined; the interpolation
    fraction is in ``[0, 1]`` by construction. The additive shift is
    irrelevant downstream — the target is standardised before training.

    **Scale-free**: ``f → λf`` scales ``S`` and the threshold alike, so ``r*``
    is unchanged. This is the one target whose optimal neighbourhood weighting
    cannot be an absolute-magnitude test, which is what makes it the primary
    R-B target.

    ``S`` is whatever cumulative profile the caller built; :func:`build_targets`
    feeds it the profile of ``f**SCALE_POWER``.
    """
    thr = theta * S[-1]
    reached = (S >= thr.unsqueeze(0)).to(S.dtype)
    # S is non-decreasing and reached[-1] is identically 1, so the first 1 along
    # the radius axis always exists; argmax returns its index.
    r_first = reached.argmax(dim=0)
    idx = r_first.unsqueeze(0)
    S_at = S.gather(0, idx).squeeze(0)
    S_prev = torch.cat([torch.zeros_like(S[:1]), S[:-1]], dim=0)
    S_prev_at = S_prev.gather(0, idx).squeeze(0)
    shell = (S_at - S_prev_at).clamp_min(eps)
    return r_first.to(S.dtype) + (thr - S_prev_at) / shell


def build_targets(f, R=BALL_RADIUS, names=TARGETS):
    """``{name: (B, *Λ)}`` for the requested targets, one ball traversal."""
    for name in names:
        if name not in TARGETS:
            raise ValueError(f"unknown target {name!r}; expected one of {TARGETS}")
    S, M = ball_reduce(f, R)
    builders = {"T0": lambda: target_mean(S), "T1": lambda: target_max(M)}
    if "T2" in names:
        # T2 encloses the mass of f**SCALE_POWER — a second traversal, because
        # the power has to be taken before the shells are summed. See the
        # SCALE_POWER comment for why it is there at all.
        S_p, _ = ball_reduce(f**SCALE_POWER, R)
        builders["T2"] = lambda: target_scale(S_p)
    return {name: builders[name]() for name in names}


def ball_features(f, R=BALL_RADIUS):
    """Every ``f(x+Δ)`` in the ball, stacked: ``(n_ball, B, *Λ)``.

    The design matrix of the pre-flight's linear filter (``notes/m1_probe.md``
    §3) — the best convolution at the architecture's own reach. Ordered
    ``[Δ=0] + l1_ball_offsets(D, R)``, i.e. by ``|Δ|₁``, so the radial
    sub-family is a contiguous grouping of the columns.
    """
    D = f.ndim - 1
    _check_ball_fits(f, R)
    dims = tuple(range(1, D + 1))
    cols = [f] + [
        torch.roll(f, shifts=tuple(-d for d in dx), dims=dims)
        for dx in l1_ball_offsets(D, R)
    ]
    return torch.stack(cols, dim=0)
