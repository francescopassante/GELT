"""Tests for :mod:`gelt.probe_targets` — the M1 ablation's supervision.

The three targets are the whole experiment's premise, so what is pinned here is
exactly the premise:

* **f is gauge invariant**, so all three targets are, and the arms are being
  asked for something an equivariant network can represent.
* **T0 is exactly a linear functional of the ball** — the calibration arm only
  calibrates if a linear filter can reach R² = 1 on it in principle.
* **T1 is the ball max** and **T2 the θ-enclosing radius**, each against a
  brute-force construction rather than against itself.
* **T2 is invariant under f → λf**, T0 and T1 homogeneous of degree 1. This is
  the amplitude-invariant identity of ``notes/where_attention_can_win.md`` §6
  criterion 2, and it is why T2 is the primary target.
"""

import itertools
import math

import pytest
import torch

from gelt import SU, Z2, link_gauge_transformation, random_links
from gelt.lattice import l1_ball_offsets
from gelt.probe_targets import (
    MEAN_DECAY,
    SCALE_POWER,
    SCALE_THETA,
    action_density,
    ball_features,
    ball_reduce,
    build_targets,
    target_max,
    target_mean,
    target_scale,
)


def _omega(L, D, nc, seed):
    torch.manual_seed(seed)
    raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        L**D, nc, nc, dtype=torch.float64
    )
    Q, _ = torch.linalg.qr(raw)
    return Q.reshape(*([L] * D), nc, nc)


def _links(L, D, gg, B, dtype):
    return torch.stack([random_links(L, D, gg, dtype=dtype) for _ in range(B)])


# ---------------------------------------------------------------------------
# f is a gauge invariant
# ---------------------------------------------------------------------------


def test_action_density_is_gauge_invariant_su2():
    L, D, nc = 6, 3, 2
    gg, dtype = SU(nc), torch.complex128
    U = _links(L, D, gg, 2, dtype)
    omega = _omega(L, D, nc, seed=3)
    U_g = torch.stack([link_gauge_transformation(u, omega, gg) for u in U])

    f, f_g = action_density(U, gg), action_density(U_g, gg)
    assert f.shape == (2, L, L, L)
    assert torch.allclose(f, f_g, atol=1e-13), (f - f_g).abs().max().item()
    # Re Tr P / nc <= 1, so the density is non-negative — the property that
    # makes T2's cumulative profile monotone and r* well defined.
    assert (f >= 0).all()


def test_action_density_is_gauge_invariant_z2():
    L, D = 6, 3
    gg, dtype = Z2(), torch.float64
    U = _links(L, D, gg, 2, dtype)
    omega = torch.where(
        torch.rand(*([L] * D), 1, 1) < 0.5,
        torch.tensor(-1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )
    U_g = torch.stack([link_gauge_transformation(u, omega, gg) for u in U])
    assert torch.allclose(action_density(U, gg), action_density(U_g, gg), atol=1e-14)
    assert (action_density(U, gg) >= 0).all()


# ---------------------------------------------------------------------------
# The reductions against brute force
# ---------------------------------------------------------------------------


def _brute_ball(f, R):
    """Every ``f(x+Δ)`` with ``|Δ|₁ ≤ R``, enumerated by hand from a hypercube.

    Independent of ``l1_ball_offsets``: it filters the full ``[-R, R]^D`` box by
    the L1 norm, so a bug in the offset generator cannot hide behind itself.
    """
    D = f.ndim - 1
    out = []
    for dx in itertools.product(range(-R, R + 1), repeat=D):
        if sum(abs(d) for d in dx) <= R:
            out.append(
                (dx, torch.roll(f, tuple(-d for d in dx), dims=tuple(range(1, D + 1))))
            )
    return out


def test_cumulative_shell_sums_match_brute_force():
    torch.manual_seed(1)
    L, D, R = 9, 3, 4
    f = torch.rand(2, L, L, L, dtype=torch.float64)
    S, M = ball_reduce(f, R)
    brute = _brute_ball(f, R)
    assert S.shape == (R + 1, 2, L, L, L)
    for r in range(R + 1):
        want = sum(t for dx, t in brute if sum(abs(d) for d in dx) <= r)
        assert torch.allclose(S[r], want, atol=1e-12), r
    want_max = torch.stack([t for _, t in brute], dim=0).amax(dim=0)
    assert torch.allclose(M, want_max, atol=1e-14)


def test_ball_features_are_the_ball_in_radius_order():
    torch.manual_seed(2)
    L, D, R = 9, 3, 3
    f = torch.rand(1, L, L, L, dtype=torch.float64)
    X = ball_features(f, R)
    offsets = [(0,) * D] + l1_ball_offsets(D, R)
    assert X.shape[0] == len(offsets) == len(_brute_ball(f, R))
    radii = [sum(abs(d) for d in dx) for dx in offsets]
    assert radii == sorted(radii)
    for i, dx in enumerate(offsets):
        want = torch.roll(f, tuple(-d for d in dx), dims=(1, 2, 3))
        assert torch.allclose(X[i], want, atol=1e-14), dx


def test_T0_is_exactly_a_linear_filter_over_the_ball():
    """The calibration arm's premise: T0 lies in the span of ``ball_features``.

    If this ever fails, R-A stops being a calibration and a linear-filter R² < 1
    on T0 would be a fact about the target rather than about the fit.
    """
    torch.manual_seed(3)
    L, D, R = 9, 3, 4
    f = torch.rand(2, L, L, L, dtype=torch.float64)
    S, _ = ball_reduce(f, R)
    X = ball_features(f, R)
    offsets = [(0,) * D] + l1_ball_offsets(D, R)
    c = torch.tensor(
        [
            (-1.0) ** sum(abs(d) for d in dx)
            * math.exp(-sum(abs(d) for d in dx) / MEAN_DECAY)
            for dx in offsets
        ],
        dtype=f.dtype,
    )
    want = (c.view(-1, *([1] * (X.ndim - 1))) * X).sum(dim=0)
    assert torch.allclose(target_mean(S), want, atol=1e-12)


def test_T2_recovers_the_enclosing_radius_of_a_uniform_field():
    """A field that is 1 everywhere makes every shell sum its own site count,
    so r* is a number that can be written down: in 3D the cumulative counts are
    1, 7, 25, 63, 129, the half-mass threshold is 64.5, and the interpolant
    crosses it 1.5/66 of the way into the r = 4 shell."""
    L, D, R = 9, 3, 4
    f = torch.ones(1, L, L, L, dtype=torch.float64)
    S, _ = ball_reduce(f, R)
    assert S[:, 0, 0, 0, 0].tolist() == [1, 7, 25, 63, 129]
    thr = SCALE_THETA * 129
    want = 4 + (thr - 63) / 66
    assert torch.allclose(target_scale(S), torch.full_like(f, want), atol=1e-12)


def test_T2_is_scale_free_and_T0_T1_are_degree_one():
    """f → λf leaves the enclosing radius alone and scales the other two.

    The one structural property that separates T2 from every absolute-magnitude
    test — and the reason the pre-registered primary reading is on T2.
    """
    torch.manual_seed(4)
    L, R, lam = 9, 4, 17.3
    f = torch.rand(2, L, L, L, dtype=torch.float64) + 0.1
    a = build_targets(f, R)
    b = build_targets(lam * f, R)
    assert torch.allclose(b["T2"], a["T2"], atol=1e-10)
    assert torch.allclose(b["T0"], lam * a["T0"], atol=1e-10)
    assert torch.allclose(b["T1"], lam * a["T1"], atol=1e-10)


def test_T1_is_the_ball_max_and_dispatch_agrees():
    torch.manual_seed(5)
    L, R = 9, 2
    f = torch.rand(2, L, L, L, dtype=torch.float64)
    S, M = ball_reduce(f, R)
    want = torch.stack([t for _, t in _brute_ball(f, R)], dim=0).amax(dim=0)
    assert torch.allclose(target_max(M), want, atol=1e-14)
    built = build_targets(f, R)
    assert torch.allclose(built["T1"], want, atol=1e-14)
    assert torch.allclose(built["T0"], target_mean(S), atol=1e-14)
    # T2 encloses the mass of f**SCALE_POWER, so it reads a *different* profile
    # from the one T0 reads — pinned here because the whole point of the power
    # is that dropping it silently turns T2 back into a near-linear target.
    S_p, _ = ball_reduce(f**SCALE_POWER, R)
    assert torch.allclose(built["T2"], target_scale(S_p), atol=1e-14)
    assert not torch.allclose(built["T2"], target_scale(S), atol=1e-6)


def test_ball_must_fit_in_the_periodic_lattice():
    """L < 2R+1 aliases the ball onto itself; that must fail loudly."""
    f = torch.rand(1, 8, 8, 8, dtype=torch.float64)
    with pytest.raises(ValueError, match="does not fit"):
        ball_reduce(f, 4)
    with pytest.raises(ValueError, match="does not fit"):
        ball_features(f, 4)
    ball_reduce(f, 3)  # 2·3+1 = 7 <= 8, fine


def test_unknown_target_name_is_rejected():
    f = torch.rand(1, 9, 9, 9, dtype=torch.float64)
    with pytest.raises(ValueError, match="unknown target"):
        build_targets(f, 4, names=("T0", "T9"))


# ---------------------------------------------------------------------------
# The bounded-receptive-field criterion, as arithmetic
# ---------------------------------------------------------------------------


def _reachable(step, layers):
    """The set of offsets a stack of ``layers`` layers with per-layer support
    ``step`` can reach, by repeated Minkowski sum."""
    reach = {(0, 0, 0)}
    for _ in range(layers):
        reach = {
            tuple(a + b for a, b in zip(u, v)) for u in reach for v in step
        }
    return reach


def test_both_architectures_reach_the_whole_target_ball():
    """Criterion 3 of ``notes/where_attention_can_win.md`` §6, checked as
    integer set arithmetic rather than asserted in prose.

    The targets are supported on the L1-ball of radius 4. GELT (R = 2 per
    layer) covers it after 2 of its 4 layers; the L-CNN (axis-aligned hops of
    length ≤ K = 2) after 3 of its 4. Neither arm is geometry-limited, so a
    difference between them is about the offset *weights* — which is the whole
    experiment. If this ever fails, R-C is measuring receptive field.
    """
    D, R, K, layers = 3, 2, 2, 4
    ball4 = {(0, 0, 0)} | {tuple(o) for o in l1_ball_offsets(D, 4)}

    gelt_step = {(0, 0, 0)} | {tuple(o) for o in l1_ball_offsets(D, R)}
    lcnn_step = {(0, 0, 0)}
    for mu in range(D):
        for k in range(1, K + 1):
            for s in (+1, -1):
                v = [0] * D
                v[mu] = s * k
                lcnn_step.add(tuple(v))

    assert ball4 <= _reachable(gelt_step, 2)
    assert ball4 <= _reachable(lcnn_step, 3)
    assert ball4 <= _reachable(gelt_step, layers)
    assert ball4 <= _reachable(lcnn_step, layers)
    # …and neither covers it in one layer, so depth is doing real work in both.
    assert not ball4 <= _reachable(gelt_step, 1)
    assert not ball4 <= _reachable(lcnn_step, 1)
