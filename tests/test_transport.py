"""Tests for build_transport_average and l1_ball_offsets.

build_transport_average returns the full signed L1-ball: every offset 0 < |Δx|₁ ≤ R,
including mixed-sign offsets.  The DP step uses U_μ for Δx_μ > 0 and U†_μ(x−ê_μ)
for Δx_μ < 0; sub-offsets always have strictly smaller |Δx|₁ and compatible signs,
so a single |Δx|₁-ordered pass is enough.

Verification strategy:
  - Counts and base cases for every component sign (+/−) and every dimension.
  - Brute-force checks for the simplest non-trivial offsets in each octant
    pattern: (1,1) all-positive, (−1,−1) all-negative, (1,−1) mixed-sign.
  - Octant relation T_{−Δx}(x) == dagger(T_Δx(x−Δx)) verified on every offset.
    Although the DP no longer relies on this trick (each offset is computed
    independently), the equality is a property of the math and a strong
    consistency check.
  - Gauge covariance T'_Δx(x) = Ω(x) · T_Δx(x) · Ω†(x+Δx) on every offset,
    for Z₂ (real) and for nc=2 complex with unitary Ω.  Z₂ is self-inverse so
    dagger bugs are invisible there; the complex case is the real audit.
"""

import pytest
import torch

from gelt.lattice import (
    Z2,
    GaugeGroup,
    build_transport_average,
    l1_ball_offsets,
    link_gauge_transformation,
    random_links,
)


def _at(T, offsets, dx):
    """Look up T_Δx in the (n_offsets, *Λ, nc, nc) tensor by offset tuple."""
    return T[offsets.index(dx)]


def _build(U, R, gaugegroup):
    """Build transports for a single config while requiring batched input."""
    return build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gaugegroup)[0]


# ---------------------------------------------------------------------------
# Mock gauge group: arbitrary nc=2 complex matrices.
# Z₂ elements are real and self-inverse, so dagger errors are invisible there.
# ---------------------------------------------------------------------------


class _Gl2(GaugeGroup):
    """Arbitrary complex 2×2 matrices — for testing dagger paths only."""

    name = "Gl2"
    nc = 2

    def random(self, shape, dtype=torch.complex64):
        nc = self.nc
        return (torch.randn(*shape, nc, nc) + 1j * torch.randn(*shape, nc, nc)).to(
            dtype
        )

    def dagger(self, U):
        return U.conj().transpose(-1, -2)


def _unitary_omega(L, D, nc, seed):
    """Random unitary Ω of shape (*Λ, nc, nc)."""
    torch.manual_seed(seed)
    raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        L**D, nc, nc, dtype=torch.float64
    )
    Q, _ = torch.linalg.qr(raw)
    return Q.reshape(*([L] * D), nc, nc)


# ---------------------------------------------------------------------------
# l1_ball_offsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,R,expected",
    [
        (2, 1, 4),
        (2, 2, 12),
        (3, 1, 6),
        (4, 1, 8),
        (4, 2, 40),
    ],
)
def test_l1_ball_offset_count(D, R, expected):
    assert len(l1_ball_offsets(D, R)) == expected


def test_l1_ball_offsets_sorted_by_norm():
    norms = [sum(abs(d) for d in dx) for dx in l1_ball_offsets(D=2, R=3)]
    assert norms == sorted(norms)


def test_l1_ball_offsets_no_zero():
    for D in (2, 3, 4):
        assert (0,) * D not in l1_ball_offsets(D, R=2)


# ---------------------------------------------------------------------------
# build_transport_average: count + signs covered
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,R,expected",
    [
        (2, 1, 4),
        (2, 2, 12),
        (3, 1, 6),
        (4, 1, 8),
        (4, 2, 40),
    ],
)
def test_full_l1_ball_count(D, R, expected):
    gaugegroup = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=D, gaugegroup=gaugegroup)
    T = _build(U, R=R, gaugegroup=gaugegroup)
    assert T.shape[0] == expected


def test_offset_axis_matches_l1_ball():
    """Offset axis length matches l1_ball_offsets exactly; that helper IS the index."""
    gaugegroup = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=3, gaugegroup=gaugegroup)
    T = _build(U, R=2, gaugegroup=gaugegroup)
    assert T.shape[0] == len(l1_ball_offsets(D=3, R=2))


# ---------------------------------------------------------------------------
# Batched DP: one pass over (N, D, *Λ, nc, nc) must match N independent passes.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("R", [1, 2])
def test_batched_matches_unbatched_z2(R):
    """Batched DP and per-config DP must produce bit-identical transports."""
    gaugegroup = Z2()
    L, D, N = 4, 2, 5
    torch.manual_seed(9)
    configs = torch.stack(
        [
            random_links(L=L, D=D, gaugegroup=gaugegroup, dtype=torch.float64)
            for _ in range(N)
        ],
        dim=0,
    )

    T_batched = build_transport_average(configs, R=R, gaugegroup=gaugegroup)
    T_per = torch.stack(
        [
            _build(configs[n], R=R, gaugegroup=gaugegroup)
            for n in range(N)
        ],
        dim=0,
    )

    assert T_batched.shape == T_per.shape
    assert torch.allclose(T_batched, T_per, atol=0.0)


def test_batched_matches_unbatched_complex():
    """Same equivalence at nc=2 complex — catches dagger/roll bugs in the batched path."""
    gaugegroup = _Gl2()
    L, D, nc, N = 4, 2, 2, 3
    torch.manual_seed(10)
    configs = torch.randn(N, D, L, L, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        N, D, L, L, nc, nc, dtype=torch.float64
    )

    T_batched = build_transport_average(configs, R=2, gaugegroup=gaugegroup)
    T_per = torch.stack(
        [
            _build(configs[n], R=2, gaugegroup=gaugegroup)
            for n in range(N)
        ],
        dim=0,
    )

    assert T_batched.shape == T_per.shape
    assert torch.allclose(T_batched, T_per, atol=1e-12)


# ---------------------------------------------------------------------------
# Base cases (|Δx|₁ = 1): one entry per ±ê_μ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mu", [0, 1])
def test_base_case_positive(mu):
    """T_{+ê_μ}(x) == U_μ(x)."""
    gaugegroup = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=2, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=1, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=2, R=1)

    dx = tuple(1 if i == mu else 0 for i in range(2))
    assert torch.allclose(_at(T, offsets, dx), U[mu], atol=0.0)


@pytest.mark.parametrize("mu", [0, 1])
def test_base_case_negative(mu):
    """T_{−ê_μ}(x) == U†_μ(x−ê_μ)."""
    gaugegroup = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=2, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=1, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=2, R=1)

    dx = tuple(-1 if i == mu else 0 for i in range(2))
    expected = gaugegroup.dagger(torch.roll(U[mu], shifts=1, dims=mu))
    assert torch.allclose(_at(T, offsets, dx), expected, atol=0.0)


# ---------------------------------------------------------------------------
# Brute-force checks at |Δx|₁ = 2, one per octant pattern
# ---------------------------------------------------------------------------


def test_brute_force_positive_l1_2():
    """T_{(1,1)}(x) = (U_0(x) @ U_1(x+ê_0) + U_1(x) @ U_0(x+ê_1)) / 2.

    Two shortest paths, averaged (N_Δx = 2!/(1!·1!) = 2).
    """
    gaugegroup = Z2()
    torch.manual_seed(1)
    U = random_links(L=6, D=2, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=2, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=2, R=2)

    expected = (
        U[0] @ torch.roll(U[1], shifts=-1, dims=0)
        + U[1] @ torch.roll(U[0], shifts=-1, dims=1)
    ) / 2
    assert torch.allclose(_at(T, offsets, (1, 1)), expected, atol=1e-12)


def test_brute_force_negative_l1_2():
    """Two shortest paths from x to x − (1,1):
    P1: x → x−ê_0 → x−ê_0−ê_1    links: U†_0(x−ê_0), U†_1(x−ê_0−ê_1)
    P2: x → x−ê_1 → x−ê_0−ê_1    links: U†_1(x−ê_1), U†_0(x−ê_0−ê_1)
    """
    gaugegroup = Z2()
    torch.manual_seed(2)
    U = random_links(L=6, D=2, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=2, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=2, R=2)

    path1 = gaugegroup.dagger(torch.roll(U[0], shifts=1, dims=0)) @ gaugegroup.dagger(
        torch.roll(torch.roll(U[1], shifts=1, dims=0), shifts=1, dims=1)
    )
    path2 = gaugegroup.dagger(torch.roll(U[1], shifts=1, dims=1)) @ gaugegroup.dagger(
        torch.roll(torch.roll(U[0], shifts=1, dims=0), shifts=1, dims=1)
    )
    expected = (path1 + path2) / 2
    assert torch.allclose(_at(T, offsets, (-1, -1)), expected, atol=1e-12)


def test_brute_force_mixed_l1_2():
    """T_{(1,−1)}(x): two shortest paths
    P1: x → x+ê_0 → x+ê_0−ê_1    links: U_0(x), U†_1(x+ê_0−ê_1)
    P2: x → x−ê_1 → x+ê_0−ê_1    links: U†_1(x−ê_1), U_0(x−ê_1)
    """
    gaugegroup = Z2()
    torch.manual_seed(3)
    U = random_links(L=6, D=2, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=2, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=2, R=2)

    # U_0(x) @ U†_1(x + ê_0 − ê_1): roll U[1] by (-1, +1) along (0, 1) brings (x + ê_0 − ê_1) → x.
    U1d_shift = gaugegroup.dagger(
        torch.roll(torch.roll(U[1], shifts=-1, dims=0), shifts=1, dims=1)
    )
    path1 = U[0] @ U1d_shift

    # U†_1(x − ê_1) @ U_0(x − ê_1): roll along dim 1 by +1.
    U1d_at_xm1 = gaugegroup.dagger(torch.roll(U[1], shifts=1, dims=1))
    U0_at_xm1 = torch.roll(U[0], shifts=1, dims=1)
    path2 = U1d_at_xm1 @ U0_at_xm1

    expected = (path1 + path2) / 2
    assert torch.allclose(_at(T, offsets, (1, -1)), expected, atol=1e-12)


def test_3d_base_cases():
    """All ±ê_μ entries present in 3D."""
    gaugegroup = Z2()
    torch.manual_seed(6)
    U = random_links(L=4, D=3, gaugegroup=gaugegroup)
    T = _build(U, R=1, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=3, R=1)

    assert T.shape[0] == 6
    for mu in range(3):
        pos = tuple(1 if i == mu else 0 for i in range(3))
        neg = tuple(-1 if i == mu else 0 for i in range(3))
        assert torch.allclose(_at(T, offsets, pos), U[mu])
        assert torch.allclose(
            _at(T, offsets, neg),
            gaugegroup.dagger(torch.roll(U[mu], shifts=1, dims=mu)),
        )


# ---------------------------------------------------------------------------
# Octant relation: T_{−Δx}(x) == dagger(T_Δx(x − Δx)) for every offset.
# Math property — not used in the DP, but a strong consistency check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("R", [1, 2])
def test_octant_relation_every_offset(R):
    gaugegroup = Z2()
    torch.manual_seed(3)
    D = 2
    U = random_links(L=6, D=D, gaugegroup=gaugegroup, dtype=torch.float64)
    T = _build(U, R=R, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=D, R=R)

    for i, dx in enumerate(offsets):
        neg_dx = tuple(-d for d in dx)
        manual = gaugegroup.dagger(torch.roll(T[i], shifts=dx, dims=tuple(range(D))))
        assert torch.allclose(_at(T, offsets, neg_dx), manual, atol=1e-12), (
            f"Octant relation failed for dx={dx}"
        )


def test_octant_relation_mixed_complex():
    """Same identity at nc=2 complex — the case where dagger errors would show."""
    gaugegroup = _Gl2()
    L, D, nc = 4, 2, 2
    torch.manual_seed(8)
    U = torch.randn(D, L, L, nc, nc, dtype=torch.complex128)
    U += 1j * torch.randn(D, L, L, nc, nc, dtype=torch.complex128).imag

    T = _build(U, R=2, gaugegroup=gaugegroup)
    offsets = l1_ball_offsets(D=D, R=2)

    for i, dx in enumerate(offsets):
        neg_dx = tuple(-d for d in dx)
        manual = gaugegroup.dagger(torch.roll(T[i], shifts=dx, dims=tuple(range(D))))
        assert torch.allclose(_at(T, offsets, neg_dx), manual, atol=1e-10), (
            f"Octant relation (complex) failed for dx={dx}"
        )


# ---------------------------------------------------------------------------
# Gauge covariance: T'_Δx(x) = Ω(x) · T_Δx(x) · Ω†(x+Δx) on every offset.
# Ω must be unitary for |Δx|₁ > 1 so Ω†Ω cancels at intermediate path sites.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("R", [1, 2])
def test_gauge_covariance_z2(R):
    gaugegroup = Z2()
    L, D = 6, 2
    torch.manual_seed(4)
    U = random_links(L=L, D=D, gaugegroup=gaugegroup, dtype=torch.float64)
    omega = gaugegroup.random((L, L), dtype=torch.float64)

    T = _build(U, R=R, gaugegroup=gaugegroup)
    T_prime = _build(
        link_gauge_transformation(U, omega, gaugegroup), R=R, gaugegroup=gaugegroup
    )
    offsets = l1_ball_offsets(D=D, R=R)

    for i, dx in enumerate(offsets):
        omega_xdx = torch.roll(
            omega, shifts=tuple(-d for d in dx), dims=tuple(range(D))
        )
        expected = omega @ T[i] @ gaugegroup.dagger(omega_xdx)
        assert torch.allclose(T_prime[i], expected, atol=1e-12), (
            f"Z₂ gauge covariance violated for dx={dx}"
        )


@pytest.mark.parametrize("R", [1, 2])
def test_gauge_covariance_complex(R):
    """Full L1-ball covariance with nc=2 complex links and unitary Ω."""
    gaugegroup = _Gl2()
    L, D, nc = 4, 2, 2
    torch.manual_seed(5)
    U = torch.randn(D, L, L, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        D, L, L, nc, nc, dtype=torch.float64
    )
    omega = _unitary_omega(L, D, nc, seed=5)

    T = _build(U, R=R, gaugegroup=gaugegroup)
    T_prime = _build(
        link_gauge_transformation(U, omega, gaugegroup), R=R, gaugegroup=gaugegroup
    )
    offsets = l1_ball_offsets(D=D, R=R)

    for i, dx in enumerate(offsets):
        omega_xdx = torch.roll(
            omega, shifts=tuple(-d for d in dx), dims=tuple(range(D))
        )
        expected = omega @ T[i] @ gaugegroup.dagger(omega_xdx)
        assert torch.allclose(T_prime[i], expected, atol=1e-9), (
            f"Complex gauge covariance violated for dx={dx}"
        )
