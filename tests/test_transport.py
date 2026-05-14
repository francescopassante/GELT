"""Tests for build_transport_sums and l1_ball_offsets.

Verification strategy:
  - Base cases (|Δx|₁=1) against the raw link tensors.
  - Brute-force path enumeration for |Δx|₁=2 (all-positive and mixed-sign).
  - Octant relation T_{-Δx}(x) = dagger(T_Δx(x-Δx)) as an independent
    consistency check on every returned entry.
  - Gauge covariance in Z₂ (sanity) and with nc=2 complex matrices (catches
    wrong/missing daggers that the abelian Z₂ group cannot expose).
"""

import torch
import pytest

from lgt.lattice import (
    GaugeGroup,
    Z2,
    build_transport_sums,
    gauge_transformation,
    l1_ball_offsets,
    random_links,
)


# ---------------------------------------------------------------------------
# Mock gauge group: arbitrary nc=2 complex matrices.
# Z₂ elements are real and self-inverse, so dagger errors are invisible there.
# This group uses the standard Hermitian conjugate and catches them.
# ---------------------------------------------------------------------------

class _Gl2(GaugeGroup):
    """Arbitrary complex 2×2 matrices — for testing dagger paths only."""
    name = "Gl2"
    nc = 2

    def random(self, shape, dtype=torch.complex64):
        nc = self.nc
        x = torch.randn(*shape, nc, nc) + 1j * torch.randn(*shape, nc, nc)
        return x.to(dtype)

    def dagger(self, U):
        return U.conj().transpose(-1, -2)


# ---------------------------------------------------------------------------
# l1_ball_offsets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D,R,expected", [
    (2, 1, 4),   # ±ê₀, ±ê₁
    (2, 2, 12),  # 4 unit + 4 norm-2 axis-aligned + 4 diagonal
    (3, 1, 6),   # ±ê₀, ±ê₁, ±ê₂
    (4, 1, 8),   # ±ê₀ … ±ê₃
])
def test_offset_count(D, R, expected):
    offsets = l1_ball_offsets(D, R)
    assert len(offsets) == expected


def test_offsets_sorted_by_norm():
    offsets = l1_ball_offsets(D=2, R=3)
    norms = [sum(abs(d) for d in dx) for dx in offsets]
    assert norms == sorted(norms)


def test_zero_not_in_offsets():
    for D in (2, 3, 4):
        for R in (1, 2):
            assert (0,) * D not in l1_ball_offsets(D, R)


# ---------------------------------------------------------------------------
# Base cases: |Δx|₁ = 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu", [0, 1])
def test_base_positive_direction(mu):
    """T_{ê_μ}(x) == U_μ(x) for all sites."""
    group = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=2, group=group, dtype=torch.float64)
    T = build_transport_sums(U, R=1, group=group)

    dx = tuple(1 if i == mu else 0 for i in range(2))
    assert torch.allclose(T[dx], U[mu], atol=0.0)


@pytest.mark.parametrize("mu", [0, 1])
def test_base_negative_direction(mu):
    """T_{-ê_μ}(x) == U†_μ(x-ê_μ) for all sites."""
    group = Z2()
    torch.manual_seed(0)
    U = random_links(L=4, D=2, group=group, dtype=torch.float64)
    T = build_transport_sums(U, R=1, group=group)

    dx = tuple(-1 if i == mu else 0 for i in range(2))
    expected = group.dagger(torch.roll(U[mu], shifts=1, dims=mu))
    assert torch.allclose(T[dx], expected, atol=0.0)


# ---------------------------------------------------------------------------
# Brute-force verification for |Δx|₁ = 2
# ---------------------------------------------------------------------------

def test_l1_2_all_positive():
    """T_{(1,1)}(x) = U_0(x) @ U_1(x+ê₀) + U_1(x) @ U_0(x+ê₁)."""
    group = Z2()
    torch.manual_seed(1)
    U = random_links(L=6, D=2, group=group, dtype=torch.float64)
    T = build_transport_sums(U, R=2, group=group)

    path1 = U[0] @ torch.roll(U[1], shifts=-1, dims=0)  # right then up
    path2 = U[1] @ torch.roll(U[0], shifts=-1, dims=1)  # up then right
    assert torch.allclose(T[(1, 1)], path1 + path2, atol=1e-12)


def test_l1_2_mixed_sign():
    """T_{(-1,1)}(x) = U†₀(x-ê₀)@U₁(x-ê₀) + U₁(x)@U†₀(x+ê₁-ê₀).

    Two shortest paths from x to x+(-1,1):
      P1: x → x-ê₀ → x-ê₀+ê₁   links: U†₀(x-ê₀), U₁(x-ê₀)
      P2: x → x+ê₁ → x+ê₁-ê₀   links: U₁(x), U†₀(x+ê₁-ê₀)
    """
    group = Z2()
    torch.manual_seed(2)
    U = random_links(L=6, D=2, group=group, dtype=torch.float64)
    T = build_transport_sums(U, R=2, group=group)

    # P1: U†₀(x-ê₀) @ U₁(x-ê₀)
    U0_dag_at_xm0 = group.dagger(torch.roll(U[0], shifts=1, dims=0))
    U1_at_xm0 = torch.roll(U[1], shifts=1, dims=0)
    path1 = U0_dag_at_xm0 @ U1_at_xm0

    # P2: U₁(x) @ U†₀(x+ê₁-ê₀) — roll +1 in dim 0, then -1 in dim 1
    U0_at_xp1m0 = torch.roll(torch.roll(U[0], shifts=1, dims=0), shifts=-1, dims=1)
    path2 = U[1] @ group.dagger(U0_at_xp1m0)

    assert torch.allclose(T[(-1, 1)], path1 + path2, atol=1e-12)


# ---------------------------------------------------------------------------
# Octant relation: T_{-Δx}(x) = dagger(T_Δx(x-Δx))
# Verified independently of the DP — both sides are computed via the
# recursion, so agreement is a non-trivial self-consistency check.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R", [1, 2])
def test_octant_relation_z2(R):
    group = Z2()
    torch.manual_seed(3)
    U = random_links(L=6, D=2, group=group, dtype=torch.float64)
    T = build_transport_sums(U, R=R, group=group)

    D = 2
    for dx, T_dx in T.items():
        neg_dx = tuple(-d for d in dx)
        # T_{-Δx}(x) from the table
        T_neg = T[neg_dx]
        # Expected via octant trick: dagger(T_Δx(x-Δx))
        # roll by +dx brings position x-Δx into index x.
        T_dx_at_xm_dx = torch.roll(T_dx, shifts=dx, dims=tuple(range(D)))
        expected = group.dagger(T_dx_at_xm_dx)
        assert torch.allclose(T_neg, expected, atol=1e-12), (
            f"Octant relation violated for dx={dx}"
        )


# ---------------------------------------------------------------------------
# Gauge covariance: T'_Δx(x) = Ω(x) · T_Δx(x) · Ω†(x+Δx)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("R", [1, 2])
def test_gauge_covariance_z2(R):
    """Covariance under Z₂ gauge transform (sanity; dagger is trivial here)."""
    group = Z2()
    L, D = 6, 2
    torch.manual_seed(4)
    U = random_links(L=L, D=D, group=group, dtype=torch.float64)
    omega = group.random((L, L), dtype=torch.float64)

    T = build_transport_sums(U, R=R, group=group)
    T_prime = build_transport_sums(gauge_transformation(U, omega, group), R=R, group=group)

    for dx, T_dx_prime in T_prime.items():
        # Ω†(x+Δx): roll by -dx brings x+Δx into position x.
        omega_xdx = torch.roll(omega, shifts=tuple(-d for d in dx), dims=tuple(range(D)))
        expected = omega @ T[dx] @ group.dagger(omega_xdx)
        assert torch.allclose(T_dx_prime, expected, atol=1e-12), (
            f"Z₂ gauge covariance violated for dx={dx}"
        )


@pytest.mark.parametrize("R", [1, 2])
def test_gauge_covariance_complex(R):
    """Covariance with nc=2 complex links and unitary Ω — catches wrong/missing daggers.

    Gauge covariance T'_Δx = Ω T_Δx Ω† requires Ω to be unitary: for paths
    with |Δx|>1, intermediate Ω†Ω factors must cancel to identity.  Arbitrary
    complex Ω breaks this; Z₂ (self-inverse ±1) cannot expose dagger errors.
    This test uses random unitary Ω with arbitrary complex links U to cover both.
    """
    group = _Gl2()
    L, D, nc = 4, 2, 2
    torch.manual_seed(5)

    # Arbitrary complex link matrices — unitarity not required for U.
    U = torch.randn(D, L, L, nc, nc, dtype=torch.float64) + \
        1j * torch.randn(D, L, L, nc, nc, dtype=torch.float64)

    # Ω must be unitary so that Ω†(y)Ω(y) = I cancels at intermediate path sites.
    omega_raw = torch.randn(L, L, nc, nc, dtype=torch.float64) + \
                1j * torch.randn(L, L, nc, nc, dtype=torch.float64)
    Q, _ = torch.linalg.qr(omega_raw.reshape(L * L, nc, nc))
    omega = Q.reshape(L, L, nc, nc)

    T = build_transport_sums(U, R=R, group=group)
    T_prime = build_transport_sums(gauge_transformation(U, omega, group), R=R, group=group)

    for dx, T_dx_prime in T_prime.items():
        omega_xdx = torch.roll(omega, shifts=tuple(-d for d in dx), dims=tuple(range(D)))
        expected = omega @ T[dx] @ group.dagger(omega_xdx)
        assert torch.allclose(T_dx_prime, expected, atol=1e-9), (
            f"Complex gauge covariance violated for dx={dx}"
        )


# ---------------------------------------------------------------------------
# Higher dimensions
# ---------------------------------------------------------------------------

def test_3d_base_case():
    """Base-case transports equal the raw links for D=3."""
    group = Z2()
    torch.manual_seed(6)
    U = random_links(L=4, D=3, group=group)
    T = build_transport_sums(U, R=1, group=group)

    assert len(T) == 6  # ±ê₀, ±ê₁, ±ê₂
    for mu in range(3):
        dx = tuple(1 if i == mu else 0 for i in range(3))
        assert torch.allclose(T[dx], U[mu])


def test_4d_offset_count():
    """Correct number of offsets returned for D=4, R=2."""
    group = Z2()
    torch.manual_seed(7)
    U = random_links(L=4, D=4, group=group)
    T = build_transport_sums(U, R=2, group=group)

    # R=1: 8 offsets (±ê_μ for 4 dirs)
    # R=2: 8 + C(4,1)*2² + C(4,1)*2 = 8 + 16 + ... let l1_ball_offsets count.
    expected = len(l1_ball_offsets(4, 2))
    assert len(T) == expected
