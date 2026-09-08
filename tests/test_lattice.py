"""Unit tests for lattice primitives.

Run with:  pytest test_lattice.py -v
"""

import pytest
import torch

from gelt.lattice import (
    SU,
    Z2,
    action,
    link_gauge_transformation,
    plaquette_tensor,
    random_links,
    topological_charge_density,
)


@pytest.fixture
def z2():
    return Z2()


def _random_omega(L: int, D: int, gaugegroup, dtype, seed: int = 42) -> torch.Tensor:
    """Sample a random gauge transformation Ω of shape (*Λ, nc, nc)."""
    torch.manual_seed(seed)
    return gaugegroup.random((L,) * D, dtype=dtype)


# ---------------------------------------------------------------------------
# Z₂ plaquette invariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,D", [(4, 2), (6, 2), (4, 3)])
def test_plaquette_bitexact_z2(z2, L, D):
    """Z₂ plaquettes are bit-exact after any gauge transformation (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=1)

    P_before = plaquette_tensor(U.unsqueeze(0), z2)[0]
    U_prime = link_gauge_transformation(U, omega, z2)
    P_after = plaquette_tensor(U_prime.unsqueeze(0), z2)[0]

    assert torch.equal(P_before, P_after), (
        f"Plaquettes not bit-exact after Z₂ gauge transform (L={L}, D={D}); "
        f"max diff = {(P_before - P_after).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Action invariance (general — holds for all unitary groups)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,D,beta", [(4, 2, 1.0), (6, 2, 2.5), (4, 3, 0.5)])
def test_action_invariant_z2(z2, L, D, beta):
    """Wilson action is invariant under gauge transformation (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=2)

    S_before = action(U.unsqueeze(0), z2, beta=beta)[0]
    U_prime = link_gauge_transformation(U, omega, z2)
    S_after = action(U_prime.unsqueeze(0), z2, beta=beta)[0]

    assert torch.equal(S_before, S_after), (
        f"Action not invariant under Z₂ gauge transform "
        f"(L={L}, D={D}, β={beta}); diff = {(S_before - S_after).abs().item()}"
    )


# ---------------------------------------------------------------------------
# Plaquette covariance: P'(x) = Ω(x) P(x) Ω†(x)
# This is the general identity for any unitary group; for Z₂ it reduces
# to the bit-exact test above, but the explicit form guards porting to SU(N).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("L,D", [(4, 2), (6, 2)])
def test_plaquette_covariance_z2(z2, L, D):
    """P'(x) = Ω(x) P(x) Ω†(x) holds exactly for Z₂ (float64)."""
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=3)

    P = plaquette_tensor(U.unsqueeze(0), z2)[0]
    U_prime = link_gauge_transformation(U, omega, z2)
    P_prime = plaquette_tensor(U_prime.unsqueeze(0), z2)[0]

    # Expected: omega[None] @ P @ dagger(omega)[None]
    # P has shape (n_pairs, *Λ, nc, nc); omega has shape (*Λ, nc, nc)
    P_expected = omega @ P @ z2.dagger(omega)  # broadcasts over n_pairs leading dim

    assert torch.allclose(P_prime, P_expected, atol=0.0), (
        f"Plaquette covariance P'=ΩPΩ† violated (L={L}, D={D}); "
        f"max diff = {(P_prime - P_expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Anisotropic action (β_t = β·ξ on temporal planes, β_s = β/ξ on spatial)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xi", [1.0, 2.0, 3.5])
def test_action_anisotropic_gauge_invariant_su2(xi):
    """The anisotropic Wilson action is gauge invariant for any ξ (SU(2))."""
    L, D = 4, 4
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(L, D, su2, dtype=torch.complex128)
    omega = _random_omega(L, D, su2, torch.complex128, seed=5)

    S_before = action(U.unsqueeze(0), su2, beta=2.0, xi=xi)[0]
    U_prime = link_gauge_transformation(U, omega, su2)
    S_after = action(U_prime.unsqueeze(0), su2, beta=2.0, xi=xi)[0]
    assert torch.allclose(S_before, S_after, atol=1e-10), (S_before, S_after, xi)


def test_action_anisotropic_bitexact_invariant_z2(z2):
    """Z₂: anisotropic action is bit-exact gauge invariant (float64)."""
    L, D = 4, 4
    torch.manual_seed(0)
    U = random_links(L, D, z2, dtype=torch.float64)
    omega = _random_omega(L, D, z2, torch.float64, seed=6)

    S_before = action(U.unsqueeze(0), z2, beta=2.0, xi=2.5)[0]
    U_prime = link_gauge_transformation(U, omega, z2)
    S_after = action(U_prime.unsqueeze(0), z2, beta=2.0, xi=2.5)[0]
    assert torch.equal(S_before, S_after)


def test_action_xi1_matches_isotropic_su2():
    """ξ=1 reproduces the isotropic action exactly (backward compatibility)."""
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(4, 4, su2, dtype=torch.complex128, N=3)
    assert torch.allclose(
        action(U, su2, beta=2.0), action(U, su2, beta=2.0, xi=1.0), atol=0.0
    )


def test_random_links_noncubic_shape():
    """``Lt`` produces a non-cubic lattice with the temporal extent on axis 0."""
    su2 = SU(2)
    U = random_links(5, 4, su2, dtype=torch.complex128, Lt=9)
    assert U.shape == (4, 9, 5, 5, 5, 2, 2)
    U_batched = random_links(5, 4, su2, dtype=torch.complex128, N=2, Lt=9)
    assert U_batched.shape == (2, 4, 9, 5, 5, 5, 2, 2)


# ---------------------------------------------------------------------------
# Topological charge density (D=4 only)
# ---------------------------------------------------------------------------


def test_topo_charge_invariant_su2():
    """q_x is gauge invariant: F→ΩFΩ† leaves Tr[FF] unchanged (SU(2), complex128)."""
    L, D = 4, 4
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(L, D, su2, dtype=torch.complex128)
    omega = _random_omega(L, D, su2, torch.complex128, seed=4)

    q_before = topological_charge_density(U.unsqueeze(0), su2)[0]
    U_prime = link_gauge_transformation(U, omega, su2)
    q_after = topological_charge_density(U_prime.unsqueeze(0), su2)[0]

    assert torch.allclose(q_before, q_after, atol=1e-12), (
        f"Topological charge density not gauge invariant (SU(2)); "
        f"max diff = {(q_before - q_after).abs().max().item()}"
    )


def test_topo_charge_nonzero_su2_zero_z2(z2):
    """q_x is generically nonzero for SU(2) but identically zero for Z₂."""
    L, D = 4, 4
    su2 = SU(2)
    torch.manual_seed(0)
    U_su2 = random_links(L, D, su2, dtype=torch.complex128)
    q_su2 = topological_charge_density(U_su2.unsqueeze(0), su2)[0]
    assert q_su2.abs().max() > 1e-6, "Expected nonzero q_x for SU(2) links."

    U_z2 = random_links(L, D, z2, dtype=torch.float64)
    q_z2 = topological_charge_density(U_z2.unsqueeze(0), z2)[0]
    assert q_z2.abs().max() < 1e-12, "Expected identically zero q_x for Z₂ links."


def test_topo_charge_requires_4d(z2):
    """Topological charge density rejects D≠4."""
    U = random_links(4, 3, z2, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="D=4"):
        topological_charge_density(U, z2)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------


def test_output_shape_preserved(z2):
    """link_gauge_transformation returns a tensor with the same shape as U."""
    L, D = 5, 2
    torch.manual_seed(0)
    U = random_links(L, D, z2)
    omega = _random_omega(L, D, z2, torch.float32)
    U_prime = link_gauge_transformation(U, omega, z2)
    assert U_prime.shape == U.shape


# ---------------------------------------------------------------------------
# SU(2) reunitarisation: the closed-form polar factor is a drop-in for the SVD
# ---------------------------------------------------------------------------


def test_su2_project_matches_svd_polar():
    """``SU(2).project`` takes a closed-form route (``_polar_factor_2x2``) rather
    than ``linalg.svd``, because APE smearing reprojects millions of links per
    training step. It must be the *same map*: the polar factor of M, rescaled by
    det^(1/2). Checked against the SVD route it replaced, on matrices far from
    the group (the hard case — squaring M squares its condition number, which is
    why the closed form works in double precision internally)."""
    torch.manual_seed(0)
    gg = SU(2)
    M = torch.randn(4096, 2, 2, dtype=torch.complex64)

    W, _, Vh = torch.linalg.svd(M)
    svd_polar = W @ Vh
    reference = svd_polar / torch.linalg.det(svd_polar).pow(0.5).unsqueeze(
        -1
    ).unsqueeze(-1)

    Q = gg.project(M)
    assert (Q - reference).abs().max() < 1e-4  # complex64 output of an
    #                                           ill-conditioned polar factor
    eye = torch.eye(2, dtype=Q.dtype)
    assert (Q @ gg.dagger(Q) - eye).abs().max() < 1e-6
    assert (torch.linalg.det(Q) - 1).abs().max() < 1e-6


def test_su2_project_agrees_with_svd_on_near_group_input():
    """On the input the smearing hot path actually feeds it — a weighted sum of
    near-parallel group elements — the two routes agree to complex64 rounding,
    so switching them changes no measured number."""
    torch.manual_seed(1)
    gg = SU(2)
    U = gg.random((20000,), dtype=torch.complex64)
    staples = sum(gg.random((20000,), dtype=torch.complex64) for _ in range(4))
    V = 0.5 * U + 0.125 * staples

    W, _, Vh = torch.linalg.svd(V)
    svd_polar = W @ Vh
    reference = svd_polar / torch.linalg.det(svd_polar).pow(0.5).unsqueeze(
        -1
    ).unsqueeze(-1)
    assert (gg.project(V) - reference).abs().max() < 1e-5


def test_su3_project_still_uses_the_general_route():
    """The closed form is 2×2-only; nc ≠ 2 must keep the SVD path."""
    torch.manual_seed(2)
    gg = SU(3)
    Q = gg.project(torch.randn(512, 3, 3, dtype=torch.complex128))
    eye = torch.eye(3, dtype=Q.dtype)
    assert (Q @ gg.dagger(Q) - eye).abs().max() < 1e-12
    assert (torch.linalg.det(Q) - 1).abs().max() < 1e-12
