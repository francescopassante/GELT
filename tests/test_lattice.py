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
