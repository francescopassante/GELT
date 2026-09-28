"""The spatial cubic group O_h on links (gelt/cubic.py) — S4's premises.

What the A₁⁺⁺ projection of the learned operator rests on:
  - the 48 maps are an exact action of O_h: composing two of them on links is,
    bit for bit, the element whose matrix is the product (so the reflection and
    permutation conventions agree with the matrices the characters are read
    from);
  - each map is a symmetry of the anisotropic Wilson action, and the classical
    0⁺⁺ operator — APE ladder included — is invariant under all 48 (the unit
    test S4 task 1 asks for; it also shows the smearing commutes with g);
  - the character table is the O_h one (orthonormality, Σ d² = 48), and the
    projectors put known operators where they belong: one plane's plaquette sum
    in A1g ⊕ Eg with the plane average as its A1g part, and Σ Im Tr U_k — a
    polar vector — entirely in T1u;
  - the regular-representation projectors behind ``irrep_fractions`` are a
    resolution of the identity into orthogonal projectors, so the C(0) shares
    add up to 1 exactly on any sample, and land where the operator is.
"""

import pytest
import torch

from gelt.cubic import (
    IRREP_DIM,
    IRREPS,
    apply_cubic,
    cubic_elements,
    cubic_matrix,
    irrep_characters,
    irrep_fractions,
    irrep_projections,
    regular_projectors,
)
from gelt.glueball import smearing_operator_basis, zero_momentum
from gelt.lattice import SU, action, random_links, rectangular_wilson_loop

ELEMENTS = cubic_elements()


def test_elements_are_the_48_signed_permutations():
    assert len(ELEMENTS) == 48
    mats = {tuple(cubic_matrix(g).flatten().tolist()) for g in ELEMENTS}
    assert len(mats) == 48
    assert ELEMENTS[0].perm == (1, 2, 3) and ELEMENTS[0].flips == ()
    for g in ELEMENTS:
        M = cubic_matrix(g)
        assert torch.equal(M.T @ M, torch.eye(3, dtype=M.dtype))


def test_character_table_is_orthonormal():
    chis = [irrep_characters(g) for g in ELEMENTS]
    for a in IRREPS:
        for b in IRREPS:
            s = sum(c[a] * c[b] for c in chis)
            assert s == (48 if a == b else 0), (a, b, s)
    assert sum(IRREP_DIM[r[:-1]] ** 2 for r in IRREPS) == 48
    # dimension = character of the identity
    for r in IRREPS:
        assert chis[0][r] == IRREP_DIM[r[:-1]]


def test_action_on_links_is_a_group_action():
    """g₂(g₁U) = (M₂M₁)U exactly, for every pair — conventions agree with the matrices."""
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(3, 4, su2, dtype=torch.complex128, Lt=2).unsqueeze(0)
    by_matrix = {tuple(cubic_matrix(g).flatten().tolist()): g for g in ELEMENTS}
    moved = {g: apply_cubic(U, su2, g, batch_dims=1) for g in ELEMENTS}
    for g1 in ELEMENTS:
        for g2 in ELEMENTS[::5]:
            g12 = by_matrix[tuple((cubic_matrix(g2) @ cubic_matrix(g1)).flatten().tolist())]
            assert torch.equal(apply_cubic(moved[g1], su2, g2, batch_dims=1), moved[g12]), (
                g1.label, g2.label)
    # and a non-identity element really moves the links
    assert not torch.equal(moved[ELEMENTS[1]], U)


def test_action_and_classical_operator_invariant_under_all_48():
    """S4 task 1: Wilson action (ξ = 3) and the classical Ō(t) at every APE level."""
    su2 = SU(2)
    torch.manual_seed(1)
    U = torch.stack([random_links(4, 4, su2, dtype=torch.complex128, Lt=3)
                     for _ in range(2)])
    levels = [0, 1, 3]
    S0 = action(U, su2, beta=2.4, xi=3.0)
    O0 = smearing_operator_basis(U, su2, levels, alpha=0.5)  # (n_lv, B, Lt)
    for g in ELEMENTS:
        gU = apply_cubic(U, su2, g, batch_dims=1)
        torch.testing.assert_close(action(gU, su2, beta=2.4, xi=3.0), S0,
                                   rtol=0, atol=1e-10, msg=g.label)
        torch.testing.assert_close(smearing_operator_basis(gU, su2, levels, alpha=0.5),
                                   O0, rtol=0, atol=1e-10, msg=g.label)


def _obar_g(U, group, f):
    return torch.stack([f(apply_cubic(U, group, g, batch_dims=1)) for g in ELEMENTS])


def test_one_plane_is_A1g_plus_Eg():
    su2 = SU(2)
    torch.manual_seed(2)
    U = torch.stack([random_links(3, 4, su2, dtype=torch.complex128, Lt=2)
                     for _ in range(2)])

    def plane(mu, nu):
        return lambda V: zero_momentum(rectangular_wilson_loop(V, su2, 1, 1, mu, nu))

    proj = irrep_projections(_obar_g(U, su2, plane(1, 2)), ELEMENTS)
    avg = (plane(1, 2)(U) + plane(1, 3)(U) + plane(2, 3)(U)) / 3
    torch.testing.assert_close(proj["A1g"], avg, rtol=0, atol=1e-12)
    torch.testing.assert_close(proj["Eg"], plane(1, 2)(U) - avg, rtol=0, atol=1e-12)
    for r in IRREPS:
        if r not in ("A1g", "Eg"):
            assert proj[r].abs().max() < 1e-12, r
    # the projections resolve the identity
    torch.testing.assert_close(sum(proj.values()), plane(1, 2)(U), rtol=0, atol=1e-12)
    frac = irrep_fractions(_obar_g(U, su2, plane(1, 2)), ELEMENTS)
    assert abs(sum(frac.values()) - 1) < 1e-12
    assert abs(frac["A1g"] + frac["Eg"] - 1) < 1e-12 and frac["Eg"] > 0.1


def test_polar_vector_is_T1u():
    """Σ_x Im Tr U_1: odd under x₁ → −x₁ (the link is daggered), carried by
    permutations like a coordinate — a polar vector. Needs Im Tr ≠ 0, so SU(3)."""
    su3 = SU(3)
    torch.manual_seed(3)
    U = random_links(3, 4, su3, dtype=torch.complex128, Lt=2).unsqueeze(0)

    def imtr(k):
        return lambda V: V[:, k].diagonal(dim1=-2, dim2=-1).sum(-1).imag.sum(dim=(2, 3, 4))

    obar = imtr(1)(U)
    assert obar.abs().max() > 1e-3  # the test is not vacuous
    proj = irrep_projections(_obar_g(U, su3, imtr(1)), ELEMENTS)
    torch.testing.assert_close(proj["T1u"], obar, rtol=0, atol=1e-12)
    for r in IRREPS:
        if r != "T1u":
            assert proj[r].abs().max() < 1e-12, r
    assert abs(irrep_fractions(_obar_g(U, su3, imtr(1)), ELEMENTS)["T1u"] - 1) < 1e-12


def test_regular_projectors_resolve_the_identity():
    P = regular_projectors(ELEMENTS)
    eye = torch.eye(48, dtype=torch.float64)
    torch.testing.assert_close(sum(P.values()), eye, rtol=0, atol=1e-12)
    for a in IRREPS:
        torch.testing.assert_close(P[a], P[a].T, rtol=0, atol=0)
        torch.testing.assert_close(P[a] @ P[a], P[a], rtol=0, atol=1e-12)
        # rank = d_Γ², the multiplicity of Γ in the regular representation
        assert round(P[a].trace().item()) == IRREP_DIM[a[:-1]] ** 2
        for b in IRREPS:
            if a != b:
                assert (P[a] @ P[b]).abs().max() < 1e-12


def test_projection_refuses_a_subset():
    with pytest.raises(ValueError):
        irrep_projections(torch.zeros(3, 2), ELEMENTS[:3])
