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


@pytest.mark.parametrize("definition", ["clover", "plaquette"])
def test_topo_charge_invariant_su2(definition):
    """q_x is gauge invariant: F→ΩFΩ† leaves Tr[FF] unchanged (SU(2), complex128)."""
    L, D = 4, 4
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(L, D, su2, dtype=torch.complex128)
    omega = _random_omega(L, D, su2, torch.complex128, seed=4)

    q_before = topological_charge_density(U.unsqueeze(0), su2, definition=definition)[0]
    U_prime = link_gauge_transformation(U, omega, su2)
    q_after = topological_charge_density(U_prime.unsqueeze(0), su2, definition=definition)[0]

    assert torch.allclose(q_before, q_after, atol=1e-12), (
        f"Topological charge density not gauge invariant (SU(2), {definition}); "
        f"max diff = {(q_before - q_after).abs().max().item()}"
    )


@pytest.mark.parametrize("definition", ["clover", "plaquette"])
def test_topo_charge_nonzero_su2_zero_z2(z2, definition):
    """q_x is generically nonzero for SU(2) but identically zero for Z₂."""
    L, D = 4, 4
    su2 = SU(2)
    torch.manual_seed(0)
    U_su2 = random_links(L, D, su2, dtype=torch.complex128)
    q_su2 = topological_charge_density(U_su2.unsqueeze(0), su2, definition=definition)[0]
    assert q_su2.abs().max() > 1e-6, "Expected nonzero q_x for SU(2) links."

    U_z2 = random_links(L, D, z2, dtype=torch.float64)
    q_z2 = topological_charge_density(U_z2.unsqueeze(0), z2, definition=definition)[0]
    assert q_z2.abs().max() < 1e-12, "Expected identically zero q_x for Z₂ links."


def test_topo_charge_requires_4d(z2):
    """Topological charge density rejects D≠4."""
    U = random_links(4, 3, z2, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="D=4"):
        topological_charge_density(U, z2)


def test_topo_charge_rejects_unknown_definition():
    """An unknown discretisation name is an error, not a silent fallback."""
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(4, 4, su2, dtype=torch.complex128).unsqueeze(0)
    with pytest.raises(ValueError, match="clover"):
        topological_charge_density(U, su2, definition="clovr")


def test_topo_charge_clover_refuses_precomputed_plaquettes():
    """The ``plaquettes`` shortcut is plaquette-only: the clover leaves are
    similarity transforms of plaquettes at four different basepoints, so reusing
    the tensor would silently evaluate F in the wrong colour frame."""
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(4, 4, su2, dtype=torch.complex128).unsqueeze(0)
    P = plaquette_tensor(U, su2)
    with pytest.raises(ValueError, match="clover"):
        topological_charge_density(U, su2, plaquettes=P)
    # …and it is still accepted, and exact, for the naive definition.
    q_ref = topological_charge_density(U, su2, definition="plaquette")
    q_pre = topological_charge_density(U, su2, plaquettes=P, definition="plaquette")
    assert torch.equal(q_ref, q_pre)


# ---------------------------------------------------------------------------
# Parity: the clover charge is odd under lattice reflections, the naive one is not
# ---------------------------------------------------------------------------


def _reflect_field(T: torch.Tensor, axis: int) -> torch.Tensor:
    """``x_axis → (−x_axis) mod L`` on a field whose lattice axis is ``axis``.

    ``flip`` gives ``T[L−1−i]``; the extra ``roll(+1)`` turns that into
    ``T[(−i) mod L]``, i.e. a reflection about the origin rather than about the
    half-integer point between sites.
    """
    return torch.roll(torch.flip(T, dims=[axis]), shifts=1, dims=axis)


def _reflect_links(U: torch.Tensor, gaugegroup, d: int = 1) -> torch.Tensor:
    """Exact lattice reflection ``P: x_d → −x_d`` applied to links ``(D,*Λ,nc,nc)``.

    A link in direction ν ≠ d is carried along: ``U'_ν(y) = U_ν(Py)``. The link
    along d is reversed as well as moved, so it becomes the dagger of the link on
    the other side: ``U'_d(y) = U_d(P(y+d̂))†``, and ``P(y+d̂) = Py − d̂`` — hence
    shift *down* by one first and reflect after. Getting that order wrong still
    produces a plausible-looking map that does not preserve the Wilson action,
    which is why the test checks the action first.
    """
    D = U.shape[0]
    out = []
    for nu in range(D):
        if nu != d:
            out.append(_reflect_field(U[nu], d))
        else:
            back = torch.roll(U[nu], shifts=+1, dims=d)  # U_d(y − d̂)
            out.append(gaugegroup.dagger(_reflect_field(back, d)))
    return torch.stack(out, dim=0)


def test_reflection_preserves_wilson_action():
    """The reflection map used below is an exact symmetry of the Wilson action."""
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(6, 4, su2, dtype=torch.complex128)
    U_refl = _reflect_links(U, su2, d=1)

    S = action(U.unsqueeze(0), su2, beta=2.4)[0]
    S_refl = action(U_refl.unsqueeze(0), su2, beta=2.4)[0]
    assert torch.allclose(S, S_refl, atol=1e-9), (
        f"Reflection is not a symmetry of the action: {S.item():.8f} → "
        f"{S_refl.item():.8f}"
    )


def test_clover_charge_is_parity_odd():
    """q_clov(x) is exactly odd under reflection, site by site and summed.

    Parity-oddness is the property the naive density lacks (see the companion
    test), and it is what makes the clover density a usable regression target.
    The flow-free topology study this was built for is closed
    (``notes/where_attention_can_win.md`` §4); the fix it forced is kept because
    the naive density was simply wrong.
    """
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(6, 4, su2, dtype=torch.complex128)
    U_refl = _reflect_links(U, su2, d=1)

    q = topological_charge_density(U.unsqueeze(0), su2)[0]
    q_refl = topological_charge_density(U_refl.unsqueeze(0), su2)[0]

    # Parity-odd means q'(y) = −q(Py) at every site, not merely Q' = −Q.
    site_residual = (q_refl + _reflect_field(q, 1)).abs().max().item()
    assert site_residual < 1e-14, (
        f"Clover density is not parity-odd site by site: max residual "
        f"{site_residual:.3e}"
    )
    Q, Q_refl = q.sum().item(), q_refl.sum().item()
    assert abs(Q) > 1e-3, "Test config has no charge to speak of; pick another seed."
    assert abs(Q + Q_refl) < 1e-12, f"Q is not parity-odd: {Q:+.6f} → {Q_refl:+.6f}"


def test_plaquette_charge_is_not_parity_odd():
    """…and the naive density is not — the defect that motivates the clover.

    Caveat 7 of CLAUDE.md, reproduced exactly: the sign of Q flips but the
    magnitude does not, on a configuration whose action the same map preserves to
    all printed digits.
    """
    su2 = SU(2)
    torch.manual_seed(0)
    U = random_links(6, 4, su2, dtype=torch.complex128)
    U_refl = _reflect_links(U, su2, d=1)

    q = topological_charge_density(U.unsqueeze(0), su2, definition="plaquette")[0]
    q_refl = topological_charge_density(U_refl.unsqueeze(0), su2, definition="plaquette")[0]
    Q, Q_refl = q.sum().item(), q_refl.sum().item()

    # Not a near-miss: |Q + Q'| is of the same order as |Q| itself.
    assert abs(Q + Q_refl) > 0.1 * abs(Q), (
        f"The naive density unexpectedly looks parity-odd ({Q:+.6f} → "
        f"{Q_refl:+.6f}); if this ever starts passing the definition changed."
    )


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
