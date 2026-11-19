"""The force-matching identity behind gelt/vacuum.py, each piece against a known answer.

- the generators are su(N) with Tr(T^a T^b) = δ/2, and σ/2 at N = 2;
- the staple forces are minus the Lie gradient of the anisotropic Wilson action
  (autograd through ``lattice.action``, complex128);
- per-link force norms are gauge invariant;
- ``lie_grad`` / ``basis_gradients`` agree with a finite difference along an
  exact group geodesic, through APE smearing and through the transport;
- the classical bases are gauge invariant;
- **Stein's identity on the marginal of one timeslice**: for any smooth h of the
  slice, E[∇h · (F_s + F_temp)] + E[Δh] = 0 exactly (compact group, Haar
  measure, no boundary), at any Lt. With h the plaquette sum, Δh = −4 C_F h.
  This checks the sign and normalisation of the whole score — and the identity
  is violated by tens of σ if F_temp's sign is flipped, so the test has power;
- the frozen-slice sweep never writes a frozen link and does move the others.
"""

import functools
import math

import pytest
import torch

from gelt.lattice import SU, action, link_gauge_transformation, random_links
from gelt.sampler import heatbath_overrelaxation_sweep
from gelt.vacuum import (
    basis_gradients,
    bilinear_basis,
    casimir_fundamental,
    frozen_slice_mask,
    lie_generators,
    lie_grad,
    loop_basis,
    spatial_force,
    spatial_slices,
    su2_exp,
    temporal_force,
)

G = SU(2)


def test_generators_su2_and_su3():
    for nc in (2, 3):
        T = lie_generators(nc)
        assert T.shape[0] == nc * nc - 1
        tr = torch.einsum("aij,bji->ab", T, T)
        assert torch.allclose(tr, 0.5 * torch.eye(nc * nc - 1, dtype=tr.dtype), atol=1e-14)
        assert torch.allclose(T, T.conj().transpose(-1, -2))
        assert T.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-14
    s = torch.tensor([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
                     dtype=torch.complex128)
    assert torch.allclose(lie_generators(2), s / 2)
    assert abs(casimir_fundamental(2) - 0.75) < 1e-15


def test_su2_exp_is_the_matrix_exponential():
    torch.manual_seed(0)
    X = torch.randn(50, 3, dtype=torch.float64) * 2
    E = su2_exp(X)
    ref = torch.matrix_exp(1j * torch.einsum("na,aij->nij", X.to(torch.complex128),
                                             lie_generators(2)))
    assert (E - ref).abs().max() < 1e-12
    assert (su2_exp(torch.zeros(3, dtype=torch.float64)) - torch.eye(2)).abs().max() < 1e-15


@pytest.mark.parametrize("xi", [1.0, 2.5])
def test_forces_are_minus_the_lie_gradient_of_the_action(xi):
    torch.manual_seed(1)
    beta = 2.3
    U4 = random_links(3, 4, G, dtype=torch.float64, N=2, Lt=4)  # (2, 4, 4, 3,3,3, 2,2)
    g, _ = lie_grad(lambda U: action(U, G, beta, xi=xi), U4)  # (2, 4, 4, 3,3,3, 3)
    minus_dS = -g[:, 1:].movedim(2, 1).reshape(8, 3, 3, 3, 3, 3)  # spatial links, slice layout
    F = temporal_force(U4, G, beta, xi) + spatial_force(spatial_slices(U4), G, beta / xi)
    assert (F - minus_dS).abs().max() < 1e-12


def test_force_norm_is_gauge_invariant():
    torch.manual_seed(2)
    U4 = random_links(3, 4, G, dtype=torch.float64, Lt=4)  # (4, 4, 3,3,3, 2,2)
    omega = G.random((4, 3, 3, 3), dtype=torch.float64)
    Ug = link_gauge_transformation(U4, omega, G)
    for f in (lambda U: temporal_force(U[None], G, 2.4, 3.0),
              lambda U: spatial_force(spatial_slices(U[None]), G, 0.8)):
        n0, n1 = f(U4).norm(dim=-1), f(Ug).norm(dim=-1)
        assert (n0 - n1).abs().max() < 1e-12


def _geodesic_check(ops_fn, U3, X, h=1e-4):
    """Σ grad·X against a central difference of ops along exp(i s X·T) U."""
    grads = basis_gradients(U3, ops_fn)  # (K, B, d, *Λ, 3)
    pred = (grads * X).flatten(2).sum(-1)  # (K, B)
    up = ops_fn(su2_exp(h * X) @ U3).T
    dn = ops_fn(su2_exp(-h * X) @ U3).T
    fd = (up - dn) / (2 * h)
    return pred, fd


def test_lie_gradient_matches_a_finite_difference_through_smearing_and_transport():
    torch.manual_seed(3)
    U3 = random_links(4, 3, G, dtype=torch.float64, N=2)
    X = torch.randn(U3.shape[:-2] + (3,), dtype=torch.float64)
    for fn in (lambda U: loop_basis(U, G, ((1, 1), (1, 2)), (0, 2))[0],
               lambda U: bilinear_basis(U, G, R=2)[0]):
        pred, fd = _geodesic_check(fn, U3, X)
        assert torch.allclose(pred, fd, rtol=1e-6, atol=1e-6), (pred, fd)


def test_network_lie_gradient_through_transport_and_double_backprop():
    """Φ_net = Σ_x GELT(W(U), T(U)): its Lie gradient against a finite difference
    along an exact geodesic (the derivative must see the transport, not only the
    plaquettes), and the force-matching loss has a nonzero parameter gradient."""
    from gelt.blocks import GELT
    from gelt.lattice import build_transport_average, plaquette_tensor

    torch.manual_seed(7)
    net = GELT(gaugegroup=G, L=4, D=3, R=2, nhead=2, gemhsa_layers=2, d_qkv=6,
               dtype=torch.complex128, mlp_hidden=8, mlp_out=1, reduction="none",
               init_scale=1.0, mlp_zero_init=False, d_model=6, in_channels=3).double()

    def phi(U):
        return net(plaquette_tensor(U, G), build_transport_average(U, 2, G)).flatten(1).sum(1)

    U3 = random_links(4, 3, G, dtype=torch.float64, N=1)
    X = torch.randn(U3.shape[:-2] + (3,), dtype=torch.float64)
    g, _ = lie_grad(phi, U3)
    h = 1e-4
    fd = (phi(su2_exp(h * X) @ U3) - phi(su2_exp(-h * X) @ U3)) / (2 * h)
    assert torch.allclose((g * X).sum(), fd.sum(), rtol=1e-6, atol=1e-8)

    # the transport matters: freezing it changes the gradient
    T0 = build_transport_average(U3, 2, G)
    g_noT, _ = lie_grad(lambda U: net(plaquette_tensor(U, G), T0).flatten(1).sum(1), U3)
    assert (g - g_noT).abs().max() > 1e-6

    target = torch.randn_like(g)
    gc, _ = lie_grad(phi, U3, create_graph=True)
    loss = ((gc - target) ** 2).mean()
    loss.backward()
    gn = sum(float(p.grad.abs().sum()) for p in net.parameters() if p.grad is not None)
    assert gn > 0


def test_classical_bases_are_gauge_invariant():
    torch.manual_seed(4)
    U3 = random_links(4, 3, G, dtype=torch.float64)
    omega = G.random((4, 4, 4), dtype=torch.float64)
    Ug = link_gauge_transformation(U3, omega, G)
    for fn in (lambda U: loop_basis(U, G, ((1, 1), (1, 2), (2, 2)), (0, 1))[0],
               lambda U: bilinear_basis(U, G, R=3)[0]):
        a, b = fn(U3[None]), fn(Ug[None])
        assert torch.allclose(a, b, rtol=1e-10, atol=1e-10)


@pytest.fixture(scope="module")
def small_ensemble():
    """A short exact SU(2) chain on 4 × 4³, anisotropic (ξ = 2)."""
    torch.manual_seed(5)
    beta, xi = 2.4, 2.0
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=1, xi=xi)
    U = random_links(4, 4, G, dtype=torch.float64, Lt=4)
    for _ in range(40):
        U, _ = sweep(U, G, beta)
    cfgs = []
    for _ in range(160):
        U, _ = sweep(U, G, beta)
        cfgs.append(U.clone())
    return torch.stack(cfgs), beta, xi


def _stein_terms(U4, beta, xi, sign=1.0):
    U3 = spatial_slices(U4)
    grad_h = spatial_force(U3, G, float(G.nc))  # ∇ Σ_p Re Tr P
    F = spatial_force(U3, G, beta / xi) + sign * temporal_force(U4, G, beta, xi)
    lhs = (grad_h * F).flatten(1).sum(1)
    from gelt.lattice import plaquette_tensor
    P = plaquette_tensor(U3, G)
    h = P.diagonal(dim1=-2, dim2=-1).sum(-1).real.flatten(1).sum(1)
    return lhs - 4 * casimir_fundamental(G.nc) * h  # E[·] = 0 per slice


def test_stein_identity_on_the_timeslice_marginal(small_ensemble):
    cfgs, beta, xi = small_ensemble
    q = _stein_terms(cfgs, beta, xi).view(cfgs.shape[0], -1).mean(1)  # per config
    n = q.shape[0]
    z = q.mean() / (q.std() / math.sqrt(n))
    assert abs(z) < 4, f"Stein identity violated: mean {q.mean():.3f}, z = {z:.2f}"
    q_bad = _stein_terms(cfgs, beta, xi, sign=-1.0).view(n, -1).mean(1)
    z_bad = q_bad.mean() / (q_bad.std() / math.sqrt(n))
    assert abs(z_bad) > 20, f"no power: the flipped force gives z = {z_bad:.1f}"


def test_frozen_slice_is_never_written():
    torch.manual_seed(6)
    U = random_links(3, 4, G, dtype=torch.float64, Lt=4)
    fr = frozen_slice_mask(U.shape[:-2], t0=2)
    U1, _ = heatbath_overrelaxation_sweep(U, G, 2.4, n_or=2, xi=3.0, frozen=fr)
    assert torch.equal(U1[1:, 2], U[1:, 2])
    assert (U1[0, 2] - U[0, 2]).abs().max() > 1e-3  # temporal links at t0 are free
    assert (U1[1:, 1] - U[1:, 1]).abs().max() > 1e-3
