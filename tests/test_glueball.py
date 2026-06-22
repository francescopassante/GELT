"""Tests for the classical glueball baseline (gelt/glueball.py).

What matters physically:
  - APE smearing is gauge covariant (smear(U^Ω) = (smear U)^Ω) and stays on the
    group. Z₂ is self-inverse so dagger/projection bugs hide there; the SU(2)
    complex case is the real check.
  - The glueball operator is gauge invariant (it is a sum of Wilson loops).
  - The correlator / effective-mass arithmetic recovers a known mass from a
    synthetic single-exponential correlator.
"""

import math

import pytest
import torch

from gelt.glueball import (
    ape_smear,
    connected_correlator,
    effective_mass,
    glueball_operator,
    jackknife_effective_mass,
    zero_momentum,
)
from gelt.lattice import SU, Z2, link_gauge_transformation, random_links


def _random_omega(gaugegroup, shape, dtype):
    """One Haar group element per site, shape (*Λ, nc, nc)."""
    return gaugegroup.random(shape, dtype=dtype)


@pytest.mark.parametrize(
    "gaugegroup, dtype",
    [(Z2(), torch.float64), (SU(2), torch.complex128)],
)
def test_ape_smear_gauge_covariant(gaugegroup, dtype):
    L, D = 4, 4
    U = random_links(L, D, gaugegroup, dtype=dtype)  # (D, *Λ, nc, nc)
    omega = _random_omega(gaugegroup, (L,) * D, dtype)

    # α = 0.7 keeps Z₂'s V = ±0.3 ± {0, 0.35, 0.7} away from 0, where the
    # projection's sign(0) tie-break is an inherent (measure-zero) ambiguity.
    alpha = 0.7
    smeared = ape_smear(U.unsqueeze(0), gaugegroup, alpha=alpha, n_steps=2)[0]
    Ug = link_gauge_transformation(U, omega, gaugegroup)
    smeared_g = ape_smear(Ug.unsqueeze(0), gaugegroup, alpha=alpha, n_steps=2)[0]
    expected = link_gauge_transformation(smeared, omega, gaugegroup)

    assert torch.allclose(smeared_g, expected, atol=1e-8)


def test_ape_smear_stays_on_group():
    L, D = 4, 4
    g = SU(2)
    U = random_links(L, D, g, dtype=torch.complex128)
    smeared = ape_smear(U.unsqueeze(0), g, alpha=0.6, n_steps=3)[0]
    # Spatial links must be unitary with unit determinant.
    eye = torch.eye(2, dtype=torch.complex128)
    for mu in range(1, D):  # axis 0 (time) is left untouched by smearing
        UUd = smeared[mu] @ g.dagger(smeared[mu])
        assert torch.allclose(UUd, eye.expand_as(UUd), atol=1e-8)
        det = torch.linalg.det(smeared[mu])
        assert torch.allclose(det, torch.ones_like(det), atol=1e-8)


def test_glueball_operator_gauge_invariant():
    L, D = 4, 4
    g = SU(2)
    U = random_links(L, D, g, dtype=torch.complex128).unsqueeze(0)
    omega = _random_omega(g, (L,) * D, torch.complex128)
    Ug = link_gauge_transformation(U[0], omega, g).unsqueeze(0)

    O = glueball_operator(U, g)
    Og = glueball_operator(Ug, g)
    assert torch.allclose(O, Og, atol=1e-8)


def test_zero_momentum_shape():
    L, D = 4, 4
    g = SU(2)
    U = random_links(L, D, g, dtype=torch.complex128, N=3)  # (3, D, *Λ, nc, nc)
    Obar = zero_momentum(glueball_operator(U, g))
    assert Obar.shape == (3, L)


def test_effective_mass_recovers_known_mass():
    # Synthetic single-exponential correlator C(Δ) = exp(-m Δ).
    m = 0.7
    Nt = 8
    C = torch.exp(-m * torch.arange(Nt, dtype=torch.float64))
    meff = effective_mass(C)
    assert torch.allclose(meff, torch.full_like(meff, m), atol=1e-10)


def test_connected_correlator_constant_operator_is_zero():
    # A config-independent Ō has no connected signal: C(0) = 0.
    Obar = torch.ones(5, 6, dtype=torch.float64)
    C = connected_correlator(Obar)
    assert torch.allclose(C, torch.zeros_like(C), atol=1e-12)


def test_jackknife_shapes_and_positive_error():
    # Synthetic ensemble O(b, t) = r_b · w(t) with r centred and w(t) > 0, so
    # the connected correlator C(Δ) ∝ Σ_t w(t+Δ)w(t) > 0 for every Δ and
    # m_eff is finite (random data would give log of negative numbers → nan).
    torch.manual_seed(0)
    Nt = 8
    r = torch.randn(20, dtype=torch.float64)
    r = r - r.mean()
    w = torch.exp(-0.5 * torch.arange(Nt, dtype=torch.float64))
    Obar = r[:, None] * w[None, :]
    mean, err = jackknife_effective_mass(Obar)
    assert mean.shape == (Nt - 1,)
    assert err.shape == (Nt - 1,)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(err).all()
    assert (err >= 0).all()
