"""Cooling correctness: covariance, group membership, action descent, and the
migration of Q towards integers (the reason cooling exists at all)."""

import functools

import torch

from gelt.lattice import (
    SU,
    action,
    link_gauge_transformation,
    random_links,
    topological_charge_density,
)
from gelt.sampler import heatbath_overrelaxation_sweep, mcmc_ensemble
from gelt.topology import cool, cooled_charge_density


def _random_omega(L, D, gaugegroup, dtype=torch.complex128):
    """Haar-ish site-local gauge transformation Ω(x)."""
    shape = (L,) * D + (gaugegroup.nc, gaugegroup.nc)
    M = torch.randn(shape, dtype=dtype)
    return gaugegroup.project(M)


def test_cool_is_gauge_covariant():
    """cool(U^Ω) == (cool U)^Ω — cooling commutes with gauge transformations."""
    torch.manual_seed(0)
    g = SU(2)
    L, D = 4, 4
    U = random_links(L, D, g, dtype=torch.float64).unsqueeze(0)
    omega = _random_omega(L, D, g)

    cooled_then_rotated = link_gauge_transformation(
        cool(U, g, n_steps=2)[0], omega, g
    )
    rotated_then_cooled = cool(
        link_gauge_transformation(U[0], omega, g).unsqueeze(0), g, n_steps=2
    )[0]
    assert torch.allclose(cooled_then_rotated, rotated_then_cooled, atol=1e-10)


def test_cool_stays_on_the_group():
    torch.manual_seed(1)
    g = SU(2)
    U = random_links(4, 4, g, dtype=torch.float64).unsqueeze(0)
    Uc = cool(U, g, n_steps=3)
    eye = torch.eye(g.nc, dtype=Uc.dtype).expand_as(Uc)
    assert torch.allclose(Uc @ g.dagger(Uc), eye, atol=1e-10)
    assert torch.allclose(torch.linalg.det(Uc), torch.ones_like(torch.linalg.det(Uc)), atol=1e-10)


def test_cool_lowers_the_action_monotonically():
    """The defining property: cooling descends towards a classical solution."""
    torch.manual_seed(2)
    g = SU(2)
    U = random_links(6, 4, g, dtype=torch.float64).unsqueeze(0)
    actions = [action(U, g, beta=1.0)[0].item()]  # action() is batched
    for _ in range(4):
        U = cool(U, g, n_steps=1)
        actions.append(action(U, g, beta=1.0)[0].item())
    assert all(b <= a + 1e-9 for a, b in zip(actions, actions[1:])), actions
    assert actions[-1] < actions[0]


def test_cool_smears_every_direction_including_time():
    """Cooling must touch the time links — that is what distinguishes it from
    the spatial APE smearing used for spectroscopy."""
    torch.manual_seed(3)
    g = SU(2)
    U = random_links(4, 4, g, dtype=torch.float64).unsqueeze(0)
    Uc = cool(U, g, n_steps=1)
    assert not torch.allclose(Uc[:, 0], U[:, 0])  # time links moved


def test_cooling_drives_Q_towards_integers():
    """On a thermalised ensemble, cooling should strip the UV fluctuation that
    keeps the naive charge far from an integer. This is the physics assertion:
    without it, q(x) is not a topological field and cannot serve as ground
    truth for a localization study."""
    torch.manual_seed(4)
    g = SU(2)
    sweep = functools.partial(heatbath_overrelaxation_sweep, n_or=2)
    configs, _ = mcmc_ensemble(
        L=6, D=4, gaugegroup=g, beta=2.4, n_configs=4,
        n_therm=80, n_skip=4, sweep_fn=sweep,
    )
    configs = configs.to(torch.complex128)

    Q_raw = topological_charge_density(configs, g).flatten(start_dim=1).sum(dim=1)
    _, Q_cool = cooled_charge_density(configs, g, n_steps=20)

    dev_raw = (Q_raw - Q_raw.round()).abs().mean().item()
    dev_cool = (Q_cool - Q_cool.round()).abs().mean().item()
    assert dev_cool < dev_raw, f"cooling did not integerise Q: {dev_raw} -> {dev_cool}"
