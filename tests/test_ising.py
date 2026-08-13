"""Correctness of the dual-Ising ground-truth layer (`gelt.ising`).

The load-bearing test is :func:`test_duality_predicts_gauge_plaquette`: a
*parameter-free* prediction for the Z₂ gauge mean plaquette, computed from an
Ising simulation, checked against a gauge simulation run by entirely separate
code. Nothing is fitted, so it can only pass if the Ising sweep, the duality
map β* = −½ ln tanh β and the free-energy derivation are all simultaneously
right. The exact-enumeration test below pins the sweep on its own, so a failure
of the duality test cannot be blamed on the sampler.
"""

import itertools
import math

import pytest
import torch

from gelt.ising import (
    GAUGE_BETA_C,
    ISING_BETA_C,
    dual_beta,
    gauge_beta,
    heatbath_sweep,
    ising_measure,
    mean_bond_energy,
    neighbour_sum,
    predicted_plaquette,
    random_spins,
    wall_observables,
)
from gelt.lattice import Z2, plaquette_tensor
from gelt.sampler import mcmc_ensemble, z2_heatbath_sweep


# --------------------------------------------------------------------------
# the duality map
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.2, 0.5, 0.745, 0.76, 1.3])
def test_duality_is_an_involution(beta):
    assert gauge_beta(dual_beta(beta)) == pytest.approx(beta, rel=1e-12)


@pytest.mark.parametrize("beta", [0.2, 0.5, 0.745, 1.3])
def test_symmetric_form_of_the_duality(beta):
    """β* = −½ ln tanh β is equivalent to sinh 2β · sinh 2β* = 1."""
    bs = dual_beta(beta)
    assert math.sinh(2 * beta) * math.sinh(2 * bs) == pytest.approx(1.0, rel=1e-12)


def test_gauge_critical_coupling_matches_the_scripts():
    """The dual of the Ising β*_c is the β_c the Z₂ scripts hardcode as 0.7614."""
    assert GAUGE_BETA_C == pytest.approx(0.7614133, abs=1e-6)
    assert dual_beta(GAUGE_BETA_C) == pytest.approx(ISING_BETA_C, rel=1e-12)


def test_predicted_plaquette_reproduces_the_strong_coupling_series():
    """⟨P⟩ ≈ t + 2t⁵ at small β, the 2 counting cubes per plaquette in 3D.

    Feeding the *low-temperature* Ising series 1 − ⟨ss⟩ ≈ 4t⁶ into the duality
    relation must reproduce the *strong-coupling* gauge series. The two series
    are computed on opposite sides of the duality and know nothing about each
    other, so agreement of the t⁵ coefficient is a real check on the algebra.
    """
    for beta in (0.05, 0.08, 0.10):
        t = math.tanh(beta)
        ss = 1.0 - 4.0 * t**6
        assert predicted_plaquette(beta, ss) == pytest.approx(t + 2 * t**5, rel=2e-3)


# --------------------------------------------------------------------------
# the sweep, against exact enumeration
# --------------------------------------------------------------------------


def _exact_2d_bond_energy(L: int, beta: float) -> float:
    """⟨s_i s_j⟩ per bond on a periodic L×L Ising lattice, by enumeration."""
    n = L * L
    states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=n)))
    s = states.view(-1, L, L)
    bonds = (s * s.roll(-1, dims=1)).sum(dim=(1, 2)) + (
        s * s.roll(-1, dims=2)
    ).sum(dim=(1, 2))
    w = torch.exp(beta * (bonds - bonds.max()).double())
    return float((w * bonds.double()).sum() / w.sum() / (2 * n))


@pytest.mark.parametrize("beta", [0.20, 0.35])
def test_heatbath_reproduces_exact_2d_enumeration(beta):
    """The sweep samples the Boltzmann distribution — checked exactly in 2D.

    A 4×4 periodic lattice has 2^16 states, so ⟨s_i s_j⟩ is available in closed
    form. The sweep is dimension-generic, so pinning it in 2D pins the 3D code
    path too; 2D is used because it is the largest lattice that can be
    enumerated. β stays below the 2D critical 0.4407 so the chain decorrelates
    in a few sweeps and the test is fast and stable.
    """
    exact = _exact_2d_bond_energy(4, beta)

    torch.manual_seed(0)
    s = random_spins((4, 4), n_replicas=512)
    for _ in range(50):  # thermalise
        s = heatbath_sweep(s, beta)
    vals = []
    for _ in range(200):
        for _ in range(2):
            s = heatbath_sweep(s, beta)
        vals.append(mean_bond_energy(s).mean())
    est = float(torch.stack(vals).mean())
    # 512 replicas × 200 samples: the statistical error is well under 1e-3.
    assert est == pytest.approx(exact, abs=3e-3)


def test_heatbath_rejects_odd_extents():
    """A checkerboard update is only valid on a bipartite lattice."""
    s = random_spins((4, 5), n_replicas=1)
    with pytest.raises(ValueError, match="even extents"):
        heatbath_sweep(s, 0.2)


def test_neighbour_sum_is_the_stencil_it_claims_to_be():
    s = random_spins((4, 6, 4), n_replicas=3)
    h = neighbour_sum(s)
    manual = torch.zeros_like(s)
    for ax in (1, 2, 3):
        manual = manual + s.roll(-1, dims=ax) + s.roll(1, dims=ax)
    assert torch.equal(h, manual)
    assert h.abs().max() <= 6  # D = 3 → six neighbours


def test_spins_stay_on_the_group():
    torch.manual_seed(1)
    s = random_spins((4, 4, 4), n_replicas=8)
    for _ in range(5):
        s = heatbath_sweep(s, 0.3)
    assert torch.equal(s.abs(), torch.ones_like(s))


# --------------------------------------------------------------------------
# wall observables
# --------------------------------------------------------------------------


def test_wall_observables_shapes_and_sign_fixing():
    torch.manual_seed(2)
    Lt, L = 8, 4
    s = random_spins((Lt, L, L), n_replicas=5, ordered=True)
    s[2] = -s[2]  # one replica in the negative sector
    obs = wall_observables(s, time_axis=0)
    for k in ("m", "e_t", "e_s"):
        assert obs[k].shape == (5, Lt)
    # all-(+1) walls are +L²; sign fixing must map the flipped replica back
    assert torch.allclose(obs["m"], torch.full((5, Lt), float(L * L)))
    # every bond satisfied in a uniform configuration
    assert torch.allclose(obs["e_t"], torch.full((5, Lt), float(L * L)))
    assert torch.allclose(obs["e_s"], torch.full((5, Lt), float(2 * L * L)))


def test_ising_measure_layout_is_chain_major():
    out = ising_measure(
        0.3, (6, 4, 4), n_replicas=3, n_measure=4, n_therm=5, n_skip=1, seed=0
    )
    assert out["m"].shape == (12, 6)
    assert out["bond_energy"].shape == (12,)
    assert torch.equal(
        out["chain"], torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    )


# --------------------------------------------------------------------------
# the duality itself, end to end
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.50])
def test_duality_predicts_gauge_plaquette(beta):
    """⟨P⟩(β) = tanh β + [1 − ⟨ss⟩(β*)] / sinh 2β, both sides simulated.

    The gauge side uses the package's own Z₂ heat-bath and plaquette tensor;
    the Ising side uses this module. Nothing is tuned. β = 0.5 is deep in the
    strong-coupling phase, so ξ ≪ L = 8 and the torus's topological sectors —
    the only thing the infinite-volume duality relation omits — contribute at
    O(e^{−L/ξ}), far below the tolerance here.
    """
    group = Z2()
    torch.manual_seed(3)
    n_cfg = 120
    configs, _acc = mcmc_ensemble(
        L=8, D=3, gaugegroup=group, beta=beta, n_configs=n_cfg,
        n_therm=200, n_skip=5, sweep_fn=z2_heatbath_sweep,
    )
    p = plaquette_tensor(configs, group)[..., 0, 0].real
    P = p.mean().item()
    # Per-configuration scatter → the error on ⟨P⟩. The test is statistics
    # limited (a full-precision version of this check runs in
    # scripts/dual_ground_truth.py against the N=2000 production ensembles),
    # so assert against the measured resolution rather than a magic constant.
    P_err = p.mean(dim=(1, 2, 3, 4)).std().item() / n_cfg**0.5

    out = ising_measure(
        dual_beta(beta), (8, 8, 8), n_replicas=64, n_measure=120,
        n_therm=200, n_skip=5, seed=4,
    )
    ss = out["bond_energy"]
    pred = predicted_plaquette(beta, float(ss.mean()))
    pred_err = (ss.std().item() / ss.numel() ** 0.5) / math.sinh(2 * beta)

    sigma = (P_err**2 + pred_err**2) ** 0.5
    assert abs(pred - P) < 4 * sigma, (
        f"duality prediction {pred:.6f} vs measured {P:.6f} "
        f"= {(pred - P) / sigma:+.1f}σ"
    )
