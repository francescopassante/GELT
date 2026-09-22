"""Correctness checks for the SU(2) heat-bath + overrelaxation sampler.

Heat-bath and overrelaxation are exact (no accept/reject tuning), so they admit
sharp tests:
  - overrelaxation conserves the Wilson action to machine precision and stays
    on the group (it is a microcanonical reflection);
  - heat-bath stays on the group and reproduces the *exact* 2D SU(2) mean
    plaquette I₂(β)/I₁(β) — the same continuum-free benchmark used by
    ``validate_sampler_su2.py``, but here as an automated regression test.
"""

import math

import torch

from gelt.lattice import SU, action, random_links
from gelt.sampler import (
    heatbath_overrelaxation_sweep,
    heatbath_sweep,
    overrelaxation_sweep,
)

GROUP = SU(2)


def _is_su2(U: torch.Tensor, tol: float = 1e-10) -> bool:
    """Every matrix in U is unitary with unit determinant."""
    eye = torch.eye(2, dtype=U.dtype, device=U.device)
    unitary = torch.allclose(U @ GROUP.dagger(U), eye.expand_as(U), atol=tol)
    det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
    unit_det = torch.allclose(det, torch.ones_like(det), atol=tol)
    return unitary and unit_det


def test_overrelaxation_preserves_action():
    """The microcanonical reflection leaves the Wilson action unchanged."""
    torch.manual_seed(0)
    U = random_links(6, 3, GROUP, dtype=torch.complex128)  # (D, *Λ, 2, 2)
    S_before = action(U.unsqueeze(0), GROUP, beta=2.0)
    U_new, acc = overrelaxation_sweep(U, GROUP, beta=2.0)
    S_after = action(U_new.unsqueeze(0), GROUP, beta=2.0)
    assert acc == 1.0
    assert torch.allclose(S_before, S_after, atol=1e-9), (S_before, S_after)
    assert _is_su2(U_new)


def test_heatbath_stays_on_group():
    torch.manual_seed(1)
    U = random_links(6, 3, GROUP, dtype=torch.complex128)
    U_new, acc = heatbath_sweep(U, GROUP, beta=2.0)
    assert acc == 1.0
    assert _is_su2(U_new)


def test_anisotropic_xi1_matches_isotropic():
    """xi=1.0 must reproduce the isotropic sweep bit-for-bit (backward compat).

    The staple is bit-identical at ξ=1 and the RNG draws match under the same
    seed, so the produced configuration must be identical."""
    L, beta = 6, 2.0
    torch.manual_seed(7)
    U = random_links(L, 4, GROUP, dtype=torch.complex128)

    torch.manual_seed(123)
    U_default, _ = heatbath_overrelaxation_sweep(U, GROUP, beta, n_or=2)
    torch.manual_seed(123)
    U_xi1, _ = heatbath_overrelaxation_sweep(U, GROUP, beta, n_or=2, xi=1.0)
    assert torch.equal(U_default, U_xi1)


def test_overrelaxation_preserves_anisotropic_action():
    """The reflection conserves the *anisotropic* action and stays on the group."""
    torch.manual_seed(0)
    xi = 2.5
    U = random_links(6, 4, GROUP, dtype=torch.complex128)
    for _ in range(5):  # get on a representative slice of configuration space
        U, _ = heatbath_sweep(U, GROUP, beta=2.0, xi=xi)
    S_before = action(U.unsqueeze(0), GROUP, beta=2.0, xi=xi)
    U_new, acc = overrelaxation_sweep(U, GROUP, beta=2.0, xi=xi)
    S_after = action(U_new.unsqueeze(0), GROUP, beta=2.0, xi=xi)
    assert acc == 1.0
    assert torch.allclose(S_before, S_after, atol=1e-9), (S_before, S_after)
    assert _is_su2(U_new)


def test_heatbath_anisotropic_stays_on_group():
    torch.manual_seed(1)
    U = random_links(6, 4, GROUP, dtype=torch.complex128)
    U_new, acc = heatbath_sweep(U, GROUP, beta=2.0, xi=3.0)
    assert acc == 1.0
    assert _is_su2(U_new)


def test_heatbath_mean_plaquette_matches_exact_2d():
    """2D SU(2): ⟨Re Tr P / 2⟩ → I₂(β)/I₁(β) = I₀(β)/I₁(β) − 2/β."""
    torch.manual_seed(2)
    L, beta = 8, 2.0
    U = random_links(L, 2, GROUP, dtype=torch.float64)  # (2, L, L, 2, 2)
    n_plaq = L * L  # one plaquette per site in 2D

    for _ in range(150):  # thermalise
        U, _ = heatbath_overrelaxation_sweep(U, GROUP, beta, n_or=3)

    plaqs = []
    for _ in range(400):  # measure
        U, _ = heatbath_overrelaxation_sweep(U, GROUP, beta, n_or=3)
        S = action(U.unsqueeze(0), GROUP, beta=beta)  # β (n_plaq − Σ ReTrP/2)
        plaqs.append(1.0 - (S / beta).item() / n_plaq)  # ⟨ReTrP/2⟩

    measured = sum(plaqs) / len(plaqs)
    b = torch.tensor(beta, dtype=torch.float64)
    exact = (torch.special.i0(b) / torch.special.i1(b) - 2.0 / b).item()
    assert abs(measured - exact) < 0.01, (measured, exact)


def test_z2_heatbath_mean_plaquette_matches_exact_2d():
    """2D Z₂: ⟨P⟩ → tanh(β), the exact result. This is the automated analogue
    of validate_sampler_z2.py's 2D panel, and the correctness check that the
    heat-bath's P(U=+1) = σ(2βs) has the right sign and normalization."""
    from gelt.lattice import Z2
    from gelt.sampler import z2_heatbath_sweep

    torch.manual_seed(3)
    g = Z2()
    L, beta = 8, 0.6
    U = random_links(L, 2, g, dtype=torch.float64)
    n_plaq = L * L

    for _ in range(100):  # thermalise
        U, acc = z2_heatbath_sweep(U, g, beta)
        assert acc == 1.0  # rejection-free by construction

    plaqs = []
    for _ in range(400):
        U, _ = z2_heatbath_sweep(U, g, beta)
        S = action(U.unsqueeze(0), g, beta=beta)
        plaqs.append(1.0 - (S / beta).item() / n_plaq)

    measured = sum(plaqs) / len(plaqs)
    exact = math.tanh(beta)
    assert abs(measured - exact) < 0.02, (measured, exact)


def test_z2_heatbath_stays_on_the_group():
    from gelt.lattice import Z2
    from gelt.sampler import z2_heatbath_sweep

    torch.manual_seed(4)
    g = Z2()
    U = random_links(6, 3, g, dtype=torch.float64)
    U, _ = z2_heatbath_sweep(U, g, 0.75)
    assert torch.all((U == 1.0) | (U == -1.0))


def test_multichain_metropolis_mean_plaquette_matches_exact_2d():
    """The batched sampler, against the same exact 2D SU(2) benchmark.

    ``metropolis_sweep_multichain`` is what generates the 1+1D Wilson-loop
    datasets of ``scripts/wilson_regression_data.py``, and every MSE downstream
    is only as trustworthy as the ensemble under it. Three couplings at once,
    because the whole point of the batched sweep is that each chain carries its
    own β, and a broadcast bug would show up as the *ladder* being wrong rather
    than the sampler.

    Independent chains from independent random starts, so there is no
    autocorrelation to window: the error on the mean is the naive one.
    """
    from gelt.sampler import _su2_proposal_paper, metropolis_sweep_multichain

    torch.manual_seed(11)
    L, n_chains = 8, 160
    betas_grid = [0.5, 2.0, 4.0]
    betas = torch.tensor(betas_grid, dtype=torch.float64).repeat_interleave(n_chains)
    U = random_links(L, 2, GROUP, dtype=torch.float64, N=betas.numel())

    for _ in range(400):  # thermalise every chain
        U, acc = metropolis_sweep_multichain(
            U, GROUP, betas, propose_fn=_su2_proposal_paper,
            n_hits=10, epsilon=0.5,
        )
    assert U.shape[0] == betas.numel()
    assert _is_su2(U)
    # A stuck chain would satisfy the plaquette test at small β by accident.
    assert acc.min().item() > 0.2, acc.min().item()

    # β = 1 in `action` makes it Σ_p (1 − Re Tr P / 2), i.e. the plaquette count
    # minus the observable — independent of the sampling β, which is the point.
    plaq = (1.0 - action(U, GROUP, beta=1.0) / (L * L)).view(len(betas_grid), n_chains)
    for k, b in enumerate(betas_grid):
        bt = torch.tensor(b, dtype=torch.float64)
        exact = (torch.special.i0(bt) / torch.special.i1(bt) - 2.0 / bt).item()
        err = plaq[k].std().item() / math.sqrt(n_chains)
        assert abs(plaq[k].mean().item() - exact) < 4 * err + 0.005, (
            b, plaq[k].mean().item(), exact
        )


def test_multichain_metropolis_respects_per_chain_beta():
    """Each chain's coupling is its own: ⟨Re Tr P⟩ must increase with β.

    The broadcast in the accept/reject is the one place a ``(B,)`` β could be
    silently reduced to a scalar — every shape would still line up, and at a
    single coupling the result would be indistinguishable.
    """
    from gelt.sampler import _su2_proposal_paper, metropolis_sweep_multichain

    torch.manual_seed(12)
    L, n_chains = 6, 48
    grid = [0.2, 1.0, 3.0, 6.0]
    betas = torch.tensor(grid, dtype=torch.float64).repeat_interleave(n_chains)
    U = random_links(L, 2, GROUP, dtype=torch.float64, N=betas.numel())
    for _ in range(250):
        U, _ = metropolis_sweep_multichain(
            U, GROUP, betas, propose_fn=_su2_proposal_paper,
            n_hits=10, epsilon=0.5,
        )
    plaq = (1.0 - action(U, GROUP, beta=1.0) / (L * L)).view(len(grid), n_chains)
    means = plaq.mean(dim=1).tolist()
    assert all(a < b for a, b in zip(means, means[1:])), means


def test_paper_proposal_stays_on_su2_and_is_symmetric():
    """``V = exp(i Σ T^a X^a)`` is in SU(2), and ``X → −X`` gives ``V†``.

    The symmetry is what lets the Metropolis acceptance drop the Hastings ratio;
    it is a property of the *kernel*, so it is checked directly rather than
    inferred from the sampler reproducing a known mean.
    """
    from gelt.sampler import _su2_from_quaternion, _su2_proposal_paper

    torch.manual_seed(13)
    U = random_links(4, 2, GROUP, dtype=torch.float64)[0]
    torch.manual_seed(99)
    V_U = _su2_proposal_paper(U, GROUP, epsilon=0.5)
    V = V_U @ GROUP.dagger(U)
    assert _is_su2(V)

    # The same X with the sign flipped is V†: build both by hand.
    X = torch.randn(5, 3, dtype=torch.float64)
    theta = 0.5 * X.norm(dim=-1)
    unit = X / X.norm(dim=-1, keepdim=True)
    Vp = _su2_from_quaternion(torch.cos(theta), torch.sin(theta)[:, None] * unit)
    Vm = _su2_from_quaternion(torch.cos(theta), -torch.sin(theta)[:, None] * unit)
    assert torch.allclose(Vm, GROUP.dagger(Vp), atol=1e-14)
