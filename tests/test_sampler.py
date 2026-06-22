"""Correctness checks for the SU(2) heat-bath + overrelaxation sampler.

Heat-bath and overrelaxation are exact (no accept/reject tuning), so they admit
sharp tests:
  - overrelaxation conserves the Wilson action to machine precision and stays
    on the group (it is a microcanonical reflection);
  - heat-bath stays on the group and reproduces the *exact* 2D SU(2) mean
    plaquette I₂(β)/I₁(β) — the same continuum-free benchmark used by
    ``validate_sampler_su2.py``, but here as an automated regression test.
"""

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
