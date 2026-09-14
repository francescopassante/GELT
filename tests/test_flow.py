"""Unit tests for the Wilson gradient flow (``gelt.flow``).

Run with:  pytest tests/test_flow.py -v
"""

import math

import pytest
import torch

from gelt.flow import (
    flow_drift,
    flow_trajectory,
    group_exp,
    traceless_antihermitian,
    wilson_flow,
)
from gelt.lattice import SU, Z2, action, link_gauge_transformation, random_links


@pytest.fixture
def su2():
    return SU(2)


def _links(L=4, D=4, group=None, seed=0, dtype=torch.complex128):
    torch.manual_seed(seed)
    return random_links(L, D, group, dtype=dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# The algebra: projection and exponential
# ---------------------------------------------------------------------------


def test_ta_projection_is_traceless_antihermitian():
    """[M]_TA lands in su(N) for an arbitrary (non-group) M."""
    torch.manual_seed(0)
    M = torch.randn(50, 3, 3, dtype=torch.complex128)
    A = traceless_antihermitian(M, 3)
    assert (A + A.conj().transpose(-1, -2)).abs().max() < 1e-14
    assert A.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-14


@pytest.mark.parametrize("nc", [2, 3])
def test_group_exp_lands_on_the_group(nc):
    """exp of a Lie-algebra element is unitary with unit determinant."""
    torch.manual_seed(0)
    X = traceless_antihermitian(torch.randn(200, nc, nc, dtype=torch.complex128), nc)
    V = group_exp(X, nc)
    eye = torch.eye(nc, dtype=V.dtype)
    assert (V @ V.conj().transpose(-1, -2) - eye).abs().max() < 1e-12
    assert (torch.linalg.det(V) - 1).abs().max() < 1e-12


def test_group_exp_su2_matches_matrix_exp():
    """The closed-form nc=2 route agrees with the general one, θ → 0 included."""
    torch.manual_seed(0)
    X = traceless_antihermitian(torch.randn(200, 2, 2, dtype=torch.complex128), 2)
    X = torch.cat([X, torch.zeros(1, 2, 2, dtype=torch.complex128)])  # θ = 0
    assert (group_exp(X, 2) - torch.linalg.matrix_exp(X)).abs().max() < 1e-13


def test_flow_drift_is_in_the_algebra(su2):
    """Z_μ(x) is anti-Hermitian and traceless, so the step stays on the group."""
    Z = flow_drift(_links(group=su2), su2)
    assert (Z + Z.conj().transpose(-1, -2)).abs().max() < 1e-13
    assert Z.diagonal(dim1=-2, dim2=-1).sum(-1).abs().max() < 1e-13


# ---------------------------------------------------------------------------
# The flow itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("integrator", ["rk3", "euler"])
def test_flow_preserves_the_group(su2, integrator):
    """Flowed links are still SU(2) after a long trajectory, with no reprojection."""
    V = wilson_flow(_links(group=su2), su2, t=1.0, eps=0.02, integrator=integrator)
    eye = torch.eye(2, dtype=V.dtype)
    assert (V @ su2.dagger(V) - eye).abs().max() < 1e-13
    assert (torch.linalg.det(V) - 1).abs().max() < 1e-13


def test_flow_is_gauge_covariant(su2):
    """flow(ΩU) = Ω flow(U): the flowed links transform like links under the same Ω.

    This is what makes ``q_clov`` of a flowed configuration a legitimate gauge
    invariant target for an equivariant network.
    """
    L, D = 4, 4
    torch.manual_seed(0)
    U = random_links(L, D, su2, dtype=torch.complex128)
    omega = su2.random((L,) * D, dtype=torch.complex128)

    flow_then_gauge = link_gauge_transformation(
        wilson_flow(U.unsqueeze(0), su2, t=0.3)[0], omega, su2
    )
    gauge_then_flow = wilson_flow(
        link_gauge_transformation(U, omega, su2).unsqueeze(0), su2, t=0.3
    )[0]
    drift = (flow_then_gauge - gauge_then_flow).abs().max().item()
    assert drift < 1e-13, f"Flow is not gauge covariant; max drift = {drift:.3e}"


def test_flow_decreases_the_action(su2):
    """The flow is gradient descent on S_W, so the action falls monotonically."""
    U = _links(group=su2)
    S = [action(U, su2, beta=2.4)[0].item()]
    V = U
    for _ in range(5):
        V = wilson_flow(V, su2, t=0.1, eps=0.02)
        S.append(action(V, su2, beta=2.4)[0].item())
    assert all(b < a for a, b in zip(S, S[1:])), f"Action not monotonic: {S}"


def test_flow_time_zero_is_the_identity(su2):
    U = _links(group=su2)
    V = wilson_flow(U, su2, t=0.0)
    assert torch.equal(U, V) and V is not U


# ---------------------------------------------------------------------------
# The normalisation — what fixes r_sm = √(8t)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_mode", [1, 2])
def test_linearised_flow_is_the_unit_heat_kernel(su2, n_mode):
    """A small transverse plane wave decays as exp(−k̂² t), k̂² = 4 sin²(k/2).

    The drift carries no β and no stray factor of two; the check that this is
    actually so is the linearised limit, where the flow equation is the lattice
    heat equation with *unit* coefficient — which is what makes flow time
    ``t/a²`` and the smoothing radius ``r_sm/a = √(8 t/a²)``, the relation the
    whole receptive-field argument of ``notes/flow_free_topology.md`` rests on.

    Setup: links along direction 0 carrying an abelian σ₃ phase that depends on
    ``x₁`` only.  The field is transverse (its lattice divergence vanishes), so
    the gauge term in the linearised drift drops and the decay is pure.
    """
    L, amp, t = 8, 1e-4, 1.0
    k = 2 * math.pi * n_mode / L
    x1 = torch.arange(L, dtype=torch.float64)
    sigma3 = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
    theta = (amp * torch.cos(k * x1)).view(1, L, 1, 1)[..., None, None]

    links = torch.eye(2, dtype=torch.complex128).expand(4, L, L, L, L, 2, 2).clone()
    links[0] = torch.matrix_exp(1j * theta * sigma3).expand(L, L, L, L, 2, 2)

    V = wilson_flow(links.unsqueeze(0), su2, t=t, eps=0.02)[0]
    a_out = V[0, 0, :, 0, 0, 0, 0].log().imag  # the σ₃ phase along x₁
    measured = ((a_out * torch.cos(k * x1)).sum() / (L / 2) / amp).item()

    expected = math.exp(-4 * math.sin(k / 2) ** 2 * t)
    assert abs(measured - expected) < 1e-5, (
        f"Linearised decay {measured:.6f} != exp(−k̂²t) = {expected:.6f}; the "
        f"flow-time normalisation is off, and with it r_sm = √(8t)."
    )


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------


def test_rk3_is_third_order_and_euler_first(su2):
    """Halving the step buys ~8× for RK3 and ~2× for Euler, and they agree in the
    limit — the cross-check the design record asks for."""
    U = _links(group=su2)
    t = 0.4
    ref = wilson_flow(U, su2, t, eps=0.0025, integrator="rk3")

    def err(integrator, eps):
        V = wilson_flow(U, su2, t, eps=eps, integrator=integrator)
        return (V - ref).abs().max().item()

    # RK3: local O(h⁴) ⇒ global O(h³); measured ratios are 8.5 and 7.3.
    assert err("rk3", 0.05) / err("rk3", 0.025) > 5.0
    # Euler: global O(h), so halving the step roughly halves the error.
    r_euler = err("euler", 0.05) / err("euler", 0.025)
    assert 1.5 < r_euler < 2.5, f"Euler does not look first order (ratio {r_euler})"
    # And a fine Euler run agrees with RK3 — different integrators, same flow.
    assert err("euler", 0.0005) < 2e-3


def test_rk3_beats_euler_at_equal_cost(su2):
    """RK3 at eps costs three drift evaluations; Euler at eps/3 costs the same.

    Recorded because it is the reason the production path is RK3: at matched
    cost the error is smaller by orders of magnitude, not by a constant.
    """
    U = _links(group=su2)
    t = 0.4
    ref = wilson_flow(U, su2, t, eps=0.0025, integrator="rk3")
    e_rk3 = (wilson_flow(U, su2, t, eps=0.06) - ref).abs().max().item()
    e_eul = (
        wilson_flow(U, su2, t, eps=0.02, integrator="euler") - ref
    ).abs().max().item()
    assert e_rk3 < e_eul / 10


# ---------------------------------------------------------------------------
# The target ladder
# ---------------------------------------------------------------------------


def test_flow_trajectory_matches_standalone_flows(su2):
    """One trajectory, five snapshots — each identical to a direct flow to that t.

    The ladder rungs are multiples of the step size here, which is the production
    case (``t/a² ∈ {0.5,1,2,4,8}`` at ``eps = 0.02``): the segmented integration
    then walks exactly the same grid as a standalone run.
    """
    U = _links(group=su2)
    times = [0.1, 0.2, 0.4]
    snaps = flow_trajectory(U, su2, times, eps=0.05)
    assert len(snaps) == len(times)
    for t, V in zip(times, snaps):
        direct = wilson_flow(U, su2, t, eps=0.05)
        assert (V - direct).abs().max() < 1e-12, f"ladder rung t={t} drifted"


def test_flow_trajectory_preserves_caller_order_and_duplicates(su2):
    U = _links(group=su2)
    snaps = flow_trajectory(U, su2, [0.2, 0.0, 0.1, 0.2], eps=0.05)
    assert torch.equal(snaps[1], U)  # t = 0 is the input itself
    assert torch.equal(snaps[0], snaps[3])  # the duplicate is the same snapshot
    assert (snaps[2] - wilson_flow(U, su2, 0.1, eps=0.05)).abs().max() < 1e-12


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_flow_rejects_z2():
    """Z₂ has no Lie algebra: [M]_TA of a real 1×1 is zero and the flow would be a
    silent no-op, so it is an error instead."""
    z2 = Z2()
    U = random_links(4, 4, z2, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="continuous gauge group"):
        wilson_flow(U, z2, t=0.1)


def test_flow_rejects_unknown_integrator(su2):
    with pytest.raises(ValueError, match="integrator"):
        wilson_flow(_links(group=su2), su2, t=0.1, integrator="rk4")


def test_flow_rejects_negative_time(su2):
    with pytest.raises(ValueError, match="non-negative"):
        wilson_flow(_links(group=su2), su2, t=-0.1)


def test_flow_lands_exactly_on_t(su2):
    """A flow time that is not a multiple of eps still lands on t, by shrinking
    the step rather than overshooting — the rungs must be comparable across β."""
    U = _links(group=su2)
    coarse = wilson_flow(U, su2, t=0.15, eps=0.1)  # 2 steps of 0.075, not 0.1 + 0.05
    assert torch.equal(coarse, wilson_flow(U, su2, t=0.15, eps=0.075))

    # And it is 0.15 that was reached, not eps-rounded up to 0.2: the residual
    # against a fine run at 0.15 is the integrator's own truncation error, an
    # order of magnitude below the distance between the two flow times.
    fine_15 = wilson_flow(U, su2, t=0.15, eps=0.005)
    fine_20 = wilson_flow(U, su2, t=0.20, eps=0.005)
    assert (coarse - fine_15).abs().max() < 0.1 * (fine_20 - fine_15).abs().max()


def test_flow_su3_runs_and_stays_on_the_group():
    """The nc ≥ 3 path goes through matrix_exp; check it is a flow, not a no-op."""
    su3 = SU(3)
    U = _links(L=3, group=su3, dtype=torch.complex128)
    V = wilson_flow(U, su3, t=0.2, eps=0.05)
    eye = torch.eye(3, dtype=V.dtype)
    assert (V @ su3.dagger(V) - eye).abs().max() < 1e-12
    assert (torch.linalg.det(V) - 1).abs().max() < 1e-12
    assert (V - U).abs().max() > 1e-3
    assert action(V, su3, 6.0)[0] < action(U, su3, 6.0)[0]
