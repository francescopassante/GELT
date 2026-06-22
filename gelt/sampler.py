"""Monte Carlo sampler for lattice gauge theory.

Phase 0: single-site Metropolis with checkerboard vectorisation.
The staple interface is designed for extension to heat-bath + overrelaxation
for U(1)/SU(2)/SU(3) without restructuring the sweep loop.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gelt.lattice import SU, Z2, GaugeGroup, random_links


def staple_sum(
    U: torch.Tensor,
    mu: int,
    gaugegroup: GaugeGroup,
    nu_dirs: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Sum of staples for every site along direction ``mu``.

    The local Wilson action for link U_μ(x) is
        S_local = -(β/nc) Re Tr[ U_μ(x) · A_μ(x) ]
    where the staple sum A_μ(x) is:
        Σ_{ν≠μ} [  U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)          (forward staple)
                  + U_ν†(x+μ̂-ν̂) · U_μ†(x-ν̂) · U_ν(x-ν̂) ]  (backward staple)

    Parameters
    ----------
    U     : ``(D, *Λ, nc, nc)``
    mu    : direction index
    gaugegroup : gauge group (used for ``dagger``)
    nu_dirs : directions ν to sum over (defaults to all ν ≠ μ). Pass the
        spatial directions only to build spatial staples for APE smearing.

    Returns
    -------
    Tensor of shape ``(*Λ, nc, nc)``.
    """
    D = U.shape[0]
    if nu_dirs is None:
        nu_dirs = range(D)
    A = torch.zeros_like(U[mu])
    for nu in nu_dirs:
        if nu == mu:
            continue
        Umu = U[mu]
        Unu = U[nu]
        # Forward: U_ν(x+μ̂) · U_μ†(x+ν̂) · U_ν†(x)
        Unu_fwd = torch.roll(Unu, shifts=-1, dims=mu)  # U_ν(x + μ̂)
        Umu_nu = torch.roll(Umu, shifts=-1, dims=nu)  # U_μ(x + ν̂)
        A = A + Unu_fwd @ gaugegroup.dagger(Umu_nu) @ gaugegroup.dagger(Unu)
        # Backward: U_ν†(x+μ̂-ν̂) · U_μ†(x-ν̂) · U_ν(x-ν̂)
        Unu_bwd = torch.roll(torch.roll(Unu, shifts=-1, dims=mu), shifts=+1, dims=nu)
        Umu_negnu = torch.roll(Umu, shifts=+1, dims=nu)  # U_μ(x - ν̂)
        Unu_negnu = torch.roll(Unu, shifts=+1, dims=nu)  # U_ν(x - ν̂)
        A = A + gaugegroup.dagger(Unu_bwd) @ gaugegroup.dagger(Umu_negnu) @ Unu_negnu
    return A


def _re_tr(M: torch.Tensor) -> torch.Tensor:
    """Re Tr for a batch of matrices: ``(*batch, nc, nc)`` → ``(*batch)``."""
    return M.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real


def _site_parity(spatial_shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Checkerboard parity (0 or 1) for each site. Shape: ``(*spatial_shape)``."""
    coords = torch.meshgrid(
        *[torch.arange(s, device=device) for s in spatial_shape],
        indexing="ij",
    )
    return sum(coords) % 2


def _su2_from_quaternion(a0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Build SU(2) matrices from quaternion components V = a_0·I + i(a·σ).

    ``a0`` is the real scalar part (shape ``(*B)``) and ``a`` the three real
    vector components (a_1, a_2, a_3) (shape ``(*B, 3)``); no unit-norm
    constraint is imposed here (callers supply normalised quaternions). The
    complex dtype follows ``a0``'s precision (float64 → complex128).
    """
    dtype = torch.complex128 if a0.dtype == torch.float64 else torch.complex64
    # V = [[a_0 + i a_3,  i a_1 + a_2],
    #      [i a_1 - a_2,  a_0 - i a_3]]
    V = torch.empty(*a0.shape, 2, 2, dtype=dtype, device=a0.device)
    V[..., 0, 0] = torch.complex(a0, a[..., 2])
    V[..., 0, 1] = torch.complex(a[..., 1], a[..., 0])
    V[..., 1, 0] = torch.complex(-a[..., 1], a[..., 0])
    V[..., 1, 1] = torch.complex(a0, -a[..., 2])
    return V


def _random_su2_near_identity(
    spatial_shape: Tuple[int, ...],
    epsilon: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Random SU(2) elements close to the identity, one per site.

    Uses the quaternionic parametrisation V = a_0 · I + i (a · σ) with
    a_0² + |a|² = 1. Each component a_k (k = 1,2,3) is drawn uniformly
    from ``[-ε, ε]`` and a_0 is set from the unit-norm constraint, so
    V → I as ε → 0.

    The kernel is symmetric in V ↔ V† (since V† = a_0 · I − i (a · σ)
    corresponds to a → −a, which leaves the density invariant), so the
    Metropolis acceptance reduces to the usual exp(−ΔS) form with no
    Hastings correction. ε must satisfy 3ε² ≤ 1 so that |a|² ≤ 1 for
    every sample; the worst case at ε = 1/√3 yields V = ±a · σ purely
    off-identity. Sensible ε ≈ 0.2 – 0.4 keeps acceptance around 50–70%
    at typical β.

    Returns a tensor of shape ``(*spatial_shape, 2, 2)`` with the given
    complex ``dtype``.
    """
    if 3 * epsilon * epsilon > 1.0:
        raise ValueError(
            f"epsilon={epsilon} violates 3·ε² ≤ 1; |a|² may exceed 1 and "
            "produce a non-unitary proposal. Use ε ≤ 1/√3 ≈ 0.577."
        )
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    a = (
        torch.rand(*spatial_shape, 3, dtype=real_dtype, device=device) * 2 - 1
    ) * epsilon
    a0 = torch.sqrt(1.0 - (a * a).sum(dim=-1))
    return _su2_from_quaternion(a0, a)


def _z2_proposal(
    U_mu: torch.Tensor, gaugegroup: GaugeGroup, epsilon: Optional[float] = None
) -> torch.Tensor:
    """Z₂ proposal: the unique non-identity element U' = −U.

    Deterministic, so ``epsilon`` is ignored and ``n_hits`` > 1 only toggles
    the link back and forth — keep ``n_hits = 1`` for Z₂.
    """
    return -U_mu


def _su2_proposal(
    U_mu: torch.Tensor, gaugegroup: GaugeGroup, epsilon: float = 0.3
) -> torch.Tensor:
    """SU(2) proposal: U' = V · U with V a random SU(2) element near the
    identity (see :func:`_random_su2_near_identity`).

    ``epsilon`` is the width of the proposal neighbourhood; tune it to keep
    acceptance around 50%.
    """
    if gaugegroup.nc != 2:
        raise NotImplementedError(
            f"_su2_proposal only supports SU(2), got nc={gaugegroup.nc}. "
            "For SU(N≥3) use Cabibbo–Marinari (heat-bath on SU(2) sub-blocks)."
        )
    spatial_shape = U_mu.shape[:-2]
    V = _random_su2_near_identity(spatial_shape, epsilon, U_mu.dtype, U_mu.device)
    return V @ U_mu


# Registry: maps GaugeGroup subclass → default Metropolis proposal kernel.
# Add one line here when a new group's proposal is ready. The accept/reject
# machinery in ``metropolis_sweep`` is group-agnostic; only the proposal
# changes between groups.
_PROPOSAL_FN: dict = {
    Z2: _z2_proposal,
    SU: _su2_proposal,
    # U1: _u1_proposal,   ← future
}


def metropolis_sweep(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    beta: float,
    propose_fn=None,
    n_hits: int = 1,
    epsilon: float = 0.3,
) -> Tuple[torch.Tensor, float]:
    """One full Metropolis sweep (all directions, both checkerboard parities).

    The accept/reject machinery is group-agnostic — the staple sum, local
    action, and the ΔS = (β / N_c) Re Tr[(U − U') · A] formula are identical
    for every gauge group. Only the proposal kernel changes between groups,
    so it is injected via ``propose_fn`` (dispatched by group type when
    ``None``): U' = −U for Z₂, U' = V · U with V near the identity for SU(2).

    Checkerboard structure: for a fixed direction μ, sites of the same parity
    do not share any plaquette through same-direction links, so their updates
    commute. The even sweep then the odd sweep is equivalent to a sequential
    site-by-site update but fully vectorised.

    Parameters
    ----------
    U          : ``(D, *Λ, nc, nc)`` — not modified in-place
    gaugegroup : gauge group
    beta       : inverse coupling
    propose_fn : proposal kernel ``(U_mu, gaugegroup, epsilon) → U_proposed``.
                 If ``None``, dispatches via ``_PROPOSAL_FN[type(gaugegroup)]``.
    n_hits     : Metropolis hits per link per sweep. The staple is the
                 expensive piece and is unchanged between hits at the
                 same site/parity, so multiple hits per sweep improve
                 mixing at almost no extra cost. Leave at 1 for Z₂, whose
                 proposal is deterministic.
    epsilon    : width of the proposal neighbourhood for continuous groups;
                 ignored by deterministic proposals (Z₂).

    Returns
    -------
    (U_new, acceptance_rate)
    """
    if propose_fn is None:
        propose_fn = _PROPOSAL_FN.get(type(gaugegroup))
        if propose_fn is None:
            raise NotImplementedError(
                f"No Metropolis proposal registered for {type(gaugegroup).__name__}. "
                f"Pass propose_fn= explicitly or add an entry to sampler._PROPOSAL_FN."
            )

    D = U.shape[0]
    spatial_shape = U.shape[1:-2]
    nc = gaugegroup.nc
    device = U.device

    parity = _site_parity(spatial_shape, device)  # (*Λ)

    U = U.clone()
    total_proposed = 0
    total_accepted = 0

    for mu in range(D):
        for par in (0, 1):
            # Staple depends only on links of the opposite parity in
            # direction μ (plus all links in directions ν ≠ μ); recompute
            # once per (μ, parity) and reuse across the n_hits.
            A = staple_sum(U, mu, gaugegroup)  # (*Λ, nc, nc)
            site_mask = parity == par  # (*Λ)

            for _hit in range(n_hits):
                U_mu = U[mu]
                U_proposed = propose_fn(U_mu, gaugegroup, epsilon)

                # ΔS = (β/nc) Re Tr[(U − U') · A]  > 0 means action increases
                dS = (beta / nc) * _re_tr((U_mu - U_proposed) @ A)  # (*Λ)

                rand = torch.rand(spatial_shape, device=device)
                accept = (dS <= 0) | (rand < torch.exp(-dS.clamp(min=0)))

                update_mask = accept & site_mask  # (*Λ)

                total_accepted += update_mask.sum().item()
                total_proposed += site_mask.sum().item()

                # [..., None, None] adds two extra dimension to broadcast update_mask with U_mu (or U_proposed)
                U[mu] = torch.where(update_mask[..., None, None], U_proposed, U_mu)

    return U, total_accepted / total_proposed


def _su2_decompose_staple(
    A: torch.Tensor, gaugegroup: GaugeGroup, eps: float = 1e-12
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Factor an SU(2) staple sum as ``A = k · V`` with ``V ∈ SU(2)``.

    Any sum of SU(2) matrices keeps the quaternion form c_0·I + i(c·σ) with
    real c, so it is a non-negative real multiple of an SU(2) element:
    ``k = √det A`` (real ≥ 0) and ``V = A / k``. Returns ``(k, V†)`` — ``V†``
    is what both the heat-bath (``U' = W·V†``) and the overrelaxation reflection
    (``U' = V†·U†·V†``) need. ``k`` is clamped at ``eps`` to keep the division
    well-defined when the staples cancel (k → 0, the disordered limit).

    Parameters
    ----------
    A : ``(*Λ, 2, 2)`` staple sum from :func:`staple_sum`.

    Returns
    -------
    (k, V_dag) : ``k`` of shape ``(*Λ)``, ``V_dag`` of shape ``(*Λ, 2, 2)``.
    """
    det = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    k = det.real.clamp_min(eps).sqrt()  # (*Λ); det is real ≥ 0 for quaternion A
    V_dag = gaugegroup.dagger(A) / k[..., None, None]
    return k, V_dag


def _project_su2(M: torch.Tensor) -> torch.Tensor:
    """Nearest SU(2) to ``M``: keep its quaternion part and renormalise.

    Every SU(2) element is ``[[α, β], [−β̄, ᾱ]]`` with ``|α|² + |β|² = 1``; the
    least-squares projection of a 2×2 ``M`` onto that form is
    ``α = (M₀₀ + M̄₁₁)/2``, ``β = (M₀₁ − M̄₁₀)/2``, renormalised. This closed
    form is cheaper than the general SVD/polar in :meth:`SU.project` (and avoids
    its ``linalg`` ops); overrelaxation calls it every sweep to shed the
    round-off it would otherwise amplify off the group manifold.
    """
    alpha = 0.5 * (M[..., 0, 0] + M[..., 1, 1].conj())
    beta = 0.5 * (M[..., 0, 1] - M[..., 1, 0].conj())
    norm = (alpha.abs() ** 2 + beta.abs() ** 2).sqrt().clamp_min(1e-12)
    alpha, beta = alpha / norm, beta / norm
    out = torch.empty_like(M)
    out[..., 0, 0], out[..., 0, 1] = alpha, beta
    out[..., 1, 0], out[..., 1, 1] = -beta.conj(), alpha.conj()
    return out


def _sample_su2_w0(a: torch.Tensor, max_iter: int = 100) -> torch.Tensor:
    """Sample the scalar part w_0 ∈ [−1, 1] of the heat-bath SU(2) element.

    The local weight for ``W = U·V`` is ``∝ exp(a·w_0)`` with the SU(2) Haar
    measure contributing ``√(1−w_0²)``, so w_0 has density
    ``∝ √(1−w_0²) · exp(a·w_0)``. Creutz (1980): propose ``x ∝ exp(a·x)`` on
    ``[−1, 1]`` via inverse-CDF (written ``x = 1 + ln[r + (1−r)e^{−2a}] / a``
    to stay overflow-free at large ``a``) and accept with probability
    ``√(1−x²)`` (the test ``r'² ≤ 1 − x²``). Unlike Kennedy–Pendleton this is
    robust at *every* ``a = β·|staple|`` — including the small-staple sites of
    coarse / low-dimensional lattices — and stays efficient at the large ``a``
    of thermalised 4D SU(2). The rejection loop is vectorised over the lattice;
    only not-yet-accepted sites are resampled.

    Parameters
    ----------
    a : ``(*Λ)`` non-negative heat-bath parameter, one per site.

    Returns
    -------
    ``(*Λ)`` real tensor of w_0 values.
    """
    shape = a.shape
    device = a.device
    a = a.clamp_min(1e-6)  # guard the 1/a at fully disordered sites
    e_m2a = torch.exp(-2 * a)
    w0 = torch.zeros(shape, dtype=a.dtype, device=device)
    accepted = torch.zeros(shape, dtype=torch.bool, device=device)
    for _ in range(max_iter):
        r = torch.rand(shape, dtype=a.dtype, device=device)
        rp = torch.rand(shape, dtype=a.dtype, device=device)
        x = 1 + torch.log(r + (1 - r) * e_m2a) / a  # x ∝ exp(a·x) on [−1, 1]
        hit = (rp * rp <= 1 - x * x) & ~accepted
        w0 = torch.where(hit, x, w0)
        accepted = accepted | hit
        if bool(accepted.all()):
            break
    if not bool(accepted.all()):
        raise RuntimeError(
            "SU(2) heat-bath w_0 sampling failed to converge in "
            f"{max_iter} iterations (unexpected — Creutz accepts at every a)."
        )
    return w0


def _su2_heatbath_links(
    A: torch.Tensor, beta: float, gaugegroup: GaugeGroup
) -> torch.Tensor:
    """Draw a fresh SU(2) link per site from its heat-bath distribution.

    With ``A = k·V`` (see :func:`_su2_decompose_staple`), the new link is
    ``U' = W·V†`` where ``W ∈ SU(2)`` is drawn with weight ``∝ exp(β·k·w_0)``:
    w_0 from :func:`_sample_su2_w0` and the 3-vector uniform on the sphere of
    radius ``√(1−w_0²)``. Computed for *every* site; the caller applies it only
    to the active checkerboard parity.
    """
    k, V_dag = _su2_decompose_staple(A, gaugegroup)
    w0 = _sample_su2_w0(beta * k)  # (*Λ)
    norm = (1 - w0 * w0).clamp_min(0).sqrt()  # |w| = √(1−w_0²)
    g = torch.randn(*w0.shape, 3, dtype=w0.dtype, device=w0.device)
    g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # uniform direction
    W = _su2_from_quaternion(w0, norm[..., None] * g)
    return W @ V_dag


def _su2_local_sweep(U: torch.Tensor, gaugegroup: GaugeGroup, update) -> torch.Tensor:
    """One checkerboard sweep applying a per-site SU(2) update to every link.

    ``update(A, U_mu) -> U_new`` maps the staple sum ``A`` and the current links
    ``U_mu`` of a direction to the proposed links for all sites; only the active
    checkerboard parity is written. Same-parity links along μ share no staple, so
    they update independently, and the staple is recomputed once per (μ, parity).
    The heat-bath and overrelaxation sweeps differ *only* in ``update``.

    SU(2) only — for SU(N≥3) use Cabibbo–Marinari (heat-bath on SU(2) subgroups).
    """
    if not (isinstance(gaugegroup, SU) and gaugegroup.nc == 2):
        raise NotImplementedError(
            f"SU(2) heat-bath / overrelaxation only, got {gaugegroup}. "
            "For SU(N≥3) use Cabibbo–Marinari on SU(2) sub-blocks."
        )
    D = U.shape[0]
    parity = _site_parity(U.shape[1:-2], U.device)
    U = U.clone()
    for mu in range(D):
        for par in (0, 1):
            A = staple_sum(U, mu, gaugegroup)
            mask = (parity == par)[..., None, None]
            U[mu] = torch.where(mask, update(A, U[mu]), U[mu])
    return U


def heatbath_sweep(
    U: torch.Tensor, gaugegroup: GaugeGroup, beta: float
) -> Tuple[torch.Tensor, float]:
    """One SU(2) heat-bath sweep — each link drawn from its exact local weight.

    There is no accept/reject (Creutz sampling), so decorrelation is far better
    than Metropolis with no critical-slowing tuning. Returns acceptance 1.0
    (heat-bath always "accepts"). SU(2) only.
    """
    U = _su2_local_sweep(
        U, gaugegroup, lambda A, U_mu: _su2_heatbath_links(A, beta, gaugegroup)
    )
    return U, 1.0


def overrelaxation_sweep(
    U: torch.Tensor, gaugegroup: GaugeGroup, beta: Optional[float] = None
) -> Tuple[torch.Tensor, float]:
    """One SU(2) overrelaxation sweep — a microcanonical, action-preserving move.

    The local action depends on the link only through ``Re Tr(U·A) = k·Re Tr(U·V)``.
    Reflecting ``W = U·V → W†`` leaves ``Re Tr`` invariant (SU(2) traces are
    real) but moves the link maximally far, so ``U' = V†·U†·V†`` conserves the
    action exactly while decorrelating the long-wavelength modes that heat-bath
    alone leaves slow. Deterministic — no randomness, no rejection — and
    therefore not ergodic on its own: interleave with :func:`heatbath_sweep`.
    ``beta`` is accepted for sweep-signature compatibility but unused (the move
    is independent of β). SU(2) only.

    The reflection is an isometry on the SU(2) manifold but *expansive* off it:
    it does not damp round-off, so the tiny non-unitarity grows geometrically
    (≈ ×15 per sweep) and would blow the links up within a handful of sweeps.
    Each updated link is therefore re-projected onto SU(2) via
    :func:`_project_su2` — a no-op (to round-off) on an already-on-group link,
    so the action is still conserved, but it keeps the ensemble on the group.
    """

    def reflect(A, U_mu):
        _, V_dag = _su2_decompose_staple(A, gaugegroup)
        return _project_su2(V_dag @ gaugegroup.dagger(U_mu) @ V_dag)

    return _su2_local_sweep(U, gaugegroup, reflect), 1.0


def heatbath_overrelaxation_sweep(
    U: torch.Tensor, gaugegroup: GaugeGroup, beta: float, n_or: int = 4
) -> Tuple[torch.Tensor, float]:
    """One heat-bath sweep followed by ``n_or`` overrelaxation sweeps (SU(2)).

    The standard production recipe: heat-bath supplies ergodicity, the cheap
    deterministic overrelaxation sweeps accelerate decorrelation of the slow
    modes. Pin ``n_or`` from a sampler call site with
    ``functools.partial(heatbath_overrelaxation_sweep, n_or=...)``. Registered
    nowhere by default — pass it as ``sweep_fn=`` to :func:`mcmc_ensemble` to
    opt SU(2) ensembles into heat-bath instead of Metropolis.
    """
    U, _ = heatbath_sweep(U, gaugegroup, beta)
    for _ in range(n_or):
        U, _ = overrelaxation_sweep(U, gaugegroup)
    return U, 1.0


def haar_ensemble(
    L: int,
    D: int,
    gaugegroup: GaugeGroup,
    beta: float,
    n_configs: int,
    n_therm: int = 0,
    n_skip: int = 1,
    sweep_fn=None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Haar-uniform ensemble — no dynamics, ignores beta.

    Shares the sampler interface with ``mcmc_ensemble`` so it can be passed
    as ``sampler=haar_ensemble`` anywhere a sampler callable is expected.
    Useful as a baseline or for architecture sanity-checks before MC is set up.
    """
    return random_links(L, D, gaugegroup, dtype=dtype, N=n_configs), 1.0


# Registry: maps GaugeGroup subclass → default sweep function (the *algorithm*).
# Z₂ and SU(2) both use the single group-agnostic ``metropolis_sweep`` (the
# group enters only through ``_PROPOSAL_FN``). This registry is the extension
# point for genuinely different algorithms — heat-bath, overrelaxation,
# Cabibbo–Marinari — which are not accept/reject and so cannot reuse
# ``metropolis_sweep``. Both SU(2) and SU(3) hash to the same `SU` key;
# ``metropolis_sweep`` raises ``NotImplementedError`` (via ``_su2_proposal``)
# for nc ≠ 2, so SU(3) callers see a clear error until a Cabibbo–Marinari
# sweep is wired in.
_SWEEP_FN: dict = {
    Z2: metropolis_sweep,
    SU: metropolis_sweep,
    # U1:  heatbath_sweep,   ← future
    # SU3: cabibbo_marinari_sweep,
}


def mcmc_ensemble(
    L: int,
    D: int,
    gaugegroup: GaugeGroup,
    beta: float,
    n_configs: int,
    n_therm: int = 200,
    n_skip: int = 5,
    sweep_fn=None,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    progress: bool = True,
) -> Tuple[torch.Tensor, float]:
    """Generate a thermalized ensemble of gauge field configurations.

    Starts from a Haar-random configuration, runs ``n_therm`` thermalisation
    sweeps, then collects one configuration every ``n_skip`` sweeps.

    Parameters
    ----------
    L, D       : lattice size and number of dimensions
    gaugegroup : gauge group
    beta       : inverse coupling (Boltzmann weight ~ exp(−β S))
    n_configs  : number of configurations to collect
    n_therm    : thermalisation sweeps before collection begins
    n_skip     : sweeps between collected configurations (decorrelation)
    progress   : show a tqdm progress bar over the thermalisation and
                 production sweeps (disable for quiet runs / tests)
    sweep_fn   : single-sweep callable ``(U, gaugegroup, beta) → (U_new, acc)``.
                 If ``None``, dispatches automatically via ``_SWEEP_FN[type(gaugegroup)]``.
                 To pin a custom sweep from a call site that only accepts a sampler
                 argument, use ``functools.partial``::

                     sampler = functools.partial(mcmc_ensemble, sweep_fn=my_sweep)
                     build_plaquette_datasets(..., sampler=sampler)
    dtype, device : passed to ``random_links``

    Returns
    -------
    (configs, mean_acceptance)
        ``configs``        : ``(n_configs, D, *Λ, nc, nc)`` on CPU
        ``mean_acceptance``: mean acceptance rate over production run
    """
    if sweep_fn is None:
        sweep_fn = _SWEEP_FN.get(type(gaugegroup))
        if sweep_fn is None:
            raise NotImplementedError(
                f"No sweep function registered for {type(gaugegroup).__name__}. "
                f"Pass sweep_fn= explicitly or add an entry to sampler._SWEEP_FN."
            )

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    U = random_links(L, D, gaugegroup, dtype=dtype).to(device)

    for _ in tqdm(
        range(n_therm), desc="thermalising", disable=not progress, leave=False
    ):
        U, _ = sweep_fn(U, gaugegroup, beta)

    configs: List[torch.Tensor] = []
    acc_rates: List[float] = []
    for i in tqdm(
        range(n_configs * n_skip),
        desc=f"sampling {n_configs} configs",
        disable=not progress,
        leave=False,
    ):
        U, acc = sweep_fn(U, gaugegroup, beta)
        if (i + 1) % n_skip == 0:
            configs.append(U.cpu())
            acc_rates.append(acc)

    return torch.stack(configs), sum(acc_rates) / len(acc_rates)


def integrated_autocorrelation_time(
    series, c: float = 6.0, max_lag: Optional[int] = None
) -> Tuple[np.ndarray, float, int]:
    """Normalised autocorrelation ρ(t) and integrated autocorrelation time τ_int
    of a scalar Markov-chain observable, with Madras–Sokal automatic windowing.

    For a chain ``x_i`` (one scalar per Monte-Carlo step),
        ρ(t)  = ⟨(x_i − x̄)(x_{i+t} − x̄)⟩ / ⟨(x_i − x̄)²⟩,
        τ_int = ½ + Σ_{t=1}^{W} ρ(t),
    where the window ``W`` is the smallest lag with ``W ≥ c · τ_int(W)`` (Sokal;
    ``c ≈ 6`` balances the bias of truncating too early against the variance of
    summing the noisy tail). τ_int counts correlated steps: samples drawn
    ``n_skip ≳ 2·τ_int`` apart are effectively independent, and the error on the
    mean scales as ``σ / √(N / 2τ_int)``. A perfectly mixed chain gives
    ρ(t≥1) ≈ 0 and τ_int → ½ (every step independent).

    Parameters
    ----------
    series : 1-D array-like, one scalar per chain step (numpy or a CPU tensor).
    c : Sokal windowing constant.
    max_lag : largest lag evaluated (defaults to ``len(series) // 4``).

    Returns
    -------
    (rho, tau_int, window) : ρ as a numpy array over lags 0…max_lag, the τ_int
        estimate, and the chosen window ``W``.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if max_lag is None:
        max_lag = n // 4
    delta = x - x.mean()
    var = float(np.mean(delta * delta))
    rho = np.ones(max_lag + 1)
    if var > 0:
        for t in range(1, max_lag + 1):
            rho[t] = np.mean(delta[: n - t] * delta[t:]) / var
    # Sokal automatic window: stop at the first W with W ≥ c · τ_int(W).
    tau, window = 0.5, max_lag
    for t in range(1, max_lag + 1):
        tau += rho[t]
        if t >= c * tau:
            window = t
            break
    tau_int = 0.5 + float(rho[1 : window + 1].sum())
    return rho, tau_int, window
