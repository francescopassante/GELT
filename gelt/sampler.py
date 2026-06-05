"""Monte Carlo sampler for lattice gauge theory.

Phase 0: single-site Metropolis with checkerboard vectorisation.
The staple interface is designed for extension to heat-bath + overrelaxation
for U(1)/SU(2)/SU(3) without restructuring the sweep loop.
"""

from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from gelt.lattice import SU, Z2, GaugeGroup, random_links


def staple_sum(U: torch.Tensor, mu: int, gaugegroup: GaugeGroup) -> torch.Tensor:
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

    Returns
    -------
    Tensor of shape ``(*Λ, nc, nc)``.
    """
    D = U.shape[0]
    A = torch.zeros_like(U[mu])
    for nu in range(D):
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
    # V = [[a_0 + i a_3,  i a_1 + a_2],
    #      [i a_1 - a_2,  a_0 - i a_3]]
    V = torch.empty(*spatial_shape, 2, 2, dtype=dtype, device=device)
    V[..., 0, 0] = torch.complex(a0, a[..., 2])
    V[..., 0, 1] = torch.complex(a[..., 1], a[..., 0])
    V[..., 1, 0] = torch.complex(-a[..., 1], a[..., 0])
    V[..., 1, 1] = torch.complex(a0, -a[..., 2])
    return V


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
