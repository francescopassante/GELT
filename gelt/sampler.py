"""Monte Carlo sampler for lattice gauge theory.

Phase 0: single-site Metropolis with checkerboard vectorisation.
The staple interface is designed for extension to heat-bath + overrelaxation
for U(1)/SU(2)/SU(3) without restructuring the sweep loop.
"""

from typing import List, Optional, Tuple

import torch

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


def metropolis_sweep(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    beta: float,
) -> Tuple[torch.Tensor, float]:
    """One full Metropolis sweep (all directions, both checkerboard parities).

    For Z₂ the proposal is the unique non-identity element U' = −U. To extend
    to U(1)/SU(N), replace the ``U_proposed = -U_mu`` line with a group-valued
    random proposal drawn from a neighbourhood of U_mu.

    Checkerboard structure: for a fixed direction μ, sites of the same parity
    do not share any plaquette through same-direction links, so their updates
    commute. The even sweep then the odd sweep is equivalent to a sequential
    site-by-site update but fully vectorised.

    Parameters
    ----------
    U          : ``(D, *Λ, nc, nc)`` — not modified in-place
    gaugegroup : gauge group
    beta       : inverse coupling

    Returns
    -------
    (U_new, acceptance_rate)
    """
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
            A = staple_sum(U, mu, gaugegroup)  # (*Λ, nc, nc)
            U_mu = U[mu]
            U_proposed = -U_mu  # Z₂: only proposal is the flip

            # ΔS = (β/nc) Re Tr[(U − U') · A]  > 0 means action increases
            dS = (beta / nc) * _re_tr((U_mu - U_proposed) @ A)  # (*Λ)

            rand = torch.rand(spatial_shape).to(device)
            accept = (dS <= 0) | (rand < torch.exp(-dS.clamp(min=0)))

            site_mask = parity == par  # (*Λ)
            update_mask = accept & site_mask  # (*Λ)

            total_accepted += update_mask.sum().item()
            total_proposed += site_mask.sum().item()

            # [..., None, None] adds two extra dimension to broadcast update_mask with U_mu (or U_proposed)
            U[mu] = torch.where(update_mask[..., None, None], U_proposed, U_mu)

    return U, total_accepted / total_proposed


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
    a = (torch.rand(*spatial_shape, 3, dtype=real_dtype, device=device) * 2 - 1) * epsilon
    a0 = torch.sqrt(1.0 - (a * a).sum(dim=-1))
    # V = [[a_0 + i a_3,  i a_1 + a_2],
    #      [i a_1 - a_2,  a_0 - i a_3]]
    V = torch.empty(*spatial_shape, 2, 2, dtype=dtype, device=device)
    V[..., 0, 0] = torch.complex(a0, a[..., 2])
    V[..., 0, 1] = torch.complex(a[..., 1], a[..., 0])
    V[..., 1, 0] = torch.complex(-a[..., 1], a[..., 0])
    V[..., 1, 1] = torch.complex(a0, -a[..., 2])
    return V


def su2_metropolis_sweep(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    beta: float,
    epsilon: float = 0.3,
    n_hits: int = 1,
) -> Tuple[torch.Tensor, float]:
    """One full Metropolis sweep for SU(2), checkerboard-vectorised.

    Mirrors :func:`metropolis_sweep` but the proposal is U' = V · U where
    V is a random SU(2) element close to the identity (see
    :func:`_random_su2_near_identity`). The staple sum, local action, and
    the ΔS = (β / N_c) Re Tr[(U − U') · A] formula are identical to the
    Z₂ case — only the proposal kernel changes.

    For a fixed direction μ, same-parity sites do not share any plaquette
    through same-direction links, so their updates commute and can be
    applied in parallel — the same trick as the Z₂ sweep.

    Parameters
    ----------
    U          : ``(D, *Λ, 2, 2)`` — not modified in-place
    gaugegroup : must be ``SU(2)``
    beta       : inverse coupling
    epsilon    : width of the proposal neighbourhood around the identity;
                 tune to keep acceptance around 50%
    n_hits     : Metropolis hits per link per sweep. The staple is the
                 expensive piece and is unchanged between hits at the
                 same site/parity, so multiple hits per sweep improve
                 mixing at almost no extra cost.

    Returns
    -------
    (U_new, acceptance_rate)
    """
    if gaugegroup.nc != 2:
        raise NotImplementedError(
            f"su2_metropolis_sweep only supports SU(2), got nc={gaugegroup.nc}. "
            "For SU(N≥3) use Cabibbo–Marinari (heat-bath on SU(2) sub-blocks)."
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
                V = _random_su2_near_identity(spatial_shape, epsilon, U.dtype, device)
                U_proposed = V @ U_mu

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


# Registry: maps GaugeGroup subclass → default sweep function.
# Add one line here when a new group/algorithm pair is ready.
# Both SU(2) and SU(3) hash to the same `SU` key; ``su2_metropolis_sweep``
# raises ``NotImplementedError`` for nc ≠ 2, so SU(3) callers will see a
# clear error until a Cabibbo–Marinari sweep is wired in.
_SWEEP_FN: dict = {
    Z2: metropolis_sweep,
    SU: su2_metropolis_sweep,
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

    for _ in range(n_therm):
        U, _ = sweep_fn(U, gaugegroup, beta)

    configs: List[torch.Tensor] = []
    acc_rates: List[float] = []
    for i in range(n_configs * n_skip):
        U, acc = sweep_fn(U, gaugegroup, beta)
        if (i + 1) % n_skip == 0:
            configs.append(U.cpu())
            acc_rates.append(acc)

    return torch.stack(configs), sum(acc_rates) / len(acc_rates)
