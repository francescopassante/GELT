"""Lattice gauge theory primitives as pure tensor operations.

Conventions
-----------
- Link tensor shape: ``(D, L, ..., L, nc, nc)`` where the leading axis indexes
  the spatial direction ``μ ∈ {0, ..., D-1}``, the middle ``D`` axes are the
  spatial coordinates, and the trailing ``(nc, nc)`` axes are the matrix
  representation of the link in the gauge group's defining representation.
- Even for Z₂ (where ``nc = 1``) the trailing color axes are kept so that
  every operation generalises verbatim to U(1) / SU(N). The dagger is written
  out explicitly for the same reason.
- Plaquette tensor shape: ``(D(D-1)/2, L, ..., L, nc, nc)``, ordered by
  ``(μ, ν)`` pairs with ``μ < ν`` lexicographically.
- Plaquette convention:
  ``P_{μν}(x) = U_μ(x) · U_ν(x + μ̂) · U_μ†(x + ν̂) · U_ν†(x)``.
- Periodic boundary conditions throughout (``torch.roll`` for shifts).
"""

import itertools
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


class GaugeGroup(ABC):
    """Abstract gauge group, parametrised by its defining-representation dimension nc."""

    name: str
    nc: int

    def __str__(self) -> str:
        return self.name

    @abstractmethod
    def random(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Sample group elements as a tensor of shape ``shape + (nc, nc)`` (Haar)."""

    @abstractmethod
    def dagger(self, U: torch.Tensor) -> torch.Tensor:
        """Hermitian conjugate (= group inverse for unitary groups)."""


class Z2(GaugeGroup):
    name = "Z2"
    nc = 1

    def random(self, shape, dtype=torch.float32):
        signs = (torch.randint(0, 2, shape, dtype=torch.int64) * 2 - 1).to(dtype)
        # Add the trailing (nc, nc) = (1, 1) color axes so that the layout matches U(1)/SU(N).
        return signs.unsqueeze(-1).unsqueeze(-1)

    def dagger(self, U):
        # Identity for real 1×1 matrices, but written explicitly for portability.
        return U.conj().transpose(-1, -2)


class SU(GaugeGroup):
    def __init__(self, n: int):
        self.name = f"SU({n})"
        self.nc = n

    def random(self, shape, dtype=torch.complex64):
        # Accept either a real or complex dtype as the precision specifier;
        # the result is always complex (real dtypes are promoted).
        complex_dtype = {
            torch.float32: torch.complex64,
            torch.float64: torch.complex128,
            torch.complex64: torch.complex64,
            torch.complex128: torch.complex128,
        }[dtype]
        # Haar sampling on U(N) via QR of a complex Ginibre matrix with the
        # Mezzadri diagonal-phase correction: PyTorch's QR fixes the
        # phase of Q by a convention that is not Haar-uniform, so we absorb
        # the phases of diag(R) back into Q's columns.
        z = torch.randn(*shape, self.nc, self.nc, dtype=complex_dtype)
        q, r = torch.linalg.qr(z)
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        q = q * (d / d.abs()).unsqueeze(-2)
        # q is Haar on U(N), with det on the unit circle; divide by det^(1/nc)
        # (principal branch) to project onto SU(N).
        det = torch.linalg.det(q)
        return q / det.pow(1 / self.nc).unsqueeze(-1).unsqueeze(-1)

    def dagger(self, U):
        return U.conj().transpose(-1, -2)


def random_links(
    L: int,
    D: int,
    gaugegroup: GaugeGroup,
    dtype: torch.dtype = torch.float32,
    N: Optional[int] = None,
) -> torch.Tensor:
    """Sample Haar-random link configuration(s).

    Without ``N``: returns shape ``(D, L, ..., L, nc, nc)``.
    With ``N``:    returns shape ``(N, D, L, ..., L, nc, nc)``.
    """
    shape = (D,) + (L,) * D
    if N is not None:
        shape = (N,) + shape
    return gaugegroup.random(shape, dtype=dtype)


def plaquette_tensor(U: torch.Tensor, gaugegroup: GaugeGroup) -> torch.Tensor:
    """Compute every 1×1 plaquette ``P_{μν}(x)`` for ``μ < ν``.

    Parameters
    ----------
    U
        Batched links of shape ``(B, D, *Λ, nc, nc)``.
    gaugegroup
        Gauge group (used for the dagger operation).

    Returns
    -------
    Tensor of shape ``(B, n_pairs, *Λ, nc, nc)`` with ``n_pairs = D(D-1)/2``.
    """
    D = U.shape[1]
    pairs = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]
    plaqs = []
    for mu, nu in pairs:
        # U[:, mu] has shape (B, *Λ, nc, nc); spatial axis mu sits at index mu+1.
        # torch.roll(t, -1, dims=mu+1) brings the value at x+μ̂ to index x.
        Umu = U[:, mu]
        Unu = U[:, nu]
        Unu_shift_mu = torch.roll(Unu, shifts=-1, dims=mu + 1)  # U_ν(x + μ̂)
        Umu_shift_nu = torch.roll(Umu, shifts=-1, dims=nu + 1)  # U_μ(x + ν̂)
        P = (
            Umu
            @ Unu_shift_mu
            @ gaugegroup.dagger(Umu_shift_nu)
            @ gaugegroup.dagger(Unu)
        )
        plaqs.append(P)
    return torch.stack(plaqs, dim=1)


def action(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    beta: float = 1.0,
    plaquettes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Wilson action ``S = β Σ_p (1 − Re Tr P_p / N_c)`` for a batch of configs.

    Parameters
    ----------
    U
        Batched link tensor of shape ``(B, D, *Λ, nc, nc)``.
        Ignored when ``plaquettes`` is provided.
    gaugegroup
        Gauge group (used to compute plaquettes if needed).
    beta
        Coupling.
    plaquettes
        Pre-computed batched plaquette tensor of shape ``(B, n_pairs, *Λ, nc, nc)``;
        if ``None`` it is computed from ``U``.

    Returns
    -------
    Tensor of shape ``(B,)`` — one scalar action value per configuration.
    """
    P = plaquettes if plaquettes is not None else plaquette_tensor(U, gaugegroup)
    re_tr_over_nc = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / gaugegroup.nc
    # re_tr_over_nc: (B, n_pairs, *Λ) — sum over all plaquettes per config.
    n_plaq_per_config = re_tr_over_nc[0].numel()
    # equivalent to beta (sum_p 1 - P_p) per config
    return beta * (n_plaq_per_config - re_tr_over_nc.flatten(1).sum(1))


def _permutation_sign(perm: Tuple[int, ...]) -> int:
    """Sign (±1) of a permutation, by counting inversions."""
    sign = 1
    p = list(perm)
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] > p[j]:
                sign = -sign
    return sign


def topological_charge_density(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    plaquettes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plaquette ("naive") topological charge density q_x at every site.

        q_x = (ε_{μνρσ} / 32π²) Tr[ F_{μν}(x) · F_{ρσ}(x) ],
        F_{μν}(x) = (P_{μν}(x) − P_{μν}†(x)) / (2i),

    summed over all four indices μ, ν, ρ, σ ∈ {0,1,2,3}.  ``F_{μν}`` is the
    Hermitian (anti-Hermitian-part-of-the-plaquette) lattice field strength
    built from the 1×1 plaquette ``P_{μν}``; it is antisymmetric in its plane
    indices (``F_{νμ} = −F_{μν}``, since ``P_{νμ} = P_{μν}†``).

    Defined only in D = 4 — the Levi-Civita symbol ε_{μνρσ} needs exactly four
    directions. The density vanishes identically for Z₂/real links (``P`` is
    its own dagger), and the total charge ``Q = Σ_x q_x`` only carries
    topological meaning for non-abelian SU(N≥2).

    Parameters
    ----------
    U
        Batched link tensor ``(B, D, *Λ, nc, nc)`` (used to read D, and to
        compute the plaquettes if ``plaquettes`` is not supplied).
    gaugegroup
        Gauge group (used for the dagger).
    plaquettes
        Optional precomputed ``(B, n_pairs, *Λ, nc, nc)`` plaquette tensor.

    Returns
    -------
    Real tensor of shape ``(B, *Λ)`` — q_x at every site of every config.
    """
    D = U.shape[1]
    if D != 4:
        raise ValueError(
            f"Topological charge density is defined only in D=4, got D={D}."
        )
    P = plaquettes if plaquettes is not None else plaquette_tensor(U, gaugegroup)

    # F_{μν} = (P_{μν} − P_{μν}†)/(2i) for each μ<ν plane (the order
    # plaquette_tensor stacks them in); fill in F_{νμ} = −F_{μν}. Dividing by
    # the imaginary 2i promotes Z₂'s real (and identically antisymmetric-zero)
    # plaquettes to complex, matching SU(N).
    pairs = [(mu, nu) for mu in range(D) for nu in range(mu + 1, D)]
    F: Dict[Tuple[int, int], torch.Tensor] = {}
    for idx, (mu, nu) in enumerate(pairs):
        Pmn = P[:, idx]
        f = (Pmn - gaugegroup.dagger(Pmn)) / 2j
        F[(mu, nu)] = f
        F[(nu, mu)] = -f

    # q_x = (1/32π²) Σ_{μνρσ} ε_{μνρσ} Tr[F_{μν} F_{ρσ}]. itertools.permutations
    # enumerates exactly the 24 all-distinct index tuples (every other term has
    # ε = 0). The imaginary parts cancel in the antisymmetric sum; take .real.
    q: Optional[torch.Tensor] = None
    for mu, nu, rho, sigma in itertools.permutations(range(D)):
        sign = _permutation_sign((mu, nu, rho, sigma))
        prod = F[(mu, nu)] @ F[(rho, sigma)]
        tr = prod.diagonal(dim1=-2, dim2=-1).sum(dim=-1)  # (B, *Λ), complex
        term = sign * tr
        q = term if q is None else q + term

    return q.real / (32.0 * math.pi**2)


def topological_charge(
    U: torch.Tensor,
    gaugegroup: GaugeGroup,
    plaquettes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Total topological charge ``Q = Σ_x q_x`` (one scalar per config).

    Sums :func:`topological_charge_density` over all sites; returns shape
    ``(B,)``. This is the scalar regression target analogous to :func:`action`.
    """
    q = topological_charge_density(U, gaugegroup, plaquettes=plaquettes)
    return q.flatten(1).sum(1)


def link_gauge_transformation(
    U: torch.Tensor,
    omega: torch.Tensor,
    gaugegroup: GaugeGroup,
) -> torch.Tensor:
    """Apply a site-local gauge transformation Ω to a link configuration.
    For each direction μ:
        U'_μ(x) = Ω(x) · U_μ(x) · Ω†(x + μ̂)
    """

    D = U.shape[0]
    out = []
    for mu in range(D):
        omega_shifted = torch.roll(omega, shifts=-1, dims=mu)  # Ω(x + μ̂)
        out.append(omega @ U[mu] @ gaugegroup.dagger(omega_shifted))
    return torch.stack(out, dim=0)


def local_gauge_transformation(W, omega, gaugegroup):
    """Adjoint-field gauge transform: W_g(x) = Ω(x) · W(x) · Ω†(x).

    W : (B, C, *Λ, nc, nc); omega : (*Λ, nc, nc). Broadcasts over (B, C).
    """
    omega_b = omega.unsqueeze(0).unsqueeze(0)  # (1, 1, *Λ, nc, nc)
    return omega_b @ W @ gaugegroup.dagger(omega_b)


def rectangular_wilson_loop(
    config: torch.Tensor,
    gaugegroup: GaugeGroup,
    R: int,
    T: int,
    mu: int,
    nu: int,
) -> torch.Tensor:
    """Re Tr W_μν(x; R, T) / nc at every site for the R×T rectangular Wilson loop.

    The loop at site x traverses:
      Segment 1: R forward steps in μ:  U_μ(x + i·μ̂),            i = 0 … R-1
      Segment 2: T forward steps in ν:  U_ν(x + R·μ̂ + i·ν̂),      i = 0 … T-1
      Segment 3: R backward steps in μ: U†_μ(x + k·μ̂ + T·ν̂),     k = R-1 … 0
      Segment 4: T backward steps in ν: U†_ν(x + k·ν̂),            k = T-1 … 0

    At R=T=1 this reduces to the standard plaquette P_μν(x).

    Parameters
    ----------
    config : ``(B, D, *Λ, nc, nc)`` batched link tensor.
    R : side length along μ in lattice units.
    T : side length along ν in lattice units.
    mu, nu : plane directions, distinct, in ``[0, D)``.

    Returns
    -------
    Real tensor of shape ``(B, *Λ)``: Re Tr W(x; R, T) / nc at every site x
    for each configuration in the batch.
    """
    D = config.shape[1]
    if not (0 <= mu < D and 0 <= nu < D and mu != nu):
        raise ValueError(f"Invalid directions mu={mu}, nu={nu} for D={D}.")

    # torch.eye broadcasts to (B, *Λ, nc, nc) on the first matmul.
    loop = torch.eye(gaugegroup.nc, dtype=config.dtype, device=config.device)

    # Spatial axes sit at positions 1..D inside config[:, d] — offset dims by 1.
    # Segment 1: U_μ(x + i·μ̂), i = 0 … R-1.
    for i in range(R):
        loop = loop @ torch.roll(config[:, mu], shifts=-i, dims=mu + 1)

    # Segment 2: U_ν(x + R·μ̂ + i·ν̂), i = 0 … T-1.
    for i in range(T):
        loop = loop @ torch.roll(
            torch.roll(config[:, nu], shifts=-R, dims=mu + 1), shifts=-i, dims=nu + 1
        )

    # Segment 3: U†_μ(x + k·μ̂ + T·ν̂), k = R-1 … 0.
    for k in range(R - 1, -1, -1):
        loop = loop @ gaugegroup.dagger(
            torch.roll(
                torch.roll(config[:, mu], shifts=-k, dims=mu + 1),
                shifts=-T,
                dims=nu + 1,
            )
        )

    # Segment 4: U†_ν(x + k·ν̂), k = T-1 … 0.
    for k in range(T - 1, -1, -1):
        loop = loop @ gaugegroup.dagger(
            torch.roll(config[:, nu], shifts=-k, dims=nu + 1)
        )

    return loop.diagonal(dim1=-2, dim2=-1).sum(dim=-1).real / gaugegroup.nc


def l1_ball_offsets(D: int, R: int) -> List[Tuple[int, ...]]:
    """All non-zero offsets Δx with |Δx|₁ ≤ R, sorted by L1 norm.

    Parameters
    ----------
    D
        Number of lattice directions.
    R
        Manhattan radius.

    Returns
    -------
    List of offset tuples ``(Δx₀, …, Δx_{D-1})``, sorted so that entries
    with smaller ``|Δx|₁`` come first (ties broken by lexicographic order).
    The ordering guarantees that when building the DP table, every
    sub-step offset ``Δx ± ê_μ`` is already present when ``Δx`` is reached.
    """
    return sorted(
        (
            dx
            for dx in itertools.product(range(-R, R + 1), repeat=D)
            if 0 < sum(abs(d) for d in dx) <= R
        ),
        key=lambda dx: sum(abs(d) for d in dx),
    )


def build_transport_average(
    U: torch.Tensor,
    R: int,
    gaugegroup: GaugeGroup,
    mode: str = "average",
) -> torch.Tensor:
    """Parallel transports for **every** offset 0 < |Δx|₁ ≤ R.


    ``mode="average"`` (default, the architecture's design choice).
    For each signed lattice offset Δx, the entry is the **average** over all
    shortest lattice paths from x to x+Δx:

        T_Δx(x)  =  (1 / N_Δx)  Σ_{P : x→x+Δx, |P|=|Δx|₁}  U_P

    with ``N_Δx = |Δx|₁! / Π_μ |Δx_μ|!`` the multinomial number of shortest
    paths. The DP builds the unnormalised sum and divides by ``N_Δx`` at the
    end. Preserves lattice rotation symmetry on top of gauge covariance.

    ``mode="single"`` (A/B variant for the path-averaging diagnostic).
    For each Δx, the entry is the transport along **one canonical shortest
    path** — at every DP step the lowest-index nonzero direction is taken,
    so the path walks |Δx_0| signed steps in direction 0, then |Δx_1| in
    direction 1, etc. Gauge covariance and translation equivariance still
    hold (each individual link step is gauge-covariant). 90°-rotation
    equivariance is **broken** — the path picks a preferred direction order.
    No normalisation (a single path, not a mean).

    Either way, normalising / not normalising by a scalar preserves gauge
    covariance: under site-local Ω,

        T_Δx(x)  →  Ω(x) · T_Δx(x) · Ω†(x+Δx)

    The DP recursion mixes forward and backward links per component sign:

        T_Δx(x) = Σ_{μ : Δx_μ > 0}  U_μ(x) · T_{Δx − ê_μ}(x + ê_μ)
                + Σ_{μ : Δx_μ < 0}  U†_μ(x − ê_μ) · T_{Δx + ê_μ}(x − ê_μ)

    Under ``mode="single"`` the outer sum collapses to its first nonzero
    branch (lowest-index μ), so every DP step picks one predecessor instead
    of D-many.

    Sub-offsets ``Δx ∓ ê_μ`` always have strictly smaller L1 norm and the
    same component signs (just one zeroed out, possibly), so ordering the
    iteration by ``|Δx|_1`` guarantees every sub-step is in the table when
    needed.

    The full table covers every offset the G-Attn block iterates over —
    positive, purely-negative, and mixed-sign — uniformly.  The octant trick
    ``T_{−Δx}(x) = dagger(T_Δx(x − Δx))`` still holds as a property of the
    math and is exercised by the test suite, but is not relied on at build
    time: a single auditable surface for the gauge-implementation stress test
    (notes/architecture.md §7.2) is worth more than the 2× memory saving for now,
    later we'll maybe switch to a half table for memory efficiency.

    Parameters
    ----------
    U
        Batched link tensor of shape ``(N, D, *Λ, nc, nc)`` with a leading
        configuration batch axis.
    R
        Manhattan radius.
    gaugegroup
        Gauge group (used for the backward-link daggers).
    mode
        ``"average"`` (default) or ``"single"`` — see above.
    Returns
    -------
    Stacked transport tensor in canonical offset order — shape
    ``(N, n_offsets, *Λ, nc, nc)``. The offset axis is ordered by
    :func:`l1_ball_offsets` ``(D, R)``: sorted by ``|Δx|₁`` then
    lexicographically. Use that helper to look up the index for a given Δx.
    """
    if mode not in ("average", "single"):
        raise ValueError(f"mode must be 'average' or 'single', got {mode!r}")
    if U.ndim < 6:
        raise ValueError(
            "U must be batched with shape (N, D, *Λ, nc, nc). "
            "For a single config, pass U.unsqueeze(0)."
        )

    N = U.shape[0]
    D = U.shape[1]
    spatial_shape = U.shape[2:-2]
    nc = U.shape[-1]

    # Spatial axis μ counted from the *end*. Color axes are always the trailing
    # two, so this index works whether or not a batch axis is present.
    def sdim(mu: int) -> int:
        return mu - D - 2

    # Identity is the DP base for the zero offset; broadcast over (N, *Λ).
    identity = (
        torch.eye(nc, dtype=U.dtype, device=U.device)
        .expand(N, *spatial_shape, nc, nc)
        .contiguous()
    )

    # Pre-compute U†_μ(x − ê_μ) once per direction: roll by +1 along the
    # spatial-μ axis brings the link value at site x − ê_μ to index x, then dagger.
    U_back: List[torch.Tensor] = [
        gaugegroup.dagger(torch.roll(U[:, mu], shifts=1, dims=sdim(mu)))
        for mu in range(D)
    ]

    offsets = l1_ball_offsets(D, R)
    zero: Tuple[int, ...] = (0,) * D
    table: Dict[Tuple[int, ...], torch.Tensor] = {zero: identity}

    # offsets are pre-sorted by |Δx|_1 so every sub-step is in the table.
    for dx in offsets:
        t: Optional[torch.Tensor] = None
        for mu in range(D):
            if dx[mu] > 0:
                # prev_dx = dx but with dx[mu] = dx[mu] - 1
                prev_dx = tuple(v - 1 if i == mu else v for i, v in enumerate(dx))
                # U_μ(x) · T_{Δx−ê_μ}(x+ê_μ)
                contrib = U[:, mu] @ torch.roll(
                    table[prev_dx], shifts=-1, dims=sdim(mu)
                )
            elif dx[mu] < 0:
                # prev_dx = dx but with dx[mu] = dx[mu] + 1
                prev_dx = tuple(v + 1 if i == mu else v for i, v in enumerate(dx))
                # U†_μ(x−ê_μ) · T_{Δx+ê_μ}(x−ê_μ)
                contrib = U_back[mu] @ torch.roll(
                    table[prev_dx], shifts=1, dims=sdim(mu)
                )
            else:
                continue
            t = contrib if t is None else t + contrib
            if mode == "single":
                # Collapse the sum to its first nonzero branch: canonical
                # shortest path = lowest-index nonzero direction at every step.
                break

        table[dx] = t

    # Stack offsets at axis 1 so the result is (N, n_off, *Λ, nc, nc).
    stacked = torch.stack([table[dx] for dx in offsets], dim=1)
    del table  # release the per-offset references; the stacked tensor owns the data now.

    # Single-path mode: one path per offset, no average to normalise away.
    if mode == "single":
        return stacked

    # Normalise each offset slice by the number of shortest paths N_Δx.
    # N_Δx is the multinomial coefficient |Δx|₁! / Π_μ |Δx_μ|!: the |Δx|₁ steps
    # of the path are partitioned into groups of identical moves (|Δx_μ| moves
    # in each direction μ). With N_Δx = 1 for the base |Δx|₁ = 1 offsets,
    # T_{±ê_μ} is left unchanged by the division.
    n_paths_list = []
    for dx in offsets:
        n = math.factorial(sum(abs(d) for d in dx))
        for d in dx:
            n //= math.factorial(abs(d))
        n_paths_list.append(n)
    real_dtype = (
        torch.float64 if U.dtype in (torch.float64, torch.complex128) else torch.float32
    )
    norms = torch.tensor(n_paths_list, dtype=real_dtype, device=U.device)
    # Broadcast across (batch, n_off, *Λ, nc, nc): target shape (1, n_off, 1, ..., 1).
    bcast_shape = [1] * stacked.ndim
    bcast_shape[1] = -1
    stacked = stacked / norms.view(bcast_shape)

    return stacked
