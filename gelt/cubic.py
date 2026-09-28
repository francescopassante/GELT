"""The spatial cubic group O_h acting on gauge links, and its ten irreps.

Supervisor's S4 (``notes/prof_notes.md``): gauge invariance and zero momentum do
not put an operator in 0⁺⁺ — on the cubic lattice it must also be a scalar under
the spatial cubic group with parity, O_h. The classical operators are scalars by
construction; the network's Ō is not, because nothing ties its per-plane input
channels or its per-axis RoPE frequencies together. This module supplies the
exact symmetry transformations of a configuration and the character table that
turns 48 evaluations Ō(gU) into the operator's irrep decomposition.

O_h is the 48 **signed permutations** of the spatial axes. The time axis is
untouched, so the anisotropy (β_t ≠ β_s) is respected and the transfer-matrix
reading of Ō(t) survives: every element maps a timeslice onto itself.

An element is ``CubicElement(perm, flips)`` acting as *reflect every axis in
``flips``, then permute*, i.e. on coordinates ``y = P·F·x`` with
``F = diag(±1)`` and ``P[a, perm[a]] = 1``. Both steps are index operations and
daggers only, so ``apply_cubic`` is exact in any precision.

Characters. With ``M = P·F`` the 3×3 signed permutation matrix, parity is
``det M`` and the proper rotation is ``R = det(M)·M`` (the inversion is −𝟙).
For the rotation group O, with σ the underlying permutation of the axes:

- A₁: 1;  A₂: sign(σ);  E: (fixed points of σ) − 1;  T₁: tr R;  T₂: sign(σ)·tr R.

(Checked on the five classes: E, 8C₃, 3C₂ = C₄², 6C₄, 6C₂'.) The ``g`` irreps of
O_h carry the O character, the ``u`` irreps the same times ``det M``. All ten
characters are real and class functions, so the projector
``P_Γ = (d_Γ/48) Σ_g χ_Γ(g)·g`` does not care whether the operator is carried
by ``g`` or ``g⁻¹``, and ``Σ_Γ P_Γ = 1`` exactly.

Layout: links are ``(*batch, D, *Λ, nc, nc)``; direction ``k`` runs along
lattice axis ``k`` (the repo's convention), which sits at tensor dimension
``batch_dims + 1 + k``.
"""

from itertools import permutations, product
from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch

from .lattice import GaugeGroup

IRREPS = ("A1g", "A2g", "Eg", "T1g", "T2g", "A1u", "A2u", "Eu", "T1u", "T2u")
IRREP_DIM = {"A1": 1, "A2": 1, "E": 2, "T1": 3, "T2": 3}


class CubicElement(NamedTuple):
    """Reflect the spatial axes in ``flips``, then permute them by ``perm``.

    ``perm`` is over the spatial axes (1..D−1): new axis ``spatial[a]`` is old
    axis ``perm[a]``. ``flips`` is a sorted tuple of spatial axes.
    """

    perm: Tuple[int, ...]
    flips: Tuple[int, ...]

    @property
    def label(self) -> str:
        return "p" + "".join(str(a) for a in self.perm) + (
            "_r" + "".join(str(a) for a in self.flips) if self.flips else "")


def cubic_elements(D: int = 4, time_axis: int = 0) -> List[CubicElement]:
    """The 48 elements of O_h on the spatial axes, identity first.

    Only ``time_axis = 0`` is supported: the spectroscopy code fixes time as
    lattice axis 0 throughout, and a second convention here would be a second
    meaning for the same links.
    """
    if time_axis != 0:
        raise ValueError("time is lattice axis 0 throughout the spectroscopy code")
    if D != 4:
        raise ValueError(f"O_h needs three spatial axes (D = 4), got D = {D}")
    spatial = tuple(range(1, D))
    out = []
    for perm in permutations(spatial):
        for bits in product((0, 1), repeat=len(spatial)):
            flips = tuple(a for a, b in zip(spatial, bits) if b)
            out.append(CubicElement(perm, flips))
    out.sort(key=lambda g: (g.perm != spatial or bool(g.flips), g.perm, g.flips))
    return out


def cubic_matrix(g: CubicElement) -> torch.Tensor:
    """The 3×3 integer matrix ``M = P·F`` with ``y = M x`` on spatial coordinates."""
    spatial = sorted(g.perm)
    n = len(spatial)
    F = torch.diag(torch.tensor(
        [-1 if a in g.flips else 1 for a in spatial], dtype=torch.int64))
    P = torch.zeros(n, n, dtype=torch.int64)
    for a, src in enumerate(g.perm):
        P[a, spatial.index(src)] = 1
    return P @ F


def _perm_sign(perm: Sequence[int]) -> int:
    inversions = sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm))
                     if perm[i] > perm[j])
    return -1 if inversions % 2 else 1


def irrep_characters(g: CubicElement) -> Dict[str, int]:
    """The ten O_h characters of ``g`` (see the module docstring)."""
    M = cubic_matrix(g)
    det = int(round(torch.linalg.det(M.double()).item()))
    trR = det * int(M.trace().item())
    sgn = _perm_sign(g.perm)
    fixed = sum(1 for a, src in enumerate(g.perm) if src == sorted(g.perm)[a])
    chi_O = {"A1": 1, "A2": sgn, "E": fixed - 1, "T1": trR, "T2": sgn * trR}
    out = {}
    for name, c in chi_O.items():
        out[name + "g"] = c
        out[name + "u"] = det * c
    return out


def irrep_projections(obar_g: torch.Tensor, elements: Sequence[CubicElement]
                      ) -> Dict[str, torch.Tensor]:
    """``P_Γ Ō = (d_Γ/48) Σ_g χ_Γ(g) Ō_g`` for every irrep Γ.

    ``obar_g`` : ``(n_elements, ...)`` with ``obar_g[i] = Ō(g_i U)``. Needs the
    whole group (all 48 elements): a projector over a subset is not one. The
    ten outputs sum to ``obar_g[identity]`` exactly (up to round-off).
    """
    if len(elements) != 48 or len(set(elements)) != 48:
        raise ValueError(f"irrep projection needs all 48 elements, got "
                         f"{len(set(elements))} distinct")
    if obar_g.shape[0] != len(elements):
        raise ValueError("obar_g's leading axis must enumerate `elements`")
    chis = [irrep_characters(g) for g in elements]
    out = {}
    for irrep in IRREPS:
        w = torch.tensor([c[irrep] for c in chis], dtype=obar_g.dtype,
                         device=obar_g.device)
        d = IRREP_DIM[irrep[:-1]]
        out[irrep] = (d / 48.0) * torch.tensordot(w, obar_g, dims=1)
    return out


def regular_projectors(elements: Sequence[CubicElement]) -> Dict[str, torch.Tensor]:
    """The isotypic projectors of the regular representation, one 48×48 per irrep.

    Acting on an orbit vector ``v_k = Ō(g_k U)``, ``(Π_Γ v)_h = (P_Γ Ō)(g_h U)``:
    ``Π_Γ[h, k] = (d_Γ/48) χ_Γ(g_k g_h⁻¹)``, with the product read off the
    matrices (``g₂(g₁U) = (M₂M₁)U``, pinned in ``tests/test_cubic.py``). The Π_Γ
    are symmetric, idempotent, mutually orthogonal and sum to 𝟙, so they split
    ``‖v‖²`` into ten parts that add up exactly.
    """
    if len(elements) != 48 or len(set(elements)) != 48:
        raise ValueError("the regular representation needs all 48 elements")
    mats = [cubic_matrix(g) for g in elements]
    index = {tuple(M.flatten().tolist()): i for i, M in enumerate(mats)}
    chis = [irrep_characters(g) for g in elements]
    # rel[h, k] = index of g_k g_h⁻¹ (signed permutations: M⁻¹ = Mᵀ)
    rel = torch.tensor([[index[tuple((mats[k] @ mats[h].T).flatten().tolist())]
                         for k in range(48)] for h in range(48)])
    out = {}
    for irrep in IRREPS:
        chi = torch.tensor([c[irrep] for c in chis], dtype=torch.float64)
        out[irrep] = (IRREP_DIM[irrep[:-1]] / 48.0) * chi[rel]
    return out


def irrep_fractions(obar_g: torch.Tensor, elements: Sequence[CubicElement]
                    ) -> Dict[str, float]:
    """Each irrep's share of the connected C(0), on the O_h-augmented sample.

    ``obar_g`` : ``(48, ...)``, ``obar_g[k] = Ō(g_k U)`` over any sample axes.
    The per-irrep C(0)s of the naive projections ``P_Γ Ō`` do not add up on a
    finite sample (the cross terms vanish only in expectation). Augmenting the
    sample by its whole orbit — which the ensemble measure allows — makes them
    orthogonal exactly: with ``δv`` the orbit vector minus the augmented mean
    (a constant vector, i.e. pure A1g), ``f_Γ = ‖Π_Γ δv‖² / ‖δv‖²`` and
    ``Σ_Γ f_Γ = 1``. ``f_A1g`` is ``C(0)`` of the projected operator over the
    orbit-averaged ``C(0)`` of the unprojected one.
    """
    v = obar_g.reshape(obar_g.shape[0], -1).double()
    v = v - v.mean()
    total = v.pow(2).sum()
    return {r: ((P.to(v.device) @ v).pow(2).sum() / total).item()
            for r, P in regular_projectors(elements).items()}


# ── the action on links ──────────────────────────────────────────────────────
def _reflect_field(T: torch.Tensor, dim: int) -> torch.Tensor:
    """``x → (−x) mod L`` along tensor dimension ``dim`` (about the origin).

    ``flip`` gives ``T[L−1−i]``; the extra ``roll(+1)`` turns that into
    ``T[(−i) mod L]``, a reflection about a site rather than about the
    half-integer point between sites.
    """
    return torch.roll(torch.flip(T, dims=[dim]), shifts=1, dims=dim)


def reflect_links(U: torch.Tensor, gaugegroup: GaugeGroup, axis: int,
                  batch_dims: int = 0) -> torch.Tensor:
    """Exact lattice reflection ``x_axis → −x_axis`` of links ``(*batch, D, *Λ, nc, nc)``.

    Links along ν ≠ axis are carried along: ``U'_ν(y) = U_ν(Py)``. The link along
    the reflected axis is reversed as well as moved, so it becomes the dagger of
    the link on the other side: ``U'_a(y) = U_a(P(y + â))†``, and
    ``P(y + â) = Py − â`` — hence shift *down* by one first and reflect after.
    The other order is a plausible map that does not preserve the Wilson action
    (``tests/test_lattice.py`` checks the action first for the same reason).
    """
    dim = batch_dims + axis  # lattice axis `axis` once the direction dim is selected
    D = U.shape[batch_dims]
    out = []
    for nu in range(D):
        field = U.select(batch_dims, nu)
        if nu != axis:
            out.append(_reflect_field(field, dim))
        else:
            back = torch.roll(field, shifts=+1, dims=dim)  # U_a(y − â)
            out.append(gaugegroup.dagger(_reflect_field(back, dim)))
    return torch.stack(out, dim=batch_dims)


def permute_links(U: torch.Tensor, perm: Sequence[int],
                  batch_dims: int = 0) -> torch.Tensor:
    """Relabel the axes: new axis ``a`` is old axis ``perm[a]`` (``perm`` over all D).

    Both the direction index and the lattice axis are permuted together, so the
    link that pointed along ``perm[a]`` at ``x`` points along ``a`` at ``y``
    with ``y_a = x_{perm[a]}`` — a relabelling, trivially a symmetry.
    """
    D = U.shape[batch_dims]
    if sorted(perm) != list(range(D)):
        raise ValueError(f"perm must be a permutation of range({D}), got {perm}")
    idx = torch.tensor(list(perm), device=U.device)
    U = U.index_select(batch_dims, idx)
    first = batch_dims + 1
    dims = list(range(U.dim()))
    dims[first:first + D] = [first + p for p in perm]
    return U.permute(dims)


def apply_cubic(U: torch.Tensor, gaugegroup: GaugeGroup, g: CubicElement,
                batch_dims: int = 0) -> torch.Tensor:
    """``gU``: reflect the axes in ``g.flips``, then permute by ``g.perm``."""
    for a in g.flips:
        U = reflect_links(U, gaugegroup, a, batch_dims=batch_dims)
    perm = (0,) + tuple(g.perm)
    if perm != tuple(range(len(perm))):
        U = permute_links(U, perm, batch_dims=batch_dims)
    return U.contiguous()
