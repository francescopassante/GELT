"""
=========================================================================
Vortex-geometry targets — the 3D Z₂ candidate.
=========================================================================

The design record is ``notes/where_attention_can_win.md`` §9; the rule it has
to clear is §6's four criteria plus the non-circularity one T1's withdrawal
added. Everything here is built from one gauge-invariant bit per plaquette,

    v(p, x) = [ Re Tr P_p(x) / nc < 0 ]  ,

the **vortex indicator**. In Z₂ that is exactly the plaquette sign, and in 3D
the plaquettes are dual to links: the Bianchi identity (the product of the six
plaquettes bounding a cube is 1, because every link of the cube appears twice)
forces an even number of negative faces per cube, so the negative plaquettes
form **closed loops on the dual lattice**. Their percolation is the confinement
mechanism of the 3D Z₂ theory.

======  ==========================  ==============================================
 name    reduction                   what it demands of the offset weights
======  ==========================  ==============================================
 V1      log(1 + largest cluster     a *global* connected-component size, reached
         touching x)                 by routing along the line. The primary.
 V2      # vortex plaquettes in      nothing — it *is* a convolution of the
         the L1-ball of radius R     indicator field. Calibration.
======  ==========================  ==============================================

**V2 is a ball count, not a local BFS.** §9.2 of the note first wrote it as a
truncated line length; writing the code is what made the distinction matter. A
calibration arm has to be *exactly* linear in the input field — that is the
whole of its job, the role T0 plays in ``gelt/probe_targets.py`` — and a
BFS-truncated length is not. The truncated length is the **classical local
arm** instead (:func:`local_component_size`), i.e. the ceiling the pre-flight
measures the architectures against, and it is deliberately not a target.

The asymmetry that makes this candidate worth running is measured in §9.3 and
lives in ``gelt.lattice.build_transport_average``, not here: in Z₂ a
path-averaged transport is a hard vortex mask, ``T_Δ² = (1 + P_enclosed)/2 ∈
{0,1}``, while a single-path or axis-aligned transport is ±1 and acts as the
identity. This module builds the supervision; that fact is what the arms vary.

**D = 3 only.** In four dimensions the negative plaquettes form closed
*surfaces*, not loops, and connected components of the dual graph are a
different object with a different physics story. Every entry point raises
rather than quietly generalising.
"""

import itertools

import numpy as np
import torch

from gelt.lattice import plaquette_tensor
from gelt.probe_targets import ball_reduce

# The planes, in ``plaquette_tensor``'s own order: (μ, ν) with μ < ν
# lexicographic. ``THIRD_DIR[p]`` is the direction the plane p does not span,
# which is the direction the dual link of plaquette p points along.
PLANE_PAIRS = tuple(itertools.combinations(range(3), 2))  # (0,1), (0,2), (1,2)
THIRD_DIR = tuple(3 - mu - nu for mu, nu in PLANE_PAIRS)  # 2, 1, 0
N_PLANES = len(PLANE_PAIRS)

# The radius the calibration target V2 is defined on. Kept equal to
# ``gelt.probe_targets.BALL_RADIUS`` on purpose: the two probes' calibration
# arms then sit at the same reach, and both are well inside the Manhattan 8
# that four layers of either architecture reach.
BALL_RADIUS = 4

TARGETS = ("V1", "V2")


def _check_3d(v):
    if v.ndim != 5 or v.shape[1] != N_PLANES:
        raise ValueError(
            f"expected a 3D vortex field of shape (B, {N_PLANES}, Lx, Ly, Lz), "
            f"got {tuple(v.shape)}. The dual-loop picture is D = 3 only — in 4D "
            f"the negative plaquettes form closed surfaces, not loops."
        )


def _cube_faces():
    """The six ``(plane, Δ)`` faces bounding the cube based at a site.

    Face ``(q, Δ)`` of the cube at ``y`` is the plaquette ``P_q(y + Δ)``. A cube
    is spanned by all three directions, so for each plane ``q`` it has two
    parallel faces, separated by the direction ``q`` does not span.
    """
    faces = []
    for q in range(N_PLANES):
        rho = THIRD_DIR[q]
        for shift in (0, 1):
            delta = [0, 0, 0]
            delta[rho] = shift
            faces.append((q, tuple(delta)))
    return tuple(faces)


CUBE_FACES = _cube_faces()


def _neighbour_spec():
    """``{p: ((q, Δ), …)}`` — the plaquettes sharing a cube with ``(p, x)``.

    Plaquette ``(p, x)`` bounds exactly two cubes: the one based at ``x`` and
    the one based at ``x − ρ̂_p``, with ``ρ_p = THIRD_DIR[p]``. Its neighbours in
    the dual graph are the other five faces of each, ten distinct offsets in
    total. Two of the twelve ``(back, shift)`` combinations reproduce the
    plaquette itself and are dropped.
    """
    spec = {}
    for p in range(N_PLANES):
        rho_p = THIRD_DIR[p]
        out = []
        for back in (0, 1):
            for q, face_delta in CUBE_FACES:
                delta = list(face_delta)
                delta[rho_p] -= back
                delta = tuple(delta)
                if q == p and delta == (0, 0, 0):
                    continue  # the plaquette itself
                out.append((q, delta))
        spec[p] = tuple(out)
    return spec


NEIGHBOUR_SPEC = _neighbour_spec()


def vortex_field(U, gaugegroup, plaquettes=None):
    """``v(p, x) = [Re Tr P_p(x)/nc < 0]`` for a batch of configurations.

    ``U`` : ``(B, D, *Λ, nc, nc)``. Returns a bool tensor ``(B, n_pairs, *Λ)``.
    ``plaquettes`` short-circuits the :func:`plaquette_tensor` call when the
    caller already holds ``(B, n_pairs, *Λ, nc, nc)`` — the network's own input
    channels, so the supervision is a function of exactly what it is fed.

    Gauge invariant for every group, because the trace of a plaquette is. The
    *dual-loop* reading of it is Z₂'s alone — see :func:`closure_defect`.
    """
    P = plaquette_tensor(U, gaugegroup) if plaquettes is None else plaquettes
    nc = P.shape[-1]
    tr = P.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc  # (B, n_pairs, *Λ)
    v = tr < 0
    _check_3d(v)
    return v


def closure_defect(v):
    """Per-cube parity of the vortex faces — **zero everywhere, in Z₂**.

    Returns ``(B, *Λ)`` of ``0``/``1``: the number of negative faces of the cube
    based at each site, mod 2. In Z₂ the plaquette *is* its sign, every link of
    a cube appears in exactly two of its six faces, and the identity is exact —
    so a nonzero entry means either the face enumeration in :data:`CUBE_FACES`
    is wrong or the field did not come from a gauge configuration. It is a
    cheap, total self-check on the same enumeration :data:`NEIGHBOUR_SPEC` is
    built from, which is why the pre-flight runs it as a hard gate before
    anything else.

    **Z₂ only as a gate.** For SU(N) the plaquettes still satisfy a Bianchi
    identity, but ``[Re Tr P < 0]`` is not a homomorphic image of it — the sign
    of a trace is not a centre element — so this parity is generically 1 on
    half the cubes and means nothing. The candidate this module serves is a Z₂
    one; the check is not a portability claim.
    """
    _check_3d(v)
    dims = (1, 2, 3)
    parity = torch.zeros(v.shape[0], *v.shape[2:], dtype=torch.int64,
                         device=v.device)
    for q, delta in CUBE_FACES:
        parity += torch.roll(
            v[:, q].to(torch.int64), shifts=tuple(-d for d in delta), dims=dims
        )
    return parity % 2


def cluster_labels(v, max_iter=256):
    """Connected components of the dual vortex graph, as a label field.

    Returns ``(B, n_pairs, *Λ)`` int64: each vortex plaquette carries the
    smallest flat index in its component, each non-vortex plaquette carries its
    own index (an isolated self-loop, so the pointer jumps below stay in range).
    Components never cross the configuration axis — every neighbour is a spatial
    roll — so one label space for the whole batch is safe.

    Shiloach–Vishkin, and **the hook has to be applied to the root**, which is
    the one detail that decides whether this terminates. Each round:

    1. every plaquette proposes the smallest label in its closed neighbourhood;
    2. each proposal is scattered onto the **root of the proposer's tree** with
       an ``amin`` reduction — so when one plaquette of a tree touches a smaller
       tree, the *whole* tree follows on the next step;
    3. the pointer forest is compressed to its roots (``lab ← lab[lab]`` to a
       fixed point).

    Relabelling only the boundary plaquette instead of its root is the obvious
    version and it is ``O(diameter)``: measured on a 48×24×24 production
    configuration it was still 223 components after 20 rounds and never
    converged, because the percolating vortex line is a few thousand dual links
    long. With the root hook it converges in **6**.

    Labels only ever decrease (``lab[k] ≤ k`` is preserved by all three steps)
    and are bounded below, so the loop terminates; ``max_iter`` is a guard
    against a bug, not a tolerance, and it raises rather than returning a
    partial labelling.
    """
    _check_3d(v)
    dims = (1, 2, 3)
    idx = torch.arange(v.numel(), device=v.device).reshape(v.shape)
    big = v.numel()
    lab = idx.clone()
    for _ in range(max_iter):
        planes = []
        for p in range(N_PLANES):
            m = torch.where(v[:, p], lab[:, p], big)
            for q, delta in NEIGHBOUR_SPEC[p]:
                shifts = tuple(-d for d in delta)
                nb_lab = torch.roll(lab[:, q], shifts=shifts, dims=dims)
                nb_v = torch.roll(v[:, q], shifts=shifts, dims=dims)
                m = torch.minimum(m, torch.where(nb_v, nb_lab, big))
            planes.append(torch.where(v[:, p], m, idx[:, p]))
        proposal = torch.stack(planes, dim=1).reshape(-1)
        lab_flat = lab.reshape(-1)
        flat = lab_flat.clone()
        flat.scatter_reduce_(0, lab_flat, proposal, reduce="amin", include_self=True)
        while True:  # compress to the roots of the current forest
            nxt = flat[flat]
            if torch.equal(nxt, flat):
                break
            flat = nxt
        m = flat.reshape(v.shape)
        if torch.equal(m, lab):
            return lab
        lab = m
    raise RuntimeError(
        f"cluster_labels did not converge in {max_iter} rounds — with a root "
        f"hook and full compression that is a bug, not a large lattice."
    )


def cluster_size_field(v, lab=None):
    """``(B, n_pairs, *Λ)`` int64 — the size of each plaquette's cluster, 0 off.

    Sizes are counted per *component*, i.e. in dual links (equivalently, in
    vortex plaquettes), which is the natural unit of vortex line length.
    """
    _check_3d(v)
    lab = cluster_labels(v) if lab is None else lab
    counts = torch.bincount(lab[v].reshape(-1), minlength=v.numel())
    return torch.where(v, counts[lab], torch.zeros_like(lab))


def largest_cluster_mask(v, lab=None):
    """``(B, n_pairs, *Λ)`` bool — membership of *this configuration's* largest
    cluster.

    Per configuration, because "the biggest one" is a statement about one
    configuration's vortex network and the ensemble average of the answer is not
    the answer. The binary reading of the pre-flight is an AUC against this.
    """
    _check_3d(v)
    lab = cluster_labels(v) if lab is None else lab
    out = torch.zeros_like(v)
    counts = torch.bincount(lab[v].reshape(-1), minlength=v.numel())
    for b in range(v.shape[0]):
        lab_b, v_b = lab[b], v[b]
        present = torch.unique(lab_b[v_b])
        if present.numel() == 0:
            continue
        out[b] = v_b & (lab_b == present[counts[present].argmax()])
    return out


def _to_site(field, v):
    """``(B, n_pairs, *Λ) → (B, *Λ)``: the max over the planes based at a site.

    The head of both architectures is one scalar per site, so a per-plaquette
    quantity has to be reduced. The max over the three plaquettes based at ``x``
    is the reduction V1 is defined with; ``notes/where_attention_can_win.md``
    §9.7 point 5 records the residual circularity risk it carries (a max over
    three objects) and the masked-loss alternative that removes it.
    """
    masked = torch.where(v, field, torch.zeros_like(field))
    return masked.max(dim=1).values


def target_cluster_size(v, lab=None):
    """V1 — ``log(1 + |C(x)|)`` per site, 0 where no plaquette at x is a vortex.

    ``|C(x)|`` is the size of the largest vortex cluster touching any of the
    three plaquettes based at ``x``. The log is not cosmetic: cluster sizes span
    three decades within one configuration, and an MSE on the raw size would be
    a fit to the percolating cluster alone.

    **Global by construction** — no local algorithm computes it, which is
    criterion 4 of ``notes/where_attention_can_win.md`` §6 in the strong sense.
    """
    _check_3d(v)
    sizes = cluster_size_field(v, lab).to(torch.float64)
    return torch.log1p(_to_site(sizes, v))


def target_ball_length(v, R=BALL_RADIUS):
    """V2 — the number of vortex plaquettes in the L1-ball of radius ``R``.

    Exactly ``Σ_Δ 1 · n(x+Δ)`` with ``n(x) = Σ_p v(p, x)``: a convolution with a
    unit kernel, so a linear filter over the same ball reproduces it to machine
    precision. That is the point — this is the calibration arm, and reading W-A
    on it is how a capacity or optimisation difference between the arms
    announces itself before any other reading is believed.
    """
    _check_3d(v)
    n = v.to(torch.float64).sum(dim=1)  # (B, *Λ), values 0…3
    S, _ = ball_reduce(n, R)
    return S[-1]


def build_targets(v, names=TARGETS, R=BALL_RADIUS, lab=None):
    """``{name: (B, *Λ)}`` for the requested targets."""
    for name in names:
        if name not in TARGETS:
            raise ValueError(f"unknown target {name!r}; expected one of {TARGETS}")
    lab = cluster_labels(v) if (lab is None and "V1" in names) else lab
    builders = {
        "V1": lambda: target_cluster_size(v, lab),
        "V2": lambda: target_ball_length(v, R),
    }
    return {name: builders[name]() for name in names}


# ── The classical local arm ──────────────────────────────────────────────────
# Not a target. This is the strongest method the *baseline family* has at the
# architecture's own reach, and ``notes/where_attention_can_win.md`` §5 makes
# measuring it the first gate of every A/B: a task whose local version is
# already solved has no headroom to measure an architecture in.

def _compact_neighbour_table(v_np):
    """``(ids, coords, nbr)`` for one configuration's vortex plaquettes.

    ``ids`` is the ``(n_pairs, *Λ)`` compact index field (−1 off the vortices),
    ``coords`` is ``(n_vort, 4)`` of ``(plane, x, y, z)``, and ``nbr`` is
    ``(n_vort, 10)`` of compact neighbour indices (−1 where the neighbouring
    plaquette is not a vortex). Built by rolling the index field, so the graph
    the BFS walks is the same :data:`NEIGHBOUR_SPEC` the label propagation uses.
    """
    ids = np.full(v_np.shape, -1, dtype=np.int64)
    coords = np.argwhere(v_np)
    ids[tuple(coords.T)] = np.arange(len(coords))
    n_nb = len(NEIGHBOUR_SPEC[0])
    nbr = np.full((len(coords), n_nb), -1, dtype=np.int64)
    for p in range(N_PLANES):
        sel = coords[:, 0] == p
        if not sel.any():
            continue
        site = tuple(coords[sel, 1 + d] for d in range(3))
        for j, (q, delta) in enumerate(NEIGHBOUR_SPEC[p]):
            rolled = np.roll(ids[q], shift=tuple(-d for d in delta), axis=(0, 1, 2))
            nbr[sel, j] = rolled[site]
    return ids, coords, nbr


def local_component_size(v, R, max_hops=None, verbose=False):
    """``(B, n_pairs, *Λ)`` int64 — the vortex line length reachable within ``R``.

    For each vortex plaquette, the number of vortex plaquettes connected to it
    by a path in the dual graph **whose plaquettes all lie inside the L1-ball of
    radius R around its base site** (periodic distance). Connectivity is
    computed from ball-internal information only: no global label is consulted.

    ``max_hops`` additionally caps the path *length*, and it is the parameter
    that makes this a fair ceiling rather than an unreachable one. With it
    unset the BFS runs to convergence inside the ball — it follows the line for
    as many steps as the line is long — while a ``k``-layer network gets exactly
    ``k`` rounds of message passing. A vortex line of length 50 inside the ball
    needs ~50 sequential steps, so the uncapped arm is not in the function class
    of a 4-layer network at all, and comparing against it measures depth rather
    than inductive bias. **Matched depth means ``max_hops = n_layers``.**
    ``notes/where_attention_can_win.md`` §6 criterion 3 says "at the
    architecture's own reach"; reach has two dimensions and the spatial one was
    the only one checked.

    Per-seed BFS rather than a propagation, because the quantity is *reachable
    set size from one node*, which no per-node relaxation computes. The frontier
    is capped by the ball, so the cost is the number of vortices in a ball —
    ~3% of 3·|ball| at production couplings — and not the cluster size.
    """
    _check_3d(v)
    L = np.array(v.shape[2:])
    out = torch.zeros(v.shape, dtype=torch.int64)
    for b in range(v.shape[0]):
        v_np = v[b].cpu().numpy()
        _, coords, nbr = _compact_neighbour_table(v_np)
        sites = coords[:, 1:]
        sizes = np.zeros(len(coords), dtype=np.int64)
        for k in range(len(coords)):
            origin = sites[k]
            seen = {k}
            frontier = [k]
            hops = 0
            while frontier and (max_hops is None or hops < max_hops):
                nxt = []
                for i in frontier:
                    for j in nbr[i]:
                        if j < 0 or j in seen:
                            continue
                        d = np.abs((sites[j] - origin + L // 2) % L - L // 2).sum()
                        if d <= R:
                            seen.add(int(j))
                            nxt.append(int(j))
                frontier = nxt
                hops += 1
            sizes[k] = len(seen)
        if len(coords):
            out[b][tuple(torch.from_numpy(coords).T)] = torch.from_numpy(sizes)
        if verbose:
            print(f"    local BFS: config {b}, {len(coords)} vortex plaquettes")
    return out


def local_size_site_feature(v, R, max_hops=None):
    """V1's own reduction applied to :func:`local_component_size`.

    ``log(1 + max over the three plaquettes at x)`` — the same shape as V1, so
    the classical arm and the target are compared site by site with no extra
    convention in between.
    """
    return torch.log1p(
        _to_site(local_component_size(v, R, max_hops=max_hops).to(torch.float64), v)
    )
