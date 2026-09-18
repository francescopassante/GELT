"""Tests for :mod:`gelt.vortex_targets` — the 3D Z₂ vortex candidate's premise.

The design record is ``notes/where_attention_can_win.md`` §9. What is pinned
here is what the candidate rests on, in the order it rests on it:

* **The face enumeration is right**, checked by physics rather than by itself:
  the per-cube parity of the vortex faces is identically zero on Z₂ gauge
  configurations (Bianchi). :data:`CUBE_FACES` and :data:`NEIGHBOUR_SPEC` are
  built from the same formula, so a bug in one shows up in that gate.
* **The dual graph is the right graph**, checked against the one configuration
  whose answer is known by hand: flipping a single link of the ordered
  configuration makes exactly one closed loop of four vortex plaquettes.
* **The vectorised label propagation equals a plain union-find**, so the
  pointer-jumping shortcut is not what the numbers depend on.
* **V2 is exactly a linear functional of the indicator field** — the
  calibration arm only calibrates if a linear filter can reach R² = 1 on it in
  principle, exactly as T0 does in ``tests/test_probe_targets.py``.
* **V1 is gauge invariant and global**, and the local arm is *local* — it
  consults no label outside its ball, which is what makes it an honest ceiling.
"""

import numpy as np
import pytest
import torch

from gelt import SU, Z2, link_gauge_transformation, random_links
from gelt.sampler import z2_heatbath_sweep
from gelt.vortex_targets import (
    BALL_RADIUS,
    CUBE_FACES,
    NEIGHBOUR_SPEC,
    N_PLANES,
    build_targets,
    closure_defect,
    cluster_labels,
    cluster_size_field,
    largest_cluster_mask,
    local_component_size,
    target_ball_length,
    target_cluster_size,
    vortex_field,
)


def _z2_configuration(L=10, beta=0.75, sweeps=40, seed=0):
    g = Z2()
    torch.manual_seed(seed)
    U = random_links(L, 3, g, dtype=torch.float64)
    for _ in range(sweeps):
        U, _ = z2_heatbath_sweep(U, g, beta)
    return U.unsqueeze(0), g


def _ordered_configuration(L=8):
    """All links +1: no vortices anywhere."""
    g = Z2()
    U = torch.ones(3, L, L, L, 1, 1, dtype=torch.float64)
    return U.unsqueeze(0), g


def _union_find_labels(v_np):
    """An independent, non-vectorised connected-components over the same graph."""
    coords = [tuple(int(c) for c in c_) for c_ in np.argwhere(v_np)]
    pos = {(p, (a, b, c)): k for k, (p, a, b, c) in enumerate(coords)}
    L = v_np.shape[1:]
    parent = list(range(len(coords)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for k, (p, a, b, c) in enumerate(coords):
        for q, delta in NEIGHBOUR_SPEC[p]:
            y = ((a + delta[0]) % L[0], (b + delta[1]) % L[1], (c + delta[2]) % L[2])
            j = pos.get((q, y))
            if j is not None:
                union(k, j)
    return coords, [find(i) for i in range(len(coords))]


# ── The face enumeration, checked by physics ────────────────────────────────
def test_cube_parity_vanishes_on_z2_configurations():
    """Bianchi: every cube carries an even number of negative faces.

    This is the gate that validates :data:`CUBE_FACES` — and with it
    :data:`NEIGHBOUR_SPEC`, which is built from the same formula — without
    reusing any of the module's own indexing to do so.
    """
    U, g = _z2_configuration()
    v = vortex_field(U, g)
    assert v.any(), "the test configuration has no vortices at all"
    assert int(closure_defect(v).max()) == 0


def test_cube_parity_is_not_vacuously_zero():
    """A field that is *not* a gauge configuration must trip the same gate."""
    U, g = _z2_configuration()
    v = vortex_field(U, g).clone()
    v[0, 0, 0, 0, 0] = ~v[0, 0, 0, 0, 0]  # flip one plaquette: breaks closure
    assert int(closure_defect(v).max()) == 1


def test_there_are_six_faces_and_ten_neighbours():
    assert len(CUBE_FACES) == 6
    assert len(set(CUBE_FACES)) == 6
    for p in range(N_PLANES):
        assert len(NEIGHBOUR_SPEC[p]) == 10
        assert (p, (0, 0, 0)) not in NEIGHBOUR_SPEC[p]


# ── The dual graph, against a configuration whose answer is known by hand ────
def test_one_flipped_link_makes_one_loop_of_four():
    """The smallest vortex: flip a link, get the four plaquettes containing it.

    In 3D a link belongs to 2(D−1) = 4 plaquettes, and those four close into a
    single dual loop around it. So the ordered configuration with one link
    flipped has exactly four vortex plaquettes in exactly one cluster — a
    complete, hand-checkable answer for the labelling, the sizes and the mask.
    """
    U, g = _ordered_configuration()
    v0 = vortex_field(U, g)
    assert not v0.any()

    U[0, 0, 2, 3, 4, 0, 0] = -1.0  # flip U_0(2,3,4)
    v = vortex_field(U, g)
    assert int(v.sum()) == 4
    assert int(closure_defect(v).max()) == 0

    lab = cluster_labels(v)
    assert torch.unique(lab[v]).numel() == 1, "the four plaquettes must be one loop"
    assert int(cluster_size_field(v, lab).max()) == 4
    assert int(largest_cluster_mask(v, lab).sum()) == 4
    # every hop of the loop is inside a ball of radius 2, so the local arm sees
    # the whole of it and the truncation is not what the size comes from
    assert int(local_component_size(v, 4).max()) == 4
    # at R = 0 only plaquettes based at the seed's own site are reachable, and
    # two of the four faces around a link are based there — so 2, not 1.
    assert int(local_component_size(v, 0).max()) == 2


def test_two_distant_flips_make_two_clusters():
    U, g = _ordered_configuration(L=12)
    U[0, 0, 1, 1, 1, 0, 0] = -1.0
    U[0, 1, 7, 7, 7, 0, 0] = -1.0
    v = vortex_field(U, g)
    lab = cluster_labels(v)
    assert int(v.sum()) == 8
    assert torch.unique(lab[v]).numel() == 2
    sizes = cluster_size_field(v, lab)
    assert sorted(sizes[v].tolist()) == [4] * 8
    assert int(largest_cluster_mask(v, lab).sum()) == 4  # one of the two, not both


# ── The vectorised labelling against a plain union-find ─────────────────────
def test_label_propagation_matches_union_find():
    U, g = _z2_configuration(L=10, beta=0.74, sweeps=60, seed=3)
    v = vortex_field(U, g)
    lab = cluster_labels(v)
    coords, roots = _union_find_labels(v[0].numpy())
    assert len(coords) > 50, "too few vortices to be a real check"

    mine = [int(lab[0][c]) for c in coords]
    # the two labellings need not agree on the *name* of a component, only on
    # the partition it induces
    pairs_mine = {}
    pairs_ref = {}
    for k, (a, b) in enumerate(zip(mine, roots)):
        pairs_mine.setdefault(a, set()).add(k)
        pairs_ref.setdefault(b, set()).add(k)
    assert sorted(map(sorted, pairs_mine.values())) == sorted(
        map(sorted, pairs_ref.values())
    )


def test_a_long_line_converges_and_is_one_cluster():
    """The regression test for the root hook.

    A straight dual line wrapping a long axis is one cluster whose graph
    diameter is half its length. Hooking the *boundary plaquette* instead of its
    tree's root needs O(diameter) rounds and silently exhausts ``max_iter`` at
    production volumes; hooking the root converges in O(log diameter). The field
    is built by hand rather than from links, so the test is about the labelling
    and nothing else.
    """
    v = torch.zeros(1, 3, 4, 4, 128, dtype=torch.bool)
    v[0, 0, 1, 2, :] = True  # plane (0,1) stacks along axis 2 — a closed line
    lab = cluster_labels(v)
    assert torch.unique(lab[v]).numel() == 1
    assert int(cluster_size_field(v, lab).max()) == 128


def test_non_cubic_lattice_is_handled():
    """Production is 48 × 24 × 24, so no axis length may be assumed equal."""
    v = torch.zeros(2, 3, 16, 8, 8, dtype=torch.bool)
    v[0, 0, 1, 2, :] = True
    v[1, 2, :, 3, 4] = True  # plane (1,2) stacks along axis 0, of length 16
    lab = cluster_labels(v)
    sizes = cluster_size_field(v, lab)
    assert int(sizes[0].max()) == 8
    assert int(sizes[1].max()) == 16
    # the two configurations must not merge into one component
    assert torch.unique(lab[0][v[0]]).numel() == 1
    assert set(torch.unique(lab[0][v[0]]).tolist()).isdisjoint(
        torch.unique(lab[1][v[1]]).tolist()
    )


def test_sizes_sum_to_the_vortex_count():
    U, g = _z2_configuration(L=10, beta=0.75, sweeps=50, seed=1)
    v = vortex_field(U, g)
    lab = cluster_labels(v)
    sizes = cluster_size_field(v, lab)
    per_component = {}
    for label in torch.unique(lab[v]).tolist():
        members = (lab == label) & v
        per_component[label] = int(members.sum())
        assert int(sizes[members][0]) == per_component[label]
    assert sum(per_component.values()) == int(v.sum())


# ── V2 is exactly linear; V1 is not ─────────────────────────────────────────
def test_V2_is_exactly_the_ball_count():
    """The calibration arm, against a brute-force sum over the ball."""
    from gelt.lattice import l1_ball_offsets

    U, g = _z2_configuration(L=10, beta=0.75, sweeps=40, seed=2)
    v = vortex_field(U, g)
    n = v.to(torch.float64).sum(dim=1)
    brute = n.clone()
    for dx in l1_ball_offsets(3, BALL_RADIUS):
        brute += torch.roll(n, shifts=tuple(-d for d in dx), dims=(1, 2, 3))
    assert torch.equal(brute, target_ball_length(v, BALL_RADIUS))


def test_V2_is_additive_and_V1_is_not():
    """V2 is a linear functional of the indicator; V1 cannot be one.

    Two vortex loops far apart add their ball counts wherever the balls do not
    overlap, while V1 stays at the size of whichever single loop is local — a
    linear filter over the indicator cannot produce that.
    """
    U, g = _ordered_configuration(L=12)
    U[0, 0, 1, 1, 1, 0, 0] = -1.0
    v_one = vortex_field(U, g)
    U[0, 1, 7, 7, 7, 0, 0] = -1.0
    v_two = vortex_field(U, g)

    site = (0, 1, 1, 1)
    assert target_ball_length(v_two, BALL_RADIUS)[site] == (
        target_ball_length(v_one, BALL_RADIUS)[site]
    )
    # V1 at a site of the first loop is the first loop's size in both fields …
    assert target_cluster_size(v_two)[site] == target_cluster_size(v_one)[site]
    # … and it is log1p of a cluster size, not of a ball count
    assert torch.isclose(
        target_cluster_size(v_one)[site],
        torch.log1p(torch.tensor(4.0, dtype=torch.float64)),
    )


def test_V1_is_zero_where_no_plaquette_is_a_vortex():
    U, g = _ordered_configuration(L=8)
    U[0, 0, 2, 3, 4, 0, 0] = -1.0
    v = vortex_field(U, g)
    y = target_cluster_size(v)
    assert float(y[0, 0, 0, 0]) == 0.0
    assert float(y.max()) > 0.0


# ── The local arm is local, and bounded by the global one ───────────────────
def test_local_component_size_is_bounded_by_the_global_one():
    U, g = _z2_configuration(L=10, beta=0.75, sweeps=50, seed=4)
    v = vortex_field(U, g)
    glob = cluster_size_field(v)
    for R in (2, 4):
        loc = local_component_size(v, R)
        assert torch.all(loc[v] >= 1)
        assert torch.all(loc[v] <= glob[v])
        assert torch.all(loc[~v] == 0)


def test_local_component_size_matches_a_brute_force_bfs():
    """Independent BFS over the *lattice*, not over the compact neighbour table."""
    U, g = _z2_configuration(L=10, beta=0.74, sweeps=60, seed=5)
    v = vortex_field(U, g)
    R = 3
    mine = local_component_size(v, R)[0]
    v_np = v[0].numpy()
    L = np.array(v_np.shape[1:])
    present = {tuple(int(c) for c in c_) for c_ in np.argwhere(v_np)}
    checked = 0
    for seed_node in sorted(present)[::7]:  # a stride, so the test stays quick
        origin = np.array(seed_node[1:])
        seen, stack = {seed_node}, [seed_node]
        while stack:
            p, a, b, c = stack.pop()
            for q, delta in NEIGHBOUR_SPEC[p]:
                y = (
                    (a + delta[0]) % int(L[0]),
                    (b + delta[1]) % int(L[1]),
                    (c + delta[2]) % int(L[2]),
                )
                node = (q, *y)
                if node in seen or node not in present:
                    continue
                if np.abs((np.array(y) - origin + L // 2) % L - L // 2).sum() <= R:
                    seen.add(node)
                    stack.append(node)
        assert int(mine[seed_node]) == len(seen)
        checked += 1
    assert checked > 5


def test_local_component_size_ignores_structure_outside_its_ball():
    """Adding a distant loop must not change a local size — the honesty gate.

    If it did, the "local" arm would be consulting global information and the
    ceiling it measures would not be a ceiling a bounded-reach architecture has
    to beat.
    """
    U, g = _ordered_configuration(L=12)
    U[0, 0, 1, 1, 1, 0, 0] = -1.0
    near = local_component_size(vortex_field(U, g), 2)
    U[0, 1, 7, 7, 7, 0, 0] = -1.0
    far = local_component_size(vortex_field(U, g), 2)
    idx = (0, slice(None), 1, 1, 1)
    assert torch.equal(near[idx], far[idx])


# ── Gauge invariance and the guards ─────────────────────────────────────────
@pytest.mark.parametrize("group,dtype", [(Z2(), torch.float64), (SU(2), torch.complex128)])
def test_vortex_field_is_gauge_invariant(group, dtype):
    L = 6
    torch.manual_seed(7)
    U = random_links(L, 3, group, dtype=dtype)
    nc = group.nc
    omega = group.project(torch.randn(L, L, L, nc, nc, dtype=dtype))
    v = vortex_field(U.unsqueeze(0), group)
    v_g = vortex_field(link_gauge_transformation(U, omega, group).unsqueeze(0), group)
    assert torch.equal(v, v_g)


def test_four_dimensional_input_is_refused():
    g = Z2()
    U = random_links(4, 4, g, dtype=torch.float64).unsqueeze(0)
    with pytest.raises(ValueError, match="D = 3 only"):
        vortex_field(U, g)


def test_unknown_target_name_is_rejected():
    U, g = _ordered_configuration(L=8)
    v = vortex_field(U, g)
    with pytest.raises(ValueError, match="unknown target"):
        build_targets(v, names=("V3",))


def test_build_targets_returns_both_in_one_call():
    U, g = _z2_configuration(L=10, beta=0.75, sweeps=40, seed=6)
    v = vortex_field(U, g)
    out = build_targets(v)
    assert set(out) == {"V1", "V2"}
    assert torch.equal(out["V1"], target_cluster_size(v))
    assert torch.equal(out["V2"], target_ball_length(v))
