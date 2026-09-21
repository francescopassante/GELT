"""Tests for :mod:`gelt.lcnn_reference` — the authors' L-CNN as a baseline arm.

``tests/test_lcnn.py`` pins *our* L-CNN against the definitions it implements.
This file pins the wrapper around the vendored one, and the things it pins are
the ones that would otherwise make the new baseline quietly wrong rather than
loudly broken:

* **the layout conversion**, bit-exact both ways — a flattened lattice with a
  split complex axis and the links packed into the field tensor is four chances
  to transpose something and still get plausible numbers out;
* **gauge invariance of the per-site readout**, SU(2) complex128 and Z₂
  float64, on stacked multi-level inputs, exactly as for our implementation —
  if this fails the arm is not an L-CNN at all;
* **the kernel-size convention**. Theirs counts ``kernel_size`` as a range
  ``[-(k-1), k-1]``, ours counts ``K`` hops per axis. ``kernel_size = K + 1`` is
  the translation, and getting it wrong halves the baseline's receptive field
  per layer while every other check still passes. The support of a layer is
  measured here as integer set arithmetic, not asserted in prose.
* **the parameter formula**, because the matched width against their
  parametrisation is *not* the matched width against ours (their kernel is
  quadratic in the input channel count) and the width has to be chosen before
  any GPU time is spent.
"""

import pytest
import torch

from gelt import SU, Z2, link_gauge_transformation, plaquette_tensor, random_links
from gelt.lcnn_reference import (
    LCNNRef,
    from_ref_layout,
    reference_dof_count,
    to_ref_layout,
)


def _real_dofs(model):
    """Parameter count in real degrees of freedom (a complex weight is two)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def _stacked_plaquettes(U, gg, n_levels):
    """``n_levels`` copies of the plaquette input on the channel axis.

    Same device as ``tests/test_lcnn.py``: smearing is gauge covariant, so plain
    copies exercise the channel bookkeeping the shootout's stacked input needs
    without dragging ``ape_smear`` (and caveat 1's Z₂ defect) into the assertion.
    """
    return torch.cat([plaquette_tensor(U, gg)] * n_levels, dim=1)


def _su2_omega(L, D, nc, seed):
    torch.manual_seed(seed)
    raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        L**D, nc, nc, dtype=torch.float64
    )
    Q, _ = torch.linalg.qr(raw)
    return Q.reshape(*([L] * D), nc, nc)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_layout_round_trip_is_exact(dtype):
    """Ours → theirs → ours is the identity, bit for bit."""
    torch.manual_seed(3)
    x = torch.randn(2, 5, 4, 3, 6, 2, 2, dtype=dtype)
    r = to_ref_layout(x)
    assert r.shape == (2, 4 * 3 * 6, 5, 2, 2, 2)
    assert torch.equal(from_ref_layout(r, (4, 3, 6)), x)


def test_layout_promotes_a_real_field_and_drops_the_zero_imaginary_part():
    """Z₂ carries a real dtype here; their layers are complex throughout."""
    torch.manual_seed(4)
    x = torch.randn(2, 3, 4, 4, 4, 1, 1, dtype=torch.float64)
    r = to_ref_layout(x)
    assert r.dtype == torch.float64
    assert torch.equal(r[..., 1], torch.zeros_like(r[..., 1]))
    assert torch.equal(from_ref_layout(r, (4, 4, 4), complex_out=False), x)


def test_layout_puts_the_lattice_in_the_order_the_reference_shift_assumes():
    """Their ``shift`` views ``(B, X, …)`` as ``(B, *dims, …)`` and rolls.

    That is only the right neighbour if the flattening was row-major over the
    lattice axes in the same order — which is what this checks directly, by
    rolling on both sides and comparing.
    """
    torch.manual_seed(5)
    lattice = (4, 3, 6)
    x = torch.randn(2, 2, *lattice, 1, 1, dtype=torch.complex128)
    r = to_ref_layout(x)

    from gelt.lcnn_reference import reference_layers

    for axis in range(len(lattice)):
        rolled_ref = reference_layers().shift(r, axis, +1, list(lattice))
        rolled_ours = to_ref_layout(torch.roll(x, +1, dims=2 + axis))
        assert torch.equal(rolled_ref, rolled_ours)


# ---------------------------------------------------------------------------
# Gauge invariance of the per-site readout
# ---------------------------------------------------------------------------


def test_reference_lcnn_per_site_output_is_gauge_invariant_su2():
    """SU(2), complex128, D = 3: the per-timeslice glueball configuration."""
    torch.manual_seed(11)
    L, D, K, nc, n_levels = 4, 3, 2, 2, 4
    gg, dtype = SU(nc), torch.complex128

    U = torch.stack(
        [random_links(L=L, D=D, gaugegroup=gg, dtype=dtype) for _ in range(2)]
    )
    omega = _su2_omega(L, D, nc, seed=11)
    U_g = torch.stack([link_gauge_transformation(u, omega, gg) for u in U])

    W, W_g = _stacked_plaquettes(U, gg, n_levels), _stacked_plaquettes(U_g, gg, n_levels)

    model = LCNNRef(
        gg, L, D, K=K, c_hidden=3, n_layers=2, dtype=dtype, reduction="none",
        in_channels=3 * n_levels,
    )
    # The head is zero-initialised by the training scripts; a zero output is
    # invariant for trivial reasons, so randomise it before asserting anything.
    with torch.no_grad():
        model.head_fc2.weight.normal_()
        model.head_fc2.bias.normal_()

    out, out_g = model(W, U), model(W_g, U_g)
    assert torch.allclose(out, out_g, atol=1e-10), (out - out_g).abs().max()
    assert out.abs().max() > 1e-6, "output is ~0: the invariance check is vacuous"


def test_reference_lcnn_per_site_output_is_gauge_invariant_z2():
    """Z₂, float64, nc = 1 — the group where a missed dagger cannot be seen."""
    torch.manual_seed(12)
    L, D, K, n_levels = 4, 3, 2, 2
    gg, dtype = Z2(), torch.float64

    U = torch.stack(
        [random_links(L=L, D=D, gaugegroup=gg, dtype=dtype) for _ in range(2)]
    )
    omega = torch.where(
        torch.rand(*([L] * D), 1, 1) < 0.5,
        torch.tensor(-1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )
    U_g = torch.stack([link_gauge_transformation(u, omega, gg) for u in U])

    W, W_g = _stacked_plaquettes(U, gg, n_levels), _stacked_plaquettes(U_g, gg, n_levels)

    model = LCNNRef(
        gg, L, D, K=K, c_hidden=3, n_layers=2, dtype=dtype, reduction="none",
        in_channels=3 * n_levels,
    )
    with torch.no_grad():
        model.head_fc2.weight.normal_()
        model.head_fc2.bias.normal_()

    out, out_g = model(W, U), model(W_g, U_g)
    assert torch.allclose(out, out_g, atol=1e-12), (out - out_g).abs().max()
    assert out.abs().max() > 1e-6


def test_non_cubic_lattice_is_accepted():
    """The Z₂ probe box is 48 × 24 × 24; an int is only the cubic shorthand."""
    torch.manual_seed(13)
    lattice, D = (6, 4, 4), 3
    gg, dtype = Z2(), torch.float64
    U = random_links(L=lattice[1], D=D, gaugegroup=gg, dtype=dtype, Lt=lattice[0])
    U = U.unsqueeze(0)
    W = plaquette_tensor(U, gg)
    model = LCNNRef(
        gg, lattice, D, K=1, c_hidden=2, n_layers=1, dtype=dtype, reduction="none",
    )
    assert model(W, U).shape == (1, *lattice)


# ---------------------------------------------------------------------------
# The kernel-size convention, and the support it buys
# ---------------------------------------------------------------------------


def test_our_K_maps_to_their_kernel_size_plus_one():
    """``kernel_size = K + 1``, and the transported-term count then agrees.

    Their ``t_w_size`` counts ``1 + Σ_axes(|a| + |b|)`` copies per channel over
    ``kernel_range = [-(k-1), k-1]``; ours counts ``1 + D·K·2``. The two are the
    same number only under this translation — which is the whole point, because
    an unmatched receptive field would make the baseline weaker by construction.
    """
    from gelt.lcnn import LConv

    D, K, c_in = 3, 2, 3
    gg, dtype = SU(2), torch.complex128
    ours = LConv(gg, c_in, 4, D, K, dtype=dtype)
    theirs = LCNNRef(gg, 4, D, K=K, c_hidden=4, n_layers=1, dtype=dtype).convs[0]

    assert theirs.kernel_range == [[-(K), K]] * D
    n_terms_theirs = 1 + sum(abs(a) + abs(b) for a, b in theirs.kernel_range)
    assert n_terms_theirs == ours.n_shifts == 1 + D * K * 2 == 13
    # And the weight's transported axis is that count over the augmented input.
    assert theirs.weight.shape == (4, 2 * c_in + 1, 2 * c_in * n_terms_theirs + 1)


@pytest.mark.parametrize("K", [1, 2])
def test_support_of_one_reference_layer_is_exactly_the_kernel_range(K):
    """A delta in the input reaches ``x ± k·μ̂`` for every ``k ≤ K``, and nothing else.

    Read through the layer's *linear* term. Their init zeroes the whole unit row
    and unit column (``weight[:, :, -1]`` and ``weight[:, -1, :]``), i.e. the
    bias and residual terms, so at initialisation the only surviving terms are
    products ``w(x)·t_w(x)`` which vanish wherever the local field does — the
    support would read as a single site and the test would pass vacuously. The
    unit-local row is switched on deliberately here: with it, the output is a
    plain sum over the transported copies and its support *is* the kernel range.
    """
    torch.manual_seed(17)
    L, D = 7, 2
    gg, dtype = Z2(), torch.float64
    U = torch.ones(1, D, L, L, 1, 1, dtype=dtype)      # trivial gauge field
    W = torch.zeros(1, 1, L, L, 1, 1, dtype=dtype)
    centre = (3, 3)
    W[0, 0, centre[0], centre[1], 0, 0] = 1.0

    model = LCNNRef(gg, L, D, K=K, c_hidden=1, n_layers=1, dtype=dtype,
                    reduction="none", in_channels=1)
    conv = model.convs[0]
    with torch.no_grad():
        conv.weight.zero_()
        # The unit-local row against every transported copy — but *not* its
        # last column, which is the unit-against-unit term: that one is a
        # constant added at every site and would paint the whole lattice.
        conv.weight[:, -1, :-1] = 1.0

    from gelt.lcnn_reference import reference_layers, to_ref_layout

    ref = reference_layers()
    x = ref.repack_x(to_ref_layout(U), to_ref_layout(W))
    out = ref.unpack_x(conv(x), D)[1]
    field = from_ref_layout(out, (L, L)).abs().sum(dim=(1, -2, -1))[0]

    expected = {centre}
    for axis in range(D):
        for k in range(1, K + 1):
            for sign in (+1, -1):
                site = list(centre)
                site[axis] = (site[axis] + sign * k) % L
                expected.add(tuple(site))
    got = {tuple(i.tolist()) for i in (field > 1e-12).nonzero()}
    assert got == expected, f"support {sorted(got)} != {sorted(expected)}"


# ---------------------------------------------------------------------------
# Parameter budget
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "D,K,c_in,c_hidden,n_layers", [(3, 2, 3, 4, 4), (3, 2, 12, 2, 4), (3, 1, 3, 6, 2)]
)
def test_reference_dof_formula_matches_the_built_model(D, K, c_in, c_hidden, n_layers):
    """The width for a matched-parameter arm is chosen from this formula.

    It has to agree with the model to the parameter, because it is what
    ``scripts/z2_dof_table.py`` and the note's matched-width argument are read
    from before any run exists.
    """
    gg, dtype = SU(2), torch.complex128
    model = LCNNRef(
        gg, 4, D, K=K, c_hidden=c_hidden, n_layers=n_layers, dtype=dtype,
        in_channels=c_in, reduction="none",
    )
    predicted = reference_dof_count(D, K, c_in, c_hidden, n_layers)
    assert predicted == _real_dofs(model)


def test_their_kernel_is_quadratic_in_the_input_width():
    """The fact that forces a different matched width than ours.

    Their first layer costs ``(2c_in+1)·(2c_in(1+2DK)+1)`` per output channel,
    so a task feeding many input channels (the glueball stack's 12) spends its
    whole budget there. This is the arithmetic behind the note's c_hidden
    choices; if it ever stops holding, those choices have to be revisited.
    """
    D, K = 3, 2
    at_3 = reference_dof_count(D, K, c_in=3, c_hidden=1, n_layers=1)
    at_12 = reference_dof_count(D, K, c_in=12, c_hidden=1, n_layers=1)
    head = 2 * 32 + 32 + 32 + 1
    assert (at_12 - head) > 13 * (at_3 - head)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_gradient_checkpointing_is_exact():
    """It is what makes the production batch fit; it must change nothing."""
    torch.manual_seed(21)
    L, D, K = 4, 3, 2
    gg, dtype = SU(2), torch.complex128
    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype).unsqueeze(0)
    W = plaquette_tensor(U, gg)

    kw = dict(
        gaugegroup=gg, L=L, D=D, K=K, c_hidden=3, n_layers=2, dtype=dtype,
        reduction="none", in_channels=3,
    )
    torch.manual_seed(99)
    plain = LCNNRef(**kw)
    torch.manual_seed(99)
    ckpt = LCNNRef(**kw, grad_checkpoint=True)
    ckpt.train()

    out_plain = plain(W, U)
    out_ckpt = ckpt(W, U)
    assert torch.allclose(out_plain, out_ckpt, atol=1e-12)

    out_plain.sum().backward()
    out_ckpt.sum().backward()
    for a, b in zip(plain.parameters(), ckpt.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-10)


def test_axis_transports_are_rejected_with_a_message_that_says_why():
    """Their layer transports internally; feeding it our (B, D, K, …) is a bug."""
    from gelt.lcnn import build_axis_transports

    torch.manual_seed(23)
    L, D, K = 4, 3, 2
    gg, dtype = SU(2), torch.complex128
    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype).unsqueeze(0)
    W = plaquette_tensor(U, gg)
    T = build_axis_transports(U, K, gg)

    model = LCNNRef(gg, L, D, K=K, c_hidden=2, n_layers=1, dtype=dtype,
                    reduction="none", in_channels=3)
    with pytest.raises(ValueError, match="raw links"):
        model(W, T)


# ---------------------------------------------------------------------------
# Is our implementation a sub-family of theirs, and by how much?
# ---------------------------------------------------------------------------


def _their_transported_basis(layer, x, ref, D, lattice, t_w):
    """Their ``augT`` columns, read off one at a time.

    Their output is ``Σ_{v,w} weight[u,v,w] · augL_v @ augT_w``, so a one-hot on
    the *unit* local row (``v = -1``, the identity) makes output channel 0 equal
    to ``augT_w`` alone. Reading the basis rather than deriving it is what keeps
    this test from re-implementing their shift ordering and then testing its own
    re-implementation.
    """
    basis = []
    saved = layer.weight.data.clone()
    for w in range(t_w):
        with torch.no_grad():
            layer.weight.data.zero_()
            layer.weight.data[0, -1, w] = 1.0
            layer.unit_tensors = {}
        out = ref.unpack_x(layer(x), D)[1]
        basis.append(from_ref_layout(out, lattice)[0, 0])
    with torch.no_grad():
        layer.weight.data.copy_(saved)
        layer.unit_tensors = {}
    return basis


def test_our_lcb_reproduces_their_kernel_exactly_at_sufficient_width():
    """``gelt.lcnn``'s L-Conv + L-Bilin **is** their merged kernel, factored.

    Constructive: take a random full ``weight[n_out, 2c_in+1, t_w]`` of theirs
    and build an ``(ω, β)`` of ours that reproduces it, with our L-Conv width
    ``c_out = t_w − 1``. ω selects the transported basis one direction per
    channel and β carries their kernel verbatim.

    So the two are **not** different function classes: ours is the same family
    with the ``(local, transported)`` kernel constrained to a subspace of
    dimension set by the L-Conv width — which is exactly why the width matters
    and why the next test exists.
    """
    from gelt.lcnn import LBilin, LConv, build_axis_transports
    from gelt.lcnn_reference import reference_layers

    torch.manual_seed(0)
    D, K, L, nc, c_in, n_out = 2, 1, 5, 2, 1, 3
    dtype, gg = torch.complex128, SU(nc)
    ref = reference_layers()

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype).unsqueeze(0)
    W = plaquette_tensor(U, gg)[:, :c_in]
    T = build_axis_transports(U, K, gg)

    n_terms = 1 + 2 * D * K
    t_w, w_in = 2 * c_in * n_terms + 1, 2 * c_in + 1
    c_out = t_w - 1

    theirs = ref.LConvBilin(
        dims=[L] * D, kernel_size=K + 1, dilation=1, n_in=c_in, n_out=n_out,
        nc=nc, use_symmetric=True,
    ).to(torch.float64)
    theirs.unit_matrix = theirs.unit_matrix.to(torch.float64)
    with torch.no_grad():
        theirs.weight.normal_()          # a full kernel: no zeroed rows

    x = ref.repack_x(to_ref_layout(U), to_ref_layout(W))
    basis = _their_transported_basis(theirs, x, ref, D, (L,) * D, t_w)
    target = from_ref_layout(ref.unpack_x(theirs(x), D)[1], (L,) * D)

    conv = LConv(gg, c_in, c_out, D, K, dtype=dtype)
    bil = LBilin(gg, c_in, c_out, n_out, dtype=dtype)

    # One L-Conv output channel per transported basis direction. The channel
    # order is ours and the column order is theirs (they enumerate an axis as
    # −1…−K then +1…+K, we enumerate +1, −1, +2, −2), so the correspondence is
    # *discovered* by matching the two bases rather than assumed.
    with torch.no_grad():
        conv.w.zero_()
        slots = [(1 + j, s) for j in range(c_in) for s in range(conv.n_shifts)]
        slots += [
            (1 + c_in + j, s) for j in range(c_in) for s in range(conv.n_shifts)
        ]
        for m, (a, s) in enumerate(slots):
            conv.w[m, a, s] = 1.0
    ours_basis = conv(W, T)[0]

    perm = {}
    for w in range(t_w - 1):                      # the last column is the unit
        hits = [
            m for m in range(c_out)
            if torch.allclose(ours_basis[m], basis[w], atol=1e-11)
        ]
        assert len(hits) == 1, f"column {w} matched {hits}"
        perm[w] = hits[0]
    identity = torch.eye(nc, dtype=dtype).expand(*(L,) * D, nc, nc)
    assert torch.allclose(basis[-1], identity, atol=1e-12)

    # Their local augmentation is [w, w†, 1]; ours is [1, W, W†]. Their
    # transported one ends with the unit; our right-hand augmentation starts
    # with it.
    with torch.no_grad():
        bil.w.zero_()
        for i in range(n_out):
            for v in range(w_in):
                v_ours = 0 if v == w_in - 1 else 1 + v
                for w in range(t_w):
                    m_ours = 0 if w == t_w - 1 else 1 + perm[w]
                    bil.w[i, v_ours, m_ours] = theirs.weight[i, v, w].to(dtype)

    ours = bil(W, conv(W, T))
    err = (ours - target).abs().max().item()
    assert err < 1e-10, err
    assert target.abs().max() > 1e-3, "target is ~0: the reconstruction is vacuous"


def test_the_production_widths_are_far_below_that_threshold():
    """How much of their family our arm could reach — the number to quote.

    The reconstruction above needs an L-Conv width of ``t_w − 1 = 2·c_in·(1+2DK)``.
    Our arms run at a small fraction of it, so the reimplementation is a *strict*
    restriction at production widths, not a cosmetic difference: its kernel over
    the transported basis is confined to the row space of ω (and its conjugate),
    at most ``2·c_out + 1`` of the ``t_w`` available directions.

    This is arithmetic, and it is here so that the claim in
    ``notes/lcnn_reference_switch.md`` §1 cannot drift from the widths the arms
    actually use.
    """
    D, K = 3, 2
    n_terms = 1 + 2 * D * K
    # The M1 probe: 3 plaquette channels in, c_hidden = 6.
    assert 2 * 3 * n_terms == 78
    assert 2 * 6 + 1 < 78
    # The glueball shootout: 12 stacked smeared channels in, c_hidden = 5.
    assert 2 * 12 * n_terms == 312
    assert 2 * 5 + 1 < 312
