"""Tests for the L-CNN, :mod:`gelt.lcnn`, as the glueball task's baseline.

``gelt/lcnn.py`` is the Favoni et al. (2012.12901) reference implementation and
has always mirrored GELT's I/O. What is pinned here is what the matched-parameter
shootout of ``reports/curve`` §7 needs on top of that:

* **gauge invariance of the per-site scalar field** ``reduction="none"`` produces
  — the object the Rayleigh loss correlates in time. Checked on *stacked
  multi-level* inputs (``in_channels`` ≠ D(D-1)/2), because that is the shape the
  shootout feeds, and a channel-axis bug would otherwise only show up after a
  training run.
* **gradient checkpointing is exact** — it is what makes batch 6 fit on the V100,
  and L-Bilin's channel-pair outer product runs inside the recompute.
* **the parameter match itself**, against the two trained GELT glueball nets. The
  comparison is a statement about architecture only as long as the budgets agree,
  so the defaults that make them agree are a test, not a comment.
"""

import pytest
import torch

from gelt import SU, Z2, link_gauge_transformation, plaquette_tensor, random_links
from gelt.blocks import GELT
from gelt.lcnn import LCNN, build_axis_transports


def _real_dofs(model):
    """Parameter count in real degrees of freedom (a complex weight is two)."""
    return sum(
        p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
    )


def _stacked_plaquettes(U, gg, n_levels):
    """``n_levels`` copies of the plaquette input on the channel axis.

    The shootout stacks one 3-channel block per APE smearing level; smearing is
    gauge *covariant*, so for an invariance test plain copies exercise the same
    channel bookkeeping without dragging ``ape_smear`` (and its α = 0.5 Z₂
    defect, caveat 1) into the assertion.
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
# Gauge invariance of the per-site readout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["relu", "softplus"])
def test_lcnn_per_site_output_is_gauge_invariant_su2(gate):
    """SU(2), complex128, D = 3: the per-timeslice glueball configuration."""
    torch.manual_seed(11)
    L, D, K, nc, n_levels = 4, 3, 2, 2, 4
    gg, dtype = SU(nc), torch.complex128

    U = torch.stack(
        [random_links(L=L, D=D, gaugegroup=gg, dtype=dtype) for _ in range(2)]
    )
    omega = _su2_omega(L, D, nc, seed=11)
    U_g = torch.stack([link_gauge_transformation(u, omega, gg) for u in U])

    model = LCNN(
        gaugegroup=gg, L=L, D=D, K=K, c_hidden=6, n_layers=4, dtype=dtype,
        reduction="none", gate=gate, in_channels=3 * n_levels,
    )

    out = model(_stacked_plaquettes(U, gg, n_levels), build_axis_transports(U, K, gg))
    out_g = model(
        _stacked_plaquettes(U_g, gg, n_levels), build_axis_transports(U_g, K, gg)
    )
    assert out.shape == (2, L, L, L)
    assert torch.allclose(out_g, out, atol=1e-11)


def test_lcnn_per_site_output_is_gauge_invariant_z2():
    """Z₂ in float64: nc = 1, Ω a site-local sign, everything real."""
    torch.manual_seed(5)
    L, D, K, n_levels = 4, 3, 2, 4
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

    model = LCNN(
        gaugegroup=gg, L=L, D=D, K=K, c_hidden=4, n_layers=3, dtype=dtype,
        reduction="none", in_channels=3 * n_levels,
    )

    out = model(_stacked_plaquettes(U, gg, n_levels), build_axis_transports(U, K, gg))
    out_g = model(
        _stacked_plaquettes(U_g, gg, n_levels), build_axis_transports(U_g, K, gg)
    )
    assert torch.allclose(out_g, out, atol=1e-12)


# ---------------------------------------------------------------------------
# The knobs the shootout added
# ---------------------------------------------------------------------------


def test_in_channels_defaults_to_the_plaquette_count():
    """Omitting ``in_channels`` must leave the reference model untouched."""
    gg = SU(2)
    for D in (2, 3, 4):
        ref = LCNN(gaugegroup=gg, L=4, D=D, K=1, c_hidden=3, n_layers=2)
        assert ref.c_in_plaq == D * (D - 1) // 2
        # First L-Conv sees the augmented input 2·C_in + 1.
        assert ref.lcb_blocks[0].lconv.c_prime == 2 * (D * (D - 1) // 2) + 1

    wide = LCNN(gaugegroup=gg, L=4, D=3, K=1, c_hidden=3, n_layers=2, in_channels=12)
    assert wide.c_in_plaq == 12
    assert wide.lcb_blocks[0].lconv.c_prime == 25


def test_grad_checkpointing_gives_the_same_gradients():
    torch.manual_seed(2)
    L, D, K, nc = 4, 3, 2, 2
    gg, dtype = SU(nc), torch.complex128

    U = torch.stack([random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)])
    W = _stacked_plaquettes(U, gg, 2)
    T = build_axis_transports(U, K, gg)

    model = LCNN(
        gaugegroup=gg, L=L, D=D, K=K, c_hidden=4, n_layers=3, dtype=dtype,
        reduction="none", in_channels=6,
    ).train()

    def grads(flag):
        model.grad_checkpoint = flag
        for p in model.parameters():
            p.grad = None
        model(W, T).pow(2).sum().backward()
        return [p.grad.clone() for p in model.parameters()]

    for g_ckpt, g_plain in zip(grads(True), grads(False)):
        assert torch.equal(g_ckpt, g_plain)


def test_init_scale_scales_the_output_linearly():
    """The Rayleigh loss is scale-invariant, so this knob may only set λ."""
    L, D, K = 4, 3, 1
    gg, dtype = SU(2), torch.complex128
    U = torch.stack([random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)])
    W, T = _stacked_plaquettes(U, gg, 1), build_axis_transports(U, K, gg)

    kw = dict(
        gaugegroup=gg, L=L, D=D, K=K, c_hidden=3, n_layers=2, dtype=dtype,
        reduction="none",
    )
    torch.manual_seed(0)
    plain = LCNN(**kw)
    torch.manual_seed(0)
    scaled = LCNN(**kw, init_scale=10.0)

    # Only the output layer is scaled, and it is affine ⇒ exactly ×10.
    assert torch.allclose(scaled(W, T), 10.0 * plain(W, T), atol=1e-10)


# ---------------------------------------------------------------------------
# The matched-parameter claim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "d_model, n_levels, c_hidden, tolerance",
    [(16, 4, 5, 0.15), (24, 7, 6, 0.15)],
)
def test_lcnn_default_geometry_matches_the_glueball_gelt_budget(
    d_model, n_levels, c_hidden, tolerance
):
    """The L-CNN geometry is matched to the GELT net it is compared against.

    K = 2 (symmetric, so ±2 per layer — the same Manhattan-8 reach as R = 2 over
    four GEMHSA layers) with 4 layers, at `c_hidden = 5` against the
    d_model = 16 / 4-level operator the shootout actually trains, and
    `c_hidden = 6` against the d_model = 24 / 7-level one. If a change to either
    architecture moves the budgets apart, the shootout stops being
    matched-parameter and this fails before a V100 hour is spent.
    """
    gg = SU(2)
    lcnn = LCNN(
        gaugegroup=gg, L=12, D=3, K=2, c_hidden=c_hidden, n_layers=4,
        dtype=torch.complex64, mlp_hidden=32, mlp_out=1, reduction="none",
        in_channels=3 * n_levels,
    )
    gelt = GELT(
        gaugegroup=gg, L=12, D=3, R=2, nhead=2, gemhsa_layers=4, d_qkv=6,
        d_model=d_model, dtype=torch.complex64, mlp_hidden=32, mlp_out=1,
        reduction="none", mlp_zero_init=False, in_channels=3 * n_levels,
    )
    ratio = _real_dofs(lcnn) / _real_dofs(gelt)
    assert abs(ratio - 1.0) <= tolerance, (
        f"L-CNN/GELT real-DOF ratio {ratio:.3f} at d_model={d_model}, "
        f"{n_levels} smear levels, c_hidden={c_hidden} — no longer "
        f"matched-parameter"
    )


# ---------------------------------------------------------------------------
# The optimised L-Conv / L-Bilin paths vs the definitions they implement
# ---------------------------------------------------------------------------


def _naive_lconv(lc, W, T):
    """Eq. 5 written out: augment first, transport every augmented channel with
    broadcast per-channel products, then contract. The module instead transports
    the raw channels, folds the colour axis, and reaches the W† half through
    (Σ conj(ω) S)† — all exact identities, which is what this pins."""
    from gelt.lcnn import _augment

    gg = lc.gaugegroup
    A = _augment(W, gg)  # (B, c_prime, *Λ, nc, nc)
    terms = [A]
    for mu in range(lc.D):
        for k in range(1, lc.K + 1):
            Uk = T[:, mu, k - 1].to(W.dtype)
            X = torch.roll(A, -k, dims=2 + mu)
            terms.append(Uk.unsqueeze(1) @ X @ gg.dagger(Uk).unsqueeze(1))
            if lc.symmetric:
                Vk = gg.dagger(torch.roll(Uk, k, dims=1 + mu))
                Y = torch.roll(A, k, dims=2 + mu)
                terms.append(Vk.unsqueeze(1) @ Y @ gg.dagger(Vk).unsqueeze(1))
    S = torch.stack(terms, dim=2)  # (B, c_prime, n_shifts, *Λ, nc, nc)
    return torch.einsum("ijs,bjs...->bi...", lc.w, S)


def _naive_lbilin(lb, W_left, W_right):
    """Eq. 6 written out: the full (c_left × c_right) channel-pair product."""
    from gelt.lcnn import _augment

    L_aug = _augment(W_left, lb.gaugegroup)
    R_aug = _augment(W_right, lb.gaugegroup)
    prod = L_aug.unsqueeze(2) @ R_aug.unsqueeze(1)
    return torch.einsum("ijk,bjk...->bi...", lb.w, prod)


@pytest.mark.parametrize("symmetric", [True, False])
def test_lconv_matches_the_naive_definition(symmetric):
    from gelt.lcnn import LConv

    torch.manual_seed(4)
    L, D, K, nc = 4, 3, 2, 2
    gg, dtype = SU(nc), torch.complex128
    U = torch.stack([random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)])
    T = build_axis_transports(U, K, gg)
    W = torch.randn(1, 3, L, L, L, nc, nc, dtype=dtype)

    lc = LConv(gg, c_in=3, c_out=4, D=D, K=K, dtype=dtype, symmetric=symmetric)
    assert torch.allclose(lc(W, T), _naive_lconv(lc, W, T), atol=1e-12)


def test_lbilin_matches_the_naive_definition():
    from gelt.lcnn import LBilin

    torch.manual_seed(6)
    L, nc = 4, 2
    gg, dtype = SU(nc), torch.complex128
    W1 = torch.randn(1, 3, L, L, nc, nc, dtype=dtype)
    W2 = torch.randn(1, 2, L, L, nc, nc, dtype=dtype)

    lb = LBilin(gg, c_in_left=3, c_in_right=2, c_out=4, dtype=dtype)
    assert torch.allclose(lb(W1, W2), _naive_lbilin(lb, W1, W2), atol=1e-12)


def test_the_kernel_reaches_both_directions():
    """A layer must see x − k·μ̂ as well as x + k·μ̂.

    The W† channels do not stand in for the backward hops — daggering commutes
    with the adjoint transport — so an asymmetric kernel gives the stack a
    one-sided cone. Perturb one site and check which outputs move.
    """
    from gelt.lcnn import LConv

    torch.manual_seed(0)
    L, D, K, nc = 6, 2, 1, 2
    gg, dtype = SU(nc), torch.complex128
    U = torch.stack([random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)])
    T = build_axis_transports(U, K, gg)
    W = torch.randn(1, 2, L, L, nc, nc, dtype=dtype)

    for symmetric, expected in ((True, {(0, 0), (1, 0), (5, 0), (0, 1), (0, 5)}),
                                (False, {(0, 0), (5, 0), (0, 5)})):
        lc = LConv(gg, c_in=2, c_out=2, D=D, K=K, dtype=dtype, symmetric=symmetric)
        base = lc(W, T)
        Wp = W.clone()
        Wp[0, :, 0, 0] += 1.0
        moved = (lc(Wp, T) - base).abs().amax(dim=(0, 1, -1, -2)) > 1e-12
        assert {tuple(ix) for ix in moved.nonzero().tolist()} == expected
