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
    "d_model, n_levels, tolerance",
    [(16, 4, 0.15), (24, 7, 0.15)],
)
def test_lcnn_default_geometry_matches_the_glueball_gelt_budget(
    d_model, n_levels, tolerance
):
    """The L-CNN defaults of ``train_glueball.py`` sit within 15% of both nets.

    K = 2, c_hidden = 6, 4 layers is the one geometry that brackets the
    d_model = 16 / 4-level and d_model = 24 / 7-level GELT operators. If a
    change to either architecture moves the budgets apart, the shootout stops
    being matched-parameter and this fails before a V100 hour is spent.
    """
    gg = SU(2)
    lcnn = LCNN(
        gaugegroup=gg, L=12, D=3, K=2, c_hidden=6, n_layers=4,
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
        f"{n_levels} smear levels — no longer matched-parameter"
    )
