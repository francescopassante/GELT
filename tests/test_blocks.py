"""Tests for :class:`gelt.blocks.GEMHSA`.

The central guarantee of the block is gauge equivariance under local Ω:
forward(Ω·W·Ω†, T(Ω·U)) must equal Ω · forward(W, T(U)) · Ω† at machine ε.
This pins down all four moving parts at once — the adjoint transport
T·X·T†, the Frobenius-product score, the softmax over offsets, and the
residual + L-Act gate — because any broken piece would mix in a residual
Ω(y) factor at some other site.
"""

import torch

from gelt import (
    SU,
    Z2,
    build_transport_average,
    link_gauge_transformation,
    local_gauge_transformation,
    random_links,
)
from gelt.blocks_bias import GELT, GEMHSA, ChannelLift


def _unitary_omega(L, D, nc, seed):
    """Random unitary Ω of shape (*Λ, nc, nc) in complex128."""
    torch.manual_seed(seed)
    raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        L**D, nc, nc, dtype=torch.float64
    )
    Q, _ = torch.linalg.qr(raw)
    return Q.reshape(*([L] * D), nc, nc)


# ---------------------------------------------------------------------------
# Gauge equivariance (the main guarantee)
# ---------------------------------------------------------------------------


def test_gemhsa_gauge_equivariance_sun():
    """SU(2) double-precision: forward(W_g, T_g) == Ω · forward(W, T) · Ω†."""
    torch.manual_seed(7)
    L, D, R, C, H, nc = 4, 2, 2, 3, 2, 2
    gg = SU(nc)
    dtype = torch.complex128

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)  # (D, *Λ, nc, nc)
    W = torch.randn(1, C, *([L] * D), nc, nc, dtype=torch.float64) + 1j * torch.randn(
        1, C, *([L] * D), nc, nc, dtype=torch.float64
    )

    omega = _unitary_omega(L, D, nc, seed=7)
    U_g = link_gauge_transformation(U, omega, gg)
    W_g = local_gauge_transformation(W, omega, gg)

    T = build_transport_average(
        U.unsqueeze(0), R=R, gaugegroup=gg
    )  # (1, n_off, *Λ, nc, nc)
    T_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

    block = GEMHSA(gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, dtype=dtype).to(
        dtype
    )
    # Exercise the orbit bias and the gate branch by setting non-trivial values.
    with torch.no_grad():
        block.b_h.copy_(torch.randn_like(block.b_h) * 0.1)

    out = block(W, T)
    out_g = block(W_g, T_g)
    expected = local_gauge_transformation(out, omega, gg)

    assert torch.allclose(
        out_g, expected, atol=1e-9
    ), f"max diff = {(out_g - expected).abs().max().item():.3e}"


def test_gemhsa_gauge_equivariance_softplus_gate():
    """Same guarantee with the softplus gate branch."""
    torch.manual_seed(11)
    L, D, R, C, H, nc = 4, 2, 2, 2, 2, 2
    gg = SU(nc)
    dtype = torch.complex128

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
    W = torch.randn(1, C, *([L] * D), nc, nc, dtype=torch.float64) + 1j * torch.randn(
        1, C, *([L] * D), nc, nc, dtype=torch.float64
    )
    omega = _unitary_omega(L, D, nc, seed=11)
    U_g = link_gauge_transformation(U, omega, gg)
    W_g = local_gauge_transformation(W, omega, gg)

    T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
    T_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

    block = GEMHSA(
        gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, gate="softplus", dtype=dtype
    ).to(dtype)

    out = block(W, T)
    out_g = block(W_g, T_g)
    expected = local_gauge_transformation(out, omega, gg)

    assert torch.allclose(out_g, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Shape / finiteness / backward
# ---------------------------------------------------------------------------


def test_gemhsa_shape_preserved_and_backward_finite():
    """Output shape == input shape; forward + backward produce finite tensors."""
    torch.manual_seed(0)
    L, D, R, C, H, nc = 4, 2, 2, 4, 2, 3
    gg = SU(nc)
    B = 2

    U_batch = random_links(L=L, D=D, gaugegroup=gg, N=B)
    # Exercise the batched DP path directly: one pass over the whole (B, D, *Λ, nc, nc).
    T = build_transport_average(U_batch, R=R, gaugegroup=gg)

    W = torch.randn(B, C, *([L] * D), nc, nc, dtype=torch.complex64, requires_grad=True)
    block = GEMHSA(gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H)

    out = block(W, T)
    assert out.shape == W.shape

    out.abs().sum().backward()
    assert torch.isfinite(W.grad).all()
    for name, p in block.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


# ---------------------------------------------------------------------------
# Z2 sanity (abelian — the bug would have hidden here)
# ---------------------------------------------------------------------------


def test_gemhsa_gauge_equivariance_z2():
    """Z₂ float64: same equivariance guarantee on the abelian baseline."""
    torch.manual_seed(3)
    L, D, R, C, H = 6, 2, 2, 2, 2
    gg = Z2()
    dtype = torch.float64

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
    W = torch.randn(1, C, *([L] * D), 1, 1, dtype=dtype)

    omega = gg.random((L,) * D, dtype=dtype)
    U_g = link_gauge_transformation(U, omega, gg)
    W_g = local_gauge_transformation(W, omega, gg)

    T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
    T_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

    block = GEMHSA(gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, dtype=dtype).to(
        dtype
    )

    out = block(W, T)
    out_g = block(W_g, T_g)
    expected = local_gauge_transformation(out, omega, gg)

    assert torch.allclose(out_g, expected, atol=1e-12)


def test_channel_lift_identity_extend_init():
    """Identity-extend init: first c_in output channels copy the input verbatim,
    the rest are zero. Makes ``d_model == d_input`` a no-op at init."""
    torch.manual_seed(0)
    c_in, c_out = 3, 8
    lift = ChannelLift(c_in, c_out, dtype=torch.complex64)
    W = torch.randn(2, c_in, 4, 4, 2, 2, dtype=torch.complex64)
    out = lift(W)
    assert out.shape == (2, c_out, 4, 4, 2, 2)
    assert torch.allclose(out[:, :c_in], W)
    assert torch.all(out[:, c_in:] == 0)


def test_gelt_d_model_widened_gauge_equivariant():
    """GELT with d_model > d_input stays gauge-equivariant end-to-end.

    The internal residual stream is wider than the plaquette input; the
    front-end ChannelLift must not break the W → Ω W Ω† transformation.
    """
    torch.manual_seed(13)
    L, D, R, nc = 4, 2, 2, 2
    gg = SU(nc)
    dtype = torch.complex128

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
    P = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)

    # Build plaquette input (D(D-1)/2 = 1 channel for D=2). The trace is a
    # gauge invariant, so we test by comparing scalar outputs at sites.
    from gelt.lattice import plaquette_tensor

    X = plaquette_tensor(U.unsqueeze(0), gg)  # (1, n_pairs, *Λ, nc, nc)

    omega = _unitary_omega(L, D, nc, seed=13)
    U_g = link_gauge_transformation(U, omega, gg)
    X_g = plaquette_tensor(U_g.unsqueeze(0), gg)
    P_g = build_transport_average(U_g.unsqueeze(0), R=R, gaugegroup=gg)

    model = GELT(
        gaugegroup=gg,
        L=L,
        D=D,
        R=R,
        nhead=2,
        gemhsa_layers=2,
        d_qkv=4,
        dtype=dtype,
        d_model=8,
        reduction="none",
    )

    y = model(X, P)
    y_g = model(X_g, P_g)
    # GELT readout is gauge invariant: y == y_g.
    assert torch.allclose(
        y, y_g, atol=1e-9
    ), f"max diff = {(y - y_g).abs().max().item():.3e}"


def test_gelt_z2_real_forward_backward():
    """Full GELT supports real-valued Z2 models without forcing complex kernels."""
    torch.manual_seed(5)
    L, D, R, B = 4, 2, 1, 2
    gg = Z2()
    dtype = torch.float32

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype, N=B)
    T = build_transport_average(U, R=R, gaugegroup=gg)
    W = torch.randn(B, D * (D - 1) // 2, *([L] * D), 1, 1, dtype=dtype)

    model = GELT(
        gaugegroup=gg,
        L=L,
        D=D,
        R=R,
        nhead=1,
        gemhsa_layers=1,
        d_qkv=2,
        dtype=dtype,
        reduction="none",
    )
    out = model(W, T)
    assert out.shape == (B, L, L)
    assert out.dtype == dtype

    out.square().mean().backward()
    for name, p in model.named_parameters():
        assert (
            p.grad is None or torch.isfinite(p.grad).all()
        ), f"non-finite grad on {name}"
