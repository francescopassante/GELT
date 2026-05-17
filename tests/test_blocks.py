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
    build_transport_sums,
    l1_ball_offsets,
    link_gauge_transformation,
    local_gauge_transformation,
    random_links,
)
from gelt.blocks import GEMHSA


def _stack_T(U, R, gaugegroup, offsets):
    """Build the (n_offsets, *Λ, nc, nc) transport tensor in canonical order."""
    T_dict = build_transport_sums(U, R=R, gaugegroup=gaugegroup)
    return torch.stack([T_dict[o] for o in offsets], dim=0)


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

    offsets = l1_ball_offsets(D, R)
    T = _stack_T(U, R, gg, offsets).unsqueeze(0)  # (1, n_off, *Λ, nc, nc)
    T_g = _stack_T(U_g, R, gg, offsets).unsqueeze(0)

    block = GEMHSA(gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, dtype=dtype).to(
        dtype
    )
    # Exercise the orbit bias and the gate branch by setting non-trivial values.
    with torch.no_grad():
        block.b_h.copy_(torch.randn_like(block.b_h) * 0.1)

    out = block(W, T)
    out_g = block(W_g, T_g)
    expected = local_gauge_transformation(out, omega, gg)

    assert torch.allclose(out_g, expected, atol=1e-9), (
        f"max diff = {(out_g - expected).abs().max().item():.3e}"
    )


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

    offsets = l1_ball_offsets(D, R)
    T = _stack_T(U, R, gg, offsets).unsqueeze(0)
    T_g = _stack_T(U_g, R, gg, offsets).unsqueeze(0)

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

    U_batch = torch.stack(
        [random_links(L=L, D=D, gaugegroup=gg) for _ in range(B)], dim=0
    )
    offsets = l1_ball_offsets(D, R)
    T = torch.stack([_stack_T(U_batch[b], R, gg, offsets) for b in range(B)], dim=0)

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

    offsets = l1_ball_offsets(D, R)
    T = _stack_T(U, R, gg, offsets).unsqueeze(0)
    T_g = _stack_T(U_g, R, gg, offsets).unsqueeze(0)

    block = GEMHSA(gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, dtype=dtype).to(
        dtype
    )

    out = block(W, T)
    out_g = block(W_g, T_g)
    expected = local_gauge_transformation(out, omega, gg)

    assert torch.allclose(out_g, expected, atol=1e-12)
