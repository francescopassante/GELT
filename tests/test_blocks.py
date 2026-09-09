"""Tests for the GELT block, :mod:`gelt.blocks`.

This is the only block variant left. The retired ``blocks_bias`` (the same
block with a learned offset bias on the score instead of RoPE) never produced
a published result and was deleted in the 2026-09-09 cleanup; its four
variant-agnostic cases are ported to the bottom of this file. Two things are
pinned here:

* **gauge equivariance** — ``forward(Ω·W·Ω†, T(Ω·U)) == Ω · forward(W, T(U)) · Ω†``.
* **the optimised attention path against a naive oracle.** The block fuses the
  key/value gather, hoists the Δx = 0 transport slot out of the per-layer
  forward, and folds the RoPE rotation into the query instead of rotating the
  transported keys. All three are algebraic identities, so the oracle written
  out below — two gathers, a concatenation, ``apply_rope`` on K̃, and the plain
  Frobenius score — must agree to machine ε, on the outputs *and* on the
  gradients.
"""

import math

import pytest
import torch

from gelt import (
    SU,
    Z2,
    build_transport_average,
    link_gauge_transformation,
    local_gauge_transformation,
    random_links,
)
from gelt.blocks import GELT, GEMHSA, ChannelLift


def _unitary_omega(L, D, nc, seed):
    """Random unitary Ω of shape (*Λ, nc, nc) in complex128."""
    torch.manual_seed(seed)
    raw = torch.randn(L**D, nc, nc, dtype=torch.float64) + 1j * torch.randn(
        L**D, nc, nc, dtype=torch.float64
    )
    Q, _ = torch.linalg.qr(raw)
    return Q.reshape(*([L] * D), nc, nc)


# ---------------------------------------------------------------------------
# Gauge equivariance of the RoPE variant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["relu", "softplus"])
def test_rope_gemhsa_gauge_equivariance_su2(gate):
    torch.manual_seed(7)
    L, D, R, C, H, nc = 4, 2, 2, 3, 2, 2
    gg, dtype = SU(nc), torch.complex128

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
    W = torch.randn(1, C, *([L] * D), nc, nc, dtype=dtype)
    omega = _unitary_omega(L, D, nc, seed=7)

    T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
    T_g = build_transport_average(
        link_gauge_transformation(U, omega, gg).unsqueeze(0), R=R, gaugegroup=gg
    )
    block = GEMHSA(
        gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, d_qkv=4, gate=gate,
        dtype=dtype,
    ).to(dtype)

    out = block(W, T)
    out_g = block(local_gauge_transformation(W, omega, gg), T_g)
    expected = local_gauge_transformation(out, omega, gg)
    assert torch.allclose(out_g, expected, atol=1e-10), (
        (out_g - expected).abs().max().item()
    )


def test_rope_gemhsa_gauge_equivariance_z2():
    """Z₂ in float64: nc = 1 makes Ω a site-local sign, and the block is real."""
    torch.manual_seed(3)
    L, D, R, C, H = 4, 2, 1, 2, 2
    gg, dtype = Z2(), torch.float64

    U = random_links(L=L, D=D, gaugegroup=gg, dtype=dtype)
    W = torch.randn(1, C, *([L] * D), 1, 1, dtype=dtype)
    omega = torch.where(
        torch.rand(*([L] * D), 1, 1) < 0.5,
        torch.tensor(-1.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    T = build_transport_average(U.unsqueeze(0), R=R, gaugegroup=gg)
    T_g = build_transport_average(
        link_gauge_transformation(U, omega, gg).unsqueeze(0), R=R, gaugegroup=gg
    )
    block = GEMHSA(
        gaugegroup=gg, L=L, D=D, R=R, d_input=C, nhead=H, d_qkv=2, dtype=dtype
    )

    out = block(W, T)
    out_g = block(local_gauge_transformation(W, omega, gg), T_g)
    expected = local_gauge_transformation(out, omega, gg)
    assert torch.allclose(out_g, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# The optimised attention path vs a naive oracle
# ---------------------------------------------------------------------------


def _oracle_forward(blk, W, T):
    """The pre-optimisation pipeline, written out: two gathers + a concat, the
    identity slot prepended inside the block, ``apply_rope`` on the transported
    keys, and the Frobenius score over channels and colour at once."""
    B, nc = W.shape[0], W.shape[-1]
    spatial = T.shape[2:-2]
    ident = (
        torch.eye(nc, dtype=T.dtype)
        .view(1, 1, *([1] * blk.D), nc, nc)
        .expand(B, 1, *spatial, nc, nc)
    )
    T_full = torch.cat([ident, T], dim=1)
    T_dag_full = torch.cat([ident, blk.gaugegroup.dagger(T)], dim=1)

    W_aug = blk.augment(W)
    trailing = W_aug.shape[2:]
    QKV = torch.matmul(
        blk.w_QKV.view(4 * blk.H * blk.d_qkv, blk.C_prime),
        W_aug.view(B, blk.C_prime, -1),
    ).view(B, 4, blk.H, blk.d_qkv, *trailing)
    Q, K, V, Q_v = QKV.unbind(dim=1)

    idx = tuple(blk._nbr_idx[k] for k in range(blk.D))
    nb = (slice(None),) * 3 + idx + (slice(None), slice(None))
    KV_tilde = blk.transport(
        torch.cat((K[nb], V[nb]), dim=2), T_full, T_dag_full
    )
    K_tilde, V_tilde = KV_tilde.split(blk.d_qkv, dim=2)
    K_tilde = blk.apply_rope(K_tilde)

    score = (Q.unsqueeze(3).conj() * K_tilde).sum(dim=(2, -2, -1)).real
    score = score / math.sqrt(blk.d_qkv * nc)
    alpha = torch.softmax(score, dim=2)
    V_weighted = (
        alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1) * V_tilde
    ).sum(dim=3)
    out = torch.matmul(blk.gaugegroup.dagger(Q_v), V_weighted)

    HD = blk.H * blk.d_qkv
    W_mix = torch.matmul(blk.w_mix.view(blk.C, HD), out.reshape(B, HD, -1)).view(
        B, blk.C, *trailing
    )
    tr = W_mix.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
    g = (
        torch.nn.functional.softplus(tr)
        if blk.gate == "softplus"
        else torch.nn.functional.relu(tr)
    )
    return W + g.unsqueeze(-1).unsqueeze(-1) * W_mix, score, alpha


@pytest.mark.parametrize(
    "group, dtype, D, R, d_qkv",
    [
        (SU(2), torch.complex128, 3, 2, 6),
        (SU(3), torch.complex128, 2, 2, 4),
        (Z2(), torch.float64, 2, 2, 2),
    ],
)
def test_rope_attend_matches_naive_oracle(group, dtype, D, R, d_qkv):
    torch.manual_seed(0)
    L, B, C, H = 4, 2, 6, 2
    nc = group.nc
    U = torch.stack([random_links(L, D, group, dtype=dtype) for _ in range(B)])
    T = build_transport_average(U, R, group)
    blk = GEMHSA(
        group, L, D, R, d_input=C, nhead=H, d_qkv=d_qkv, dtype=dtype
    )
    blk.store_attention = True

    W_fast = torch.randn(B, C, *([L] * D), nc, nc, dtype=dtype, requires_grad=True)
    W_ref = W_fast.detach().clone().requires_grad_(True)

    fast = blk(W_fast, T)
    ref, score_ref, alpha_ref = _oracle_forward(blk, W_ref, T)

    assert torch.allclose(fast, ref, atol=1e-12)
    assert torch.allclose(blk._last_score, score_ref, atol=1e-12)
    assert torch.allclose(blk._last_alpha, alpha_ref, atol=1e-12)

    fast.abs().pow(2).sum().backward()
    ref.abs().pow(2).sum().backward()
    assert torch.allclose(W_fast.grad, W_ref.grad, atol=1e-11)


def test_prepending_the_self_offset_is_optional():
    """``GELT.attn`` prepends the Δx = 0 slot once; a layer handed the full table
    must not prepend it again, and both routes must agree exactly."""
    torch.manual_seed(1)
    L, D, R, B, C, H = 4, 3, 1, 2, 4, 2
    gg, dtype = SU(2), torch.complex128
    U = torch.stack([random_links(L, D, gg, dtype=dtype) for _ in range(B)])
    T = build_transport_average(U, R, gg)
    blk = GEMHSA(gg, L, D, R, d_input=C, nhead=H, d_qkv=4, dtype=dtype)
    W = torch.randn(B, C, *([L] * D), 2, 2, dtype=dtype)

    T_full, T_dag_full = blk.prepend_self_offset(T)
    assert T_full.shape[1] == blk.n_offsets
    assert torch.equal(blk(W, T), blk(W, T_full, T_dag_full))

    with pytest.raises(ValueError, match="identity already prepended"):
        blk(W, T[:, :-1])


def test_introspection_is_off_by_default():
    """The stashes cost a GPU sync and a reduction over the offset-expanded
    tensors, so training must not pay for them unless asked."""
    torch.manual_seed(1)
    L, D, R, B, C = 4, 2, 1, 1, 3
    gg, dtype = SU(2), torch.complex128
    U = torch.stack([random_links(L, D, gg, dtype=dtype) for _ in range(B)])
    T = build_transport_average(U, R, gg)
    model = GELT(
        gaugegroup=gg, L=L, D=D, R=R, nhead=2, gemhsa_layers=2, d_qkv=4,
        dtype=dtype, mlp_hidden=4, mlp_out=1, reduction="none",
        mlp_zero_init=False, d_model=C, in_channels=1,
    )
    W = torch.randn(B, 1, *([L] * D), 2, 2, dtype=dtype)

    model(W, T)
    layer = model.gemhsa_models[0]
    assert not hasattr(layer, "_last_alpha")
    assert not hasattr(layer, "_last_W_act_norm")

    model.set_introspection(store_attention=True, diagnostics=True)
    model(W, T)
    for layer in model.gemhsa_models:
        assert layer._last_alpha.shape[:2] == (B, 2)
        assert isinstance(layer._last_W_act_norm, float)


# ---------------------------------------------------------------------------
# The offset gather's hand-written backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [None, 1])
def test_offset_gather_backward_matches_autograd(budget, monkeypatch):
    """``_OffsetGather`` keeps autograd's forward and replaces its backward (an
    atomic scatter-add) with the equivalent gather + reduction. ``budget=1``
    forces one offset per chunk, which must only change the summation order."""
    from gelt.blocks import _OffsetGather
    import gelt.blocks as br

    if budget is not None:
        monkeypatch.setattr(br, "_GRAD_GATHER_BUDGET", budget)

    torch.manual_seed(0)
    L, D, R, B, C, dq, nc = 5, 3, 2, 2, 3, 4, 2
    blk = GEMHSA(SU(nc), L, D, R, d_input=C, nhead=2, d_qkv=dq,
                 dtype=torch.complex128)

    X = torch.randn(B, 2, dq, L, L, L, nc, nc, dtype=torch.complex128)
    Xf = X.clone().requires_grad_(True)
    Xr = X.clone().requires_grad_(True)

    fast = _OffsetGather.apply(Xf, blk._nbr_idx, blk._nbr_idx_inv)
    indexer = (
        (slice(None),) * 3
        + tuple(blk._nbr_idx[k] for k in range(D))
        + (slice(None), slice(None))
    )
    ref = Xr[indexer]
    assert torch.equal(fast, ref)

    weight = torch.randn_like(fast)
    (fast * weight).real.sum().backward()
    (ref * weight).real.sum().backward()
    assert torch.allclose(Xf.grad, Xr.grad, atol=1e-13)


def test_offset_gather_inverse_index_is_the_forward_index_reversed():
    """The gradient index is the forward one with the offsets negated; if the two
    buffers ever disagree the gradient is silently wrong at every mixed-sign
    offset, which no shape check would catch."""
    L, D, R = 6, 3, 2
    blk = GEMHSA(SU(2), L, D, R, d_input=2, nhead=1, d_qkv=2,
                 dtype=torch.complex128)
    coords = torch.meshgrid(*[torch.arange(L) for _ in range(D)], indexing="ij")
    for i, off in enumerate(blk.offsets):
        for d in range(D):
            assert torch.equal(blk._nbr_idx[d, i], (coords[d] + off[d]) % L)
            assert torch.equal(blk._nbr_idx_inv[d, i], (coords[d] - off[d]) % L)


def test_grad_checkpointing_gives_the_same_gradients():
    """``train_glueball.py`` runs with ``GRAD_CHECKPOINT=True``, so the layer
    forward is replayed inside the backward. ``_OffsetGather`` is a custom
    autograd Function and the Δx = 0 prepend now happens outside the checkpointed
    region — both are places where a checkpointed run could diverge from a plain
    one without any error being raised."""
    torch.manual_seed(2)
    L, D, R, B = 4, 3, 1, 2
    gg, dtype = SU(2), torch.complex128
    U = torch.stack([random_links(L, D, gg, dtype=dtype) for _ in range(B)])
    T = build_transport_average(U, R, gg)
    W = torch.randn(B, 3, *([L] * D), 2, 2, dtype=dtype)

    grads = []
    for ckpt in (False, True):
        torch.manual_seed(5)
        model = GELT(
            gaugegroup=gg, L=L, D=D, R=R, nhead=2, gemhsa_layers=3, d_qkv=4,
            dtype=dtype, mlp_hidden=8, mlp_out=1, reduction="none",
            mlp_zero_init=False, d_model=6, in_channels=3,
            grad_checkpoint=ckpt,
        )
        model.train()  # checkpointing is active only in training mode
        out = model(W, T)
        out.pow(2).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])

    for plain, checkpointed in zip(*grads):
        assert torch.allclose(plain, checkpointed, atol=1e-12)


# ---------------------------------------------------------------------------
# Ported from the deleted tests/test_blocks.py (which covered the retired
# ``blocks_bias`` variant): shape/finiteness, the front-end ChannelLift, the
# widened residual stream, and the real-valued Z₂ path. They are variant-
# agnostic — nothing here touches the positional encoding — so the guarantees
# move to the one block that remains.
# ---------------------------------------------------------------------------
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
