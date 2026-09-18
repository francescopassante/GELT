"""
=========================================================================
L-CNN — Favoni, Ipp, Müller, Schuh (arXiv:2012.12901).
=========================================================================

Reference implementation of the gauge-equivariant primitives:
  * ``LConv``  — parallel-transport convolution (Eq. 5).
  * ``LBilin`` — site-local matrix-bilinear product (Eq. 6).
  * ``LCB``    — combined L-Conv + L-Bilin block (supp. Eqs. 11–12).
  * ``LAct``   — gauge-equivariant scalar gating activation (Eq. 7).
  * ``Trace``  — gauge-invariant readout (Eq. 10).
  * ``LCNN``   — full model: plaquettes → (L-CB → L-Act)^L → Trace → MLP.

The transport input expected by ``LConv`` / ``LCB`` is the axis-aligned
link-product tensor produced by :func:`build_axis_transports` — distinct
from the L1-ball shortest-path average used by the GELT block.

Conventions match ``gelt/blocks.py``:
  * Adjoint field ``W`` : ``(B, C, *Λ, nc, nc)``.
  * Link tensor ``U``  : ``(B, D, *Λ, nc, nc)``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def build_axis_transports(
    U: torch.Tensor, K: int, gaugegroup
) -> torch.Tensor:
    """Axis-aligned link products ``U^(k)_μ(x)`` for ``k ∈ [1, K]``.

    For each direction ``μ`` and shift ``k``,

        U^(k)_μ(x) = U_μ(x) · U_μ(x + μ̂) · … · U_μ(x + (k-1)·μ̂)

    is the parallel-transport matrix from ``x`` to ``x + k·μ̂``. The
    ``k = 0`` slot is the identity and is handled implicitly inside
    ``LConv`` (no transport, current-site lookup), so it is omitted here.

    Parameters
    ----------
    U : ``(B, D, *Λ, nc, nc)`` batched link tensor.
    K : maximum shift (kernel half-size).

    Returns
    -------
    ``(B, D, K, *Λ, nc, nc)`` tensor with ``out[b, μ, k-1, x] = U^(k)_μ(x)``.
    """
    B, D = U.shape[0], U.shape[1]
    per_mu = []
    for mu in range(D):
        U_mu = U[:, mu]  # (B, *Λ, nc, nc)
        axis = 1 + mu  # μ-th spatial axis on (B, *Λ, nc, nc)
        cum = U_mu
        slabs = [cum]
        for k in range(1, K):
            # U_μ(x + k·μ̂) at index x ⇒ torch.roll along the μ-axis by -k.
            shifted = torch.roll(U_mu, shifts=-k, dims=axis)
            cum = cum @ shifted
            slabs.append(cum)
        per_mu.append(torch.stack(slabs, dim=1))  # (B, K, *Λ, nc, nc)
    return torch.stack(per_mu, dim=1)  # (B, D, K, *Λ, nc, nc)


def _augment(W: torch.Tensor, gaugegroup) -> torch.Tensor:
    """Channel augmentation: ``C → C' = 2C + 1`` (identity + W + W†)."""
    B, C = W.shape[0], W.shape[1]
    spatial = W.shape[2:-2]
    nc = W.shape[-1]
    identity = (
        torch.eye(nc, dtype=W.dtype, device=W.device)
        .view(1, 1, *([1] * len(spatial)), nc, nc)
        .expand(B, 1, *spatial, nc, nc)
    )
    return torch.cat([identity, W, gaugegroup.dagger(W)], dim=1)


def _fold_channels_to_columns(X):
    """``(B, C, *Λ, nc, nc)`` → ``(B, *Λ, nc, C·nc)``: channels become columns."""
    B, C = X.shape[0], X.shape[1]
    spatial, nc = X.shape[2:-2], X.shape[-1]
    return X.movedim(1, -2).reshape(B, *spatial, nc, C * nc)


def _unfold_columns_to_rows(Y, C):
    """``(B, *Λ, nc, C·nc)`` → ``(B, *Λ, C·nc, nc)``: columns become rows."""
    B = Y.shape[0]
    spatial, nc = Y.shape[1:-2], Y.shape[-2]
    return Y.reshape(B, *spatial, nc, C, nc).movedim(-2, -3).reshape(
        B, *spatial, C * nc, nc
    )


def transport_adjoint(T, X, gaugegroup):
    """``T · X_c · T†`` for every channel ``c`` of ``X``.

    ``T`` : ``(B, *Λ, nc, nc)``; ``X`` : ``(B, C, *Λ, nc, nc)``.

    The channel axis is folded into the matrix dimension, so each side is one
    batched ``(nc × nc)·(nc × C·nc)`` product per site instead of ``C`` separate
    ``nc × nc`` ones. At ``nc = 2`` that is the difference between a few hundred
    thousand BLAS rows and tens of millions of them — the same folding the
    GEMHSA hot path uses (notes/performance_audit.md §3.4).
    """
    B, C = X.shape[0], X.shape[1]
    spatial, nc = X.shape[2:-2], X.shape[-1]
    Y = T @ _fold_channels_to_columns(X)
    Y = _unfold_columns_to_rows(Y, C) @ gaugegroup.dagger(T)
    return Y.reshape(B, *spatial, C, nc, nc).movedim(-3, 1)


class LConv(nn.Module):
    """L-Conv (Eq. 5): trainable parallel-transport convolution.

        W_{x,i}^{out} = Σ_{j, s} ω[i, j, s]
                        · T_s(x) · W_{aug,j}(x + Δ_s) · T_s†(x)

    Operates on the augmented input ``W_aug = [1, W, W†]`` (``C → 2C + 1``).
    The shift index ``s`` runs over the local term ``Δ = 0`` (one slot, as in
    the reference implementation — the earlier per-μ duplication of it was
    harmless but bought nothing) and then, per axis μ and hop ``k = 1 … K``,
    ``Δ = +k·μ̂`` and — when ``symmetric`` — ``Δ = −k·μ̂``.

    **The backward hops are not optional.** ``lge-cnn``'s own ``LConv`` and
    ``LConvBilin`` both loop over both orientations, and the W† channels of the
    augmentation do *not* stand in for them: daggering commutes with the
    adjoint transport, so a W† channel is the dagger of a *forward*-transported
    matrix, never a backward-transported one. With ``symmetric=False`` a layer
    sees only x … x + K·μ̂ and a stack sees a one-sided cone.

    The transport is applied to the ``C`` raw channels and the augmentation is
    formed afterwards, which is exact for the same reason: ``T·W†·T† =
    (T·W·T†)†`` and ``T·1·T† = 1``. Transporting the augmented ``2C + 1``
    channels instead would repeat every product twice and transport the
    identity for nothing.

    ``normalize_shifts`` is the **M2 control** of ``notes/m1_probe.md``: it
    reparameterises ω as ``g[i,j] · ω̂[i,j,s]`` with ``Σ_s |ω̂[i,j,s]| = 1``, so
    the aggregation *over offsets* is bounded exactly the way a softmax is,
    while the per-channel magnitude ``g`` stays free — which is also what GELT
    does (α sums to one over offsets; V and the channel mix carry the scale).
    It is off by default: the reference L-Conv has unbounded offset weights, and
    that is the property under test (``notes/where_attention_can_win.md`` §1,
    M2).
    """

    def __init__(
        self,
        gaugegroup,
        c_in: int,
        c_out: int,
        D: int,
        K: int,
        dtype: torch.dtype = torch.complex64,
        symmetric: bool = True,
        normalize_shifts: bool = False,
        conv_init_scale: float = 1.0,
    ):
        super().__init__()
        self.gaugegroup = gaugegroup
        self.D = D
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        self.symmetric = symmetric
        self.normalize_shifts = normalize_shifts
        self.c_prime = 2 * c_in + 1  # augmented input width
        # 1 local term + K hops per axis, in one or both orientations.
        self.n_shifts = 1 + D * K * (2 if symmetric else 1)

        # ω[i, j, s]. Variance ~ 1 / (c_prime · n_shifts) so the L-Conv output
        # starts at unit scale regardless of fan-in.
        #
        # ``conv_init_scale`` multiplies that, and exists because unit scale per
        # *layer* is not unit scale per *stack* at nc = 1. The L-Act gate is
        # ``g(Re Tr W/nc)·W``: for SU(2) the averaged trace of a 2×2 matrix is
        # small and the gate damps, but at nc = 1 the "trace" is the entry
        # itself, so the gate squares whatever it is given and the L-Bilin
        # squares it again. Measured on a Z₂ 8³ box, 4 layers, c_hidden = 6,
        # the output magnitude of the reference init:
        #
        #   layers   1        2        3        4
        #   |out|    0.36     3.6      7.8e4    1.4e21      (and inf at 48×24×24)
        #
        # and against the L-Conv weight scale s at 4 layers, three seeds:
        #
        #   s        1.00            0.70            0.50            0.20
        #   |out|    1e21 … 3e28     0.17 … 3e4      0.02 … 0.13     0.01 … 0.13
        #
        # s = 0.5 is the largest value on the stable plateau and puts the Z₂
        # stack at the same output magnitude the SU(2) reference reaches at
        # s = 1 (0.05 … 0.19). It is **not** a handicap on the baseline: it is
        # the initialisation that gives it the same starting scale the other
        # group gets for free. The default is 1.0, so every existing caller is
        # bit-identical. See ``notes/where_attention_can_win.md`` §9.5.
        n_terms = self.c_prime * self.n_shifts
        sigma = conv_init_scale / math.sqrt(n_terms)
        w = torch.randn(c_out, self.c_prime, self.n_shifts, dtype=dtype) * sigma
        self.w = nn.Parameter(w)
        if normalize_shifts:
            # The free per-channel magnitude, initialised to the L1 norm the
            # row already has, so the bounded arm's init distribution is the
            # reference one and only the *reparameterisation* differs.
            real_dtype = torch.empty(0, dtype=dtype).real.dtype
            self.w_scale = nn.Parameter(
                w.abs().sum(dim=-1).to(real_dtype)
            )

    def kernel(self):
        """ω as the contraction below should see it.

        The identity with ``self.w`` unless ``normalize_shifts`` is set, in
        which case the offset axis is L1-normalised and rescaled by the free
        per-channel magnitude ``w_scale`` — see the class docstring.
        """
        if not self.normalize_shifts:
            return self.w
        norm = self.w.abs().sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return self.w / norm * self.w_scale.unsqueeze(-1)

    def shifted_terms(self, W, U_transports):
        """The ``n_shifts`` transported copies of ``W``: ``(B, C, S, *Λ, nc, nc)``."""
        gg = self.gaugegroup
        slabs = [W]  # s = 0: the local term, no transport
        for mu in range(self.D):
            w_axis = 2 + mu  # μ-th lattice axis of (B, C, *Λ, nc, nc)
            t_axis = 1 + mu  # ... and of (B, *Λ, nc, nc)
            for k in range(1, self.K + 1):
                # U^(k)_μ(x): transports x + k·μ̂ back to x.
                Uk = U_transports[:, mu, k - 1]
                slabs.append(
                    transport_adjoint(Uk, torch.roll(W, -k, dims=w_axis), gg)
                )
                if self.symmetric:
                    # T_{−k,μ}(x) = U^(k)†_μ(x − k·μ̂) — the same link product
                    # read at the far end of the backward hop.
                    Vk = gg.dagger(torch.roll(Uk, k, dims=t_axis))
                    slabs.append(
                        transport_adjoint(Vk, torch.roll(W, k, dims=w_axis), gg)
                    )
        return torch.stack(slabs, dim=2)

    def forward(self, W: torch.Tensor, U_transports: torch.Tensor) -> torch.Tensor:
        """Run the L-Conv.

        ``W`` : ``(B, C_in, *Λ, nc, nc)``.
        ``U_transports`` : ``(B, D, K, *Λ, nc, nc)`` — k=1..K (the local slot
        needs no transport and is handled here).
        """
        B = W.shape[0]
        spatial, nc = W.shape[2:-2], W.shape[-1]
        U_transports = U_transports.to(W.dtype)

        S = self.shifted_terms(W, U_transports)  # (B, C, n_shifts, *Λ, nc, nc)
        S_flat = S.reshape(B, self.c_in * self.n_shifts, -1)

        # The augmentation is never materialised: the W channels contract with
        # ω directly, and the W† channels through
        #   Σ_j ω_j · S_j†  =  (Σ_j conj(ω_j) · S_j)† ,
        # so both halves are one GEMM and a single dagger of the *output* (c_out
        # channels) instead of one of the input (c_in · n_shifts channels).
        n = self.c_in * self.n_shifts
        w = self.kernel()
        w_S = w[:, 1 : 1 + self.c_in].reshape(self.c_out, n)
        w_Sd = w[:, 1 + self.c_in :].conj().reshape(self.c_out, n)
        mixed = torch.matmul(torch.cat([w_S, w_Sd], dim=0), S_flat)
        mixed = mixed.reshape(B, 2 * self.c_out, *spatial, nc, nc)
        out = mixed[:, : self.c_out] + self.gaugegroup.dagger(mixed[:, self.c_out :])

        # The identity channel of the augmentation transports to itself, so it
        # contributes Σ_s ω[i, 0, s] · 1 — a per-channel bias matrix.
        bias = w[:, 0].sum(dim=-1)  # (c_out,)
        identity = torch.eye(nc, dtype=W.dtype, device=W.device)
        return out + bias.view(1, -1, *([1] * len(spatial)), 1, 1) * identity


class LBilin(nn.Module):
    """L-Bilin (Eq. 6): site-local matrix-bilinear product.

        W_{x,i}^{out} = Σ_{j, j'} α[i, j, j'] · W_{aug,j}(x) · W'_{aug,j'}(x)

    Both inputs are augmented (``[1, ·, ·†]``) before the bilinear. The
    identity-channel of the augmentation gives every L-Bilin a free linear
    / residual / bias path; the dagger half gives orientation reversal.

    The channel-pair product is never materialised. Following the reference
    implementation's ``bilin_implementation = 2`` ("the good one"), α contracts
    the *right* channel axis first,

        tmp[i, j](x) = Σ_{j'} α[i, j, j'] · W'_{aug,j'}(x)   (one GEMM)
        out[i](x)    = Σ_j  W_{aug,j}(x) · tmp[i, j](x)      (one batched matmul)

    so the ``(c_left × c_right)`` matrices per site of the naive order — 169 of
    them at the shootout's width, 1.3 GiB per block at the production batch —
    never exist. The left channel axis is folded into the contraction instead.
    """

    def __init__(
        self,
        gaugegroup,
        c_in_left: int,
        c_in_right: int,
        c_out: int,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.gaugegroup = gaugegroup
        self.c_out = c_out
        self.c_left = 2 * c_in_left + 1
        self.c_right = 2 * c_in_right + 1

        sigma = 1.0 / math.sqrt(self.c_left * self.c_right)
        self.w = nn.Parameter(
            torch.randn(c_out, self.c_left, self.c_right, dtype=dtype) * sigma
        )

    def forward(self, W_left: torch.Tensor, W_right: torch.Tensor) -> torch.Tensor:
        B = W_left.shape[0]
        spatial, nc = W_left.shape[2:-2], W_left.shape[-1]
        L_aug = _augment(W_left, self.gaugegroup)   # (B, c_left, *Λ, nc, nc)
        R_aug = _augment(W_right, self.gaugegroup)  # (B, c_right, *Λ, nc, nc)

        # tmp[i, j] = Σ_{j'} α[i, j, j'] · R_aug[j'] — one GEMM over the right
        # channel axis, output (B, c_out·c_left, *Λ, nc, nc).
        w_flat = self.w.reshape(self.c_out * self.c_left, self.c_right)
        tmp = torch.matmul(w_flat, R_aug.reshape(B, self.c_right, -1))
        tmp = tmp.reshape(B, self.c_out, self.c_left, *spatial, nc, nc)

        # out[i] = Σ_j L_aug[j] · tmp[i, j]: fold (j, colour) into one
        # contraction axis so this is a single batched matmul.
        L_f = _fold_channels_to_columns(L_aug).unsqueeze(1)  # (B,1,*Λ,nc,c_left·nc)
        tmp_f = tmp.movedim(2, -3).reshape(
            B, self.c_out, *spatial, self.c_left * nc, nc
        )
        return (L_f @ tmp_f).reshape(B, self.c_out, *spatial, nc, nc)


class LCB(nn.Module):
    """Combined L-Conv + L-Bilin block.

    ``W' = LConv(W, U_transports)`` produces the transported and channel-
    mixed intermediate; ``LBilin(W, W')`` multiplies it on-site against the
    original W. This is the workhorse block of the paper — every L-Conv in
    their networks is paired with an L-Bilin, and the combined block
    doubles the maximum loop area per layer (supp. Eq. 15).
    """

    def __init__(
        self,
        gaugegroup,
        c_in: int,
        c_out: int,
        D: int,
        K: int,
        dtype: torch.dtype = torch.complex64,
        symmetric: bool = True,
        normalize_shifts: bool = False,
        conv_init_scale: float = 1.0,
    ):
        super().__init__()
        self.lconv = LConv(
            gaugegroup, c_in, c_out, D, K, dtype=dtype, symmetric=symmetric,
            normalize_shifts=normalize_shifts, conv_init_scale=conv_init_scale,
        )
        self.lbilin = LBilin(gaugegroup, c_in, c_out, c_out, dtype=dtype)

    def forward(self, W: torch.Tensor, U_transports: torch.Tensor) -> torch.Tensor:
        W_transp = self.lconv(W, U_transports)
        return self.lbilin(W, W_transp)


class LAct(nn.Module):
    """L-Act (Eq. 7): gauge-equivariant scalar gating.

        W_{x,i} → g(W_{x,i}) · W_{x,i}

    with ``g(W) = activation(Re Tr W / nc)``. Multiplying a covariant
    matrix by a gauge-invariant scalar preserves covariance.
    """

    def __init__(self, activation: str = "relu"):
        super().__init__()
        if activation not in ("relu", "softplus"):
            raise ValueError(
                f"activation must be 'relu' or 'softplus', got {activation!r}"
            )
        self.activation = activation

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        nc = W.shape[-1]
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc  # (B, C, *Λ)
        g = F.relu(trace) if self.activation == "relu" else F.softplus(trace)
        return g.unsqueeze(-1).unsqueeze(-1) * W


class Trace(nn.Module):
    """Gauge-invariant trace readout: ``(B, C, *Λ, nc, nc) → (B, 2C, *Λ)``."""

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, C, *Λ)
        imag = trace.imag if trace.is_complex() else torch.zeros_like(trace)
        return torch.cat([trace.real, imag], dim=1)  # (B, 2C, *Λ)


class LCNN(nn.Module):
    """Favoni et al L-CNN.

    Pipeline:
      1. Plaquette input ``(B, C_in, *Λ, nc, nc)`` with ``C_in = D(D-1)/2``
         (already built by the dataset) — or ``C_in = in_channels`` when a
         stacked multi-level input is passed instead.
      2. Stack of L-CB blocks (optionally followed by L-Act), each
         consuming the precomputed axis-aligned transports
         ``U^(k)_μ(x)`` from :func:`build_axis_transports`.
      3. ``Trace`` to extract gauge-invariant scalars per site.
      4. Per-site MLP head (one hidden layer). The paper explicitly avoids
         global average pooling at the end; supervision is per-site and
         the spatial reduction is applied last.
      5. Optional spatial reduction (``"sum"`` / ``"mean"`` / ``"none"``)
         to match the GELT model's reduction modes.

    ``normalize_shifts`` bounds every L-Conv's aggregation over offsets — the
    M2 control arm of ``notes/m1_probe.md`` (see :class:`LConv`).

    ``in_channels``, ``init_scale`` and ``grad_checkpoint`` are the knobs the
    matched-parameter glueball shootout needs; all three default to the
    reference behaviour.
    """

    def __init__(
        self,
        gaugegroup,
        L: int,
        D: int,
        K: int,
        c_hidden: int,
        n_layers: int,
        dtype: torch.dtype = torch.complex64,
        mlp_hidden: int = 32,
        mlp_out: int = 1,
        reduction: str = "sum",
        use_l_act: bool = True,
        gate: str = "softplus",
        in_channels: int | None = None,
        init_scale: float = 1.0,
        grad_checkpoint: bool = False,
        symmetric: bool = True,
        normalize_shifts: bool = False,
        conv_init_scale: float = 1.0,
    ):
        super().__init__()
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(
                f"reduction must be 'sum', 'mean', or 'none', got {reduction!r}"
            )
        self.reduction = reduction
        self.K = K
        self.D = D
        self.gaugegroup = gaugegroup

        # Input width. The default D(D-1)/2 is the plain plaquette input the
        # dataset builders produce; `in_channels` overrides it for a stacked
        # multi-level input (the glueball task feeds 3 spatial-plaquette
        # channels per APE smearing level, exactly as GELT does — matching the
        # *inputs* is what keeps the shootout a comparison of architectures).
        c_in_plaq = D * (D - 1) // 2 if in_channels is None else in_channels
        self.c_in_plaq = c_in_plaq
        self.c_hidden = c_hidden
        self.grad_checkpoint = grad_checkpoint

        # Stack of L-CB (+ L-Act). The first block maps C_in_plaq → c_hidden;
        # subsequent blocks keep the width at c_hidden.
        widths = [c_in_plaq] + [c_hidden] * n_layers
        self.lcb_blocks = nn.ModuleList(
            [
                LCB(
                    gaugegroup, widths[i], widths[i + 1], D, K, dtype=dtype,
                    symmetric=symmetric, normalize_shifts=normalize_shifts,
                    conv_init_scale=conv_init_scale,
                )
                for i in range(n_layers)
            ]
        )
        self.l_acts = nn.ModuleList(
            [LAct(activation=gate) if use_l_act else nn.Identity() for _ in range(n_layers)]
        )

        # Real-valued per-site head (Trace produces 2·c_hidden reals per site).
        # float64 in, float64 out: a real model (Z₂, where nc = 1) is built at
        # torch.float64 for the high-precision gauge tests, and mapping it to a
        # float32 head would fail the matmul outright.
        real_dtype = (
            torch.float64
            if dtype in (torch.complex128, torch.float64)
            else torch.float32
        )
        self.trace = Trace()
        self.head_fc1 = nn.Linear(2 * c_hidden, mlp_hidden).to(real_dtype)
        self.head_fc2 = nn.Linear(mlp_hidden, mlp_out).to(real_dtype)
        # Output scale knob, mirroring GELT's `init_scale`. The reference init
        # is scale-agnostic because the paper's losses are supervised; a
        # Rayleigh objective is invariant under O → λO, so only the scale pin
        # (train_glueball.SCALE_REG) sees λ — this exists so the pin does not
        # have to travel decades before the ratio terms dominate the gradient.
        if init_scale != 1.0:
            with torch.no_grad():
                self.head_fc2.weight.mul_(init_scale)
                self.head_fc2.bias.mul_(init_scale)

    def forward(self, W: torch.Tensor, U_transports: torch.Tensor) -> torch.Tensor:
        """W : ``(B, C_in_plaq, *Λ, nc, nc)`` — plaquettes.
        U_transports : ``(B, D, K, *Λ, nc, nc)`` — axis-aligned link products.
        """
        # Cast inputs to model dtype once (mirrors GELT.forward).
        w_dtype = self.lcb_blocks[0].lconv.w.dtype
        if W.dtype != w_dtype:
            W = W.to(w_dtype)
        if U_transports.dtype != w_dtype:
            U_transports = U_transports.to(w_dtype)

        for lcb, act in zip(self.lcb_blocks, self.l_acts):
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # Same trade as GELT.attn: recompute the block in backward
                # rather than store it. L-Bilin's (c_left, c_right) outer
                # product is the memory wall here — 13×13 channel pairs per
                # site at c_hidden = 6, 25×13 in the first block — so at the
                # glueball task's B = configs·Lt this is what makes batch 6
                # fit. use_reentrant=False: U_transports carries no grad.
                W = checkpoint(lcb, W, U_transports, use_reentrant=False)
            else:
                W = lcb(W, U_transports)
            W = act(W)

        trace = self.trace(W).movedim(1, -1)  # (B, *Λ, 2·c_hidden)
        h = F.relu(self.head_fc1(trace))
        site_out = self.head_fc2(h).squeeze(-1)  # (B, *Λ)

        if self.reduction == "none":
            return site_out
        spatial_dims = tuple(range(1, site_out.ndim))
        if self.reduction == "sum":
            return site_out.sum(dim=spatial_dims)
        return site_out.mean(dim=spatial_dims)
