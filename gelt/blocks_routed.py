"""
=========================================================================
Routed gauge-equivariant operator algebra (notes/codex_architecture.md §5).
=========================================================================

This module implements the *three-branch* architecture described in
``notes/codex_architecture.md`` §5 — the proposed direction after the
fused-``GEMHSA`` and split ``GEAttention + GEFFN`` diagnoses (§§2-4).
Each block has three parallel branches that share the augmented input
and a precomputed transport bank, but compute genuinely different
operator-algebra primitives:

* **Branch A — algebraic transport mixing (L-Conv)**. For every offset
  ``r`` in the (signed) L1-ball of radius ``R`` (including ``r = 0``),
  transport the augmented field and combine with a learned signed/
  complex linear map ``M_A[c, r, c']``:

      ``A(x)_c = Σ_{r, c'} M_A[c, r, c'] · T(x,r) · W'(x+r)_{c'} · T†(x,r)``.

  Mathematically this is exactly Favoni et al.'s L-Conv (codex §5,
  "honest framing"). Branch A's job is to provide a reliable algebraic
  basis of transported features at every site — without making the
  optimizer discover it from softmax routing alone.

* **Branch B — soft attention routing**. Same transported value bank,
  but weighted by gauge-invariant softmax attention scores
  ``Re Σ_c Tr[Q_c† · K̃_c]``. No ``Q†`` factor in the value path —
  this is pure routing, like the ``GEAttention`` block of the split
  architecture (§§4.1). Adaptive component on top of Branch A's basis.

* **Branch C — local bilinear / multilinear FFN (L-Bilin)**. Purely
  local product of projected covariant channels at every site:

      ``C(x) = w_mix · ( Q_b†(x) · V_b(x) [ · V2_b(x) ] )``.

  Degree-2 (bilinear) by default; ``trilinear=True`` switches to the
  degree-3 product ``Q_b† · V_b · V2_b`` (codex §5: "Consider local
  multilinear FFNs, e.g. ``Q† V V2``, for faster growth of operator
  degree when needed"). This is L-Bilin.

The block's residual update is

    ``W ← W + α · (g(W_res) · W_res − W)``, with
    ``W_res = W + α_A · A(W) + α_B · B(W) + α_C · C(W)``,

where ``g(·)`` is the L-Act gate, ``α`` is an outer ReZero scalar
(identity-at-init when α=0) and ``α_A, α_B, α_C`` are per-branch
ReZero scalars so each branch can engage independently. The decoupled
α's are what the codex notes argue for in §6: "SGD is not
reparameterization-invariant" — additive parallel branches break the
multiplicative coupling that traps the fused stack.

``ChannelLift``, ``Trace`` and ``MLP`` are reused verbatim from
``gelt.blocks``; the transport helper ``_transport_adjoint`` and the
augmentation / L-Act gate / neighbour index helpers come from
``gelt.blocks_new``.
"""

import math

import torch
import torch.nn as nn

from gelt.blocks import ChannelLift, MLP, Trace
from gelt.blocks_new import (
    _augment,
    _build_nbr_idx,
    _l_act_gate,
    _transport_adjoint,
)
from gelt.lattice import l1_ball_offsets


class RoutedBlock(nn.Module):
    """One layer of the three-branch routed operator algebra (§5).

    Branches share the augmented input ``W'`` (channel augmentation
    ``C → 2C + 1`` with identity + daggers, like every other block in
    this codebase) and the L1-ball offset set ``{Δx : 0 ≤ |Δx|_1 ≤ R}``
    — including the **zero offset** (codex §5, "Include the zero offset.
    The model needs an explicit identity transport path"). The transport
    tensor ``T`` arriving from the dataset builder carries only the
    non-zero offsets; the identity slot is synthesised inside ``forward``.

    Per-branch ReZero scalars ``α_A, α_B, α_C`` plus the outer ReZero
    ``α`` make the block exactly the identity at full zero-init and let
    each branch engage independently as training progresses. Default
    inits follow the codex recipe: warm-start ``α_A`` (algebraic basis
    is the "reliable backbone"), keep ``α_B`` smaller (attention layers
    on top), and ``α_C`` at the same warm start as ``α_A`` because the
    multiplicative path is what actually grows operator degree.
    """

    def __init__(
        self,
        gaugegroup,
        L,
        D,
        R,
        d_input,
        nhead,
        d_qkv=None,
        gate="softplus",
        dtype=torch.complex64,
        alpha_init: float = 0.0,
        alpha_A_init: float = 1.0,
        alpha_B_init: float = 0.0,
        alpha_C_init: float = 1.0,
        init_scale: float = 1.0,
        trilinear: bool = False,
        constructive_A: bool = True,
    ):
        super().__init__()
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate}")

        self.gaugegroup = gaugegroup
        self.D = D
        self.R = R
        self.H = nhead
        self.C = d_input
        self.d_qkv = d_input // nhead if d_qkv is None else d_qkv
        if self.d_qkv < 1:
            raise ValueError(
                f"d_qkv must be >= 1, got {self.d_qkv} "
                f"(d_input={d_input}, nhead={nhead}). "
                f"Pass d_qkv explicitly when d_input < nhead — this happens "
                f"e.g. in D=2, where d_input = D(D-1)/2 = 1."
            )
        self.gate = gate
        self.trilinear = trilinear

        # Offsets: zero self-offset PREPENDED so the algebraic transport
        # branch always has an explicit identity route (codex §5). The
        # external transport tensor only carries the non-zero offsets;
        # the identity slot is synthesised in ``forward``.
        self.offsets = [tuple([0] * D)] + l1_ball_offsets(D, R)
        self.n_offsets = len(self.offsets)
        # _nbr_idx[d, i, x] = (x[d] + Δx_i[d]) mod L. Periodic gather.
        self.register_buffer("_nbr_idx", _build_nbr_idx(self.offsets, L, D))

        # Augmented channel count: C -> C' = 2C + 1 (identity + daggers).
        self.C_prime = 2 * d_input + 1
        # Cached on-site identity for ``_augment``: (1, 1, *[1]*D, nc, nc).
        nc = gaugegroup.nc
        identity = torch.eye(nc, dtype=dtype).view(1, 1, *([1] * D), nc, nc)
        self.register_buffer("_identity", identity)

        # -----------------------------------------------------------------
        # Branch A — L-Conv (algebraic transport mixing).
        # w_A has shape (C_out, n_offsets, C_in_aug). Output channel c is
        #     A(x)_c = Σ_{r, c'} w_A[c, r, c'] · W̃[c', r](x)
        # with W̃[c', r](x) = T(x,r) · W'(x+r)_{c'} · T†(x,r).
        # -----------------------------------------------------------------
        if constructive_A:
            # Constructive init (codex §5 last bullet): start with single-
            # offset, single-channel basis routes — output channel c picks
            # one (offset, augmented-channel) pair (round-robin over the
            # n_offsets · C' basis). With α_A warm-started > 0 this gives
            # the model an L-CNN basis from epoch 0 rather than asking the
            # optimizer to discover transport routes from random init.
            w_A = torch.zeros(d_input, self.n_offsets, self.C_prime, dtype=dtype)
            basis_size = self.n_offsets * self.C_prime
            for c in range(d_input):
                flat_idx = c % basis_size
                r_idx, k_idx = divmod(flat_idx, self.C_prime)
                w_A[c, r_idx, k_idx] = 1.0
        else:
            sigma_A = 0.02 * init_scale / math.sqrt(self.n_offsets * self.C_prime)
            w_A = torch.randn(d_input, self.n_offsets, self.C_prime, dtype=dtype) * sigma_A
        self.w_A = nn.Parameter(w_A)

        # -----------------------------------------------------------------
        # Branch B — soft attention routing.
        # Fused QKV projection over the augmented channels, then gather +
        # transport K, V; score is Frobenius product, softmax over offsets;
        # value path is the attention-weighted sum of transported V's
        # (no Q† factor — that lives in Branch C).
        # -----------------------------------------------------------------
        sigma = 0.02 * init_scale / math.sqrt(self.C_prime)
        self.w_QKV_B = nn.Parameter(
            torch.randn(3, self.H, self.d_qkv, self.C_prime, dtype=dtype) * sigma
        )
        # Per-(head, offset) score bias. Untied across the lattice rotation/
        # reflection symmetries — the rectangular Wilson-loop targets need
        # axis-selective routing (same rationale as ``GEMHSA.b_h``).
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_offsets))
        self._bias_view_shape = (1, self.H, self.n_offsets) + (1,) * D
        # Mix branch-B output back to the C-channel residual stream.
        sigma_mix = 0.02 * init_scale / math.sqrt(self.H * self.d_qkv)
        self.w_mix_B = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

        # -----------------------------------------------------------------
        # Branch C — local L-Bilin (optionally trilinear).
        # Project to Q_b, V_b (and V2_b if trilinear) at every site, then
        # multiply at the same anchor: Q_b†(x) · V_b(x) [ · V2_b(x) ].
        # No transport, no neighbours, no softmax — pure local L-Bilin.
        # -----------------------------------------------------------------
        n_factors = 3 if trilinear else 2
        self.w_QV_C = nn.Parameter(
            torch.randn(n_factors, self.H, self.d_qkv, self.C_prime, dtype=dtype) * sigma
        )
        self.w_mix_C = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

        # -----------------------------------------------------------------
        # Per-branch ReZero scalars + outer ReZero gate.
        # alpha (outer) gates the whole (W_act − W) update so the block is
        # bit-exactly W → W at alpha=0. The per-branch α_A/B/C control the
        # mixture inside W_res = W + α_A·A + α_B·B + α_C·C; warm-starting
        # them > 0 engages each branch from epoch 0 (codex §6: additive
        # parallel branches break the multiplicative-coupling chicken-and-
        # egg problem of the fused stack).
        # -----------------------------------------------------------------
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        self.alpha_A = nn.Parameter(torch.full((1,), float(alpha_A_init)))
        self.alpha_B = nn.Parameter(torch.full((1,), float(alpha_B_init)))
        self.alpha_C = nn.Parameter(torch.full((1,), float(alpha_C_init)))

    # ---------------------------------------------------------------------
    # Shared transport-bank helper. Transports the AUGMENTED field per
    # offset (including the zero/self offset). Reused by branches A and B.
    # ---------------------------------------------------------------------
    def _transport_bank(self, W_aug, T, T_dag):
        """Transport ``W_aug`` for every offset (zero offset included).

        Returns ``(B, C', n_offsets, *Λ, nc, nc)``.
        """
        # Periodic neighbour gather along the L1-ball offsets.
        idx = tuple(self._nbr_idx[k] for k in range(self.D))
        # nb_indexer = (slice, slice) over (B, C'), then idx tuples over *Λ,
        # then (slice, slice) over (nc, nc).
        nb_indexer = (slice(None),) * 2 + idx + (slice(None), slice(None))
        W_nb = W_aug[nb_indexer]  # (B, C', n_off, *Λ, nc, nc)

        # ``_transport_adjoint`` expects (B, H, d, n_off, *Λ, nc, nc).
        # Treat C' as the head axis and d=1; squeeze back after.
        W_nb_h = W_nb.unsqueeze(2)  # (B, C', 1, n_off, *Λ, nc, nc)
        W_tilde = _transport_adjoint(W_nb_h, T, T_dag, self.n_offsets)
        return W_tilde.squeeze(2)  # (B, C', n_off, *Λ, nc, nc)

    # ---------------------------------------------------------------------
    # Branch A — algebraic L-Conv mixing over (offset, channel).
    # ---------------------------------------------------------------------
    def branch_A(self, W_tilde, trailing):
        """A(x)_c = Σ_{r, c'} w_A[c, r, c'] · W̃[c', r](x).

        Implemented as a single ``(C, n_off·C') @ (B, n_off·C', |Λ|·nc·nc)``
        matmul. ``W_tilde`` is (B, C', n_off, *Λ, nc, nc); reorder to
        (B, n_off, C', ...) → flatten so the contraction is one matmul.
        """
        B = W_tilde.shape[0]
        # (B, C', n_off, *Λ, nc, nc) → (B, n_off, C', *Λ, nc, nc).
        W_p = W_tilde.permute(0, 2, 1, *range(3, W_tilde.ndim))
        flat = W_p.reshape(B, self.n_offsets * self.C_prime, -1)
        w_A_flat = self.w_A.reshape(self.C, self.n_offsets * self.C_prime)
        A_out = torch.matmul(w_A_flat, flat)
        return A_out.view(B, self.C, *trailing)

    # ---------------------------------------------------------------------
    # Branch B — soft attention routing.
    # ---------------------------------------------------------------------
    def branch_B(self, W_aug, W_tilde, trailing):
        """Attention-weighted sum of transported V's (no Q† factor).

        ``W_tilde`` is the shared transport bank (B, C', n_off, *Λ, nc, nc);
        we project Q on-site from ``W_aug`` and K, V from the same bank
        by linearly mixing the augmented channels.
        """
        B = W_aug.shape[0]

        # On-site Q from W_aug: (3·H·d, C') @ (B, C', N). Slice Q only here;
        # K and V are computed from W_tilde (already-transported neighbour
        # bank) to avoid transporting Q (which is anchored at x and never
        # moved). This is equivalent to projecting Q/K/V on W_aug and
        # transporting K, V afterwards, but cheaper: K and V are read
        # directly from W_tilde.
        W_aug_flat = W_aug.view(B, self.C_prime, -1)
        w_Q_flat = self.w_QKV_B[0].reshape(self.H * self.d_qkv, self.C_prime)
        Q = torch.matmul(w_Q_flat, W_aug_flat).view(B, self.H, self.d_qkv, *trailing)

        # Project K, V from the transport bank. W_tilde: (B, C', n_off, *Λ, nc, nc).
        # Reorder to (B, C', n_off·|Λ|·nc·nc) for the per-channel linear mix,
        # then reshape back.
        nc = trailing[-1]
        spatial = trailing[:-2]
        W_tilde_flat = W_tilde.reshape(B, self.C_prime, -1)
        w_K_flat = self.w_QKV_B[1].reshape(self.H * self.d_qkv, self.C_prime)
        w_V_flat = self.w_QKV_B[2].reshape(self.H * self.d_qkv, self.C_prime)
        K_tilde = torch.matmul(w_K_flat, W_tilde_flat).view(
            B, self.H, self.d_qkv, self.n_offsets, *spatial, nc, nc
        )
        V_tilde = torch.matmul(w_V_flat, W_tilde_flat).view(
            B, self.H, self.d_qkv, self.n_offsets, *spatial, nc, nc
        )

        # Frobenius-product score: Re Σ_c Tr[Q_c† · K̃_c] / √(d_qkv · nc).
        Q_e = Q.unsqueeze(3)  # (B, H, d_qkv, 1, *Λ, nc, nc)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)
        bias = self.b_h.real if self.b_h.is_complex() else self.b_h
        score = score + bias.view(self._bias_view_shape)

        # Softmax over the offset axis.
        alpha = torch.softmax(score, dim=2)
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        # Pure-attention value (no Q† factor — that's Branch C's job).
        V_weighted = (alpha_b * V_tilde).sum(dim=3)  # (B, H, d_qkv, *Λ, nc, nc)

        # Channel mix back to C output channels.
        HD = self.H * self.d_qkv
        out_flat = V_weighted.reshape(B, HD, -1)
        w_mix_flat = self.w_mix_B.view(self.C, HD)
        return torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing)

    # ---------------------------------------------------------------------
    # Branch C — local L-Bilin (optionally trilinear).
    # ---------------------------------------------------------------------
    def branch_C(self, W_aug, trailing):
        """Local product ``Q_b†(x) · V_b(x) [ · V2_b(x) ]`` per site."""
        B = W_aug.shape[0]
        W_aug_flat = W_aug.view(B, self.C_prime, -1)
        n_factors = 3 if self.trilinear else 2
        w_QV_flat = self.w_QV_C.view(n_factors * self.H * self.d_qkv, self.C_prime)
        QV = torch.matmul(w_QV_flat, W_aug_flat)
        QV = QV.view(B, n_factors, self.H, self.d_qkv, *trailing)

        if self.trilinear:
            Q_b, V_b, V2_b = QV.unbind(dim=1)
        else:
            Q_b, V_b = QV.unbind(dim=1)
        Q_b_dag = self.gaugegroup.dagger(Q_b)
        W_prime = torch.matmul(Q_b_dag, V_b)
        if self.trilinear:
            W_prime = torch.matmul(W_prime, V2_b)

        # Channel mix back to C output channels.
        HD = self.H * self.d_qkv
        out_flat = W_prime.reshape(B, HD, -1)
        w_mix_flat = self.w_mix_C.view(self.C, HD)
        return torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing)

    # ---------------------------------------------------------------------
    # Forward pass.
    # ---------------------------------------------------------------------
    def forward(self, W, T, T_dag=None):
        """W : (B, C, *Λ, nc, nc).
        T : (B, n_offsets - 1, *Λ, nc, nc) — non-zero offsets only.
        T_dag : optional precomputed dagger of T (shared across stacked
                blocks by ``GELT.attn``).
        """
        expected_external = self.n_offsets - 1
        assert T.shape[1] == expected_external, (
            f"Expected T.shape[1] == {expected_external} (non-zero offsets), "
            f"got {T.shape[1]}"
        )

        nc = W.shape[-1]
        B = T.shape[0]
        spatial = T.shape[2:-2]
        # Synthesise the identity transport for the zero offset and
        # prepend it so T lines up with our self-offset-included
        # offsets list.
        identity_T = (
            torch.eye(nc, dtype=T.dtype, device=T.device)
            .view(1, 1, *([1] * self.D), nc, nc)
            .expand(B, 1, *spatial, nc, nc)
        )
        T = torch.cat([identity_T, T], dim=1)
        if T_dag is None:
            T_dag = self.gaugegroup.dagger(T)
        else:
            T_dag = torch.cat([identity_T, T_dag], dim=1)

        # Channel augmentation: (B, C, *Λ, nc, nc) -> (B, C', *Λ, nc, nc).
        W_aug = _augment(W, self.gaugegroup, self._identity)
        trailing = W_aug.shape[2:]  # (*Λ, nc, nc)

        # Shared transport bank: T(x,r) · W'(x+r) · T†(x,r) for every r,
        # consumed by branches A and B. Branch C is purely local and
        # doesn't need transports.
        W_tilde = self._transport_bank(W_aug, T, T_dag)

        A_out = self.branch_A(W_tilde, trailing)
        B_out = self.branch_B(W_aug, W_tilde, trailing)
        C_out = self.branch_C(W_aug, trailing)

        # Mixed residual update: α_A·A + α_B·B + α_C·C added to W, then
        # L-Act gate, then outer ReZero. At α=0 the block is exactly
        # W → W; non-zero outer α blends in the gated mixed branches.
        W_branch = (
            self.alpha_A * A_out + self.alpha_B * B_out + self.alpha_C * C_out
        )
        W_res = W + W_branch
        W_act = _l_act_gate(W_res, self.gate)
        return W + self.alpha * (W_act - W)


class GELT(nn.Module):
    """Full GELT model with the routed three-branch operator algebra.

    Pipeline:
      1. Plaquette input (built by the dataset builder, ``d_input = D(D-1)/2``).
      2. ``ChannelLift`` to widen the residual stream to ``d_model``.
      3. Stack of ``RoutedBlock``s — each runs all three branches
         (A: L-Conv, B: soft attention, C: local L-Bilin) in parallel
         and combines them with per-branch ReZero scalars plus an outer
         ReZero gate (notes/codex_architecture.md §5).
      4. ``Trace`` to read out the gauge-invariant scalar per site.
      5. ``MLP`` (one hidden layer) to mix the trace features.
      6. Spatial reduction (``"sum"``, ``"mean"``, ``"none"``).

    Knobs that matter:
      * ``alpha_A_init / alpha_B_init / alpha_C_init`` — codex §5 argues
        for warm-starting the algebraic-basis branches (A, C) above the
        attention branch (B). Default: ``α_A = α_C = 1.0``, ``α_B = 0.0``.
        The outer ``alpha_init`` defaults to ``0.0`` (identity-at-init);
        bump to e.g. ``0.5`` to engage the block from epoch 0 (same logic
        as ``train_gelt_diagnosis.py``).
      * ``constructive_A`` (default True) — initialise Branch A's L-Conv
        weights as single-offset, single-channel basis routes so the
        model starts with an L-CNN-like algebraic basis.
      * ``trilinear`` (default False) — switch Branch C to the degree-3
        local product ``Q† · V · V2`` for faster operator-degree growth.
    """

    def __init__(
        self,
        gaugegroup,
        L,
        D,
        R,
        nhead,
        n_layers: int | None = None,
        d_qkv=None,
        gate="softplus",
        dtype=torch.complex64,
        mlp_hidden=32,
        mlp_out=1,
        reduction: str = "sum",
        alpha_init: float = 0.0,
        alpha_A_init: float = 1.0,
        alpha_B_init: float = 0.0,
        alpha_C_init: float = 1.0,
        init_scale: float = 1.0,
        mlp_zero_init: bool = True,
        d_model: int | None = None,
        mlp_dropout: float = 0.0,
        trilinear: bool = False,
        constructive_A: bool = True,
        # back-compat aliases so existing training scripts keep working
        gemhsa_layers: int | None = None,
    ):
        super().__init__()
        d_input = D * (D - 1) // 2
        if d_model is None:
            d_model = d_input
        if d_model < d_input:
            raise ValueError(
                f"d_model must be >= d_input = D(D-1)/2 = {d_input}, got {d_model}."
            )
        self.d_input = d_input
        self.d_model = d_model

        if reduction not in ("sum", "mean", "none"):
            raise ValueError(
                f"reduction must be 'sum', 'mean', or 'none', got {reduction!r}"
            )
        self.reduction = reduction

        if n_layers is None:
            if gemhsa_layers is None:
                raise ValueError("Must specify n_layers (or gemhsa_layers).")
            n_layers = gemhsa_layers

        # ChannelLift: identity-extend init when d_model == d_input.
        self.lift = ChannelLift(d_input, d_model, dtype=dtype)

        self.blocks = nn.ModuleList(
            [
                RoutedBlock(
                    gaugegroup,
                    L,
                    D,
                    R,
                    d_model,
                    nhead,
                    d_qkv=d_qkv,
                    gate=gate,
                    dtype=dtype,
                    alpha_init=alpha_init,
                    alpha_A_init=alpha_A_init,
                    alpha_B_init=alpha_B_init,
                    alpha_C_init=alpha_C_init,
                    init_scale=init_scale,
                    trilinear=trilinear,
                    constructive_A=constructive_A,
                )
                for _ in range(n_layers)
            ]
        )
        # Back-compat alias for the per-layer α diagnostic in
        # ``train_gelt_diagnosis*.py`` — each RoutedBlock exposes ``.alpha``
        # (outer ReZero) and additionally ``.alpha_A/B/C``.
        self.gemhsa_models = self.blocks

        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_model, mlp_hidden, mlp_out, dropout=mlp_dropout).to(real_dtype)
        if mlp_zero_init:
            nn.init.zeros_(self.mlp.fc2.weight)
            nn.init.zeros_(self.mlp.fc2.bias)

    def attn(self, W, T, T_dag):
        """Run the routed block stack. Every block consumes ``T``."""
        for layer in self.blocks:
            W = layer(W, T, T_dag)
        return W

    def forward(self, W, T):
        first_layer = self.blocks[0]
        # All RoutedBlocks expose w_QKV_B as their first attention weight;
        # pick its dtype as the model dtype.
        w_dtype = first_layer.w_QKV_B.dtype
        if W.dtype != w_dtype:
            W = W.to(w_dtype)
        if T.dtype != w_dtype:
            T = T.to(w_dtype)
        W = self.lift(W)
        # Shared T_dag across all stacked blocks.
        T_dag = first_layer.gaugegroup.dagger(T)
        W_attn = self.attn(W, T, T_dag)
        trace = self.trace(W_attn)
        site_out = self.mlp(trace).squeeze(-1)
        if self.reduction == "none":
            return site_out
        spatial_dims = tuple(range(1, site_out.ndim))
        if self.reduction == "sum":
            return site_out.sum(dim=spatial_dims)
        return site_out.mean(dim=spatial_dims)
