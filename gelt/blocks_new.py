"""
=========================================================================
GEAttention / GEFFN / GELT — split-block gauge-equivariant transformer
                              (notes/new_architecture.md §4).
=========================================================================

This module implements the new architecture proposed in
``notes/new_architecture.md``: a transformer-faithful split of the legacy
fused ``GEMHSA`` block into two sub-blocks, mirroring the standard
"attention then FFN" pattern of a vanilla transformer.

* ``GEAttention`` (sub-block A, §4.1): pure attention. The value path is
  the attention-weighted sum of transported V's, with **no Q† factor on
  the left** — i.e. exactly what a standard MHA layer does, only with
  gauge-equivariant transport wrapping the value. This is the *routing*
  primitive.
* ``GEFFN`` (sub-block B, §4.2): purely local L-Bilin. At every site,
  ``Q_b†(x) · V_b(x)`` — no transport, no neighbours, no softmax. This
  is the *multiplicative* primitive (the operator L-CNN calls L-Bilin),
  dropped into the FFN slot of the transformer block.
* ``GELT`` stacks them in the standard transformer order
  ``W ← W + Attn(W, T); W ← W + FFN(W)``. Each sub-block has its own
  ReZero ``α`` so the gradient can decouple routing from multiplication;
  this is the fix for the 3×3 Wilson-loop wall diagnosed in §3 of the
  notes. The pattern is configurable for ablations / unequal counts
  (§8.4: "2 attention + 4 L-Bilin"). The legacy fused ``GEMHSA`` is
  still available via ``pattern="M..."`` — it is imported from
  ``gelt.blocks`` and kept around per §9.2.

``ChannelLift``, ``Trace`` and ``MLP`` are reused verbatim from
``gelt.blocks`` (the readout half of the model didn't change).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gelt.blocks import GEMHSA, ChannelLift, MLP, Trace
from gelt.lattice import l1_ball_offsets


# ---------------------------------------------------------------------------
# Shared helpers used by both sub-blocks. Kept at module scope so each
# block reads top-to-bottom without inheritance gymnastics.
# ---------------------------------------------------------------------------


def _augment(W, gaugegroup, identity_buf):
    """Channel augmentation: ``(B, C, *Λ, nc, nc) -> (B, 2C+1, *Λ, nc, nc)``.

    Prepend the on-site identity, append the daggered channels.
    ``identity_buf`` is the precomputed ``(1, 1, *[1]*D, nc, nc)`` identity
    registered as a buffer at module init — broadcast at forward time so we
    don't re-allocate ``torch.eye`` per call.
    """
    spatial = W.shape[2:-2]
    nc = W.shape[-1]
    identity = identity_buf.expand(W.shape[0], 1, *spatial, nc, nc)
    return torch.cat([identity, W, gaugegroup.dagger(W)], dim=1)


def _transport_adjoint(X_nb, T, T_dag, n_offsets):
    """Compute ``T(x) · X_nb(x, n) · T†(x)`` with ``(H, d_qkv)`` folded into
    the column dim of the right-multiplicand.

    This replaces the naive broadcast matmul ``T_b @ X_nb @ T_b_dag`` (with
    ``T_b`` broadcast over ``H``, ``d_qkv``): it would launch
    ``B·H·d_qkv·n_off·|Λ|`` tiny (nc, nc)@(nc, nc) matmuls. Here we issue
    ``B·n_off·|Λ|`` matmuls of shape ``(nc, nc) @ (nc, H·d_qkv·nc)`` — 16×
    fewer launches at the benchmark shape, with ``T`` un-broadcast. This
    works because ``T`` is the same for all heads and d_qkv channels at
    fixed (B, offset, site).

    Inputs:
        X_nb : (B, H, d, n_off, *Λ, nc, nc)
        T    : (B, n_off, *Λ, nc, nc)
        T_dag: (B, n_off, *Λ, nc, nc)  -- precomputed once per layer
    Returns:
        (B, H, d, n_off, *Λ, nc, nc), equivalent to the naive
        two-matmul broadcast version.
    """
    B = X_nb.shape[0]
    H = X_nb.shape[1]
    d = X_nb.shape[2]
    n = n_offsets
    nc = X_nb.shape[-1]
    spatial = X_nb.shape[4:-2]
    D = len(spatial)

    # Move (H, d) after the row index 'i'. Source axes:
    #   0=B, 1=H, 2=d, 3=n, 4..3+D=spatial, 4+D=i, 5+D=j
    # Target:
    #   0=B, 1=n, 2..1+D=spatial, 2+D=i, 3+D=H, 4+D=d, 5+D=j
    perm = (0, 3) + tuple(range(4, 4 + D)) + (4 + D, 1, 2, 5 + D)
    Xp = X_nb.permute(*perm)
    # Flatten (H, d, j).
    X_flat = Xp.reshape(B, n, *spatial, nc, H * d * nc)
    # Left-multiply: one big matmul per (B, n_off, x).
    L = T @ X_flat  # (B, n, *Λ, nc, H*d*nc)

    # Now right-multiply by T†. (nc, H*d*nc) -> (nc, H, d, nc) -> (nc*H*d, nc)
    L = L.reshape(B, n, *spatial, nc, H, d, nc)
    L_flat = L.reshape(B, n, *spatial, nc * H * d, nc)
    R = L_flat @ T_dag  # (B, n, *Λ, nc*H*d, nc)

    # Reshape and permute back to (B, H, d, n, *Λ, nc, nc).
    out = R.reshape(B, n, *spatial, nc, H, d, nc)
    inv_perm = (0, 3 + D, 4 + D, 1) + tuple(range(2, 2 + D)) + (2 + D, 5 + D)
    return out.permute(*inv_perm)


def _l_act_gate(W_res, gate):
    """L-Act gate ``g(W_res) · W_res`` with ``g(W) = relu/softplus(Re Tr W / nc)``.

    The gate is a real scalar per ``(B, C, *Λ)``; gauge-invariant because
    ``Re Tr`` is a class function on the adjoint orbit. Used inside every
    sub-block to provide a per-site nonlinearity that does not break
    equivariance.
    """
    nc = W_res.shape[-1]
    trace_per_chan = W_res.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
    if gate == "relu":
        g = F.relu(trace_per_chan)
    else:
        g = F.softplus(trace_per_chan)
    return g.unsqueeze(-1).unsqueeze(-1) * W_res


def _build_nbr_idx(offsets, L, D):
    """Periodic neighbour index buffer:
    ``_nbr_idx[d, i, x] = (x[d] + Δx_i[d]) mod L``.

    Returned tensor has shape ``(D, n_offsets, *Λ)`` and is meant to be
    registered as a buffer.
    """
    offset_tensor = torch.tensor(offsets, dtype=torch.long)  # (n_off, D)
    coords = torch.meshgrid(
        *[torch.arange(L) for _ in range(D)], indexing="ij"
    )  # (*Λ)
    nbr_idx = torch.stack(
        [
            (coords[d].unsqueeze(0) + offset_tensor[:, d].view(-1, *([1] * D))) % L
            for d in range(D)
        ],
        dim=0,
    )  # (D, n_offsets, *Λ)
    return nbr_idx


# ---------------------------------------------------------------------------
# GEAttention — sub-block A: pure attention, no L-Bilin (§4.1).
# ---------------------------------------------------------------------------


class GEAttention(nn.Module):
    """Pure gauge-equivariant attention (notes/new_architecture.md §4.1).

    The "routing" half of the split block. Identical scaffolding to
    ``GEMHSA`` (augment, fused QKV, transport, score, softmax) but the
    value path is the attention-weighted sum of transported V's with
    **no Q† factor on the left**:

        ``V_weighted(x) = Σ_y α(x→y) · T_xy · V_a(y) · T†_xy``

    Followed by ``channel_mix``, the L-Act gate, and a ReZero residual.

    Gauge equivariance: ``T_xy → Ω_x · T_xy · Ω†_y`` and
    ``V_a(y) → Ω_y · V_a(y) · Ω†_y`` make the per-term product
    ``Ω_x · (T · V · T†) · Ω†_x`` covariant at ``x``; the softmax weight
    ``α`` is gauge-invariant (Frobenius product of covariant-at-x
    matrices), so the sum is covariant. ✓

    Stacked with ``GEFFN`` inside ``GELT``: this block routes, ``GEFFN``
    multiplies. The split decouples the two gradient objectives that the
    fused ``GEMHSA`` collapsed into one.
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
        init_scale: float = 1.0,
    ):
        super().__init__()
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
                f"Pass d_qkv explicitly when d_input < nhead."
            )
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate}")
        self.gate = gate

        # offsets is a list of the Δx_i in the L1 ball of radius R
        self.offsets = l1_ball_offsets(D, R)
        # n_offsets for R=1 D=2 is 4; for R=1 D=3 is 6; for R=2 D=3 is 24...
        self.n_offsets = len(self.offsets)
        # _nbr_idx[d, i, x] are the coords of the neighbor of x at offset Δx_i
        self.register_buffer("_nbr_idx", _build_nbr_idx(self.offsets, L, D))

        # Per-(head, offset) score bias. Untied across the lattice rotation/
        # reflection symmetries, same rationale as in ``GEMHSA``: orbit tying
        # prevents axis selection on rectangular Wilson-loop targets.
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_offsets))
        # Precomputed reshape target for broadcasting b_h over (B, *Λ) at score time.
        self._bias_view_shape = (1, self.H, self.n_offsets) + (1,) * D

        # Channel augmentation: C -> C' = 2C + 1 (identity + daggers).
        self.C_prime = 2 * d_input + 1

        # Cached on-site identity for `augment`: shape (1, 1, *[1]*D, nc, nc),
        # broadcast at forward time. Avoids re-allocating torch.eye every step.
        nc = gaugegroup.nc
        identity = torch.eye(nc, dtype=dtype).view(1, 1, *([1] * D), nc, nc)
        self.register_buffer("_identity", identity)

        # Fused QKV projection: one (3·H·d, C') @ (B, C', N) matmul.
        # σ ≈ 0.02·init_scale / √C' (real & imag parts independently).
        sigma = 0.02 * init_scale / math.sqrt(self.C_prime)
        self.w_QKV = nn.Parameter(
            torch.randn(3, self.H, self.d_qkv, self.C_prime, dtype=dtype) * sigma
        )
        # channel mix back to C output channels.
        sigma_mix = 0.02 * init_scale / math.sqrt(self.H * self.d_qkv)
        self.w_mix = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

        # ReZero scalar α. With identity-at-init (α=0), this sub-block is
        # bit-exactly W → W and stacks cleanly. Warm-start to a small positive
        # value (e.g. 0.05) to engage routing from step 0.
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    def augment(self, W):
        # Channel augmentation: (B, C, *Λ, nc, nc) -> (B, 2C+1, *Λ, nc, nc).
        return _augment(W, self.gaugegroup, self._identity)

    def transport(self, X_nb, T, T_dag):
        """T(x) · X_nb(x, n) · T†(x). See ``_transport_adjoint`` for the
        rationale on the matmul fusion."""
        return _transport_adjoint(X_nb, T, T_dag, self.n_offsets)

    def forward(self, W, T, T_dag=None):
        """W: (B, C, *Λ, nc, nc); T: (B, n_offsets, *Λ, nc, nc).

        T_dag : optional precomputed dagger of T. When called from
                ``GELT.attn`` this is computed once and shared across all
                stacked layers; standalone callers can leave it as ``None``
                and the block will compute it lazily.
        """
        assert T.shape[1] == self.n_offsets, (
            f"Expected T.shape[1] == {self.n_offsets} (number of offsets), got {T.shape[1]}"
        )
        if T_dag is None:
            T_dag = self.gaugegroup.dagger(T)

        nc = W.shape[-1]
        # Augment, then mix channels to build Q, K, V of shape
        # (B, H, d_qkv, *Λ, nc, nc).
        W_aug = self.augment(W)  # (B, C', *Λ, nc, nc), contiguous
        B = W_aug.shape[0]
        trailing = W_aug.shape[2:]  # (*Λ, nc, nc)

        # Fused QKV: a single (3·H·d, C') @ (B, C', N) matmul, then split.
        W_aug_flat = W_aug.view(B, self.C_prime, -1)
        w_QKV_flat = self.w_QKV.view(3 * self.H * self.d_qkv, self.C_prime)
        QKV = torch.matmul(w_QKV_flat, W_aug_flat)  # (B, 3·H·d, N)
        QKV = QKV.view(B, 3, self.H, self.d_qkv, *trailing)
        Q, K, V = QKV.unbind(dim=1)

        # Neighbour gather.
        idx = tuple(self._nbr_idx[k] for k in range(self.D))
        # nb_indexer = (:, :, :, ?, :, :) -> ? across dimension *Λ selects neighbors for each lattice site
        nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))
        K_nb = K[nb_indexer]  # (B, H, d_qkv, n_off, *Λ, nc, nc)
        V_nb = V[nb_indexer]

        # Transport K and V together: T and its dagger are shared across
        # heads, channels, and (in GELT) all stacked attention layers.
        KV_nb = torch.cat((K_nb, V_nb), dim=2)
        del K_nb, V_nb
        KV_tilde = self.transport(KV_nb, T, T_dag)
        K_tilde, V_tilde = KV_tilde.split(self.d_qkv, dim=2)

        # Score = Re Σ_c Tr[Q_c† · K̃_c] / √(d_qkv · nc); implementable via
        # Frobenius product without matmul.
        Q_e = Q.unsqueeze(3)  # (B, H, d_qkv, 1, *Λ, nc, nc)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)
        # Per-offset bias b_h[h, n], broadcast over (B, *Λ). The view shape
        # is precomputed at __init__; the .real fallback handles the case
        # where module-wide ``.to(complex_dtype)`` upcast the parameter
        # (b_h is initialised real, but tests cast the whole block).
        bias = self.b_h.real if self.b_h.is_complex() else self.b_h
        score = score + bias.view(self._bias_view_shape)

        # Softmax over offsets.
        alpha = torch.softmax(score, dim=2)
        # alpha: (B, H, n_off, *Λ) → (B, H, 1, n_off, *Λ, 1, 1) to broadcast
        # over the d_qkv and the two color axes.
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        # PURE attention output: weighted sum of transported V's — no Q†.
        # This is the only structural departure from ``GEMHSA.attend``.
        V_weighted = (alpha_b * V_tilde).sum(dim=3)  # (B, H, d_qkv, *Λ, nc, nc)

        # Channel mix back to C output channels: (C, H·d) @ (B, H·d, |Λ|·nc·nc).
        HD = self.H * self.d_qkv
        out_flat = V_weighted.reshape(B, HD, -1)
        w_mix_flat = self.w_mix.view(self.C, HD)
        W_mix = torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing)

        # Residual + L-Act gate + ReZero. At α=0 the block is bit-exactly W→W;
        # during training α grows and the routed/mixed path takes over.
        W_res = W + W_mix
        W_act = _l_act_gate(W_res, self.gate)
        return W + self.alpha * (W_act - W)


# ---------------------------------------------------------------------------
# GEFFN — sub-block B: purely local L-Bilin (§4.2).
# ---------------------------------------------------------------------------


class GEFFN(nn.Module):
    """Local gauge-equivariant L-Bilin "FFN" (notes/new_architecture.md §4.2).

    The "multiplication" half of the split block. Per site, per head, per
    channel:

        ``W'(x) = Q_b†(x) · V_b(x)``

    No transport, no neighbours, no softmax. Each multiplication is a
    single uncoupled gradient objective ("pick channels of Q_b and V_b
    whose product is closest to the next sub-loop"), which is the part
    that gets lost inside the fused ``GEMHSA`` at depth (§3 of the notes).

    Gauge equivariance: ``Q_b(x) → Ω_x · Q_b(x) · Ω†_x`` ⇒
    ``Q_b†(x) → Ω_x · Q_b†(x) · Ω†_x``, so
    ``Q_b†(x) · V_b(x) → Ω_x · (Q_b† V_b) · Ω†_x``. Covariant at ``x``. ✓

    This is exactly Favoni et al.'s L-Bilin (the operator L-CNN uses for
    loop doubling), dropped into the FFN slot of the transformer block.

    NOTE on ``nhead``: there is no attention here, so ``nhead`` is a
    channel-parallelism count for the bilinear path (analogous to the
    expansion-factor / inner width of a transformer FFN). Kept as a kwarg
    so ``GELT`` can pass the same value to both sub-blocks.
    """

    def __init__(
        self,
        gaugegroup,
        L,
        D,
        d_input,
        nhead,
        d_qkv=None,
        gate="softplus",
        dtype=torch.complex64,
        alpha_init: float = 0.0,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.gaugegroup = gaugegroup
        self.D = D
        self.H = nhead
        self.C = d_input
        self.d_qkv = d_input // nhead if d_qkv is None else d_qkv
        if self.d_qkv < 1:
            raise ValueError(
                f"d_qkv must be >= 1, got {self.d_qkv} "
                f"(d_input={d_input}, nhead={nhead})."
            )
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate}")
        self.gate = gate

        # Channel augmentation: C -> C' = 2C + 1 (identity + daggers).
        self.C_prime = 2 * d_input + 1

        # Cached on-site identity for `augment`. Same role as in GEAttention.
        nc = gaugegroup.nc
        identity = torch.eye(nc, dtype=dtype).view(1, 1, *([1] * D), nc, nc)
        self.register_buffer("_identity", identity)

        # Fused QV projection: a single (2·H·d, C') @ (B, C', N) matmul, then
        # split into Q_b and V_b. K is absent — no scoring happens here.
        sigma = 0.02 * init_scale / math.sqrt(self.C_prime)
        self.w_QV = nn.Parameter(
            torch.randn(2, self.H, self.d_qkv, self.C_prime, dtype=dtype) * sigma
        )
        # channel mix back to C output channels.
        sigma_mix = 0.02 * init_scale / math.sqrt(self.H * self.d_qkv)
        self.w_mix = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

        # Per-block ReZero α. Same identity-at-init story as ``GEAttention``.
        # In the typical split-block schedule (§5), warm-start this larger
        # than the attention α so the multiplicative path engages first
        # ("the part that actually creates new loop content").
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

    def forward(self, W):
        """W: (B, C, *Λ, nc, nc) → (B, C, *Λ, nc, nc).

        Purely local: no transport, no T argument, no neighbour gather.
        """
        W_aug = _augment(W, self.gaugegroup, self._identity)
        B = W_aug.shape[0]
        trailing = W_aug.shape[2:]  # (*Λ, nc, nc)

        # Fused QV: (2·H·d, C') @ (B, C', N).
        W_aug_flat = W_aug.view(B, self.C_prime, -1)
        w_QV_flat = self.w_QV.view(2 * self.H * self.d_qkv, self.C_prime)
        QV = torch.matmul(w_QV_flat, W_aug_flat)
        QV = QV.view(B, 2, self.H, self.d_qkv, *trailing)
        Q_b, V_b = QV.unbind(dim=1)

        # Local L-Bilin: Q_b†(x) · V_b(x) per (head, channel, site). The
        # (nc, nc) matmul broadcasts naturally across (B, H, d_qkv, *Λ).
        Q_b_dag = self.gaugegroup.dagger(Q_b)
        W_prime = torch.matmul(Q_b_dag, V_b)  # (B, H, d_qkv, *Λ, nc, nc)

        # Channel mix back to C output channels: (C, H·d) @ (B, H·d, |Λ|·nc·nc).
        HD = self.H * self.d_qkv
        out_flat = W_prime.reshape(B, HD, -1)
        w_mix_flat = self.w_mix.view(self.C, HD)
        W_mix = torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing)

        # Residual + L-Act gate + ReZero. Identical structure to GEAttention
        # — preserves the identity-at-init guarantee at α=0.
        W_res = W + W_mix
        W_act = _l_act_gate(W_res, self.gate)
        return W + self.alpha * (W_act - W)


# ---------------------------------------------------------------------------
# GELT — split-block model (default) with optional fused-GEMHSA fallback.
# ---------------------------------------------------------------------------


def _parse_pattern(pattern, n_layers):
    """Resolve the sub-block execution pattern.

    ``pattern`` is a string over the alphabet ``{'A', 'F', 'M'}``: ``A``
    inserts a ``GEAttention``, ``F`` inserts a ``GEFFN``, ``M`` inserts a
    legacy fused ``GEMHSA`` (for ablations — §9.2 of the notes). If
    ``pattern`` is ``None``, default to ``"AF" * n_layers`` — the standard
    transformer "attention, then FFN" pair repeated ``n_layers`` times.
    The default matches §4.3 of ``notes/new_architecture.md``.
    """
    if pattern is None:
        pattern = "AF" * n_layers
    pattern = pattern.upper()
    bad = set(pattern) - {"A", "F", "M"}
    if bad:
        raise ValueError(
            f"pattern must be a string over {{'A','F','M'}}, got bad chars "
            f"{sorted(bad)} in {pattern!r}"
        )
    if len(pattern) == 0:
        raise ValueError("pattern must contain at least one sub-block")
    return pattern


class GELT(nn.Module):
    """Full GELT model with the split GEAttention + GEFFN stack.

    Pipeline:
      1. Compute Plaq (+ optional Poly) — done by the dataset builder.
      2. ``ChannelLift`` from ``d_input = D(D-1)/2`` to ``d_model`` so the
         residual stream isn't pinned to the small plaquette channel count.
      3. Stack of sub-blocks according to ``pattern`` (default
         ``"AF" * n_layers``): ``GEAttention`` does the routing (sum over
         transported V's); ``GEFFN`` does the local L-Bilin multiplication
         ``Q_b† · V_b``; legacy ``GEMHSA`` ("M" in the pattern) is
         available for ablations.
      4. ``Trace`` to get Re, Im parts of the trace as scalar per site.
      5. ``MLP`` with one hidden layer to mix the trace features and output
         a scalar per site for regression or classification.
      6. Spatial reduction (``reduction`` arg): ``"sum"`` for extensive
         per-config targets like the Wilson action, ``"mean"`` for the
         average Wilson loop ⟨W⟩, ``"none"`` to keep the per-site readout
         ``(B, *Λ)`` for per-site supervision (e.g. ``Re Tr W(R,T,x)/nc``).

    Layer pattern
    -------------
    The default ``pattern = "AF" * n_layers`` reproduces the standard
    transformer template: each "layer" is one attention sub-block followed
    by one FFN sub-block. Pass ``pattern`` explicitly for unequal counts
    (e.g. ``pattern="AFFAFF"`` for "2 attention + 4 L-Bilin" — see §8.4 of
    ``notes/new_architecture.md``). Include ``M`` characters to splice in
    legacy fused-``GEMHSA`` blocks for ablations.

    Back-compat
    -----------
    ``gemhsa_layers`` is accepted as an alias for ``n_layers`` so the
    existing training scripts keep working. When set with the default
    pattern, each old "GEMHSA layer" becomes an ``Attn → FFN`` pair.
    ``alpha_init`` is accepted as a single value forwarded to both
    sub-block alphas. ``self.gemhsa_models`` aliases the full ordered list
    of sub-blocks so the per-layer α diagnostic in ``train_gelt.py``
    keeps reading something useful (both Attn and FFN expose ``.alpha``).
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
        attn_alpha_init: float = 0.0,
        ffn_alpha_init: float = 0.0,
        init_scale: float = 1.0,
        mlp_zero_init: bool = True,
        d_model: int | None = None,
        mlp_dropout: float = 0.0,
        pattern: str | None = None,
        # ---- back-compat aliases (old fused-GEMHSA API) -------------------
        gemhsa_layers: int | None = None,
        alpha_init: float | None = None,
    ):
        # Plaquette input -> D(D-1)/2 plaquettes per site.
        d_input = D * (D - 1) // 2
        # Internal residual-stream width. Defaults to d_input (no lift) for
        # backward compatibility; pass d_model > d_input to widen the
        # intermediate channels via the front-end ChannelLift.
        if d_model is None:
            d_model = d_input
        if d_model < d_input:
            raise ValueError(
                f"d_model must be >= d_input = D(D-1)/2 = {d_input}, got {d_model}."
            )
        self.d_input = d_input
        self.d_model = d_model
        super(GELT, self).__init__()
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(
                f"reduction must be 'sum', 'mean', or 'none', got {reduction!r}"
            )
        self.reduction = reduction

        # ---- resolve back-compat aliases ----------------------------------
        if n_layers is None:
            if gemhsa_layers is None:
                raise ValueError("Must specify n_layers (or gemhsa_layers).")
            n_layers = gemhsa_layers
        # ``alpha_init`` (legacy, fused-GEMHSA) applies to BOTH sub-blocks
        # so old configs that bumped α off zero keep doing so.
        if alpha_init is not None:
            attn_alpha_init = alpha_init
            ffn_alpha_init = alpha_init

        # Channel lift to widen the small plaquette input to d_model. When
        # d_model == d_input the lift is the identity matrix and is a no-op
        # at init (still trainable — the model can learn to mix plaquette
        # channels even at unchanged width).
        self.lift = ChannelLift(d_input, d_model, dtype=dtype)

        # Build the sub-block stack from ``pattern``. ``layers`` is the
        # ordered ModuleList of sub-blocks in execution order. ``_kinds`` is
        # a parallel list of 'A'/'F'/'M' so ``forward`` knows whether to
        # pass T to each block (Attention and the legacy GEMHSA need T;
        # FFN is purely local).
        self.pattern = _parse_pattern(pattern, n_layers)
        self._kinds = list(self.pattern)
        layers: list[nn.Module] = []
        for kind in self._kinds:
            if kind == "A":
                layers.append(
                    GEAttention(
                        gaugegroup,
                        L,
                        D,
                        R,
                        d_model,
                        nhead,
                        d_qkv=d_qkv,
                        gate=gate,
                        dtype=dtype,
                        alpha_init=attn_alpha_init,
                        init_scale=init_scale,
                    )
                )
            elif kind == "F":
                layers.append(
                    GEFFN(
                        gaugegroup,
                        L,
                        D,
                        d_model,
                        nhead,
                        d_qkv=d_qkv,
                        gate=gate,
                        dtype=dtype,
                        alpha_init=ffn_alpha_init,
                        init_scale=init_scale,
                    )
                )
            else:  # 'M' — legacy fused GEMHSA, kept for ablations.
                layers.append(
                    GEMHSA(
                        gaugegroup,
                        L,
                        D,
                        R,
                        d_model,
                        nhead,
                        d_qkv=d_qkv,
                        gate=gate,
                        dtype=dtype,
                        alpha_init=attn_alpha_init,
                        init_scale=init_scale,
                    )
                )
        # ModuleList so the sub-block parameters are registered with PyTorch
        # and picked up by .parameters() / .to() / .state_dict().
        self.layers = nn.ModuleList(layers)
        # Back-compat alias: train_gelt.py iterates ``model.gemhsa_models``
        # to print per-layer α values. Aliases the full ordered list so the
        # diagnostic keeps printing something meaningful (Attn and FFN both
        # expose an ``alpha`` parameter).
        self.gemhsa_models = self.layers

        # Trace produces real values, so the MLP must live in the matching
        # real dtype — not the complex `dtype` of the sub-block stack. A
        # blanket `.to(complex_dtype)` on GELT would otherwise miscast it.
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_model, mlp_hidden, mlp_out, dropout=mlp_dropout).to(real_dtype)
        # Zero-init the MLP's last linear layer: at init the model outputs 0
        # at every site, so the untrained prediction is exactly 0.
        if mlp_zero_init:
            nn.init.zeros_(self.mlp.fc2.weight)
            nn.init.zeros_(self.mlp.fc2.bias)

    def attn(self, W, T, T_dag):
        """Run the sub-block stack. Attention/GEMHSA layers consume ``T``;
        FFN layers don't. Kept name ``attn`` for back-compat with callers
        that drove the old fused stack directly.
        """
        for kind, layer in zip(self._kinds, self.layers):
            if kind == "F":
                W = layer(W)
            else:
                W = layer(W, T, T_dag)
        return W

    def forward(self, W, T):
        # Cast inputs to the model's weight dtype once (hoisted out of every
        # sub-block's forward) so real-valued data (Z₂ float32 plaquettes)
        # can be fed to a complex model without a per-layer cast.
        first_layer = self.layers[0]
        # GEAttention and GEMHSA carry w_QKV; GEFFN carries w_QV. Pick
        # whichever the first layer exposes to determine the model dtype.
        if hasattr(first_layer, "w_QKV"):
            w_dtype = first_layer.w_QKV.dtype
        else:
            w_dtype = first_layer.w_QV.dtype
        if W.dtype != w_dtype:
            W = W.to(w_dtype)
        if T.dtype != w_dtype:
            T = T.to(w_dtype)
        # Lift the (small) plaquette channel count to d_model before the
        # sub-block stack. With identity-extend init this is bit-exactly the
        # input when d_model == d_input.
        W = self.lift(W)
        # T_dag is shared across all stacked attention/GEMHSA layers.
        T_dag = first_layer.gaugegroup.dagger(T)
        W_attn = self.attn(W, T, T_dag)
        trace = self.trace(W_attn)
        site_out = self.mlp(trace)  # (B, *Λ, mlp_out)
        # squeeze(-1) handles mlp_out=1 → (B, *Λ).
        site_out = site_out.squeeze(-1)
        if self.reduction == "none":
            # Per-site supervision (e.g. Re Tr W(R,T,x)/nc): keep the spatial axes.
            return site_out
        # Reduce the site-local readout over the spatial axes.
        # "sum" matches an extensive per-config target (Wilson action);
        # "mean" matches the average Wilson loop ⟨W⟩.
        spatial_dims = tuple(range(1, site_out.ndim))
        if self.reduction == "sum":
            return site_out.sum(dim=spatial_dims)
        return site_out.mean(dim=spatial_dims)
