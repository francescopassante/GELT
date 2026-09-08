"""
=========================================================================
GEMHSA / GELT — gauge-equivariant attention block and the full model.
=========================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from gelt.lattice import l1_ball_offsets


# Peak-memory budget for the offset-gather backward below: the gradient is
# re-gathered in slices of the offset axis so the transient never exceeds this,
# which matters because the backward runs while the (much larger) recomputed
# forward activations of the same layer are still alive.
_GRAD_GATHER_BUDGET = 512 * 2**20


class _OffsetGather(torch.autograd.Function):
    """Gather ``X`` at every L1-ball offset, with a *coalesced* backward.

    The forward is ``X[x + Δx_i]`` for every offset i — an advanced-index read.
    Left to autograd, its backward is ``index_put_(accumulate=True)``: a
    scatter-add in which every element of the input receives n_offsets separate
    contributions, i.e. millions of atomic read-modify-writes on GPU. Measured on
    CPU at ``(24, 4, 6, 12³, 2, 2)`` it cost **64× the forward** — by far the most
    expensive single operation in the block.

    But the forward is a pure *translation* per offset, so the gradient is itself
    a gather:

        dL/dX (x) = Σ_i  g_i(x − Δx_i)

    and indexing the offset axis alongside the lattice axes expresses the whole
    sum as one advanced-index read plus one reduction — no atomics, no Python
    loop over offsets. Same number as autograd's, up to the order in which the
    n_offsets terms are summed (float32: ~1e-6 relative; float64: exact).

    ``chunk`` slices the offset axis of the *gradient* only, to bound the
    transient; it changes nothing but the summation order.
    """

    @staticmethod
    def forward(ctx, X, nbr_idx, nbr_idx_inv):
        D = nbr_idx.shape[0]
        ctx.save_for_backward(nbr_idx_inv)
        ctx.D = D
        indexer = (
            (slice(None),) * 3
            + tuple(nbr_idx[k] for k in range(D))
            + (slice(None), slice(None))
        )
        return X[indexer]

    @staticmethod
    def backward(ctx, grad):
        (inv,) = ctx.saved_tensors
        D = ctx.D
        n_off = inv.shape[1]
        per_offset = grad.numel() // n_off * grad.element_size()
        chunk = max(1, min(n_off, _GRAD_GATHER_BUDGET // max(per_offset, 1)))
        ar = torch.arange(n_off, device=grad.device)
        out = None
        for lo in range(0, n_off, chunk):
            hi = min(lo + chunk, n_off)
            sel = (
                (slice(None),) * 3
                + (ar[lo:hi].view(-1, *([1] * D)),)
                + tuple(inv[k, lo:hi] for k in range(D))
                + (slice(None), slice(None))
            )
            part = grad[sel].sum(dim=3)
            out = part if out is None else out + part
        return out, None, None


class GEMHSA(nn.Module):
    """Gauge-equivariant multi-head self-attention block.

    The input is a batched covariant local field of shape ``(B, C, *Λ, nc, nc)``
    Every channel of W transforms in the adjoint representation, ``W → Ω W Ω†``.

    Pipeline:

    1. **augment + project.** Append the on-site identity and daggers
    (C → 2C + 1), then project to per-head Q_s, K, V, Q_v of
       shape (B, H, d_qkv, *Λ, nc, nc). There are two independent query
       projections: Q_s for the score path and Q_v for the value path.
    2. **adjoint transport.** For every Δx in the L1-ball of radius R, gather
       the neighbour fields K(x+Δx), V(x+Δx) and apply the
       shortest-path-averaged transport: K'(x+Δx->x) = T(x) · K(x+Δx) · T†(x)
    3. **relative position (RoPE) + gauge invariant score.** Before scoring,
       K's channel-pairs are rotated by an offset-dependent rotation R(Δx) in
       channel space; the query sits at offset 0 → identity, so only K is
       rotated. Applying *different* rotations to Q and K (here R(Δx) vs I) is
       what makes the score offset-selective rather than rotation-invariant.
       Then s = (1/√(nc·d)) · Re Σ_a Tr[Q_s†_a · K̃_a] per offset and per head.
       NOTE: RoPE replaces the additive per-offset bias b_h. The bias was an
       explicit convolution kernel on the logits (content-independent), which
       did all the axis selection on its own; RoPE instead injects relative
       position *through* the Q·K content product.
    4. **softmax** over the offset axis (normalizes over neighbours per
       site, per head).
    5. **multiplicative value path.** Output of the attention head is
       Σ_i α_i · Q_v†(x) · V'_i(x) — both factors are covariant at x, so the
       product is covariant; this is the L-Bilin-baked-in step that gives the
       loop-doubling expressivity argument. Q_v is its own projection,
       independent of the score-path query Q_s.
    6. **channel mix back to C** via a complex linear (H, d_qkv) → C.
        NOTE: C is the model's working width inside the GEMHSA stack. It is
        decoupled from the (small) plaquette input count D(D-1)/2 ∈ {1, 3, 6}
        by the front-end ``ChannelLift`` in ``GELT``; pass ``d_model`` to
        widen it.
    7. **residual + L-Act gate.** W_act = g(W_mix) · W_mix with
       g(W) = ReLU(Re Tr[W]/nc) or softplus(Re Tr[W]/nc);
       the block output is W_in + W_act (standard transformer-style
       residual — only the sublayer output is gated, the residual stream
       is left untouched).

    The transport T is precomputed by the dataset builder (it is a
    function of the link configuration only
    it's a tensor of shape (B, n_offsets, *Λ, nc, nc)
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
        init_scale: float = 1.0,
        qk_init_scale: float = 1.0,
    ):
        super(GEMHSA, self).__init__()
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
                f"e.g. with GELT in D=2, where d_input = D(D-1)/2 = 1."
            )
        if gate not in ("relu", "softplus"):
            raise ValueError(f"gate must be 'relu' or 'softplus', got {gate}")
        self.gate = gate

        # Introspection switches, both off by default because both cost real
        # time on the training path.
        #   store_attention — keep _last_score / _last_alpha for the
        #     interpretability program (scripts/*_attention_*.py set it).
        #     (B, H, n_off, *Λ) reals per layer, retained until the next forward.
        #   diagnostics — the _last_*_norm scalars. Each is a full reduction over
        #     an offset-expanded tensor *plus* a .item(), i.e. a GPU sync; there
        #     are ~10 per block, they run again inside every gradient-checkpoint
        #     recompute, and the two K̃/Ṽ norms alone re-read and re-materialise
        #     ~12% of the block's whole memory traffic (abs + pow + mean over
        #     (B, H, d_qkv, n_off, *Λ, nc, nc)).
        self.store_attention = False
        self.diagnostics = False

        # offsets is a list of the Δx_i in the L1 ball of radius R. The
        # Δx = 0 self-offset is prepended (transport is the identity), so
        # the attention has an explicit "attend to self" slot in addition
        # to the residual skip. External transport tensors from
        # ``build_transport_average`` only carry the non-zero offsets; the
        # block synthesises the identity for the zero offset inside
        # ``forward``.
        self.offsets = [tuple([0] * D)] + l1_ball_offsets(D, R)
        # n_offsets for R=1 D=2 is 5; for R=1 D=3 is 7; for R=2 D=3 is 25...
        # (one more than l1_ball_offsets due to the zero self-offset).
        self.n_offsets = len(self.offsets)

        # _nbr_idx[d, i, x] are the coords of the neighbor of x at offset Δx_i
        # = (x[d] + Δx_i[d]) mod L.
        offset_tensor = torch.tensor(self.offsets, dtype=torch.long)  # (n_off, D)
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
        self.register_buffer("_nbr_idx", nbr_idx)
        # The same map with the offsets reversed, (x − Δx_i) mod L. This is the
        # index the gather's *gradient* needs — see _OffsetGather.
        nbr_idx_inv = torch.stack(
            [
                (coords[d].unsqueeze(0) - offset_tensor[:, d].view(-1, *([1] * D))) % L
                for d in range(D)
            ],
            dim=0,
        )
        self.register_buffer("_nbr_idx_inv", nbr_idx_inv)

        # Relative positional encoding (RoPE) over the offset axis — replaces
        # the additive per-offset bias b_h. Rather than adding a content-
        # independent scalar to each logit (an explicit convolution kernel on
        # the logits, which tends to do all the selection work on its own), RoPE
        # injects relative position *through* the Q·K content product by
        # rotating K's channel-pairs by R(Δx). Because the query lives at the
        # centre (offset 0 → identity) and each key at offset Δx, the relative
        # rotation R(Δx) is non-trivial and offset-dependent, so the Frobenius
        # score is no longer rotation-invariant (it would be if Q and K shared
        # the same rotation). The rotation is real and acts only on the d_qkv
        # channel index → gauge invariance is untouched.
        if self.d_qkv % 2 != 0:
            raise ValueError(
                f"RoPE rotates channels in pairs, so d_qkv must be even; "
                f"got d_qkv={self.d_qkv}. Pass an even d_qkv explicitly "
                f"(note D=2 GELT has d_input = 1, so d_qkv must be set)."
            )
        self.n_pairs = self.d_qkv // 2

        # Axial N-d RoPE: each channel-pair p is tied to one lattice axis
        # (round-robin, axis = p % D) and encodes the displacement Δx along that
        # axis. _rope_disp[n, p] = Δx_n[axis(p)] is pure geometry, precomputed.
        pair_axis = [p % D for p in range(self.n_pairs)]
        rope_disp = offset_tensor[:, pair_axis].to(torch.get_default_dtype())
        self.register_buffer("_rope_disp", rope_disp)  # (n_offsets, n_pairs)
        # Learnable per-pair angular frequency θ_p. The rotation angle at offset
        # Δx is θ_p · Δx[axis(p)]. Learnable rather than the fixed geometric
        # base used for long text sequences (meaningless at Manhattan radius
        # R≈1–2): the model tunes how fast each pair winds with displacement.
        if self.n_pairs > 1:
            freq = torch.logspace(0, -1, self.n_pairs)
        else:
            freq = torch.ones(1)
        self.rope_freq = nn.Parameter(freq)

        # Channel augmentation expands
        # C -> C' = 2C + 1 by appending the identity and daggers.
        self.C_prime = 2 * d_input + 1

        # Cached on-site identity for `augment`: shape (1, 1, *[1]*D, nc, nc),
        # broadcast at forward time. Avoids re-allocating torch.eye every step.
        nc = gaugegroup.nc
        identity = torch.eye(nc, dtype=dtype).view(1, 1, *([1] * D), nc, nc)
        self.register_buffer("_identity", identity)

        # Fused QKV projection.
        # A single (4·H·d, C') @ (B, C', N) matmul instead of four separate ones.
        # There are now TWO independent Q projections:
        #   Q_s — the score-path query, paired with K in Re Tr[Q_s†·K̃].
        #   Q_v — the value-path query, the left factor in Q_v†·Ṽ.
        # Decoupling them lets the score (which neighbour to attend to) and the
        # value bilinear (what loop to build at x) use different on-site
        # combinations of the augmented channels — the L-Bilin loop-doubling
        # factor no longer has to double as the attention-similarity query.
        # The projections are initialized at scales matching their roles:
        #   σ_V  = 0.02·init_scale / √C'  (small — keeps the value path tiny so
        #                                  the residual stream W + W_act is near
        #                                  identity at init, preserving stackability)
        #   σ_QK = qk_init_scale  / √C'   (standard transformer scale by default)
        # The score Re Tr[Q_s†·K̃] / √(d·nc) is O(σ_QK²·|W|²) at init. With the
        # old tied σ ≈ 0.02·init_scale/√C', score variance ~ σ⁴ ~ 1e-8 collapsed
        # the softmax to uniform on epoch 0 — no axis-selectivity signal could
        # ever start propagating to Q/K. Decoupling lets the score channel sit
        # at O(1) magnitude immediately while V stays small. Q_v rides with the
        # score-path scale σ_QK so the value bilinear Q_v†·Ṽ ~ σ_QK·σ_V stays
        # small at init exactly as the old shared-Q value path did.
        # Order along axis 0: [Q_s, K, V, Q_v].
        sigma_v = 0.02 * init_scale / math.sqrt(self.C_prime)
        sigma_qk = qk_init_scale / math.sqrt(self.C_prime)
        w_qkv = torch.randn(4, self.H, self.d_qkv, self.C_prime, dtype=dtype)
        w_qkv[0] *= sigma_qk  # Q_s (score query)
        w_qkv[1] *= sigma_qk  # K
        w_qkv[2] *= sigma_v  # V
        w_qkv[3] *= sigma_qk  # Q_v (value query)
        self.w_QKV = nn.Parameter(w_qkv)
        # channel mix back to C output channels.
        sigma_mix = 0.02 * init_scale / math.sqrt(self.H * self.d_qkv)
        self.w_mix = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

    def augment(self, W):
        # Channel augmentation: (B, C, *Λ, nc, nc) -> (B, 2C+1, *Λ, nc, nc).
        # Prepend the identity, append the daggered channels.
        spatial = W.shape[2:-2]
        nc = W.shape[-1]
        identity = self._identity.expand(W.shape[0], 1, *spatial, nc, nc)
        return torch.cat([identity, W, self.gaugegroup.dagger(W)], dim=1)

    def transport(self, X_nb, T, T_dag):
        """Compute T(x) · X_nb(x, n) · T†(x) for every (h, d, n, x) with
        (H, d_qkv) folded into the column dim of the right-multiplicand.

        This replaces the naive broadcast matmul
        T_b @ X_nb @ T_b_dag (with T_b broadcast over H, d_qkv): it would
        launch B·H·d_qkv·n_off·|Λ| tiny (nc, nc)@(nc, nc) matmuls.
        Here we issue B·n_off·|Λ| matmuls of shape (nc, nc) @ (nc, H·d_qkv·nc)
        — 16× fewer launches at the benchmark shape, with T un-broadcast.
        This is because T is the same for all heads and d_qkv channels at fixed B, offset and site.

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
        n = self.n_offsets
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

    def apply_rope(self, K):
        """Rotate K's channel-pairs by the offset-dependent angle θ_p·Δx[axis(p)].

        ``K`` : (B, H, d_qkv, n_off, *Λ, nc, nc). The rotation mixes the two
        channels of each pair and is the identity at offset 0, so the query
        (always at the centre) is left unrotated and the score becomes
        offset-dependent. The rotation acts purely on the channel index, so the
        (nc, nc) colour matrices ride along untouched — gauge invariance holds.
        """
        B, H, dq, n = K.shape[0], K.shape[1], K.shape[2], K.shape[3]
        spatial = K.shape[4:-2]
        nc = K.shape[-1]
        # .real fallbacks: a module-wide ``.to(complex_dtype)`` (the tests cast
        # the whole block) would upcast these real tensors to complex.
        freq = self.rope_freq.real if self.rope_freq.is_complex() else self.rope_freq
        disp = self._rope_disp.real if self._rope_disp.is_complex() else self._rope_disp
        angle = disp * freq  # (n_off, n_pairs), broadcast over the pair axis
        # cos/sin → (1, 1, n_pairs, n_off, *1, 1, 1) to broadcast over B, H,
        # the lattice, and the two colour axes. reshape() copies the transpose.
        view = (1, 1, self.n_pairs, n) + (1,) * len(spatial) + (1, 1)
        cos = torch.cos(angle).t().reshape(view)
        sin = torch.sin(angle).t().reshape(view)
        # Split the d_qkv axis into (n_pairs, 2) and apply the planar rotation.
        Kp = K.reshape(B, H, self.n_pairs, 2, n, *spatial, nc, nc)
        k0 = Kp[:, :, :, 0]
        k1 = Kp[:, :, :, 1]
        r0 = cos * k0 - sin * k1
        r1 = sin * k0 + cos * k1
        return torch.stack((r0, r1), dim=3).reshape(B, H, dq, n, *spatial, nc, nc)

    def rope_score(self, Q, K_tilde):
        """Gauge-invariant score with the RoPE rotation folded into the query.

        Same number as ``Re Σ_c Tr[Q_c† · rope(K̃)_c] / √(d_qkv·nc)`` with
        ``rope`` as in :meth:`apply_rope`, but K̃ is never rotated. Writing the
        planar rotation of pair ``p`` out and using the linearity of the
        Frobenius product ``⟨A, B⟩ = Σ_ij conj(A_ij) B_ij``:

            Σ_s ⟨Q_{p,s}, rope(K̃)_{p,s}⟩
              = cos(θ_p·Δx) [⟨Q_{p,0}, K̃_{p,0}⟩ + ⟨Q_{p,1}, K̃_{p,1}⟩]
              + sin(θ_p·Δx) [⟨Q_{p,1}, K̃_{p,0}⟩ − ⟨Q_{p,0}, K̃_{p,1}⟩]

        Both brackets are Frobenius products against the *unrotated* K̃: the
        first with Q, the second with Q's pair-swapped copy
        ``Q'_{p,0} = Q_{p,1}``, ``Q'_{p,1} = −Q_{p,0}``. Q carries no offset
        axis, so swapping it costs n_offsets times less than rotating K̃ — at the
        production shape ``apply_rope`` plus the old score materialised 569 MiB
        per layer per forward where this route materialises 237 MiB.

        ``Q`` : (B, H, d_qkv, *Λ, nc, nc); ``K_tilde`` : the transported,
        *unrotated* keys (B, H, d_qkv, n_off, *Λ, nc, nc).
        """
        B, H, dq = Q.shape[0], Q.shape[1], Q.shape[2]
        spatial = Q.shape[3:-2]
        nc = Q.shape[-1]
        n = K_tilde.shape[3]
        # Q' — the within-pair swap-and-negate that turns the first bracket into
        # the second. Per-site, so this is the cheap tensor to touch.
        Qp = Q.reshape(B, H, self.n_pairs, 2, *spatial, nc, nc)
        Q_swap = torch.stack((Qp[:, :, :, 1], -Qp[:, :, :, 0]), dim=3).reshape(
            B, H, dq, *spatial, nc, nc
        )

        def brackets(Qx):
            # Frobenius product over the colour axes only: the channel axis has
            # to survive until cos/sin have been applied per pair.
            t = (Qx.unsqueeze(3).conj() * K_tilde).sum(dim=(-2, -1))
            return t.reshape(B, H, self.n_pairs, 2, n, *spatial).sum(dim=3)

        # .real fallbacks: a module-wide ``.to(complex_dtype)`` (the tests cast
        # the whole block) would upcast these real tensors to complex.
        freq = self.rope_freq.real if self.rope_freq.is_complex() else self.rope_freq
        disp = self._rope_disp.real if self._rope_disp.is_complex() else self._rope_disp
        angle = disp * freq  # (n_off, n_pairs)
        view = (1, 1, self.n_pairs, n) + (1,) * len(spatial)
        cos = torch.cos(angle).t().reshape(view)
        sin = torch.sin(angle).t().reshape(view)
        # cos, sin are real, so Re(cos·u + sin·v) = cos·Re u + sin·Re v.
        score = (cos * brackets(Q).real + sin * brackets(Q_swap).real).sum(dim=2)
        return score / math.sqrt(dq * nc)

    def attend(self, Q, KV, Q_v, T, T_dag):
        """Fully batched gauge-equivariant attention over the L1-ball.

        ``Q`` is the score-path query and ``Q_v`` the independent value-path
        query — two separate on-site projections of the augmented field. ``KV``
        is the key and value projections stacked on the head axis,
        ``(B, 2·H, d_qkv, *Λ, nc, nc)`` with K first: they share the gather and
        the transport, and the caller already has them adjacent in the fused QKV
        output, so handing them over as one view saves a concatenation the size
        of the whole offset-expanded neighbourhood.

        Single fused pass — no Python loop over offsets. Pipeline:
          1. Gather K(x+Δx_i), V(x+Δx_i)
          2. Adjoint transport: K' = T(x) · K(x+Δx) · T†(x)
          3. RoPE-rotate K̃ by R(Δx), then score Re Σ_c Tr[Q_c† · K̃_c] /
             √(d_qkv·nc) computed as a Frobenius product. Steps 2 and 3 commute
             (one acts on colour, the other on channels) and the rotation is
             folded into the query instead — see :meth:`rope_score`.
          4. Softmax over the offset axis.
          5. Value path Q_v† · V' — but α-weighted *before* the matmul
             Σ_n α_n (Q_v† Ṽ_n) = Q_v† (Σ_n α_n Ṽ_n)

        ``T`` and ``T_dag`` are shared across all GEMHSA layers in a stacked
        GELT; the dagger is computed once at the GELT level and threaded in.
        """
        nc = Q.shape[-1]
        H = self.H

        # 1. Neighbour gather.
        # (B, 2H, d_qkv, n_off, *Λ, nc, nc) -> for each lattice site and
        # neighbour, a (B, 2H, d, nc, nc) key/value tensor. One indexed read,
        # because K and V arrive stacked. _OffsetGather is the same read as
        # ``KV[(slice(None),) * 3 + (self._nbr_idx[k] for k) + (…)]`` with the
        # gradient written as a gather instead of an atomic scatter-add.
        KV_nb = _OffsetGather.apply(KV, self._nbr_idx, self._nbr_idx_inv)

        # 2. Transport: T · X · T†. T and its dagger are shared across heads,
        # channels, and (in GELT) all stacked GEMHSA layers.
        # K and V use the same transport, so they ride through it together and
        # are split off afterwards as views.
        KV_tilde = self.transport(KV_nb, T, T_dag)
        del KV_nb
        K_tilde, V_tilde = KV_tilde[:, :H], KV_tilde[:, H:]

        # 3./4. RoPE-weighted score. The rotation makes the score
        # offset-selective (Q sits at offset 0 → identity, so only the relative
        # rotation R(Δx) survives); it is applied to Q rather than to K̃ because
        # K̃ carries the offset axis and Q does not.
        score = self.rope_score(Q, K_tilde)
        # score: (B, H, n_off, *Λ)

        # 5. Softmax over offsets.
        alpha = torch.softmax(score, dim=2)

        # 6. Value path. Sum V' over n_off with α weights BEFORE the
        # Q† matmul.
        # alpha: (B, H, n_off, *Λ) → (B, H, 1, n_off, *Λ, 1, 1) to broadcast
        # over the d_qkv and the two color axes.
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        V_weighted = (alpha_b * V_tilde).sum(dim=3)  # (B, H, d_qkv, *Λ, nc, nc)
        Q_v_dag = self.gaugegroup.dagger(Q_v)  # (B, H, d_qkv, *Λ, nc, nc)
        # Two value-path branches with separate learnable scalar weights:
        #   alpha_attn * V_weighted              — transformer-style attention
        #                                          sum (no Q_v† factor).
        #   alpha_bilin * (Q_v† · V_weighted)    — L-Bilin loop-doubling step.
        # Both are gauge-equivariant at x (V_weighted is the attention-weighted
        # sum of transported V's; Q_v†·V_weighted multiplies on the left by Q_v†
        # which transforms with Ω_x on both sides). Q_v is a value-path query
        # projection independent of the score-path query Q.
        bilin = torch.matmul(Q_v_dag, V_weighted)  # (B, H, d_qkv, *Λ, nc, nc)

        # Diagnostic intermediates — read by scripts/train_gelt_diagnosis.py
        # to introspect per-layer attention state. Stored detached / no-grad so
        # they don't retain the autograd graph. Scalars for activations, full
        # tensors for score/alpha (cheap: (B, H, n_off, *Λ) reals).
        # Both stashes are opt-in (see store_attention / diagnostics in
        # __init__): the scalars each force a GPU sync and a reduction over an
        # offset-expanded tensor, which is pure overhead on a training step.
        if self.store_attention:
            with torch.no_grad():
                self._last_score = score.detach()
                self._last_alpha = alpha.detach()
        if self.diagnostics:
            with torch.no_grad():
                self._last_Q_norm = Q.detach().abs().pow(2).mean().sqrt().item()
                self._last_Q_v_norm = Q_v.detach().abs().pow(2).mean().sqrt().item()
                self._last_K_tilde_norm = (
                    K_tilde.detach().abs().pow(2).mean().sqrt().item()
                )
                self._last_V_tilde_norm = (
                    V_tilde.detach().abs().pow(2).mean().sqrt().item()
                )
                self._last_bilin_norm = bilin.detach().abs().pow(2).mean().sqrt().item()

        return bilin
        # return self.alpha_attn * V_weighted + self.alpha_bilin * bilin

    def prepend_self_offset(self, T, T_dag=None):
        """Prepend the Δx = 0 slot (transport = identity) to a transport table.

        ``build_transport_average`` materialises only the non-zero offsets, but
        the block's offset axis starts with the on-site slot. Splitting this out
        of :meth:`forward` lets ``GELT.attn`` pay the concatenation once for the
        whole stack instead of once per layer per forward pass.
        """
        nc = T.shape[-1]
        B = T.shape[0]
        spatial = T.shape[2:-2]
        identity_T = (
            torch.eye(nc, dtype=T.dtype, device=T.device)
            .view(1, 1, *([1] * self.D), nc, nc)
            .expand(B, 1, *spatial, nc, nc)
        )
        if T_dag is None:
            T_dag = self.gaugegroup.dagger(T)
        # The identity is its own dagger, so the same block prepends to both.
        return torch.cat([identity_T, T], dim=1), torch.cat([identity_T, T_dag], dim=1)

    def forward(self, W, T, T_dag=None):
        """Run the block.

        W : input field, (B, C, *Λ, nc, nc)
        T : precomputed transports, (B, n_offsets, *Λ, nc, nc)
        T_dag : optional precomputed dagger of T. When called from
                ``GELT.attn`` this is computed once and shared across all
                stacked layers; standalone callers can leave it as ``None``
                and the block will compute it lazily.

        Inputs are expected to be in the model's weight dtype already;
        ``GELT.forward`` performs the cast once on entry. Standalone callers
        that pass real-valued data into a complex model should cast first.

        Returns a tensor of the same shape as W.
        """

        # External T carries only the non-zero offsets; the Δx = 0 entry
        # (whose transport is the identity) is synthesised here — unless the
        # caller already did it. ``GELT.attn`` prepends once and reuses the
        # result across the whole stack, which is worth doing: the concatenation
        # copies the full (B, n_off, *Λ, nc, nc) transport table, and it used to
        # run twice per layer per forward (and again inside every
        # gradient-checkpoint recompute).
        if T.shape[1] == self.n_offsets:
            prepend = False
        elif T.shape[1] == self.n_offsets - 1:
            prepend = True
        else:
            raise ValueError(
                f"Expected T.shape[1] == {self.n_offsets - 1} (non-zero offsets) "
                f"or {self.n_offsets} (identity already prepended), "
                f"got {T.shape[1]}"
            )

        nc = W.shape[-1]
        if prepend:
            T, T_dag = self.prepend_self_offset(T, T_dag)
        elif T_dag is None:
            T_dag = self.gaugegroup.dagger(T)

        # Augment, then mix channels to build Q, K, V of shape
        # (B, H, d_qkv, *Λ, nc, nc).
        W_aug = self.augment(W)  # (B, C', *Λ, nc, nc), contiguous
        B = W_aug.shape[0]
        trailing = W_aug.shape[2:]  # (*Λ, nc, nc)

        # Fused QKV: a single (4·H·d, C') @ (B, C', N) matmul, then split.
        W_aug_flat = W_aug.view(B, self.C_prime, -1)
        w_QKV_flat = self.w_QKV.view(4 * self.H * self.d_qkv, self.C_prime)
        QKV = torch.matmul(w_QKV_flat, W_aug_flat)  # (B, 4·H·d, N)
        QKV = QKV.view(B, 4, self.H, self.d_qkv, *trailing)
        Q, Q_v = QKV[:, 0], QKV[:, 3]
        # K and V are adjacent along the fused axis and the (4, H) strides merge,
        # so this is a view: attend() then gathers and transports the pair in one
        # pass instead of two plus a concatenation.
        KV = QKV[:, 1:3].reshape(B, 2 * self.H, self.d_qkv, *trailing)

        # Transport, score, softmax, multiplicative value. Q is the score-path
        # query (Re Tr[Q†·K̃]); Q_v is the independent value-path query (Q_v†·Ṽ).
        out = self.attend(Q, KV, Q_v, T, T_dag)  # (B, H, d_qkv, *Λ, nc, nc)

        # Channel mix back to C output channels. Expressed as a single matmul
        # ``(C, H·d) @ (B, H·d, |Λ|·nc·nc) -> (B, C, |Λ|·nc·nc)``
        HD = self.H * self.d_qkv
        out_flat = out.reshape(B, HD, -1)
        w_mix_flat = self.w_mix.view(self.C, HD)
        W_mix = torch.matmul(w_mix_flat, out_flat).view(B, self.C, *trailing)

        # Residual + L-Act gate. Standard transformer-style: gate only the
        # sublayer output (W_mix), then add to the untouched residual W.
        # The gate is a real scalar per (B, C, x).
        trace_per_chan = W_mix.diagonal(dim1=-2, dim2=-1).sum(-1).real / nc
        if self.gate == "relu":
            g = F.relu(trace_per_chan)
        else:
            g = F.softplus(trace_per_chan)
        W_act = g.unsqueeze(-1).unsqueeze(-1) * W_mix

        # Diagnostic intermediates — residual stream magnitudes. Ratio
        # |W_act|/|W_in| tells you whether the sublayer is actually
        # contributing to the residual stream or has collapsed to ≈0.
        if self.diagnostics:
            with torch.no_grad():
                self._last_W_in_norm = W.detach().abs().pow(2).mean().sqrt().item()
                self._last_W_mix_norm = W_mix.detach().abs().pow(2).mean().sqrt().item()
                self._last_W_act_norm = W_act.detach().abs().pow(2).mean().sqrt().item()
                self._last_gate_mean = g.detach().mean().item()
                self._last_gate_std = g.detach().std(unbiased=False).item()

        return W + W_act


class Trace(nn.Module):
    """Trace block: outputs the trace of the input field as a scalar per site.

    This is a gauge-invariant quantity, so it can be used for supervised
    regression tasks or as a readout head for classification.
    """

    def forward(self, W):
        # W: (B, C, *Λ, nc, nc) -> trace over color
        trace = W.diagonal(dim1=-2, dim2=-1).sum(-1)  # (B, C, *Λ)
        imag = trace.imag if trace.is_complex() else torch.zeros_like(trace)
        out = torch.cat([trace.real, imag], dim=1)  # (B, 2C, *Λ)
        return out


class MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features, out_features, dropout: float = 0.0
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Dropout sits between fc1's ReLU and fc2. It acts on gauge-invariant
        # trace features, so it does not affect equivariance. Identity (no-op)
        # when dropout=0.0.
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        # (B, 2C, *Λ) -> (B, *Λ, 2C) so nn.Linear acts on the channel axis.
        # reshape() would reinterpret memory and scramble the per-site vectors;
        # movedim is the permutation we actually want.
        x = x.movedim(1, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ChannelLift(nn.Module):
    """Per-site complex linear over the channel axis.

    Gauge-equivariant by linearity: every output channel is a complex linear
    combination of input channels, so if every input channel transforms in
    the adjoint ``W → Ω W Ω†``, every output channel does too (the Ω, Ω†
    factors pull out of the sum because the mix coefficients are scalars).

    Used at the front of ``GELT`` to widen the plaquette channel count
    ``C_in = D(D-1)/2 ∈ {1, 3, 6}`` to a configurable ``d_model`` so the
    intermediate GEMHSA blocks do not collapse the residual stream to a
    handful of channels.

    Init is **identity-extend**: the first ``C_in`` output channels copy the
    input verbatim, the remaining ``d_model - C_in`` are zero. This makes
    ``d_model == C_in`` bit-exactly backward-compatible with the un-lifted
    model, and gives extra channels a clean starting point — the first
    GEMHSA's random Q/K/V projections + the residual ``W_lift + W_mix``
    populate them within one block.
    """

    def __init__(self, c_in: int, c_out: int, dtype=torch.complex64):
        super().__init__()
        if c_out < c_in:
            raise ValueError(
                f"ChannelLift expects c_out >= c_in (got c_in={c_in}, "
                f"c_out={c_out}). The lift is meant to widen the input."
            )
        self.c_in = c_in
        self.c_out = c_out
        weight = torch.zeros(c_out, c_in, dtype=dtype)
        weight[:c_in, :c_in] = torch.eye(c_in, dtype=dtype)
        self.weight = nn.Parameter(weight)

    def forward(self, W):
        # W: (B, C_in, *Λ, nc, nc) -> (B, C_out, *Λ, nc, nc).
        # Single matmul on the channel axis; the (nc, nc) matrices ride along
        # in the flattened trailing axis.
        B, C_in = W.shape[0], W.shape[1]
        trailing = W.shape[2:]
        W_flat = W.reshape(B, C_in, -1)
        out = torch.matmul(self.weight, W_flat)
        return out.view(B, self.c_out, *trailing)


class GELT(nn.Module):
    """Full GELT model:
    Pipeline:
      1. Compute Plaq (+ optional Poly)
      2. GEMHSA blocks with H heads and d_qkv channels per head.
      3. Trace block to get Re, Im parts of the trace as scalar per site.
      4. MLP with one hidden layer to mix the trace features and output a scalar per site for regression or classification.
      5. Spatial reduction (``reduction`` arg): ``"sum"`` for extensive
         per-config targets like the Wilson action, ``"mean"`` for the
         average Wilson loop ⟨W⟩, ``"none"`` to keep the per-site readout
         ``(B, *Λ)`` for per-site supervision (e.g. ``Re Tr W(R,T,x)/nc``).
    """

    def __init__(
        self,
        gaugegroup,
        L,
        D,
        R,
        nhead,
        gemhsa_layers,
        d_qkv=None,
        gate="softplus",
        dtype=torch.complex64,
        mlp_hidden=32,
        mlp_out=1,
        reduction: str = "sum",
        init_scale: float = 1.0,
        qk_init_scale: float = 1.0,
        mlp_zero_init: bool = True,
        d_model: int | None = None,
        mlp_dropout: float = 0.0,
        grad_checkpoint: bool = False,
        in_channels: int | None = None,
    ):
        # Plaquette input -> D(D-1)/2 plaquettes per site. Callers stacking
        # extra adjoint channels (e.g. plaquettes at several APE smearing
        # levels) pass the widened count via ``in_channels``; every channel
        # must still transform in the adjoint for the ChannelLift/GEMHSA
        # equivariance to hold.
        d_input = in_channels if in_channels is not None else D * (D - 1) // 2
        # Internal residual-stream width. Defaults to d_input (no lift) for
        # backward compatibility; pass d_model > d_input to widen the
        # intermediate channels via the front-end ChannelLift.
        if d_model is None:
            d_model = d_input
        if d_model < d_input:
            raise ValueError(
                f"d_model must be >= d_input = {d_input}, got {d_model}. "
                + ("d_input is in_channels as passed"
                   if in_channels is not None else "d_input = D(D-1)/2")
                + " — widen d_model, or feed fewer input channels."
            )
        self.d_input = d_input
        self.d_model = d_model
        super(GELT, self).__init__()
        if reduction not in ("sum", "mean", "none"):
            raise ValueError(
                f"reduction must be 'sum', 'mean', or 'none', got {reduction!r}"
            )
        self.reduction = reduction
        # Recompute each GEMHSA layer in backward instead of storing its
        # activations. The gathered K/V neighbourhoods are
        # (B, H, d, n_off, *Λ, nc, nc)-sized, so stored activations — not
        # weights or transports — are the training memory wall (V100 profile:
        # ~7 GiB/config at 4 layers without this). Costs ~one extra forward.
        self.grad_checkpoint = grad_checkpoint
        # Channel lift to widen the small plaquette input to d_model. When
        # d_model == d_input the lift is the identity matrix and is a no-op
        # at init (still trainable — the model can learn to mix plaquette
        # channels even at unchanged width).
        self.lift = ChannelLift(d_input, d_model, dtype=dtype)
        # ModuleList so the GEMHSA parameters are registered with PyTorch
        # and picked up by .parameters() / .to() / .state_dict().
        self.gemhsa_models = nn.ModuleList(
            [
                GEMHSA(
                    gaugegroup,
                    L,
                    D,
                    R,
                    d_model,
                    nhead,
                    d_qkv,
                    gate,
                    dtype,
                    init_scale=init_scale,
                    qk_init_scale=qk_init_scale,
                )
                for i in range(gemhsa_layers)
            ]
        )

        # Trace produces real values, so the MLP must live in the matching
        # real dtype — not the complex `dtype` of the GEMHSA stack. Blanket
        # `.to(complex_dtype)` on GELT would otherwise miscast the MLP.
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.mlp = MLP(2 * d_model, mlp_hidden, mlp_out, dropout=mlp_dropout).to(
            real_dtype
        )
        # Zero-init the MLP's last linear layer: at init the model outputs 0
        # at every site, so the untrained prediction is exactly 0.
        if mlp_zero_init:
            nn.init.zeros_(self.mlp.fc2.weight)
            nn.init.zeros_(self.mlp.fc2.bias)

    def set_introspection(self, store_attention=None, diagnostics=None):
        """Turn the per-layer introspection stashes on or off across the stack.

        Both are off by default because both cost time on every forward pass
        (see ``GEMHSA.__init__``). ``store_attention=True`` is what the
        interpretability scripts need: it keeps ``_last_alpha`` / ``_last_score``
        alive after the forward.
        """
        for layer in self.gemhsa_models:
            if store_attention is not None:
                layer.store_attention = store_attention
            if diagnostics is not None:
                layer.diagnostics = diagnostics
        return self

    def attn(self, W, T, T_dag):
        # The Δx = 0 slot is the same for every layer, so prepend it once here
        # rather than in each layer's forward (where it also re-ran inside every
        # gradient-checkpoint recompute).
        first = self.gemhsa_models[0]
        if T.shape[1] == first.n_offsets - 1:
            T, T_dag = first.prepend_self_offset(T, T_dag)
        for layer in self.gemhsa_models:
            if self.grad_checkpoint and self.training and torch.is_grad_enabled():
                # use_reentrant=False: T/T_dag carry no grad (only W and the
                # layer parameters do), which the non-reentrant variant
                # handles correctly. The _last_score/_last_alpha stashes are
                # rewritten during the backward recompute — same values.
                W = checkpoint(layer, W, T, T_dag, use_reentrant=False)
            else:
                W = layer(W, T, T_dag)
        return W

    def forward(self, W, T):
        # Cast inputs to the model's weight dtype once (hoisted out of every
        # GEMHSA layer's forward) so real-valued data (Z₂ float32 plaquettes)
        # can be fed to a complex model without a per-layer cast.
        first_layer = self.gemhsa_models[0]
        w_dtype = first_layer.w_QKV.dtype
        if W.dtype != w_dtype:
            W = W.to(w_dtype)
        if T.dtype != w_dtype:
            T = T.to(w_dtype)
        # Lift the (small) plaquette channel count to d_model before the
        # GEMHSA stack. With identity-extend init this is bit-exactly the
        # input when d_model == d_input.
        W = self.lift(W)
        # T_dag is shared across all stacked GEMHSA layers
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
