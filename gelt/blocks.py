"""
=========================================================================
GEMHSA / GELT — gauge-equivariant attention block and the full model.
=========================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from gelt.lattice import l1_ball_offsets


class GEMHSA(nn.Module):
    """Gauge-equivariant multi-head self-attention block.

    The input is a batched covariant local field of shape ``(B, C, *Λ, nc, nc)``
    Every channel of W transforms in the adjoint representation, ``W → Ω W Ω†``.

    Pipeline:

    1. **augment + project.** Append the on-site identity and daggers
    (C → 2C + 1), then project to per-head Q, K, V of
       shape (B, H, d_qkv, *Λ, nc, nc).
    2. **adjoint transport.** For every Δx in the L1-ball of radius R, gather
       the neighbour fields K(x+Δx), V(x+Δx) and apply the
       shortest-path-averaged transport: K'(x+Δx->x) = T(x) · K(x+Δx) · T†(x)
    3. **gauge invariant score.** s = (1/√(nc·d)) · Re Σ_a Tr[Q†_a · K̃_a] per offset
       and per head, plus a learnable real bias ``b_h[h, n]`` per (head,
       offset).
       NOTE: The bias is intentionally untied across point-group-related offsets so
       that the model can favour specific axes — orbit tying was preventing axis selection
       through the attention in the Wilson loop targets
    4. **softmax** over the offset axis (normalizes over neighbours per
       site, per head).
    5. **multiplicative value path.** Output of the attention head is
       Σ_i α_i · Q†(x) · V'_i(x) — both factors are covariant at x, so the
       product is covariant; this is the L-Bilin-baked-in step that gives the
       loop-doubling expressivity argument.
       NOTE: could use another projection for Q, more general
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
        alpha_init: float = 0.0,
        init_scale: float = 1.0,
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

        # Per-offset score bias. Previously this was tied across the lattice
        # rotation/reflection symmetries, enforcing architectural isotropy. That tying is
        # a stronger symmetry than most physical targets have: e.g. a 1×R
        # rectangular Wilson loop in the (μ,ν) can't be learned this way.
        self.b_h = nn.Parameter(torch.zeros(self.H, self.n_offsets))
        # Precomputed reshape target for broadcasting b_h over (B, *Λ) at score time.
        self._bias_view_shape = (1, self.H, self.n_offsets) + (1,) * D

        # Channel augmentation expands
        # C -> C' = 2C + 1 by appending the identity and daggers.
        self.C_prime = 2 * d_input + 1

        # Cached on-site identity for `augment`: shape (1, 1, *[1]*D, nc, nc),
        # broadcast at forward time. Avoids re-allocating torch.eye every step.
        nc = gaugegroup.nc
        identity = torch.eye(nc, dtype=dtype).view(1, 1, *([1] * D), nc, nc)
        self.register_buffer("_identity", identity)

        # Fused QKV projection.
        # a single (3·H·d, C') @ (B, C', N) matmul instead of three separate ones
        # σ ≈ 0.02·init_scale / √C' (real & imag parts independently). With
        # init_scale=1 and small w^V combined with the residual connection,
        # the block is roughly identity at init (-> stackable layers).
        sigma = 0.02 * init_scale / math.sqrt(self.C_prime)
        self.w_QKV = nn.Parameter(
            torch.randn(3, self.H, self.d_qkv, self.C_prime, dtype=dtype) * sigma
        )
        # channel mix back to C output channels.
        sigma_mix = 0.02 * init_scale / math.sqrt(self.H * self.d_qkv)
        self.w_mix = nn.Parameter(
            torch.randn(self.C, self.H, self.d_qkv, dtype=dtype) * sigma_mix
        )

        # ReZero / LayerScale: per-block learnable scalar α. alpha_init=0
        # gives identity-at-init property, but pairs badly with the MLP zero-init
        # on hard targets. Init α to a small positive value
        # (e.g. 0.05) forces the multiplicative path to contribute from the start
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))

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

    def attend(self, Q, K, V, T, T_dag):
        """Fully batched gauge-equivariant attention over the L1-ball.

        Single fused pass — no Python loop over offsets. Pipeline:
          1. Gather K(x+Δx_i), V(x+Δx_i)
          2. Adjoint transport: K' = T(x) · K(x+Δx) · T†(x)
          3. Scalar score Re Σ_c Tr[Q_c† · K̃_c] / √(d_qkv·nc) computed as a
             Frobenius product, plus the per-(head, offset) bias.
          4. Softmax over the offset axis.
          5. Value path Q† · V' — but α-weighted *before* the matmul
             Σ_n α_n (Q† Ṽ_n) = Q† (Σ_n α_n Ṽ_n)

        ``T`` and ``T_dag`` are shared across all GEMHSA layers in a stacked
        GELT; the dagger is computed once at the GELT level and threaded in.
        """
        nc = Q.shape[-1]

        # 1. Neighbour gather.
        idx = tuple(
            self._nbr_idx[k] for k in range(self.D)
        )  # (n_off, *Λ) D dimensional vectors
        # nb_indexer = (:, :, :, ?, :, :) -> ? across dimension *Λ selects neighbors for each lattice site
        nb_indexer = (slice(None),) * 3 + idx + (slice(None), slice(None))
        K_nb = K[
            nb_indexer
        ]  # (B, H, d_qkv, n_off, *Λ, nc, nc) -> for each lattice site and neighbor, a (B, H, d, nc, nc) K tensor
        V_nb = V[nb_indexer]  # same

        # 2. Transport: T · X · T†. T and its dagger are shared across heads,
        # channels, and (in GELT) all stacked GEMHSA layers.
        # K and V use the same transport. Concatenate along the channel axis,
        # then split after the transport to save one transport
        KV_nb = torch.cat((K_nb, V_nb), dim=2)
        del K_nb, V_nb
        KV_tilde = self.transport(KV_nb, T, T_dag)
        K_tilde, V_tilde = KV_tilde.split(self.d_qkv, dim=2)

        # 3. Score = Tr[Q_c† K'_c]/sqrt(Nc d_qkv); implementable via
        # Frobenius product without matmul.
        Q_e = Q.unsqueeze(3)  # (B, H, d_qkv, 1, *Λ, nc, nc)
        score = (Q_e.conj() * K_tilde).sum(dim=(2, -2, -1)).real
        score = score / math.sqrt(self.d_qkv * nc)
        # score: (B, H, n_off, *Λ)

        # 4. Per-offset bias b_h[h, n], broadcast over (B, *Λ). The view
        # shape is precomputed at __init__; the .real fallback handles the
        # case where module-wide ``.to(complex_dtype)`` upcast the parameter
        # (b_h is initialised real, but tests cast the whole block).
        # bias = self.b_h.real if self.b_h.is_complex() else self.b_h
        score = score  # + bias.view(self._bias_view_shape)

        # 5. Softmax over offsets.
        alpha = torch.softmax(score, dim=2)

        # 6. Value path. Sum V' over n_off with α weights BEFORE the
        # Q† matmul.
        # alpha: (B, H, n_off, *Λ) → (B, H, 1, n_off, *Λ, 1, 1) to broadcast
        # over the d_qkv and the two color axes.
        alpha_b = alpha.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
        V_weighted = (alpha_b * V_tilde).sum(dim=3)  # (B, H, d_qkv, *Λ, nc, nc)
        Q_dag = self.gaugegroup.dagger(Q)  # (B, H, d_qkv, *Λ, nc, nc)
        # Two value-path branches with separate learnable scalar weights:
        #   alpha_attn * V_weighted              — transformer-style attention
        #                                          sum (no Q† factor).
        #   alpha_bilin * (Q† · V_weighted)      — L-Bilin loop-doubling step.
        # Both are gauge-equivariant at x (V_weighted is the attention-weighted
        # sum of transported V's; Q†·V_weighted multiplies on the left by Q†
        # which transforms with Ω_x on both sides).
        bilin = torch.matmul(Q_dag, V_weighted)  # (B, H, d_qkv, *Λ, nc, nc)
        return bilin
        # return self.alpha_attn * V_weighted + self.alpha_bilin * bilin

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
        # (whose transport is the identity) is synthesised here.
        expected_external = self.n_offsets - 1
        assert T.shape[1] == expected_external, (
            f"Expected T.shape[1] == {expected_external} (non-zero offsets), "
            f"got {T.shape[1]}"
        )

        nc = W.shape[-1]
        B = T.shape[0]
        spatial = T.shape[2:-2]
        identity_T = (
            torch.eye(nc, dtype=T.dtype, device=T.device)
            .view(1, 1, *([1] * self.D), nc, nc)
            .expand(B, 1, *spatial, nc, nc)
        )
        T = torch.cat([identity_T, T], dim=1)
        if T_dag is None:
            T_dag = self.gaugegroup.dagger(T)
        else:
            # T_dag was computed for the external (non-zero) offsets;
            # prepend the identity (its own dagger) so it lines up with T.
            T_dag = torch.cat([identity_T, T_dag], dim=1)

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

        # Transport, score, softmax, multiplicative value.
        out = self.attend(Q, K, V, T, T_dag)  # (B, H, d_qkv, *Λ, nc, nc)

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
        return W + W_act
        # ReZero: blend toward the L-Act output with a per-block scalar α
        # (zero-init). At α=0 the block is bit-exactly the identity W → W;
        # during training α grows and the gate/mix path takes over.
        # return W + self.alpha * (W_act - W)


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
        alpha_init: float = 0.0,
        init_scale: float = 1.0,
        mlp_zero_init: bool = True,
        d_model: int | None = None,
        mlp_dropout: float = 0.0,
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
                    alpha_init=alpha_init,
                    init_scale=init_scale,
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

    def attn(self, W, T, T_dag):
        for layer in self.gemhsa_models:
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
