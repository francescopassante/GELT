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


class LConv(nn.Module):
    """L-Conv (Eq. 5): trainable parallel-transport convolution.

        W_{x,i}^{out} = Σ_{j, μ, k} ω[i, j, μ, k]
                        · U^(k)_μ(x) · W_{aug,j}(x + k·μ̂) · U^(k)†_μ(x)

    Operates on the augmented input ``W_aug = [1, W, W†]`` (``C → 2C + 1``).
    Shifts ``k = 0 … K``: ``k = 0`` is the identity-transport slot at the
    same site (μ-independent — the redundancy is harmless and matches the
    reference openpixi implementation). Negative shifts are subsumed by
    the W† channels of the augmentation.
    """

    def __init__(
        self,
        gaugegroup,
        c_in: int,
        c_out: int,
        D: int,
        K: int,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.gaugegroup = gaugegroup
        self.D = D
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        self.c_prime = 2 * c_in + 1  # augmented input width

        # ω[i, j, μ, k]. Variance ~ 1 / (c_prime · D · (K+1)) so the L-Conv
        # output starts at unit scale regardless of fan-in.
        n_terms = self.c_prime * D * (K + 1)
        sigma = 1.0 / math.sqrt(n_terms)
        self.w = nn.Parameter(
            torch.randn(c_out, self.c_prime, D, K + 1, dtype=dtype) * sigma
        )

    def forward(self, W: torch.Tensor, U_transports: torch.Tensor) -> torch.Tensor:
        """Run the L-Conv.

        ``W`` : ``(B, C_in, *Λ, nc, nc)``.
        ``U_transports`` : ``(B, D, K, *Λ, nc, nc)`` — k=1..K (the k=0 slot
        is synthesised here as the identity).
        """
        W_aug = _augment(W, self.gaugegroup)  # (B, C', *Λ, nc, nc)
        B = W_aug.shape[0]
        spatial = W_aug.shape[2:-2]
        nc = W_aug.shape[-1]
        D, K = self.D, self.K

        # Build W_shift[μ, k](x) = W_aug(x + k·μ̂) for k = 0 … K.
        # k=0 slot is W_aug itself (no shift); for k>0 roll along the
        # μ-th spatial axis (offset by 2 because of the (B, C') prefix).
        shifted_per_mu = []
        for mu in range(D):
            axis = 2 + mu
            per_k = [W_aug]
            for k in range(1, K + 1):
                per_k.append(torch.roll(W_aug, shifts=-k, dims=axis))
            shifted_per_mu.append(torch.stack(per_k, dim=2))  # (B, C', K+1, *Λ, nc, nc)
        W_shift = torch.stack(shifted_per_mu, dim=2)  # (B, C', D, K+1, *Λ, nc, nc)

        # Adjoint transport: U^(k)_μ · W_aug(x + k·μ̂) · U^(k)†_μ.
        # k=0 transport is the identity, handled by padding U_full with I.
        U_transports = U_transports.to(W_aug.dtype)
        identity_T = (
            torch.eye(nc, dtype=W_aug.dtype, device=W_aug.device)
            .view(1, 1, 1, *([1] * len(spatial)), nc, nc)
            .expand(B, D, 1, *spatial, nc, nc)
        )
        U_full = torch.cat([identity_T, U_transports], dim=2)  # (B, D, K+1, *Λ, nc, nc)
        U_dag = self.gaugegroup.dagger(U_full)

        # Broadcast U over the channel axis C' (the transport is the same
        # for every input channel at fixed (B, μ, k, x)).
        U_b = U_full.unsqueeze(1)  # (B, 1, D, K+1, *Λ, nc, nc)
        U_dag_b = U_dag.unsqueeze(1)
        W_transp = U_b @ W_shift @ U_dag_b  # (B, C', D, K+1, *Λ, nc, nc)

        # Linear mix over (j, μ, k) → C_out. Single fused matmul.
        n_terms = self.c_prime * D * (K + 1)
        w_flat = self.w.view(self.c_out, n_terms)
        W_flat = W_transp.reshape(B, n_terms, -1)
        out = torch.matmul(w_flat, W_flat).reshape(B, self.c_out, *spatial, nc, nc)
        return out


class LBilin(nn.Module):
    """L-Bilin (Eq. 6): site-local matrix-bilinear product.

        W_{x,i}^{out} = Σ_{j, j'} α[i, j, j'] · W_{aug,j}(x) · W'_{aug,j'}(x)

    Both inputs are augmented (``[1, ·, ·†]``) before the bilinear. The
    identity-channel of the augmentation gives every L-Bilin a free linear
    / residual / bias path; the dagger half gives orientation reversal.
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
        L_aug = _augment(W_left, self.gaugegroup)   # (B, c_left, *Λ, nc, nc)
        R_aug = _augment(W_right, self.gaugegroup)  # (B, c_right, *Λ, nc, nc)
        # Outer matrix product over channels: (j, j') → c_left·c_right entries
        # each of shape (nc, nc) at every site.
        L_e = L_aug.unsqueeze(2)  # (B, c_left, 1, *Λ, nc, nc)
        R_e = R_aug.unsqueeze(1)  # (B, 1, c_right, *Λ, nc, nc)
        prod = L_e @ R_e          # (B, c_left, c_right, *Λ, nc, nc)

        B = prod.shape[0]
        spatial = prod.shape[3:-2]
        nc = prod.shape[-1]
        w_flat = self.w.view(self.c_out, self.c_left * self.c_right)
        prod_flat = prod.reshape(B, self.c_left * self.c_right, -1)
        return torch.matmul(w_flat, prod_flat).reshape(
            B, self.c_out, *spatial, nc, nc
        )


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
    ):
        super().__init__()
        self.lconv = LConv(gaugegroup, c_in, c_out, D, K, dtype=dtype)
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
         (already built by the dataset).
      2. Stack of L-CB blocks (optionally followed by L-Act), each
         consuming the precomputed axis-aligned transports
         ``U^(k)_μ(x)`` from :func:`build_axis_transports`.
      3. ``Trace`` to extract gauge-invariant scalars per site.
      4. Per-site MLP head (one hidden layer). The paper explicitly avoids
         global average pooling at the end; supervision is per-site and
         the spatial reduction is applied last.
      5. Optional spatial reduction (``"sum"`` / ``"mean"`` / ``"none"``)
         to match the GELT model's reduction modes.
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

        c_in_plaq = D * (D - 1) // 2
        self.c_in_plaq = c_in_plaq
        self.c_hidden = c_hidden

        # Stack of L-CB (+ L-Act). The first block maps C_in_plaq → c_hidden;
        # subsequent blocks keep the width at c_hidden.
        widths = [c_in_plaq] + [c_hidden] * n_layers
        self.lcb_blocks = nn.ModuleList(
            [
                LCB(gaugegroup, widths[i], widths[i + 1], D, K, dtype=dtype)
                for i in range(n_layers)
            ]
        )
        self.l_acts = nn.ModuleList(
            [LAct(activation=gate) if use_l_act else nn.Identity() for _ in range(n_layers)]
        )

        # Real-valued per-site head (Trace produces 2·c_hidden reals per site).
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        self.trace = Trace()
        self.head_fc1 = nn.Linear(2 * c_hidden, mlp_hidden).to(real_dtype)
        self.head_fc2 = nn.Linear(mlp_hidden, mlp_out).to(real_dtype)

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
