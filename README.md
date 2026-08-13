# GELT — Gauge-Equivariant Lattice Transformer

Master's thesis: a gauge-equivariant attention network (GELT) for SU(N_c)
lattice gauge theory. The architecture is built on the L-CNN framework
([Favoni et al., 2021](https://arxiv.org/abs/2012.12901)) with two departures:

- **Matrix-bilinear value path.** The standard scalar-weighted value is replaced
  by `α · Q† · Ṽ`, so L-CNN's loop-doubling universality argument transfers
  directly: each block roughly doubles the maximum loop length reachable.
- **Shortest-path-averaged transport.** Parallel transport between sites is
  averaged over *all* shortest lattice paths in the L1-ball of Manhattan radius R
  (computed via a DP recursion), giving a non-axis-aligned, gauge-covariant
  receptive field without enumerating paths explicitly.

The gauge-invariant attention score `Re Tr[Q† · K̃]` is a two-loop correlator —
the natural matrix generalisation of the standard inner product `q†k`, and a
well-known observable in lattice QCD (glueball propagators, Polyakov-loop
correlators, string-tension measurements).

---

## Results so far

**0⁺⁺ glueball spectroscopy.** Trained as a variational operator on the Rayleigh
loss `−C(1)/C(0)`, GELT saturates the transfer-matrix bound on an anisotropic
SU(2) lattice (L=12, Lt=24, β=2.4, ξ=3.0) and **beats the classical multi-level
GEVP on ground-state overlap**: ΔA₀ = +0.078 ± 0.022 (3.6σ), combining the
original run with an independent fresh-ensemble replication, at a consistent
mass. Multi-level smeared input channels were the enabler — with thin-plaquette
input the operator only relearns APE×2's staple content. Written up in
`glueball_report/`.

**Attention as a lattice operator.** Because the attention score is gauge
invariant, any reduction of the attention map is a local scalar lattice
operator — so its connected correlator has a mass. Measured on 3D Z₂ near
criticality, `ξ_A = 1/m_A` tracks the classical correlation length at Pearson
**0.9966** over a factor 2.7 in ξ, surviving a non-monotonic excursion. The
random-init control also tracks ξ, so the *structural* claim is established
while the *learning* claim rests on the correlated ΔA₀ (5.4σ–23.5σ). Written up
in `attention_report/`.

Both reports also record the negative results, which are part of the record:
attention does **not** localize on topological charge, and the attention-*range*
statistic `ℓ_att` is bounded by R and centred by the ball geometry, making it
unusable as a correlation-length probe. See `notes/topological_localization.md`.

---

## Architecture overview

```
Input: link configuration U  (D, *Λ, N_c, N_c)
          │
          ▼
    Preprocessing:  1×1 plaquettes  →  D(D-1)/2 W-channels
          │
          ▼  (×n_blocks)
    GEMHSA block
    ├── Augment W → [𝟙, W, W†]
    ├── Q, Q_v, K, V projections  (per-site, per-head, gauge-covariant)
    ├── build_transport_average(U, R)  →  T_Δx(x) for |Δx|₁ ≤ R (U batched)
    ├── K̃, Ṽ = T_Δx · K · T_Δx†    (parallel transport to site x)
    ├── score = Re Tr[Q† · K̃]  +  RoPE rotation *or* learned offset bias
    ├── α = softmax(scores)
    ├── W_out += Σ_y  α_{x→y} · Q_v†_x · Ṽ_{y→x}   (matrix-bilinear value)
    ├── channel mix  →  C_out W-channels
    └── residual + L-Act gate
          │
          ▼
    Readout:  Re Tr → per-site MLP → spatial reduction
```

`_last_score` / `_last_alpha` are stashed per layer under `no_grad` — the hook
the interpretability program reads the attention out through.

---

## Repository layout

```
gelt/                    library (installed editable via pyproject.toml)
  lattice.py             GaugeGroup ABC + Z2/SU(N); plaquettes, Wilson action
                         (anisotropic), topological charge, Wilson loops,
                         l1_ball_offsets, build_transport_average
  sampler.py             Metropolis (Z2 + SU(2)), Z2 heat-bath, SU(2)
                         heat-bath + overrelaxation, mcmc_ensemble,
                         haar_ensemble, integrated_autocorrelation_time
  blocks_rope.py         GELT with rotary positional encoding  ← the trained one
  blocks_bias.py         GELT with a learned offset bias       ← the tested one
  lcnn.py                Favoni et al. L-CNN — equivariant baseline
  cnn_baseline.py        LatticeCNN — non-equivariant reference (2D only)
  glueball.py            0⁺⁺ spectroscopy: APE smearing, correlators, m_eff,
                         multi-level GEVP, cosh fits, overlap A₀, jackknife
  topology.py            cooling + cooled charge density
  data.py                dataset construction and splits

scripts/                 entry points (each self-contained; see CLAUDE.md)
tests/                   pytest — gauge invariance/equivariance, sampler
                         exactness, transport, glueball arithmetic
notes/                   design records and the run-by-run experimental log
glueball_report/         LaTeX write-up of the spectroscopy result
attention_report/        LaTeX write-up of the attention-as-operator result
```

**`CLAUDE.md` is the maintained source of truth** for module-by-module detail,
conventions, current status, and known caveats. This README is the summary.

---

## Installation

```bash
git clone git@github.com:francescopassante/GELT.git
cd GELT
uv venv && source .venv/bin/activate
uv pip install -e .
```

---

## Usage

```bash
# Sampler validation (four-panel, one per group)
python scripts/validate_sampler_z2.py
python scripts/validate_sampler_su2.py
python scripts/validate_anisotropy.py

# Classical 0⁺⁺ baseline: correlator + GEVP effective mass
python scripts/measure_glueball.py

# Train GELT as a variational glueball operator (V100-scale)
python scripts/train_glueball.py

# Cosh fits + ground-state overlap A₀ (offline, CPU)
python scripts/fit_glueball_overlap.py

# Attention readouts
python scripts/visualize_glueball_attention.py
python scripts/z2_attention_correlator.py

# Quick gauge-invariance check on the full model
python scripts/check_gelt_invariance.py

# Unit tests
pytest tests/
```

Device order is cuda → mps → cpu. `datasets/`, `*.pth`, `*.png` and `download/`
are gitignored; the training scripts cache their ensembles under `datasets/`.

---

## Tensor conventions

| Object | Shape | Notes |
|---|---|---|
| Links U | `(D, *Λ, N_c, N_c)` | direction first, color last |
| Plaquettes P | `(D(D-1)/2, *Λ, N_c, N_c)` | (μ,ν) pairs, μ < ν |
| W-channels | `(B, C, *Λ, N_c, N_c)` | batch and channel first |
| Transport T | `(N, n_offsets, *Λ, N_c, N_c)` | offsets in `l1_ball_offsets` order |

- Periodic BCs via `torch.roll` throughout — no manual index arithmetic.
- Color axes are always present, even for Z₂ (`N_c = 1`), so every matmul
  ports verbatim to U(1)/SU(N). Never broadcast across color axes implicitly.
- Wilson action: `S = β Σ_p (1 − Re Tr P_p / N_c)`. Anisotropic when `ξ ≠ 1`
  (temporal plaquettes weighted `β·ξ`, spatial `β/ξ`); `ξ = 1` is bit-exact.
- Time is lattice axis 0 throughout the spectroscopy code.
- Float32 for training; float64/complex128 for gauge-invariance unit tests.

---

## Why attention beats convolution here

For a non-equivariant CNN, predicting the Wilson action from link variables
requires the network to learn "multiply four specific link values around a
plaquette" — a product the convolutional kernel cannot express with its additive
inductive bias. R² ≈ 0 across all L confirms this on Haar-random data.
With plaquettes as input, R² ≈ 0.99: the task collapses to a linear sum.

The GELT closes this gap by construction: the matrix-bilinear value path
`Q† · Ṽ` directly encodes multiplicative loop content, and the attention scores
weight neighbors by physical relevance.

---

## References

- Favoni, Ipp, Müller, Schuh (2021). *Lattice Gauge Equivariant Convolutional Neural Networks.*
  [arXiv:2012.12901](https://arxiv.org/abs/2012.12901)
- Nagai, Tomiya (2021). *Gauge covariant neural network for 4-dimensional non-Abelian gauge theory.*
  [arXiv:2103.11965](https://arxiv.org/abs/2103.11965)
- Nagai, Ohno, Tomiya (2025). *CASK: gauge-covariant surrogate action.*
  [arXiv:2501.16955](https://arxiv.org/abs/2501.16955)
