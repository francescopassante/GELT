# GELT — Gauge-Equivariant Lattice Transformer

Master's thesis: a gauge-equivariant graph-attention network (G-GAT) for
SU(N_c) lattice gauge theory. The architecture is built on the L-CNN framework
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

The CNN in `gelt/cnn_baseline.py` serves as a non-equivariant reference.
The G-GAT is not yet implemented; the codebase is at Phase 0 of the roadmap.

---

## Architecture overview

```
Input: link configuration U  (D, *Λ, N_c, N_c)
          │
          ▼
    Preprocessing
    ├── Plaq:    1×1 plaquettes  →  D(D-1)/2 W-channels
    └── Poly:   Polyakov loops   →  D extra W-channels  [optional]
          │
          ▼  (×n_blocks)
    G-Attn block
    ├── Augment W → [𝟙, W, W†]
    ├── Q, K, V projections  (per-site, per-head, gauge-covariant)
    ├── build_transport_sums(U, R)  →  T_Δx(x) for |Δx|₁ ≤ R
    │        (DP over positive octant; negatives via octant trick)
    ├── K̃, Ṽ = T_Δx · K · T_Δx†    (parallel transport to site x)
    ├── score = Re Tr[Q† · K̃] / √(N_c · d)  +  learned position bias
    ├── α = softmax(scores)
    ├── W_out += Σ_y  α_{x→y} · Q†_x · Ṽ_{y→x}   (multiplicative value)
    ├── channel mix  →  C_out W-channels
    └── residual + L-Act
          │
          ▼
    Readout
    └── Re Tr head → MLP → scalar
```

---

## Repository layout

```
gelt/                   library (install editable via pyproject.toml)
  lattice.py            GaugeGroup ABC + Z2; plaquette, action, gauge_transformation
  sampler.py            Metropolis sweep (checkerboard); mcmc_ensemble, haar_ensemble
  data.py               build_link_datasets, build_plaquette_datasets
  cnn_baseline.py       LatticeCNN — non-equivariant reference, D=2/3/4+
  train.py              train_model, full_pipeline (early stopping, R²)

scripts/
  main.py               single-(L, β) CNN run
  L_scan.py             replay saved L-scan; absolute-MSE + R² panels
  lr_scan.py            learning-rate sweep at fixed L
  validate_sampler.py   Z₂ Metropolis sanity check (4-panel)
  visualize.py          2D lattice visualisation

tests/
  test_lattice.py       gauge-invariance unit tests (plaquette, action, gauge_transformation)
  test_data_model.py    split validation + CNN shape guards

notes/
  architecture.md       full G-GAT spec (§10 build-order checklist)
  roadmap.md            phased plan — Z₂ through SU(3), plus novel directions
  papers_review.md      literature review: L-CNN, CovResNet, CASK
  sampling.md           MC sampler strategy
  resources.md          curated reading list
  tunnel-visualization.md  exploratory notes on topological-charge visualisation
```

---

## Installation

```bash
git clone git@github.com:francescopassante/GELT.git
cd GELT
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with plain pip:
```bash
pip install -e ".[dev]"
```

---

## Usage

```bash
# CNN baseline: single (L, β) run
python scripts/main.py

# Replay saved L-scan and generate R² plots
python scripts/L_scan.py

# Validate the Z₂ Metropolis sampler
python scripts/validate_sampler.py

# Unit tests
pytest tests/

# CNN architecture summary (5×5, D=2)
python -m gelt.cnn_baseline
```

---

## Tensor conventions

| Object | Shape | Notes |
|---|---|---|
| Links U | `(D, *Λ, N_c, N_c)` | direction first, color last |
| Plaquettes P | `(D(D-1)/2, *Λ, N_c, N_c)` | (μ,ν) pairs, μ < ν |
| W-channels | `(B, C, *Λ, N_c, N_c)` | batch and channel first |

- Periodic BCs via `torch.roll` throughout — no manual index arithmetic.
- Color axes are always present, even for Z₂ (`N_c = 1`), so every matmul
  ports verbatim to U(1)/SU(N).
- Wilson action: `S = β Σ_p (1 − Re Tr P_p / N_c)`.

---

## Roadmap

| Phase | Setting | Goal |
|---|---|---|
| **0** | 2D Z₂ | Implementation validation; gauge-invariance unit tests |
| **1** | 3D Z₂ | 3D Ising duality; critical exponents (ν, β_c) |
| **2** | 4D U(1) | First continuous group; monopole condensation |
| **3** | 4D SU(2) | Replicate L-CNN benchmarks; matched-parameter shootout |
| **4** | 4D SU(2) + fermions | SLHMC surrogate action (CASK comparison) |
| **5** | 4D SU(3) | String tension, glueball mass, topological susceptibility |
| **6** | Various | Cross-β transfer, attention-as-ξ, trivializing flows, … |

Current position: **Phase 0** (post-refactor, G-GAT not yet implemented).
The next concrete step is `build_transport_sums(U, R)` — the DP routine for
shortest-path-averaged parallel transport (see `notes/architecture.md` §10).

---

## Why attention beats convolution here

For a non-equivariant CNN, predicting the Wilson action from link variables
requires the network to learn "multiply four specific link values around a
plaquette" — a product the convolutional kernel cannot express with its additive
inductive bias. R² ≈ 0 across all L confirms this on Haar-random data.
With plaquettes as input, R² ≈ 0.99: the task collapses to a linear sum.

The G-GAT closes this gap by construction: the matrix-bilinear value path
`Q† · Ṽ` directly encodes multiplicative loop content, and the attention scores
weight neighbors by physical relevance — in principle learning to look exactly
as far as the correlation length demands.

---

## References

- Favoni, Ipp, Müller, Schuh (2021). *Lattice Gauge Equivariant Convolutional Neural Networks.*
  [arXiv:2012.12901](https://arxiv.org/abs/2012.12901)
- Nagai, Tomiya (2021). *Gauge covariant neural network for 4-dimensional non-Abelian gauge theory.*
  [arXiv:2103.11965](https://arxiv.org/abs/2103.11965)
- Nagai, Ohno, Tomiya (2025). *CASK: gauge-covariant surrogate action.*
  [arXiv:2501.16955](https://arxiv.org/abs/2501.16955)
