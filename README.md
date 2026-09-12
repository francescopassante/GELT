# GELT — Gauge-Equivariant Lattice Transformer

Master's thesis: a gauge-equivariant attention network for SU(N_c) lattice gauge
theory. The architecture is built on the L-CNN framework
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
well-known observable in lattice QCD.

> **Scope of this repository (2026-09-09).** It holds what reproduces
> **`../report/main.tex`**, plus two audits of that content: the SU(2) fair fight
> and the operator decomposition. Studies that closed negative or never reached
> the paper — the dual-Ising ground truth and the ν-exponent fits, the
> topological-localization readout, the ℓ_att attention-range statistic, the
> rotation-irrep projection, the Z₂ fair fight — were **deleted**, not archived.
> They are in the history at commit `cfa0a7e` and their design records with them.

---

## Reproducing the report

`../report/main.tex`, section by section. Everything marked *GPU* is a V100-scale
run; everything marked *offline* is CPU-seconds from artifacts already in the
repo.

| main.tex | claim | how |
|---|---|---|
| §Validation and tests | "gauge equivariance is verified to machine precision in complex128" | `pytest tests/test_blocks.py` (equivariance + the naive-oracle equivalence); `python scripts/check_gelt_invariance.py` prints the drift (8.9e-16) |
| §Validation and tests, `fig:wilson_loop_regression` | CNN (~500k par.) vs GELT (~1.5k par.) on a per-site 1×2 Wilson loop | `python scripts/train_cnn.py` and `python scripts/train_gelt.py` — same D, L, group, sampler, loop and splits by construction; scatters land in `results/wilson_regression/` |
| §SU(2) Glueball spectroscopy, `tab:meff` + `fig:glueball_meff` | learned operator vs classical GEVP, m_eff(Δ) | *GPU:* `python scripts/measure_glueball.py` (ensemble + classical baseline), then `python scripts/train_glueball.py` |
| §…, ΔA₀ = +0.078 ± 0.022 (3.6σ) | cosh fits, overlap A₀, correlated differences, `fig:glueball_overlap` | *offline:* `python scripts/fit_glueball_overlap.py dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt` (and the `_ens1` dump for the replication) |
| §…, the replication | the second, independently sampled ensemble | *GPU, ~24 h:* `bash scripts/overnight_replication.sh` |
| §Attention as a physical field, `tab:train_rnd_gevp` | ξ_A and A₀, trained / random / GEVP, four β in 3D Z₂ | *GPU:* `python scripts/z2_beta_scan.py` (ensembles + classical mass), `Z2G_R=6 Z2G_N_USE=800 python scripts/train_z2_glueball.py <β>` once per β, then `python scripts/z2_attention_correlator.py`. *offline replot:* `ZAC_REPLOT=results/attention/z2_attention_correlator_diag_R6.pt python scripts/z2_attention_correlator.py` |
| §…, `tab:su2_train_rnd_gevp` | the same measurement on SU(2) at β = 2.4 | *GPU, ~40 min:* `python scripts/su2_attention_correlator.py` |

Two audits of the above, neither of which main.tex quotes:

| question | how |
|---|---|
| Is the classical comparator a straw man? | *offline:* `SFF_NOCACHE=1 python scripts/su2_fair_fight.py` reproduces the published ΔA₀; drop the flag (and give it the SU(2) ensemble) for the strengthened `deep` / `shapes` / `full` arms. Verdict in `notes/audit_2026-09-06.md` §6.4/§6.5 |
| Did the network find operator content the classical basis cannot express? | *offline:* `python scripts/operator_decomposition.py` — 12.9% of the norm² outside the span, ΔA₀ = +0.076 ± 0.019 (4.0σ). `notes/operator_decomposition.md` |
| Is the advantage the architecture, or just richer inputs? | *GPU, ~1.5 days:* `bash scripts/curve_batch.sh` (3 trained points + the untrained trace), then *offline:* one `SFF_TRUNCATE=1 SFF_BASES=1 python scripts/su2_fair_fight.py <dump>` per dump and `python scripts/input_architecture_curve.py`. A₀ against input content for classical / trained / untrained. Verdict in `notes/fable5.1_10-09_audit.md` §8.2 |

**Known caveat that touches the Z₂ table.** Projected Z₂ APE smearing has no
tunable radius at any α, and at the production `SMEAR_ALPHA = 0.5` it is not
gauge covariant: the classical Z₂ comparator is one operator, not four, and the
Z₂ networks were trained on four input channels of which three are byte-identical.
SU(2) is verified clean (1.4e-15), so the spectroscopy headline is untouched.
Full measurement and fix in `notes/audit_2026-09-06.md` §2.

---

## Layout

```
gelt/                  library (installed editable via pyproject.toml)
  lattice.py           GaugeGroup ABC + Z2/SU(N); plaquettes, Wilson action
                       (anisotropic), topological charge, Wilson loops,
                       l1_ball_offsets, build_transport_average
  sampler.py           Metropolis (Z2 + SU(2)), Z2 heat-bath, SU(2) heat-bath +
                       overrelaxation, mcmc_ensemble, haar_ensemble,
                       integrated_autocorrelation_time
  blocks.py            GEMHSA / ChannelLift / Trace / MLP / GELT — the one block
  glueball.py          0⁺⁺ spectroscopy: APE smearing, correlators, m_eff,
                       multi-level GEVP, cosh fits, overlap A₀, jackknife
  lcnn.py              Favoni et al. L-CNN — equivariant baseline
  cnn_baseline.py      LatticeCNN — non-equivariant reference
  data.py              dataset construction and splits

scripts/               entry points, flat and self-contained (table below)
tests/                 pytest: gauge invariance/equivariance, sampler exactness,
                       transport, glueball arithmetic
notes/                 design records and the run-by-run experimental log
reports/               LaTeX write-ups and their PDFs (paper / glueball / attention)
dumps/                 the two test-split Ō dumps — tracked on purpose
results/               generated figures, checkpoints, dumps (gitignored)
datasets/              cached ensembles (gitignored)
PLANS.md               future directions — proposals, none implemented
CLAUDE.md              module-by-module detail, conventions, status, caveats
```

### scripts/

| script | what it does |
|---|---|
| `validate_sampler_z2.py`, `validate_sampler_su2.py` | four-panel Metropolis sanity check against the exact mean plaquette |
| `validate_anisotropy.py` | ξ=1 regression, the ⟨P_st⟩ > ⟨P_ss⟩ split, renormalized vs bare anisotropy |
| `check_gelt_invariance.py` | sixty-second gauge-invariance check on the full model |
| `train_cnn.py`, `train_gelt.py`, `train_lcnn.py` | per-site Wilson-loop regression: CNN, GELT, L-CNN |
| `check_glueball_autocorrelation.py` | τ_int of the smeared operator — sets the production `n_skip` |
| `curve_batch.sh` | the V100 batch behind the curve: the untrained trace, the thin points, the width control |
| `input_architecture_curve.py` | assembles the per-dump fair fights into A₀(x), the figure, the LaTeX table and the pre-registered tests |
| `measure_glueball.py` | classical 0⁺⁺ baseline: correlator, GEVP m_eff, ensemble cache |
| `train_glueball.py` | GELT as a variational operator on the Rayleigh loss |
| `fit_glueball_overlap.py` | cosh fits, overlap A₀, correlated (Δm, ΔA₀) — offline |
| `overnight_replication.sh` | fresh ensemble + from-scratch training, unattended |
| `operator_decomposition.py` | O_GELT = P + r against the classical span — offline |
| `su2_fair_fight.py` | the strengthened classical arms — is the comparator fair? |
| `z2_beta_scan.py` | 3D Z₂ classical mass vs β: the ensembles and the reference ξ |
| `train_z2_glueball.py` | one variational operator per β, 3D Z₂ |
| `z2_attention_correlator.py` | ξ_A and A₀ of the attention field vs classical vs random |
| `su2_attention_correlator.py` | the same measurement on anisotropic SU(2) |
| `profile_glueball_step.py` | where one training step goes, per stage, fwd and bwd |

---

## Installation

```bash
git clone git@github.com:francescopassante/GELT.git
cd GELT
uv venv && source .venv/bin/activate
uv pip install -e .
pytest tests
```

Device order is cuda → mps → cpu. `datasets/`, `results/`, `*.pth` and `*.png`
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
- Color axes are always present, even for Z₂ (`N_c = 1`), so every matmul ports
  verbatim to U(1)/SU(N). Never broadcast across color axes implicitly.
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

GELT closes this gap by construction: the matrix-bilinear value path `Q† · Ṽ`
directly encodes multiplicative loop content, and the attention scores weight
neighbours by physical relevance.

---

## References

- Favoni, Ipp, Müller, Schuh (2021). *Lattice Gauge Equivariant Convolutional Neural Networks.*
  [arXiv:2012.12901](https://arxiv.org/abs/2012.12901)
- Morningstar, Peardon (1999). *Efficient glueball simulations on anisotropic lattices.*
  [hep-lat/9901004](https://arxiv.org/abs/hep-lat/9901004)
- Nagai, Tomiya (2021). *Gauge covariant neural network for 4-dimensional non-Abelian gauge theory.*
  [arXiv:2103.11965](https://arxiv.org/abs/2103.11965)
- Nagai, Ohno, Tomiya (2025). *CASK: gauge-covariant surrogate action.*
  [arXiv:2501.16955](https://arxiv.org/abs/2501.16955)
