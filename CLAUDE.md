# GELT — Gauge-Equivariant Neural Networks for Lattice Gauge Theory

Master's thesis codebase. Goal: build a **gauge-equivariant graph-attention
network (or transformer) (GELT - gauge equivariant lattice transformer)**
for SU(N_c) lattice gauge theory, starting from 2D Z₂ as
a debug-friendly testbed and scaling toward U(1)/SU(2)/SU(3) and 3+1D.

The architecture follows the L-CNN framework (Favoni et al. 2012.12901)
for primitives and gauge-equivariance proofs, with two departures:
(i) the L-Conv + L-Bilin stack is replaced by an attention block whose
**value path is matrix-bilinear** (`α · Q† · Ṽ`), so L-CNN's loop-doubling
universality argument transfers directly; (ii) parallel transport between
sites is **averaged over all shortest lattice paths** in the L1-ball of
Manhattan radius R (computed by a DP recursion, not enumeration), so each
block already reaches the full L1-ball receptive field with non-axis-aligned
loop content.

## Documents

All in `notes/`. (The previous `architecture.html` and `roadmap.html` were
removed pending rewrites; the spec now lives across the notes below.)

- `notes/abstract.md` — the thesis abstract: GELT as a gauge-equivariant
  attention encoder, gauge-invariant scores + matrix-bilinear value path,
  L1-ball shortest-path transport, RoPE geometric prior, and the
  interpretability program (attention vs. topology / correlation length).
- `notes/GELTsummary.md` — plain-language tour of the codebase modules
  (lattice / sampler / data / blocks) and a step-by-step walk through the
  `GELT` forward pass and the `GEMHSA` layer.
- `notes/explainability.md` — the thesis spine: *the attention map is a
  measurement*. Why equivariance makes attention physically interpretable,
  the three interpretability studies (emergent correlation length,
  localization on topological lumps, head/layer specialization), and how to
  run them (extract `_last_alpha`, ablation/intervention, validation against
  cooled `q(x)` / `ξ(β)`).
- `notes/fable_audit.md` — a code+notes audit (architecture / efficiency /
  explainability feasibility) with a prioritized to-do list. The current
  source of truth for known issues; the "Suggested next steps" section
  below mirrors it.
- `notes/papers_review.md` — full literature review of L-CNN, the
  gauge-covariant ResNet (Nagai-Tomiya 2103.11965), and CASK (2501.16955).
  Sections 0 (lattice primer) and 1 (L-CNN) are the architecture
  prerequisites.
- `notes/sampling.md` — strategy notes for the MC sampler (single-site
  Metropolis for Z₂; heat-bath + overrelaxation now implemented for SU(2),
  extension plan to U(1)/SU(3)).
- `notes/glueball_spectroscopy.md` — the plan for moving GELT from
  per-configuration regression to **0⁺⁺ glueball spectroscopy**: GELT as a
  learned variational operator trained on the Rayleigh loss `−C(1)/C(0)`
  (the converged loss *is* the glueball mass), the classical
  correlator/`m_eff` baseline it is validated against, the central role of
  spatial smearing, and the heat-bath sampler as the prerequisite long pole.
  Contains the **2026-07-01 audit** amending the §6.2 plan (per-timeslice 3D
  operator, code blockers, revised checklist — the source of truth for
  `train_glueball.py`).
- `notes/topological_localization.md` — **design record for the
  interpretability program promised in the conference abstract**: why clause 1
  (attention localizing on topological structure) needs the 4D `q(x)` model
  rather than the 3D glueball operator, the cooling ground truth (`n_cool=35`
  from the pre-flight, and why the naive charge's `Z < 1` is harmless to a rank
  statistic), the lattice/ensemble choices with reasons, the training-target
  circularity trade-off, the statistical conventions inherited from the
  glueball attention study, and the closure of clause 2 (attention range vs
  correlation length) as not measurable at `ξ_s ≲ 1`. The source of truth for
  this program; read before touching `gelt/topology.py`.
- `notes/resources.md` — curated textbooks, lecture notes, and ML-for-LGT
  papers with suggested reading order.
- `notes/tunnel-visualization.md` — exploratory notes on visualising
  what the topological-charge network learns about the QCD vacuum.

## Status

Phase 0 (2D Z₂ implementation validation), extended toward SU(2): the
Metropolis sampler, the targets, and the GELT block now all support
`nc = 2`, and `validate_sampler_su2.py` / `validate_sampler_z2.py` validate
the Metropolis sampler for each group.

The codebase was refactored from the original OO scaffolding
(`Site` / `Link` / `Plaquette` / `Lattice` classes) to **pure tensor
operations** suitable for autograd, vectorisation, and clean generalisation
to U(1)/SU(N). It was then reorganised into a proper Python package:
`gelt/` (library), `scripts/` (entry points), `tests/` (pytest).

The **GELT block** exists in **two variants** that differ only in positional
encoding (≈ 90% shared code):

- **`gelt/blocks_rope.py`** — rotary positional encoding (RoPE) on the
  attention score. **This is the trained variant** (`scripts/train_gelt.py`
  imports `GELT` from here).
- **`gelt/blocks_bias.py`** — convolutional/offset bias on the score. This
  is the variant imported by `gelt/__init__.py` (`from gelt import GELT`),
  by `tests/test_blocks.py`, and by `scripts/check_gelt_invariance.py`.

Each variant provides `GEMHSA` (single equivariant attention layer),
`ChannelLift` (front-end width lift), `Trace`, `MLP`, and `GELT` (the full
model: `ChannelLift` → stacked `GEMHSA` blocks → `Trace` → per-site `MLP` →
spatial reduction).

A second baseline now lives in **`gelt/lcnn.py`**: the Favoni et al. L-CNN
(`LConv`, `LBilin`, `LCB`, `LAct`, `Trace`, `LCNN` + `build_axis_transports`),
the matched-parameter comparison target for the GELT.

**Glueball spectroscopy program** (`notes/glueball_spectroscopy.md`): the
classical 0⁺⁺ baseline *code* (§6.1) is in place — `gelt/glueball.py`
(operator, APE smearing, connected correlator, `m_eff`, jackknife), validated
by `tests/test_glueball.py` and visualised by `scripts/measure_glueball.py`,
with `scripts/check_glueball_autocorrelation.py` fixing the production
`n_skip` from the smeared-operator `τ_int`. The SU(2) heat-bath +
overrelaxation sampler (§8, the prerequisite ensemble long pole) and the
`integrated_autocorrelation_time` diagnostic are also in place.

**Go/no-go question — is a mass discoverable? RESOLVED: yes, on an
anisotropic lattice.** The isotropic `L=12 β=2.4 N=2000` run did *not* plateau
(weak `m·a ≈ 0.8`, drowning by Δ≈3) even with the multi-level GEVP — the lattice,
not the operator basis, was the bottleneck. Adding **anisotropy** (finer
`a_t = a_s/ξ`; `staple_sum`/sweeps take `xi`, `action`/`random_links`/
`mcmc_ensemble` take `xi`/`Lt`) fixed it: on `L=12 Lt=24 β=2.4 ξ=3.0 N=2000` the
`C(Δ)` decays cleanly over ~10–12 slices and the **GEVP ground state plateaus at
m·a_t ≈ 0.33** (`m_eff(Δ=1)=0.365±0.008`, `(Δ=2)=0.333±0.011`). Caveat: the
reported `m·a_s = ξ·m·a_t ≈ 1.0` is *not* continuum physics (β_s=β/ξ=0.8 is
strong-coupling/coarse a_s); proper anisotropy tuning + continuum extrapolation
is future work. **§6.2 (`scripts/train_glueball.py` — `GELT(reduction="none")` on
the Rayleigh loss `−C(1)/C(0)`, jackknife eval, vs. classical/L-CNN curves) is now
unblocked**, validated against the classical GEVP plateau `m·a_t ≈ 0.33` on the
cached anisotropic ensemble (`datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000.pt`;
re-sample via `measure_glueball.py` if the cache is absent).
See `notes/glueball_spectroscopy.md` for the full run-by-run record.

**§6.2 constraints (audit 2026-07-01, recorded in
`notes/glueball_spectroscopy.md` § "Audit"):** the variational operator must be
**per-timeslice 3D** — `GELT(D=3, L=12)` on each timeslice's *spatial* links
(batch = config × timeslice), because any temporal receptive field voids the
transfer-matrix bound and makes the Rayleigh loss gameable toward m → 0 (the
§7 "never smear in time" rule applied to the network itself). This also
sidesteps GEMHSA's cubic-only `_nbr_idx` (which cannot ingest 24 × 12³) and the
4D transport memory wall. Further musts: `mlp_zero_init=False` (zero init ⇒
exactly zero Rayleigh gradient — training never starts), on-the-fly 3D
transport per batch, hard train/held-out split with (blocked) jackknife on
held-out configs only, and a strengthened classical anchor (GEVP at Δ=3–4 —
the Δ=1/Δ=2 points differ by ~2σ, so the plateau needs confirming) with an
anisotropic re-run of the τ_int pre-flight. The 4D program (q(x),
time-spanning Wilson loops) is untouched — the 3D restriction is a per-task
input-domain choice, not an architecture change.

**§6.2 DELIVERED (Runs 4–5, 2026-07-03, recorded in
`notes/glueball_spectroscopy.md` § "Run 4"/"Run 5"):** with thin-plaquette
input GELT learned only ≈ APE×2's staple content (depth-4 loop degree can't
rebuild iterated smearing) and was variationally redundant with the classical
basis. Feeding **multi-level smeared input channels**
(`INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`, `GELT(..., in_channels=3·n_levels)`)
fixed it: the trained operator's Rayleigh loss saturates the transfer-matrix
bound at the anchor (`val −0.6185 ↔ m·a_t ≈ 0.33`), its `m_eff` plateaus from
Δ=1, and it **beats the classical GEVP at Δ=1 by 3.9σ** (blocked jackknife of
the *difference* on shared test configs: −0.028 ± 0.007) while agreeing on the
plateau mass — the enlarged 5-op GEVP collapses onto pure GELT. Two loss
pathologies were fixed en route: chained-ratio denominators → `C(0)`, and a
`(log C(0))²` scale pin (`SCALE_REG`) for the Ō → λŌ flat direction, which the
site-coherent smeared inputs otherwise pump to float32 overflow. Still open
from the §6.2 list: the matched-parameter L-CNN baseline on the same
per-timeslice task (the fresh-ensemble replication is done — see below).

**Presentation layer (2026-07-04):** how the Run-5 result is *reported* in
LGT-standard form — masses quoted from a cosh fit (not single m_eff points)
and operator quality quoted as the **ground-state overlap fraction A₀**
(Morningstar–Peardon's metric, and exactly what the Rayleigh loss maximises).
`gelt/glueball.py` gained `gevp_ground_vector` (projected GEVP operator) and
`fit_cosh_correlator` (profiled-A grid fit); `train_glueball.py` gained
`EVAL_ONLY` and dumps the test-split Ō arrays to `datasets/…_test_obars.pt`;
`scripts/fit_glueball_overlap.py` runs the jackknifed fits, the correlated
(Δm, ΔA₀) differences, and the `glueball_overlap.png` figure offline on CPU.
**Real numbers (window Δ ∈ [2,7], 400 test configs):** GELT
m·a_t = 0.332 ± 0.027 with **A₀ = 0.903 ± 0.047**, GEVP-projected
0.340 ± 0.030 with A₀ = 0.837 ± 0.056; correlated ΔA₀ = +0.066 ± 0.031
(2.1σ), Δm consistent with 0 — same mass, more ground-state weight. Written
up in `glueball_report/glueball_spectroscopy.tex` § "Quoting the result".

**Replication (2026-07-05):** the seed-1 phase of
`scripts/overnight_replication.sh` (fresh ensemble sampled from scratch +
from-scratch training) **reproduces the result**: GELT A₀ = 1.013 ± 0.062 vs
GEVP-projected 0.925 ± 0.071, correlated **ΔA₀ = +0.089 ± 0.030 (2.9σ)**,
m_eff difference at Δ=1 −0.038 ± 0.008 (4.5σ); combined with Run 5 the
overlap headline is **ΔA₀ = +0.078 ± 0.022 (3.6σ)**. Key lesson: the
Rayleigh floor is *ensemble-specific* — the seed-1 ensemble reads ~1σ
heavier (GEVP flat at ≈ 0.39; best val −0.5638 ↔ m ≈ 0.395, the floor of
*that* ensemble, saturated to ~0.004 like Run 5's −0.6185 ↔ 0.33), so
loss values don't replicate but operator quality does. The seed-2 phase
died 42% into sampling on `_sample_su2_w0`'s 100-iteration rejection cap —
a statistical inevitability, not a fluke (Creutz acceptance ~ √(π/2a);
anisotropic temporal staples reach a ≈ 43 → ~O(1) expected cap hits per
ensemble's ~3.4e9 draws). Fixed: `max_iter` 100 → 1000 (free on the common
path); rerun ens2 standalone if a third measurement is wanted — with two
independent ensembles agreeing it's a bonus, not a blocker. Written up in
the report § "Replication on an independent ensemble" and
`notes/glueball_spectroscopy.md` § "Replication results".

**Interpretability program (2026-08-07, `notes/topological_localization.md`):**
the attention readout is live. `scripts/visualize_glueball_attention.py` reads
the Run-5 operator's attention out as a physical field on a held-out ensemble:
the operator's quality is carried by **one local head** (L3h1 ablates at
+0.0787 ± 0.0145, 5.4σ, against nothing else above 2σ), there is **no
depth-wise coarsening** at R=2 (six of eight heads saturate at
ℓ_att = 2.000 ± 0.000; two are degenerate fixed kernels), the most distributed
head extends its reach on high-action regions (Spearman −0.564 ± 0.002 vs the
smeared density), and the anisotropic heads are locked to a **globally fixed
spatial axis** (100.0% of 1.3M site-slice samples) — so the learned operator is
a cubic-group scalar only approximately, and an explicit `A₁⁺⁺` projection over
the 24 lattice rotations is the open next step. `scripts/beta_scan.py` closed
the companion question negatively: `ξ_s ≤ 1.1` spatial lattice spacings across
β ∈ [2.1, 2.7], smaller than the integer grid the attention lives on, so the
attention-range-vs-correlation-length study is not measurable on this lattice
family (the box is *large* — L/ξ_s ≈ 12 — it is the spacing that is coarse).
A follow-up attempt to rescue the range question in **3D Z₂ near its critical
point** (where ξ genuinely reaches 5.3 lattice spacings, unlike SU(2)) also
closed negative: the operators reproduce the classical mass at R=6 but ℓ_att
sits on its *uniform* value at both R=6 (4.25/4.54 vs 4.28) and R=12
(7.4–8.6 vs 8.31), with no trend in ξ. **Methodological lesson: a range
statistic is bounded by R *and centred by the ball geometry* — always quote
ℓ_att against Σ_d(4d·d)/n_off.** The attempt did leave an exact, tested Z₂
heat-bath in `gelt/sampler.py` (`z2_heatbath_sweep`, `P(U=+1)=σ(2βs)`; the
Metropolis flip proposal's acceptance collapses to 0.02 near β_c) plus
`scripts/z2_beta_scan.py` / `train_z2_glueball.py` / `z2_attention_readout.py`.
Full record in `notes/topological_localization.md` §6.1.
`gelt/topology.py` (`cool`, `cooled_charge_density`) + `tests/test_topology.py`
+ `scripts/check_cooling.py` are the 4D cooling layer for the topology study;
`glueball.ape_smear` gained a `directions` argument (default unchanged, so the
spatial-only spectroscopy path is byte-identical). Both results are written up
in `glueball_report/glueball_spectroscopy.tex` § 10.

Known caveats (see `notes/fable_audit.md` for the full list and the
prioritized fixes):

1. **The trained variant (`blocks_rope`) is the untested one.** The tests
   and the invariance check exercise `blocks_bias`. Parametrizing the
   equivariance tests over both modules is the cheap fix.
2. **(resolved)** The dead parameters are gone: `self.alpha` (ReZero) was
   deleted from both variants (the residual stays `W + W_act`, and the
   `alpha_init` plumbing was removed from `GEMHSA`/`GELT`/`train_gelt.py`),
   and `blocks_bias`'s `b_h` (bias) was restored to the score path (the
   bias add is live again). Both variants now have only live, trainable
   parameters, so the `test_blocks` backward case passes.
3. **RoPE axis coverage:** `pair_axis = [p % D for p in range(n_pairs)]`
   only rotates `n_pairs` axes, so in 4D with small `d_qkv` some axes get
   the identity rotation. Enforce `d_qkv ≥ 2D` for full coverage.

What still does **not** exist: a full worst-case-Ω stress test (only the
quick `check_gelt_invariance.py` exists); β in the datasets (needed for the
strong correlation-length study); cooling/smearing of `q(x)`; offset-chunked
attention (the memory gate on the explainability program at physical R);
non-Z₂/non-SU(2) production samplers.

## Layout

Library lives in `gelt/`; entry-point scripts in `scripts/`; pytest in
`tests/`. The package is installed editable via `pyproject.toml`.

### `gelt/`

- **`lattice.py`** — `GaugeGroup` ABC with `Z2` and `SU(N)` implementations;
  pure tensor functions:
  - `random_links(L, D, group, dtype, Lt=None)` → `(D, *Λ, nc, nc)`. `Lt` gives
    a non-cubic `Λ = (Lt,) + (L,)*(D-1)` lattice (time = axis 0) for anisotropy.
  - `plaquette_tensor(U, group)` → `(D(D-1)/2, *Λ, nc, nc)`.
  - `action(U, group, beta=1.0, plaquettes=None, xi=1.0, time_axis=0)` → scalar
    Wilson action `β Σ_p (1 − Re Tr P / nc)`. **Anisotropic** when `xi ≠ 1`:
    temporal plaquettes (plane touching `time_axis`) weighted by `β_t = β·ξ`,
    spatial by `β_s = β/ξ` (tree-level); `xi = 1` is the bit-exact isotropic path.
  - `topological_charge_density(U, group, plaquettes=None)` → per-site
    naive (plaquette) charge density `q_x` (clover-free,
    `F_{μν} = (P − P†)/2i`).
  - `topological_charge(U, group, plaquettes=None)` → `Q = Σ_x q_x`, one
    scalar per config — the topology analogue of `action`.
  - `rectangular_wilson_loop(U, group, R, T, mu, nu)` → `Re Tr W/nc` at
    every site for the R×T loop in the (μ, ν) plane (R=T=1 is the plaquette).
  - `link_gauge_transformation(U, omega, group)` — apply site-local Ω to
    every link (`U_μ(x) → Ω(x) · U_μ(x) · Ω†(x+μ̂)`); used by the
    gauge-invariance unit tests and the GELT invariance check.
  - `local_gauge_transformation(W, omega, group)` — apply site-local Ω to
    an adjoint field (`W(x) → Ω(x) · W(x) · Ω†(x)`); used in the GELT
    equivariance tests.
  - `l1_ball_offsets(D, R)` → list of signed Δx tuples with
    `1 ≤ |Δx|_1 ≤ R`, ordered by `|Δx|_1`.
  - `build_transport_average(U, R, group, mode="average")` — DP routine that
    materialises transports `T_Δx(x)` over the full signed L1-ball. Expects
    batched links `(N, D, *Λ, nc, nc)`. `mode="average"` (default) is the
    shortest-path-averaged transport (rotation-symmetric, the architecture's
    default); `mode="single"` is a single-canonical-path variant (rotation
    symmetry broken — for A/B testing whether averaging dilutes a
    specific-path target).
- **`sampler.py`** — `staple_sum`, `metropolis_sweep` (checkerboard-
  vectorised single-site Metropolis; the proposal is routed by
  `_PROPOSAL_FN[type(group)]` — `_z2_proposal` (`U → −U`) and
  `_su2_proposal` (`U → V·U`, V near identity)); `mcmc_ensemble`
  (thermalise + decorrelate + collect, dispatches the sweep per group via
  `_SWEEP_FN`, **Metropolis is the registered default for both Z₂ and
  SU(2)**); `haar_ensemble` (Haar-uniform, ignores β — shares the sampler
  interface for sanity checks). To plug in U(1)/SU(N) later, add a proposal
  (and/or sweep) and register it in `_PROPOSAL_FN` / `_SWEEP_FN`.
  - **Anisotropy:** `staple_sum` and all sweeps take `xi=1.0, time_axis=0`; for
    `xi ≠ 1` the per-plane coupling ratio ξ^±1 is folded into the staple (β stays
    the single overall scale), so `xi = 1` is bit-exact backward-compatible. Opt
    in by binding ξ into the sweep, e.g.
    `functools.partial(heatbath_overrelaxation_sweep, n_or=4, xi=ξ)`;
    `mcmc_ensemble(..., Lt=…)` gives the matching non-cubic temporal extent. See
    `notes/glueball_spectroscopy.md` (anisotropy resolves the heavy 0⁺⁺ that an
    isotropic lattice cannot).
  - **SU(2) heat-bath + overrelaxation** (`heatbath_sweep`,
    `overrelaxation_sweep`, and the combined `heatbath_overrelaxation_sweep`
    — 1 heat-bath + `n_or` OR sweeps) is the exact, no-tuning sampler that
    beats Metropolis critical slowing; it is the prerequisite for resolvable
    spectroscopy (see `notes/glueball_spectroscopy.md` §8). It is **opt-in,
    not the registry default** (so `validate_sampler_su2.py` still tests
    Metropolis): pass it as `sweep_fn=` to `mcmc_ensemble`, e.g.
    `functools.partial(heatbath_overrelaxation_sweep, n_or=4)`. Both sweeps
    share one checkerboard skeleton (`_su2_local_sweep`), differing only in
    the per-site update: heat-bath factors the staple `A = k·V`
    (`_su2_decompose_staple`), draws the scalar part via Creutz
    (`_sample_su2_w0`), and sets `U' = W·V†`; overrelaxation reflects
    `U' = V†·U†·V†`. Overrelaxation is microcanonical (action-preserving) but
    *expansive* off the group manifold, so each reflected link is re-projected
    onto SU(2) with the closed-form `_su2_from_quaternion`-style projector
    `_project_su2` (cheaper than `SU.project`'s SVD/det). SU(2) only — SU(N≥3)
    needs Cabibbo–Marinari.
  - **`integrated_autocorrelation_time(series, c=6, max_lag=None)`** — generic
    Markov-chain diagnostic: normalised autocorrelation `ρ(t)` and `τ_int`
    (Madras–Sokal automatic windowing, the proper version of the inline
    estimate in `validate_sampler_su2.py`) of any scalar chain observable.
    Returns `(rho, tau_int, window)`; samples `n_skip ≳ 2·τ_int` apart are
    effectively independent.
- **`data.py`** — `build_plaquette_datasets(N, D, L, group, target, ...)`.
  `target` is a callable `target(configs, group) -> Tensor` (use
  `functools.partial` to bind extra args, e.g.
  `partial(rectangular_wilson_loop, R=2, T=3, mu=0, nu=1)`).
  `structured=True` (default) keeps the full `(N, n_pairs, *Λ, nc, nc)`
  matrix layout for GELT; `structured=False` calls `flatten_color` (also in
  this module) to split color axes for the CNN baseline — real groups
  give `(D · nc², *Λ)`, complex groups split real/imag for
  `(2 · D · nc², *Λ)`. With `R` set, the transport is precomputed per config
  via `build_transport_average` (honoring `transport_mode`) and the splits
  yield `(X, T, y)` triples. `save=True` writes to `datasets/`;
  `load_plaquette_datasets(prefix, datasets_dir="datasets")` reloads them.
- **`cnn_baseline.py`** — `LatticeCNN(L, D, in_channels, hidden_channels,
  kernel_size=3)`. Non-equivariant CNN baseline; uses `Conv2d`/`Conv3d` for
  D=2/3 and a roll-based `_RollConvND` for D≥4.
- **`glueball.py`** — classical 0⁺⁺ glueball spectroscopy baseline (the
  validation target the learned GELT operator will be judged against, per
  `notes/glueball_spectroscopy.md` §6.1). Time is lattice axis 0; spatial
  directions are 1..D-1.
  - `ape_smear(U, group, alpha=0.5, n_steps=1)` — spatial-only APE smearing
    (each spatial link replaced by the group projection of
    `(1−α)U + (α/n_staples)·Σ daggered spatial staples`, reusing
    `staple_sum`); time links untouched so the transfer-matrix interpretation
    holds. The crucial enabler for a reachable plateau (§7).
  - `glueball_operator(U, group, R=1, T=1)` → `(B, *Λ)` real scalar field:
    sum of spatial-plane R×T Wilson loops (a rotational scalar; R=T=1 is the
    spatial plaquette).
  - `zero_momentum(O)` → `(B, Nt)` timeslice operator `Ō(t)` (sum over
    spatial sites).
  - `connected_correlator(Obar)` → `(Nt,)` vacuum-subtracted `C(Δ)`, averaged
    over the batch and all time origins (0⁺⁺ has nonzero VEV — subtraction is
    essential).
  - `effective_mass(C)` → `m_eff(Δ) = log[C(Δ)/C(Δ+1)]`.
  - `jackknife_effective_mass(Obar)` → `(mean, err)` leave-one-out jackknife
    over configs.
  - **Multi-level GEVP** (the Morningstar–Peardon variational fix for a single
    operator's poor ground-state overlap): `smearing_operator_basis(configs,
    group, levels, ...)` → `(n_levels, B, Nt)` stack of zero-momentum operators
    at cumulative APE levels (incremental smearing); `connected_correlator_matrix(Obar)`
    → `(Nt, n_ops, n_ops)`; `gevp_eigenvalues(C, t0=1)` solves `C(Δ)v=λC(t0)v`
    via robust eigh-whitening with an eigenvalue floor (not Cholesky — low
    statistics can make `C(t0)` indefinite), returning λ descending (col 0 =
    ground state); `gevp_effective_mass(lams)` and `jackknife_gevp_effective_mass(Obar,
    t0)`. Masses are read off `Δ ≥ t0`.
  - **Fit / overlap layer** (how masses are *quoted*, vs. the m_eff plots they
    are *shown* on): `gevp_ground_vector(C, t0, td)` → the ground-state
    generalized eigenvector v₀ (same eigh-whitening), defining the *projected
    operator* `Σᵢ v₀ᵢ Ōᵢ(t)` — the optimal single operator in a basis's span
    (fixed-vector GEVP), the apples-to-apples comparator for a single learned
    operator; `fit_cosh_correlator(C, dmin, dmax, sigma)` → `(m, A, χ²)`
    least-squares fit to the periodic single-state form
    `A·[e^(−mΔ) + e^(−m(Nt−Δ))]` (profiled-A grid scan + parabolic refine —
    dependency-free, no convergence failures, safe to re-run inside every
    jackknife sample; diagonal χ² only, errors come from jackknifing the whole
    fit). The ground-state overlap fraction is `A₀ = A·(1+e^(−m·Nt))/C(0)`.
- **`lcnn.py`** — Favoni et al. L-CNN (gauge-equivariant baseline):
  `build_axis_transports` (axis-aligned link products `U^(k)_μ(x)`, the
  L-CNN transport input — distinct from GELT's L1-ball `T`); `LConv`,
  `LBilin`, `LCB` (L-Conv-Bilin block), `LAct` (trace-gated activation),
  `Trace`, and `LCNN` (stacked L-CB(+L-Act) → `Trace` → per-site MLP →
  reduction). Mirrors `GELT`'s I/O so the two are matched-parameter
  comparable.
- **`blocks_rope.py` / `blocks_bias.py`** — the two GELT variants (see
  Status). `GEMHSA`: augment (append daggers + identity) → fused Q/Q_v/K/V
  projections → adjoint transport of K, V via `T`/`T_dag` → gauge-invariant
  score `Re Tr[Q† K̃]` (+ RoPE rotation or offset bias) → softmax →
  multiplicative value `Σ α · Q_v† · Ṽ` → channel mix → residual + L-Act
  gate. `GELT.forward(W, T)` computes `T_dag` once and threads
  `(T, T_dag)` through the stack. `_last_score` / `_last_alpha` are stashed
  per layer (under `no_grad`) for the interpretability program.
- **`__init__.py`** — re-exports `GELT` (from `blocks_bias`), `LatticeCNN`,
  the dataset builders, the `lattice` primitives, and the ensembles.

### `scripts/`

Each script is self-contained: it defines its own `evaluate` / `train_model`
loop inline (there is no shared `gelt/train.py`). Device order: cuda → mps
→ cpu.

- **`train_cnn.py`** — single-(L, β) run of the CNN baseline (`LatticeCNN`);
  `haar_ensemble`, a `rectangular_wilson_loop` target, target
  standardization, matched-capacity hyperparameters.
- **`train_gelt.py`** — single-(L, β, R) run of the GELT model (imports
  `GELT` from `blocks_rope`); `structured=True`, unpacks `(X, T, y)`,
  trains on a `topological_charge_density` target, passes `T` to
  `model(X, T)`, uses a `StepLR` scheduler. The minimal reference for how
  to train the architecture.
- **`train_lcnn.py`** — single-run of the Favoni L-CNN (`gelt.lcnn.LCNN`);
  mirrors `train_gelt.py` (same loop, split, standardisation, plotting) but
  feeds the axis-aligned `build_axis_transports` instead of the L1-ball `T`.
- **`check_gelt_invariance.py`** — quick gauge-invariance check on the full
  `GELT` (from `blocks_bias`): `forward(W_g, T_g) ≈ forward(W, T)` on SU(2).
- **`validate_sampler_su2.py`** / **`validate_sampler_z2.py`** — four-panel
  sanity checks on the Metropolis sampler (one per group): thermalisation,
  2D β-scan, 3D β-scan, plaquette autocorrelation. The 2D panel compares to
  the exact mean plaquette — `I₂(β)/I₁(β)` for SU(2), `tanh(β)` for Z₂ — and
  the 3D panel shows the SU(2) confining crossover vs. the Z₂ transition near
  `β_c ≈ 0.761`. Write `sampler_validation_su2.png` / `sampler_validation_z2.png`.
- **`validate_anisotropy.py`** — validates the anisotropic SU(2) lattice: ξ=1
  reproduces the exact 2D mean plaquette (refactor correctness), a ξ-scan shows
  the plaquette split `⟨P_st⟩ > ⟨P_ss⟩`, and a Creutz-ratio ratio
  `ξ_R ≈ χ_ss/χ_st` estimates the **renormalized** anisotropy vs the bare ξ (the
  tree-level mismatch made visible; no auto-tuning). Writes
  `anisotropy_validation.png`.
- **`measure_glueball.py`** — classical 0⁺⁺ baseline (`gelt.glueball`): four
  panels validating the correlator/`m_eff` code on a synthetic known mass and
  smearing monotonicity (top row), then the real-ensemble `C(Δ)` and `m_eff(Δ)`
  comparing thin, single-smeared, and the **multi-level GEVP ground state**
  (bottom row). Defaults to an **anisotropic** run (`XI`, `LT` tunables); samples
  via SU(2) heat-bath + overrelaxation and **caches the ensemble under
  `datasets/`** (cache key includes ξ, Lt) so the GEVP analysis can be re-tuned
  without re-sampling. Reports `m·a_t` and `m·a_s = ξ·m·a_t`. Writes
  `glueball_validation.png`.
- **`fit_glueball_overlap.py`** — presentation-grade §6.2 spectroscopy
  (offline, CPU-trivial): cosh fits + ground-state overlap `A₀` for GELT, the
  *projected* classical GEVP operator (`gevp_ground_vector`), and APE×max on
  one shared fit window; the whole fit (incl. v₀) is redone inside every
  delete-block jackknife sample, and the correlated same-configs differences
  (Δm ≈ 0 = same physics, ΔA₀ > 0 = better operator) are the significance
  statements. Consumes the test-split Ō arrays `train_glueball.py` dumps to
  `datasets/…_test_obars.pt` (set `EVAL_ONLY = True` there to reproduce the
  dump from an existing checkpoint in one GPU eval pass; the dump path can be
  passed as `argv[1]`). Writes `glueball_overlap.png` (m_eff + fitted-mass
  bands; ρ(Δ) = [C(Δ)/C(0)]/cosh_ref overlap panel — flat at A₀ ⇔ pure
  ground state).
- **`overnight_replication.sh`** — ~24 h unattended V100 batch closing the
  "one ensemble" caveat: two fresh ensembles (seeds 1 and 2, sampled from
  scratch — the long pole) each with a from-scratch GELT training, giving
  three fully independent A₀ measurements (Run 5 + ens1 + ens2). Uses
  `train_glueball.py`'s env overrides (`GLUEBALL_ENSEMBLE_SEED`,
  `GLUEBALL_INIT_SEED`, `GLUEBALL_RESUME`, `GLUEBALL_EVAL_ONLY`); non-default
  seeds get a `RUN_TAG` suffix on cache / checkpoint / dump / plot names so
  Run-5 artifacts are never overwritten. Phases are failure-isolated and log
  to `logs/<phase>.log` with unbuffered output and ~30 s tqdm ticks
  (`python -u`, `TQDM_MININTERVAL=30`) so progress is watchable with
  `tail -f`; each leaves a `datasets/*_test_obars.pt` dump for
  `fit_glueball_overlap.py`.
- **`check_glueball_autocorrelation.py`** — step-1 pre-flight before
  `measure_glueball.py`: runs a long `n_skip=1` heat-bath+OR chain, builds the
  plaquette and the thin/smeared glueball operator per config, and reports
  `τ_int` (via `integrated_autocorrelation_time`) so the production `n_skip` can
  be set to `≳ 2·τ_int` of the smeared operator. Writes
  `glueball_autocorrelation.png`.

### `tests/`

- **`test_lattice.py`** — gauge-invariance checks on `plaquette_tensor` /
  `action` under `link_gauge_transformation` (bit-exact in Z₂ float64), plus
  **anisotropic-action** gauge invariance (SU(2) + Z₂, `xi ≠ 1`), the `xi = 1`
  match to the isotropic action, and the non-cubic `random_links(..., Lt=)` shape.
- **`test_data_model.py`** — split-validation and CNN-baseline shape
  guards.
- **`test_transport.py`** — coverage for `l1_ball_offsets` and
  `build_transport_average`: offset counts, brute-force per-octant pattern,
  octant-relation consistency, and gauge covariance under unitary Ω for
  both Z₂ and `nc = 2` complex.
- **`test_blocks.py`** — gauge equivariance of `GEMHSA` end-to-end
  (`forward(W_g, T_g) == Ω · forward(W, T) · Ω†`) for SU(2) in
  complex128 and Z₂ in float64; both gate branches; shape preservation
  and finite-grad backward pass on a batched SU(3) example. **Tests the
  `blocks_bias` variant** (not the trained `blocks_rope`); the full suite
  passes now that the dead parameters are resolved (see Status).
- **`test_sampler.py`** — SU(2) heat-bath + overrelaxation correctness:
  overrelaxation conserves the Wilson action to machine precision and stays
  on the group; heat-bath stays on the group and reproduces the *exact* 2D
  SU(2) mean plaquette `I₂(β)/I₁(β)` (the automated analogue of
  `validate_sampler_su2.py`'s 2D panel). Plus **anisotropy**: `xi = 1`
  reproduces the isotropic sweep bit-for-bit, and overrelaxation conserves the
  *anisotropic* action (and heat-bath stays on the group) for `xi ≠ 1`.
- **`test_glueball.py`** — classical glueball baseline correctness: APE
  smearing is gauge covariant (`smear(Uᵍ) == (smear U)ᵍ`) and stays on the
  group (SU(2) + Z₂); the glueball operator is gauge invariant; the
  correlator / `m_eff` / jackknife arithmetic recovers a known mass from a
  synthetic single-exponential correlator and gives a finite, positive-error
  jackknife band; and the **GEVP** recovers both masses of a synthetic
  two-state correlator matrix (plus basis shape/invariance and the
  single-operator matrix↔scalar consistency check). The fit/overlap layer:
  `gevp_ground_vector`'s projection exactly kills the excited state of the
  synthetic two-state model, and `fit_cosh_correlator` recovers `(m, A)` from
  an exact periodic correlator (weighted and unweighted).

## Conventions

- **Tensor layouts** (the only spec, no OO wrappers):
  - Links: `(D, L, ..., L, nc, nc)`. Direction first, spatial axes,
    color axes last.
  - Plaquettes: `(n_pairs, L, ..., L, nc, nc)` with
    `n_pairs = D(D-1)/2`, ordered by `(μ, ν)` with `μ < ν` lexicographically.
- **Color axes are always present**, even for Z₂ where `nc = 1`. Every
  product is written as `A @ B` and every inverse as `group.dagger(A)`,
  so the code ports verbatim to U(1)/SU(N).
- **Plaquette convention:**
  `P_{μν}(x) = U_μ(x) · U_ν(x + μ̂) · U_μ†(x + ν̂) · U_ν†(x)`.
- **Periodic BCs:** `torch.roll` for shifts. Never manual modulo arithmetic
  on indices (it's harder to vectorise and harder to read).
- **Wilson action:** `S = β Σ_p (1 − Re Tr P / nc)`. β defaults to 1.0,
  reproducing the legacy unnormalised form `n_plaq − Σ P` for Z₂.
- **Parallel transport:** sum over **all** shortest lattice paths in the
  L1-ball — never a single axis-aligned path (unless `mode="single"`).
  `build_transport_average` expects batched links `(N, D, *Λ, nc, nc)` and
  materialises the full signed L1-ball in one `|Δx|_1`-ordered DP pass,
  using `U_μ(x)` for `Δx_μ > 0` steps and `U†_μ(x − ê_μ)` for `Δx_μ < 0`
  steps. The octant identity `T_{−Δx}(x) = dagger(T_Δx(x − Δx))` holds
  as a math property and is a test-suite consistency check; it is **not**
  relied on at build time (a single auditable DP surface is worth more
  than the 2× memory saving from canonical-offset storage, and mixed-sign
  offsets cannot be derived from positive-octant data anyway).
- **Float32** for training; pass `dtype=torch.float64` (→ `complex128`) through
  the dataset builders for high-precision gauge-invariance unit tests and
  worst-case-Ω drift reporting.

## Running

The package is installed editable; scripts are run from the repo root.

```bash
python scripts/train_cnn.py            # single-(L, β) CNN baseline run
python scripts/train_gelt.py           # single-(L, β, R) GELT run (blocks_rope)
python scripts/train_lcnn.py           # single-run Favoni L-CNN baseline
python scripts/check_gelt_invariance.py  # quick SU(2) gauge-invariance check on GELT
python scripts/validate_sampler_su2.py # Metropolis four-panel sanity check (SU(2))
python scripts/validate_sampler_z2.py  # Metropolis four-panel sanity check (Z₂)
python scripts/validate_anisotropy.py  # anisotropic-lattice checks (ξ=1 regression, ⟨P_st⟩>⟨P_ss⟩, ξ_R)
python scripts/check_glueball_autocorrelation.py  # τ_int of the glueball operator (set n_skip)
python scripts/measure_glueball.py     # anisotropic 0⁺⁺ glueball baseline (correlator + GEVP m_eff)
python scripts/fit_glueball_overlap.py # cosh fits + overlap A₀ (offline, from train_glueball.py's test-Ō dump)
python -m gelt.cnn_baseline            # torchsummary for a 5×5 CNN
pytest tests                           # unit tests
```

`.venv/` is local (uv-style, not gitignored). `datasets/`, `*.pth`,
`*.png` are gitignored.

## The inductive-bias gap that motivates GELT

For Haar-random Z₂ links in 2D, every plaquette is ±1 with mean 0, and
plaquette pairs share either 0 or 2 links — both cases give zero
covariance, so plaquettes are independent under random ±1 links. With
`n_plaq = L²` independent ±1 contributions, `Var(action) = L²`, so an
absolute MSE that grows like L² is just the label scale growing — use
`R² = 1 − MSE / Var(y)` to compare across L. On Haar data, a CNN fed
**plaquettes** reaches R² ≈ 0.99 (the action is a linear sum of its inputs)
while a CNN fed **links** sits at R² ≈ 0 (no inductive bias for "multiply
four specific link values"). That gap is what the equivariant transport
provides. Even a perfect action regressor on Haar data is only memorising
the action *function*; β-dependent physics requires the Metropolis data
path, hence the MCMC samplers and the physical targets (action, topological
charge, Wilson loops).

## Things to keep in mind

- **Do not silently broadcast across color axes.** Every matmul should
  be explicit (`A @ B`) and every dagger explicit (`group.dagger(A)`);
  for Z₂ both are no-ops, but for U(1)/SU(N) any laxity is a bug that
  Z₂ cannot catch. The same applies to the GELT transport: a missed dagger
  or a wrong shortest-path step will pass Z₂ tests and fail at the first
  non-abelian Ω.
- **`blocks_rope` is trained but `blocks_bias` is tested.** Trust nothing
  about the RoPE variant's equivariance until the tests are parametrized
  over both modules; "should hold" is exactly what the stress test exists
  to replace.
- **`LatticeCNN` is 2D-only** (`Conv2d` with circular padding); raises
  `NotImplementedError` for `D ≠ 2`. Generalising means switching to
  `Conv3d`/`ConvNd` or factoring the convolution layer.
- **Datasets do not store β.** For multi-β training (and the strong
  correlation-length study), β should become part of the dataset so the
  model can be conditioned on it.
- **Sampler dispatch is by group type.** `metropolis_sweep` looks up
  `_PROPOSAL_FN[type(group)]` and `mcmc_ensemble` looks up
  `_SWEEP_FN[type(group)]`; adding U(1)/SU(N) is a registry entry plus the
  proposal/sweep function.
- **Do not remove comments unless asked to.**

## Suggested next steps

In priority order, per `notes/fable_audit.md` §4 (architecture work
gating the explainability program in `notes/explainability.md`):

1. ~~**Resolve the dead parameters** (`self.alpha` ReZero, `blocks_bias`'s
   `b_h`).~~ **Done:** `self.alpha` deleted (residual stays `W + W_act`),
   `b_h` restored; the `test_blocks` suite passes.
2. **Merge `blocks_bias`/`blocks_rope` into one `blocks.py`** with a
   `pos_encoding ∈ {"rope", "bias", "none"}` switch, and **parametrize the
   equivariance tests over all variants** (this is also the §7-style stress
   test the trained variant currently lacks).
3. **Enforce RoPE axis coverage** (`d_qkv ≥ 2D`; set `d_qkv=8` for D=4).
4. **Gate the `.item()` diagnostics** behind a flag (they force a GPU sync
   every layer/step); default off in training.
5. **Offset-chunked attention + on-the-fly transport** — the memory gate on
   the whole explainability program at physical R.
6. **Add cooling/smearing of `q(x)`** (prerequisite for the localization
   study) and **β in the dataset** (prerequisite for the strong correlation-
   length study).
7. **The thesis novelty** (per `notes/explainability.md`): the attention map
   as a *measurement* — emergent correlation length `ℓ_att(β)` vs `ξ(β)`,
   spatial localization on topological lumps, head/layer specialization —
   validated against lattice ground truth, plus the matched-parameter
   shootout vs. the L-CNN baseline.
