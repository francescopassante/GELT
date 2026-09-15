# GELT — Gauge-Equivariant Neural Networks for Lattice Gauge Theory

Master's thesis codebase. A **gauge-equivariant lattice transformer** for
SU(N_c) lattice gauge theory, using 2D/3D Z₂ as a debug-friendly testbed and
4D SU(2) for the physics results.

The architecture follows the L-CNN framework (Favoni et al. 2012.12901) for
primitives and gauge-equivariance proofs, with two departures: (i) the
L-Conv + L-Bilin stack is replaced by an attention block whose **value path is
matrix-bilinear** (`α · Q† · Ṽ`), so L-CNN's loop-doubling universality argument
transfers directly; (ii) parallel transport between sites is **averaged over all
shortest lattice paths** in the L1-ball of Manhattan radius R (a DP recursion,
not enumeration), so each block already reaches the full L1-ball receptive field
with non-axis-aligned loop content.

## Scope — read this first

**2026-09-09: the repo was pruned to what reproduces `../report/main.tex`**, plus
two audits of that content (the SU(2) fair fight, the operator decomposition).
`README.md` is the reproduction guide — which script produces which table and
figure — and the entry point for orientation.

Deleted, not archived (history at commit `cfa0a7e`): the dual-Ising ground truth
and the ν-exponent fits (`gelt/ising.py`, `scripts/dual_ground_truth.py`,
`notes/dual_ground_truth.md`), the topological-localization study
(`gelt/topology.py`, `scripts/topology_attention.py`, `scripts/check_cooling.py`),
the ℓ_att attention-range readouts (`scripts/z2_attention_readout.py`,
`scripts/beta_scan.py`, `scripts/visualize_glueball_attention.py`), the rotation-
irrep study, the Z₂ fair fight, `scripts/compare_transport_modes.py`, and the
`blocks_bias` block variant. When a comment in the surviving code says "retired"
or "removed in the 2026-09-09 cleanup", that is where it went.

## Documents

- `README.md` — **the reproduction guide**: main.tex section → script → artifact.
- `PLANS.md` — future directions. Proposals only; nothing there is implemented.
- `notes/glueball_spectroscopy.md` — the run-by-run record of the SU(2) 0⁺⁺
  program: the anisotropy pivot, the multi-level smeared inputs, Runs 4–5, the
  §6.2 audit, the presentation layer, the replication.
- `notes/lcnn_shootout.md` — the design record for the matched-parameter L-CNN
  baseline: what "matched" means on four axes, why it is a switch inside
  `train_glueball.py`, and the readings fixed in advance. **Run 2026-09-14:
  parity** — ΔA₀(GELT − L-CNN) = +0.007 ± 0.007 (1.1σ), same mass, same
  noise-to-signal (§9.3); the one asymmetry is robustness (§9.2).
- `notes/where_attention_can_win.md` — **the architecture question's running
  record, and the first thing to read before proposing a GELT-vs-L-CNN task.**
  Both attempts and why each closed: the 0⁺⁺ tie (structural) and the flow-free
  topology study (**stopped 2026-09-15**, measured). It separates the two
  candidate mechanisms — *relative weighting* (M1, never observed to pay) from
  *boundedness* (M2, observed three times) — states the **four criteria** a
  candidate task must now satisfy, and proposes the one experiment that targets
  M2 directly. §5's pre-flight rule is the cheap gate that stopped topology for
  ~4 h instead of ~60–100: **measure the best classical method at the
  architecture's own reach before building any training code.**
- `notes/m1_probe.md` — **attempt 3, built and pre-flighted 2026-09-15, not yet
  run.** The confound `where_attention_can_win.md` §1 missed: GELT and the
  matched L-CNN differ in *two* things (input-dependent offset weights **and**
  transport geometry), so no GELT-vs-L-CNN number has ever measured M1. The
  clean test is GELT against **GELT with the softmax frozen**, on three
  constructed per-site targets. Holds the arm table, the four-criteria audit
  (criterion 4 is knowingly violated — it is a mechanism assay, not a physics
  result), the pre-registered readings R-A…R-E, and §3.1, where the pre-flight
  **changed the design before any training code ran**. §3.2 is the production
  pre-flight (2026-09-15, both ensembles, all gates pass): T0 exactly 1.0000,
  T1 headroom 0.92, T2 0.38 — and the **radial linear filter equals the
  full-ball one to four decimals**, so the M1-free ceiling is a five-parameter
  object and nothing that separates the arms can be directional weighting.
- `notes/attention_as_operator.md` — the design record for "the attention map is
  a lattice operator": why ℓ_att failed and the correlator of the attention field
  does not, the three arms, the Z₂ result (§6.1) and its transport to SU(2) (§9).
  §8 records why the scan has four β and not five.
- `notes/operator_decomposition.md` — `O_GELT = P + r` against the classical
  span: what the network found, and (§5, 2026-09-12) the two controls it was
  missing — the strengthened spans and the untrained network.
- `notes/audit_2026-09-06.md` — the repo audit: the **Z₂ APE smearing defect**
  (§2, live caveat), the fair-fight results and the corrections they forced
  (§6.4/§6.5), the ranked plan (§4, re-ranked after the cleanup).
- `notes/performance_audit.md` — the source of truth for anything touching the
  GELT hot path: the four measured optimisations that landed, what was rejected,
  and the ranked backlog.
- `notes/papers_review.md` — literature review (L-CNN, Nagai–Tomiya, CASK).
  Sections 0 (lattice primer) and 1 (L-CNN) are the architecture prerequisites.
- `notes/resources.md` — textbooks and lecture notes, with a reading order.
- `reports/` — the LaTeX write-ups and their PDFs: `paper/` (the full draft),
  `glueball/` (spectroscopy), `attention/` (attention as operator). They pull
  figures from `results/` via `\graphicspath`.

## Status

**Wilson-loop regression** (main.tex §Validation). GELT recovers per-site Wilson
loops a CNN with orders of magnitude more parameters cannot. `train_gelt.py` and
`train_cnn.py` are deliberately the same problem (D=3, L=8, Z₂, Haar links, 1×2
loop, same splits); change one and change the other.

**0⁺⁺ glueball spectroscopy in 4D SU(2)** (main.tex §SU(2) Glueball
spectroscopy). GELT trained as a variational operator on the Rayleigh loss
`−C(1)/C(0)`, per-timeslice 3D, on an anisotropic lattice (L=12, Lt=24, β=2.4,
ξ=3.0). Against the classical multi-level GEVP on the same smeared basis and the
same held-out configurations: same mass, more ground-state weight —
**ΔA₀ = +0.078 ± 0.022 (3.6σ)** over two independent ensembles, m_eff flat from
Δ=1, and the enlarged GEVP+GELT basis collapses onto pure GELT. Two enablers,
both hard-won: **anisotropy** (the isotropic lattice cannot resolve the 0⁺⁺ at
all) and **multi-level smeared input channels** (with thin plaquettes the network
only relearns APE×2's staple content).

**Attention as a lattice operator** (main.tex §Attention as a physical field).
The attention score is gauge invariant, so any reduction of the attention map is
a local scalar lattice operator and its connected correlator has a mass. Measured
on 3D Z₂ at four β approaching β_c ≈ 0.7614 (ξ = 2 … 6) and on anisotropic SU(2)
at β = 2.4: ξ_A tracks the classical correlation length, and **training raises
the attention field's ground-state overlap** far above both the random-init
network and the classical GEVP (Z₂ ΔA₀ = +0.110 … +0.267; SU(2) +0.286 ± 0.056).
The random-init arm tracks ξ too — so the *structural* claim (equivariant
attention maps are lattice operators with a mass) is established by ξ_A, while
the *learning* claim rests on ΔA₀. "The network discovers ξ" is **not**
supportable.

**The matched-parameter L-CNN** (the baseline main.tex names). Same inputs, loss,
splits, estimator and parameter budget, only the block differs: **parity**.
ΔA₀(GELT − L-CNN) = +0.012 ± 0.009 (1.3σ) over two ensembles, Δm consistent with
zero, both beating the classical GEVP by the same margin (+0.077 ± 0.022 and
+0.067 ± 0.020). So "a learned equivariant operator beats the GEVP" is not an
attention artefact — and what attention buys has to be said precisely: the
within-layer L1-ball reach, an attention field that is itself a measurable
operator, 3.9× the step cost, and **robustness** — 3 of 9 L-CNN operators put
36–98% of C(0) on a single configuration where 0 of 14 GELT ones exceed 1.0%
(`notes/lcnn_shootout.md` §9–§9.2).

**The two audits.** (i) *Is the classical comparator a straw man?* Against the
input-matched strengthened arm (`deep`), the spectroscopy claim survives:
ΔA₀ = +0.097 ± 0.021 (4.6σ) combined. "Does real loop-shape variety close the
gap?" is **answered** (`notes/fable5.1_10-09_audit.md` §8.1, superseding
`notes/audit_2026-09-06.md` §6.5): dropping C(t0)'s near-null directions
instead of flooring them makes `full` readable, and loop shapes close about 60%
of the gap and leave a 1–2σ edge, ΔA₀ = +0.038 ± 0.027. `shapes` and
`shapes_sm` still fall back on ens1, for a reason no whitening can fix — the
GEVP at t0 = 1 maximises C(2)/C(1) while the gate tests C(2)/C(0). (ii) *What did the network find?* 12.9%/11.7%
of `O_GELT`'s norm² lies outside the span of the whole classical basis on two
ensembles, and removing it costs the entire advantage — ΔA₀ = +0.076 ± 0.019
(4.0σ) at unchanged mass. Not a contact term. `r` alone is a poor operator
(A₀ = 0.43) that wins by constructive interference. Both controls have since
run (`notes/fable5.1_10-09_audit.md` §8.3): against the strong `deep` arm the
7-level net is 12.8% outside with ΔA₀(GELT − P) = +0.097 ± 0.021, against the
21-operator `full` arm 2.5% and +0.021 ± 0.009; 80% of the residual is
rectangular loops the network rediscovered. **Untrained nets are further
outside the span (16–74%) than the trained one**, so the norm fraction is
architectural — what is learned is `Z_r/Z_G` (0.146 vs 0.047) and the sign of
ΔA₀(net − P), which flips to −0.091 ± 0.042 without training.

## Layout

Library in `gelt/`, entry points in `scripts/`, pytest in `tests/`. Installed
editable via `pyproject.toml`. Device order: cuda → mps → cpu.
`lge-cnn-master/` is vendored third-party code — the authors' own L-CNN
implementation (MIT, Favoni et al. 2012.12901), layer sources only, tracked so
`scripts/bench_lcnn_reference.py` runs anywhere the repo is cloned. Nothing in
`gelt/` imports it.

### `gelt/`

- **`lattice.py`** — `GaugeGroup` ABC with `Z2` and `SU(N)`; pure tensor
  functions. `SU.project` (nearest group element) takes a closed-form route at
  `nc == 2` (`_polar_factor_2x2`), the same map with no `linalg` kernels (~23×).
  - `random_links(L, D, group, dtype, Lt=None)` → `(D, *Λ, nc, nc)`. `Lt` gives a
    non-cubic `Λ = (Lt,) + (L,)*(D-1)` lattice (time = axis 0) for anisotropy.
  - `plaquette_tensor(U, group)` → `(D(D-1)/2, *Λ, nc, nc)`.
  - `action(U, group, beta, plaquettes=None, xi=1.0, time_axis=0)` → Wilson
    action `β Σ_p (1 − Re Tr P / nc)`. **Anisotropic** when `xi ≠ 1`: temporal
    plaquettes weighted `β_t = β·ξ`, spatial `β_s = β/ξ` (tree-level); `xi = 1` is
    the bit-exact isotropic path.
  - `topological_charge_density(U, group, plaquettes=None, definition="clover")`
    / `topological_charge` — `q_x = ε_{μνρσ} Tr[F_{μν}F_{ρσ}]/32π²` and its
    lattice sum, D = 4 only. **`"clover"` is the default**: `F_{μν} = (C − C†)/8i`
    with `C_{μν}(x)` the sum of the four leaves around `x`, all based at `x`
    (a cyclic rotation of a plaquette is a similarity transform by a link, and
    `q_x` contracts two planes at one site, so the basepoint is not free) —
    hypercubic-symmetric and **exactly parity-odd**. `"plaquette"` is the naive
    corner-based `F_{μν} = (P − P†)/2i`, L-CNN Eq. (13), kept because that is
    what the L-CNN paper regresses and because the parity test is comparative.
    The `plaquettes=` shortcut is plaquette-only and raises otherwise.
  - `rectangular_wilson_loop(U, group, R, T, mu, nu)` → `Re Tr W/nc` per site.
  - `link_gauge_transformation` / `local_gauge_transformation` — apply site-local
    Ω to links (`U_μ(x) → Ω(x) U_μ(x) Ω†(x+μ̂)`) or to an adjoint field
    (`W(x) → Ω(x) W(x) Ω†(x)`).
  - `l1_ball_offsets(D, R)` → signed Δx with `1 ≤ |Δx|₁ ≤ R`, ordered by `|Δx|₁`.
  - `build_transport_average(U, R, group, mode="average")` — DP routine
    materialising `T_Δx(x)` over the full signed L1-ball. Expects batched links
    `(N, D, *Λ, nc, nc)`. `"average"` is shortest-path-averaged (the default and
    the architecture's choice); `"single"` is one canonical path.
- **`sampler.py`** — `staple_sum`, `metropolis_sweep` (checkerboard-vectorised;
  proposal routed by `_PROPOSAL_FN[type(group)]`), `mcmc_ensemble` (thermalise +
  decorrelate + collect, sweep routed by `_SWEEP_FN`; **Metropolis is the
  registry default for both groups**), `haar_ensemble`.
  - `staple_sum(..., batched=True)` accepts a leading configuration axis; it is
    bit-identical to looping and is what lets `ape_smear` vectorise. The MCMC
    sweeps are one sequential chain and keep the unbatched form.
  - **Anisotropy:** `staple_sum` and all sweeps take `xi=1.0, time_axis=0`; the
    per-plane coupling ratio ξ^±1 is folded into the staple, so `xi = 1` is
    bit-exact backward-compatible. Opt in by binding ξ into the sweep, e.g.
    `functools.partial(heatbath_overrelaxation_sweep, n_or=4, xi=ξ)`.
  - **SU(2) heat-bath + overrelaxation** (`heatbath_sweep`,
    `overrelaxation_sweep`, `heatbath_overrelaxation_sweep`) — exact,
    tuning-free, beats Metropolis critical slowing; the prerequisite for
    resolvable spectroscopy. **Opt-in, not the registry default**: pass it as
    `sweep_fn=` to `mcmc_ensemble`. Heat-bath factors the staple `A = k·V` and
    draws the scalar part via Creutz (`_sample_su2_w0`, `max_iter=1000` — the 100
    of the original cost the ens2 replication run); overrelaxation reflects
    `U' = V†·U†·V†` and re-projects with the closed-form `_project_su2`.
    SU(2) only — SU(N≥3) needs Cabibbo–Marinari.
  - **`z2_heatbath_sweep`** — exact Z₂ heat-bath, `P(U=+1) = σ(2βs)`. Near β_c
    the Metropolis flip proposal's acceptance collapses to 0.02; this does not.
  - **`integrated_autocorrelation_time(series, c=6, max_lag=None)`** → `(rho,
    tau_int, window)`, Madras–Sokal windowing. Samples `n_skip ≳ 2·τ_int` apart
    are effectively independent.
- **`data.py`** — `build_plaquette_datasets(N, D, L, group, target, ...)`.
  `target` is a callable `target(configs, group) -> Tensor` (bind extras with
  `functools.partial`). `structured=True` keeps the `(N, n_pairs, *Λ, nc, nc)`
  matrix layout for GELT; `structured=False` calls `flatten_color` for the CNN
  baseline. With `R` set, the transport is precomputed per config and the splits
  yield `(X, T, y)` triples. `save=True` writes to `datasets/`.
- **`blocks.py`** — the one GELT block. `GEMHSA`: augment (append daggers +
  identity) → fused Q/Q_v/K/V projections → adjoint transport of K, V via
  `T`/`T_dag` → gauge-invariant score `Re Tr[Q† K̃]` + RoPE → softmax →
  multiplicative value `Σ α · Q_v† · Ṽ` → channel mix → residual + L-Act gate.
  Also `ChannelLift`, `Trace`, `MLP`, `GELT` (ChannelLift → stacked GEMHSA →
  Trace → per-site MLP → spatial reduction). `GELT.forward(W, T)` computes
  `T_dag` once and prepends the Δx = 0 transport slot once for the whole stack.
  `_last_score` / `_last_alpha` are stashed per layer under `no_grad` for the
  interpretability studies — **opt-in**, via
  `GELT.set_introspection(store_attention=True)`, along with the `_last_*_norm`
  scalars under `diagnostics=True`.
  **`alpha_mode`** selects how the offset weights are produced and nothing else:
  `"softmax"` is the architecture, `"frozen"` is the M1 ablation
  (`notes/m1_probe.md`) — α becomes `softmax` of a learned `(H, n_offsets)`
  table, Q_s/K/`rope_freq` are not allocated, and `value_path` is literally the
  same method in both modes. Still a *softmax*, so the ablation removes
  input-dependence (M1) without removing boundedness (M2).
- **`glueball.py`** — classical 0⁺⁺ spectroscopy. Time is lattice axis 0.
  - `ape_smear(U, group, alpha=0.5, n_steps=1, directions=None)` — spatial-only
    APE smearing (time links untouched, so the transfer-matrix interpretation
    holds). Vectorised over the configuration batch, sliced by a `chunk_bytes`
    budget; bit-exact under any chunking.
  - `glueball_operator` → `(B, *Λ)` sum of spatial-plane R×T Wilson loops;
    `zero_momentum` → `(B, Nt)`; `connected_correlator` → vacuum-subtracted
    `C(Δ)` (0⁺⁺ has a nonzero VEV — the subtraction is essential);
    `effective_mass`; `jackknife_effective_mass`.
  - **Multi-level GEVP**: `smearing_operator_basis`, `connected_correlator_matrix`,
    `gevp_eigenvalues` (eigh-whitening with an eigenvalue floor, not Cholesky —
    low statistics can make `C(t0)` indefinite; `truncate=True` on it and on
    `gevp_ground_vector` drops the near-null directions instead of flooring
    them), `gevp_effective_mass`, `jackknife_gevp_effective_mass`.
  - **Fit / overlap layer**: `gevp_ground_vector(C, t0, td)` → the ground-state
    generalized eigenvector, defining the *projected operator* — the optimal
    single operator in a basis's span, i.e. the apples-to-apples comparator for a
    single learned operator; `fit_cosh_correlator(C, dmin, dmax, sigma)` →
    `(m, A, χ²)` for `A·[e^(−mΔ) + e^(−m(Nt−Δ))]` (profiled-A grid scan +
    parabolic refine — dependency-free, safe inside every jackknife sample). The
    ground-state overlap fraction is `A₀ = A·(1+e^(−m·Nt))/C(0)`.
- **`probe_targets.py`** — the M1 probe's supervision (`notes/m1_probe.md`).
  `action_density` → `f(x) = 1 − Re Tr P̄(x)/nc`; `ball_reduce` → cumulative
  shell sums and the running max over the L1-ball of `BALL_RADIUS = 4` in one
  traversal; `build_targets` → **T0** (a radial convolution — the calibration
  arm, exactly linear by construction), **T1** (the ball max — position), **T2**
  (the θ = ½ enclosing radius of `f**SCALE_POWER`'s mass — position *and* scale,
  and exactly invariant under `f → λf`). `ball_features` is the pre-flight's
  design matrix. **`SCALE_POWER = 4` is a pre-flight finding**, not a taste: at
  p = 1 the 66-site outer shell concentrates, the `min{r}` selection never
  selects, and a linear filter reaches R² = 0.978.
- **`lcnn.py`** — Favoni et al. L-CNN: `build_axis_transports` (axis-aligned link
  products — distinct from GELT's L1-ball `T`), `LConv`, `LBilin`, `LCB`,
  `LAct`, `Trace`, `LCNN`. Mirrors `GELT`'s I/O so the two are
  matched-parameter comparable. `LCNN` also takes `in_channels` (the stacked
  multi-level smeared input the glueball task feeds), `init_scale` (output
  scale, GELT's knob under another name), `grad_checkpoint` and `symmetric`.
  **The L-Conv kernel is symmetric** (both orientations, as in the authors'
  own code): the W† channels do *not* subsume backward hops, because daggering
  commutes with the adjoint transport, and the one-sided version gave a stack a
  one-sided cone (`notes/lcnn_shootout.md` §8.1). The block follows the
  reference's fast contraction order — the channel-pair outer product is never
  materialised, the transport runs on the raw channels and the augmentation is
  formed afterwards, and the colour matmuls fold the channel axis (§8.2–8.4);
  both rewrites are pinned against the naive definitions in `tests/test_lcnn.py`.
  **The matched shootout is wired up but has not been run** — it is
  `GLUEBALL_ARCH=lcnn` in `train_glueball.py`, driven by `lcnn_shootout.sh`;
  design and pre-registered readings in `notes/lcnn_shootout.md`.
  `scripts/bench_lcnn_reference.py` times our block against
  `lge-cnn-master/`'s at the production shape.
  **`normalize_shifts`** (off by default) reparameterises ω as
  `g[i,j]·ω̂[i,j,s]` with `Σ_s |ω̂| = 1`, bounding the aggregation over offsets
  the way a softmax does while the per-channel magnitude stays free — the M2
  control arm of `notes/m1_probe.md` (R-D). It is the identity at
  initialisation, so the arm starts from the reference distribution.
- **`cnn_baseline.py`** — `LatticeCNN`: non-equivariant baseline; `Conv2d`/
  `Conv3d` for D=2/3 and a roll-based `_RollConvND` for D≥4.

### `scripts/`

Flat and self-contained: each defines its own `evaluate` / `train_model` inline
(there is no shared `gelt/train.py`). Cross-imports go through
`sys.path.insert(0, dirname(__file__))`, which is why they are not in
subdirectories. `README.md` has the one-line table; the details that matter:

- **`train_glueball.py`** — the variational operator. `GELT(reduction="none")` on
  the Rayleigh loss, **per-timeslice 3D** (any temporal receptive field voids the
  transfer-matrix bound and makes the loss gameable toward m → 0),
  `mlp_zero_init=False` (zero init ⇒ exactly zero Rayleigh gradient — training
  never starts), `INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`, `SCALE_REG` pinning the
  `Ō → λŌ` flat direction. Env overrides (each also `--name=value` in argv):
  `GLUEBALL_ENSEMBLE_SEED`, `GLUEBALL_INIT_SEED`, `GLUEBALL_RESUME`,
  `GLUEBALL_EVAL_ONLY`, `GLUEBALL_INPUT_SMEAR_LEVELS`, `GLUEBALL_D_MODEL`,
  `GLUEBALL_RANDOM_INIT` (the untrained, eval-only baseline), `GLUEBALL_LR`,
  `GLUEBALL_EPOCHS` and `GLUEBALL_RUN_TAG`. Artifact names are `_sm<levels>` +
  `_d<width>` + `_ens<k>` + `_rnd<k>|_init<k>` + tag, so no run overwrites
  another; the two 7-level nets are d_model 24 but predate the `_d` tag. Dumps
  the test-split Ō arrays to `datasets/…_test_obars.pt` (also kept in `dumps/`).
  **`GLUEBALL_ARCH=lcnn`** swaps in the matched-parameter L-CNN
  (`GLUEBALL_LCNN_K|_C_HIDDEN|_LAYERS|_INIT_SCALE`, defaults 2/5/4/1.0 — the
  width is matched to the net being compared against, `c_hidden=6` for a
  7-level run) and its
  axis-aligned transports, leaving ensemble, splits, inputs, loss, selection and
  jackknife identical by construction — `_build_model()` is the one constructor
  (`profile_glueball_step.py` calls it too, so the profiler cannot drift). The
  stem becomes `best_glueball_lcnn…`, the geometry slot carries
  `_k<K>c<c_hidden>l<layers>` when non-default, and the dump keeps the
  `gelt_obar` key (one learned operator against the classical span, whatever
  produced it) plus `meta["arch"]`.
- **`z2_attention_correlator.py`** — the Z₂ table. GEVP t0=1/td=2, window
  Δ∈[2,8], cosh + A₀, blocked jackknife block 20, config-scramble null,
  time-shuffle zero-mode check, correlated ΔA₀. `ZAC_REPLOT=<dump.pt>` regenerates
  every figure and table offline (no GPU, no ensembles, no checkpoints).
- **`su2_attention_correlator.py`** — the same measurement on anisotropic SU(2);
  **imports the estimator layer from `z2_attention_correlator.py`**, so identical
  conventions is a fact about the call graph. Defaults to one row at β = 2.4 (the
  only trained coupling ⇒ a true diagonal cell); `SAC_BETAS` runs a scan. Refuses
  to print aggregates (Pearson, slope, dynamic range) below three couplings
  rather than emitting NaN. **Does not run on MPS** (SU(2) projection needs
  complex `linalg.det`/`svd`) — use `SAC_DEVICE=cpu` for a smoke test.
- **`su2_fair_fight.py`** — the strengthened classical arms (`published` /
  `deep` / `shapes` / `full`), gated on reproducing the published numbers first.
  `SFF_NOCACHE=1` runs the dump-only path offline. `SFF_BASES=1` runs **every**
  arm offline from the kept per-ensemble obars cache (now write-once);
  `SFF_TRUNCATE=1` / `SFF_PRUNE=<ρ>` are the estimator knobs of
  `notes/fable5.1_10-09_audit.md` WP1; `thin` is the curve's one-operator arm.
  A single-dump run writes `su2_fair_fight_<dump stem>[_trunc][_prune<ρ>].pt`.
- **`input_architecture_curve.py`** — A₀ against *input content* for three
  traces (classical GEVP, trained GELT, untrained GELT), assembled offline from
  one `su2_fair_fight.py` output per dump; prints the pre-registered readings of
  `notes/fable5.1_10-09_audit.md` §1.2 and writes
  `results/fair_fight/input_architecture_curve.{png,pt,tex}`. Its V100 half is
  `curve_batch.sh` (untrained evals, the thin points, the width control), which
  refuses to run without CUDA and skips phases whose dump exists.
- **`lcnn_shootout.sh`** — the L-CNN batch: profile a step, a 4-point LR ×
  init-scale sweep (10-epoch runs under disposable `_sweep_*` tags), the two
  full trainings (ens0, ens1), the untrained control. `LCNN_DRY_RUN=1` lists the
  phases without a GPU; `LCNN_PARTS=2,3 LCNN_LR=… LCNN_INIT=…` runs the second
  half once the sweep has been read. Refuses to run on the CPU and skips any
  phase whose dump exists.
- **`operator_decomposition.py`** — `O_GELT = P + r` in the exact Hilbert-space
  metric `C_ab(0)`, offline from the `dumps/` Ō arrays. Prints the published
  comparison first as a gate. `--basis=<obars.pt>:<arm>` swaps the span for a
  fair-fight arm (`deep`, `full`, …; the GEVP arm then uses §8.1's truncated
  whitening, and the ladder increments are skipped for a non-nested arm),
  `--shape-span=<obars.pt>:<arm>` projects the residual `r` onto a richer span,
  `--m-ref=<float>` fixes the reference mass for the amplitude split (needed for
  an untrained net, whose own cosh fit does not converge). All three are
  repeatable — one value, or one per dump — so both ensembles combine in one
  run; `--proj-eps` cuts the span's Gram, `--out-tag` names the artifacts.
- **`probe_common.py` / `probe_preflight.py` / `train_probe.py` /
  `probe_readings.py` / `probe_batch.sh`** — the M1 probe
  (`notes/m1_probe.md`). `probe_common.py` holds ensemble, timeslice
  extraction, splits, inputs, standardisation, the five `ARMS` and the R²
  sufficient statistics, so the arms cannot drift — the same discipline that
  makes the two attention scripts share one estimator. It **samples nothing**:
  the ensembles are `train_glueball.py`'s cached ones, by the identical cache
  key, and a missing cache is an error rather than a 24-hour sampling run.
  `probe_preflight.py` is §5's gate — the best linear filter over the same
  radius-4 ball, i.e. the best *M1-free* method at matched reach — and it is
  what must be read before anything trains. `train_probe.py` runs one
  (arm, target, seed); its dump is the six per-config sums an R² is made of, so
  the correlated jackknife is exact offline and a run is kilobytes.
  `probe_readings.py` prints R-A…R-E and a LaTeX fragment; it **skips
  `run_tag`-carrying dumps**, so the LR sweep's 6-epoch runs cannot be read as
  the 20-epoch result.
- **`profile_glueball_step.py`** — where one optimizer step goes, per stage,
  forward **and backward** separately. It goes through
  `train_glueball.config_inputs`, i.e. the pipeline the training loop actually
  runs; an earlier version profiled a different one and produced the false
  "transport is 1.9%" figure. Env: `PROFILE_DIAGNOSTICS`, `PROFILE_LEGACY_SMEAR`,
  `PROFILE_MICRO`, `PROFILE_COMPILE`.

### `tests/`

- **`test_lattice.py`** — gauge invariance of `plaquette_tensor` / `action`
  (bit-exact in Z₂ float64), anisotropic-action invariance, the `xi = 1` match,
  non-cubic shapes, `SU(2).project`'s closed-form route against the SVD polar
  it replaced (far-from-group and near-group input; SU(3) still takes the general
  route), and **parity**: an exact lattice reflection (checked first to preserve
  the Wilson action to 1e-9) under which the clover density is odd site by site
  to 1.4e-17 and the plaquette density demonstrably is not.
- **`test_transport.py`** — `l1_ball_offsets` counts, brute-force per-octant
  pattern, octant-relation consistency, gauge covariance for Z₂ and `nc = 2`.
- **`test_blocks.py`** — gauge equivariance of `GEMHSA`/`GELT` (SU(2) complex128,
  both gates; Z₂ float64), the optimised attention path against a naive oracle
  (outputs, `_last_score`, `_last_alpha` and input gradients to 1e-12 for
  SU(2)/SU(3)/Z₂), `_OffsetGather`'s hand-written backward against autograd's,
  the Δx = 0 prepend, introspection off by default, gradient checkpointing, plus
  the four cases ported from the retired `blocks_bias` suite. The **frozen-α**
  arm runs the same gauntlet (oracle, equivariance at both gates, full-stack
  invariance) plus the two properties the M1 reading rests on: α is identical
  for two different fields on the same links *and the softmax arm's is not*, and
  no score path is allocated with nothing left ungradiented.
- **`test_sampler.py`** — overrelaxation conserves the Wilson action to machine
  precision and stays on the group; heat-bath reproduces the exact 2D SU(2) mean
  plaquette `I₂(β)/I₁(β)`; anisotropic versions of both.
- **`test_glueball.py`** — APE smearing is gauge covariant and stays on the group;
  the operator is gauge invariant; correlator / `m_eff` / jackknife recover a
  known mass from a synthetic correlator; the GEVP recovers both masses of a
  synthetic two-state matrix; `gevp_ground_vector` kills the excited state
  exactly; the truncated whitening is the identity on a well-conditioned C(t0)
  and survives an exactly duplicated operator; `fit_cosh_correlator` recovers
  `(m, A)`.
- **`test_lcnn.py`** — the L-CNN as the glueball baseline: gauge invariance of
  the per-site readout on stacked multi-level inputs (SU(2) complex128, both
  gates; Z₂ float64), `in_channels` defaulting to the plaquette count,
  gradient checkpointing bit-exact, `init_scale` linear in the output, the
  parameter match against both trained GELT nets (real DOFs within 15%), the
  optimised `LConv`/`LBilin` against the naive definitions they implement
  (1e-12, complex128), the **support of a layer** — that the kernel reaches
  `x − k·μ̂` as well as `x + k·μ̂` — and `normalize_shifts`: the identity at
  init, L1-bounded over offsets after any update, equivariance untouched.
- **`test_probe_targets.py`** — the M1 probe's premise: `f` is gauge invariant
  (SU(2), Z₂) and non-negative; the shell sums and the ball max against a
  brute-force enumeration of `[-R,R]^D` filtered by the L1 norm (independent of
  `l1_ball_offsets`, so a bug there cannot hide behind itself); **T0 is exactly
  a linear functional of the ball**; T2 recovers a hand-computed enclosing
  radius on a uniform field and is invariant under `f → λf` while T0/T1 are
  degree 1; the periodic-ball guard; and **both architectures reach the whole
  radius-4 target ball** (GELT after 2 of 4 layers, the L-CNN after 3) as
  integer set arithmetic — if that fails, R-C is measuring receptive field.
- **`test_data_model.py`** — split validation and CNN-baseline shape guards.

## Conventions

- **Tensor layouts** (the only spec, no OO wrappers): links
  `(D, L, …, L, nc, nc)` — direction first, spatial axes, color last;
  plaquettes `(n_pairs, L, …, L, nc, nc)` with `n_pairs = D(D-1)/2`, ordered by
  `(μ, ν)`, μ < ν lexicographically.
- **Color axes are always present**, even for Z₂ where `nc = 1`. Every product is
  `A @ B` and every inverse `group.dagger(A)`, so the code ports verbatim to
  U(1)/SU(N). For Z₂ both are no-ops; any laxity is a bug Z₂ cannot catch.
- **Plaquette convention:** `P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)`.
- **Periodic BCs:** `torch.roll`. Never manual modulo arithmetic on indices.
- **Wilson action:** `S = β Σ_p (1 − Re Tr P / nc)`.
- **Parallel transport:** sum over **all** shortest lattice paths in the L1-ball
  — never a single axis-aligned path (unless `mode="single"`). The octant
  identity `T_{−Δx}(x) = dagger(T_Δx(x − Δx))` holds as a math property and is a
  test-suite check; it is **not** relied on at build time (one auditable DP
  surface beats a 2× memory saving, and mixed-sign offsets cannot be derived from
  positive-octant data anyway).
- **Float32** for training; `dtype=torch.float64` (→ `complex128`) through the
  dataset builders for high-precision gauge-invariance tests.
- **Time is lattice axis 0** throughout the spectroscopy code.

## Performance

Full record and the algebra behind each number: `notes/performance_audit.md`.
The ens1 replication logged **7.77 s/step, 1942 s/epoch** on a 32 GB V100 for a
~5k-parameter model. Four exactly-equivalent changes landed, each measured and
each covered by a test:

- **`SU(2).project` in closed form** (`gelt.lattice._polar_factor_2x2`). The APE
  ladder was issuing ~4.5 M batched 2×2 complex SVDs *per step*. `H = M†M` is 2×2
  Hermitian PSD, so `√H = (H + √det H·𝟙)/√(tr H + 2√det H)`: two matmuls plus real
  scalar arithmetic, agreeing with the SVD route to 3.9e-7 on 248 832 real
  production matrices with a *better* unitarity residual. `H` is formed in
  float64; MPS (no float64) stays in complex64 — where it now runs at all.
- **`ape_smear` vectorised over the configuration batch** (`staple_sum` gained
  `batched=True`): **7.4×** on real configs (5.85 s → 0.79 s per step), bit-exact
  under any chunking.
- **Introspection stashes are opt-in.** The two K̃/Ṽ norms alone re-read ~12% of
  the block's memory traffic, and the ~10 `.item()` calls per block are ~10 GPU
  syncs — all of it running again inside every gradient-checkpoint recompute.
- **The GEMHSA hot path**: K and V share one gather; the Δx = 0 transport slot is
  prepended once in `GELT.attn`; the RoPE rotation is folded into the *query*
  (`GEMHSA.rope_score` — Q carries no offset axis, so n_offsets times less data);
  and the neighbour gather's backward is written as a gather (`_OffsetGather`)
  instead of the atomic scatter-add autograd emits, which measured **64× its own
  forward**. Per-layer materialised bytes 43.7 → 29.8 GiB, non-view kernels
  76 → 44, `.item()` syncs ~10 → 0. **2.16×** on forward + backward, outputs
  identical to 1.2e-7.

**Measured end to end on the V100 (2026-09-13): 7.77 → 5.04 s/step**, i.e. the
landed work is worth **1.54×** in situ (the same profiler says 1.29 s for the
matched-parameter L-CNN arm, so GELT costs 3.9× what it does — the architecture,
not a defect). The profile re-ranked the backlog twice. First: **backward is 64% of the step
and `rope_score` is the largest single stage** at 238 ms per layer — **this was
an artifact and is retracted** (`notes/performance_audit.md` §5.0(ii)): the
micro-bench timed the gather and the transport inside rope_score's lambda, since
`.detach().requires_grad_()` cuts the backward graph but not the forward work.
Isolated, rope_score is **28.5 ms**. By subtraction **the transport is ≈200 ms
per layer per forward, ~70% of the forward**, so **§5.1 (the adjoint SO(3)
representation) is #1 by a wide margin** and everything else is a few percent.
The one thing that landed from the rope_score work is the **single pass over
K̃** (1.30× on the stage, measured 2.4% of the step); `PROFILE_ROPE=1 python
scripts/profile_glueball_step.py` is the bench behind it and behind the two
§5.6 `torch.compile` obstacles (§5.0(iv)).

**The step is now fully accounted for** (§5.0(v)): per layer, transport
223.4 ms fwd / 326.6 bwd, gather 9.3 / 142.9, rope_score 23.5 / 71.6, value sum
19.1 / 12.4 — which reconstructs 97% of the forward and 107% of the backward.
**Transport is 62.8% of the step**, and §5.1's predicted ~3.5× would take it
4930 → 2720 ms, **1.8× end to end**. Second is the gather's *backward* at 13.1%
and 15.4× its own forward, ~5× off its own traffic budget — a kernel problem
(a 4-index advanced read into a 9-d tensor, in 13 chunks), not the
loop-vs-index question, and worth a look after §5.1.

## Known caveats

1. **Z₂ APE smearing is broken at the production `SMEAR_ALPHA = 0.5`**
   (`notes/audit_2026-09-06.md` §2). With two spatial staples the update is
   `V = (1−α)U + (α/2)(s₁+s₂)`: the exact identity for α < ½, a majority-vote
   automaton for α > ½, and at α = ½ exactly `V = 0` whenever the staples disagree
   with the link, where `Z2.project` sends `0 → +1` — a value that does not
   transform. Measured on production configs: the ladder **freezes after one
   step**, 99.9% of changed links are gauge-dependent tie-breaks, Ō(t) moves 1.82σ
   under a gauge transformation, 0.67% of the network's smeared input plaquettes
   flip sign. So the Z₂ classical basis is one operator, not four (the dumps
   already say `gevp_fell_back=True, n_ops=1`), and the Z₂ nets were trained on
   four channels of which three are byte-identical. **SU(2) is verified clean**
   (1.4e-15). `tests/test_glueball.py` misses it because its covariance cases use
   α = 0.6/0.7, never 0.5. Fix: α = 0.7, or unprojected fat links.
2. **RoPE axis coverage:** `pair_axis = [p % D for p in range(n_pairs)]` rotates
   only `n_pairs` axes, so `d_qkv < 2D` leaves whole axes on the identity
   rotation. Enforce `d_qkv ≥ 2D` (8 in D=4, 6 in D=3).
3. **Datasets do not store β.** Multi-β training (and any conditioning on the
   coupling) needs β to become part of the dataset.
4. **Sampler dispatch is by group type.** `metropolis_sweep` looks up
   `_PROPOSAL_FN[type(group)]`, `mcmc_ensemble` looks up `_SWEEP_FN`; adding
   U(1)/SU(N) is a registry entry plus the proposal/sweep function.
5. **No worst-case-Ω stress test.** `check_gelt_invariance.py` is the quick
   version; the systematic drift-vs-Ω study does not exist.
6. **Offset-chunked attention does not exist** — it is the memory gate on running
   the attention studies at physically large R.
7. ~~**`topological_charge_density` is not parity-odd.**~~ **Fixed 2026-09-14**
   (`notes/where_attention_can_win.md` §4.1): `definition="clover"` is now the
   default and is parity-odd to 1.4e-17 site by site. `definition="plaquette"`
   keeps the old corner-based density, which is *not* — under the exact lattice
   reflection `x₁ → (−x₁) mod L` with `U'_1(y) = U_1(P(y+1̂))†` (which preserves
   the Wilson action to all printed digits) the charge goes
   **Q: −1.3161 → +2.9888**, sign flipped, magnitude not. That is now a test
   rather than a defect, and nothing was invalidated: no production script ever
   called the function. **Note the reflection's index order** — `P(y+1̂) = Py − 1̂`,
   so the link along the reflected axis is shifted *down* before reflecting;
   the other order gives a plausible map that silently fails to preserve the
   action, which is why the test checks the action first.

## Running

```bash
python scripts/validate_sampler_z2.py        # Metropolis four-panel check (Z₂)
python scripts/validate_sampler_su2.py       # …and SU(2)
python scripts/validate_anisotropy.py        # anisotropic-lattice checks
python scripts/check_gelt_invariance.py      # 60-second gauge-invariance check
python scripts/train_cnn.py                  # CNN baseline, 1×2 Wilson loop
python scripts/train_gelt.py                 # GELT, the same problem
python scripts/train_lcnn.py                 # L-CNN baseline
python scripts/check_glueball_autocorrelation.py   # τ_int → production n_skip
python scripts/measure_glueball.py           # classical 0⁺⁺ baseline + ensemble
python scripts/train_glueball.py             # GELT as a variational operator
python scripts/fit_glueball_overlap.py [dump]      # cosh fits + A₀ (offline)
bash   scripts/overnight_replication.sh      # fresh ensemble + retraining (~24 h)
bash   scripts/curve_batch.sh                # the curve's random trace + 3 trainings
GLUEBALL_ARCH=lcnn python scripts/train_glueball.py  # the matched-parameter L-CNN
LCNN_DRY_RUN=1 bash scripts/lcnn_shootout.sh # the L-CNN batch: what would run
python scripts/bench_lcnn_reference.py       # our L-CNN block vs lge-cnn's
python scripts/operator_decomposition.py     # O = P + r (offline, seconds)
python scripts/input_architecture_curve.py   # A₀ vs input content (offline, seconds)
SFF_NOCACHE=1 python scripts/su2_fair_fight.py     # reproduce ΔA₀ offline
python scripts/z2_beta_scan.py               # Z₂ classical mass vs β
Z2G_R=6 Z2G_N_USE=800 python scripts/train_z2_glueball.py 0.756
python scripts/z2_attention_correlator.py    # the Z₂ attention table
python scripts/su2_attention_correlator.py   # the SU(2) row (~40 min)
PROBE_DRY_RUN=1 bash scripts/probe_batch.sh   # the M1 probe: what would run
python scripts/probe_preflight.py            # §5's gate (offline, seconds)
PROBE_ARM=frozen PROBE_TARGET=T2 python scripts/train_probe.py
python scripts/probe_readings.py             # R-A…R-E (offline, seconds)
PROFILE_DIAGNOSTICS=1 python scripts/profile_glueball_step.py
PROFILE_ROPE=1 python scripts/profile_glueball_step.py   # the rope_score bench
pytest tests
```

`.venv/` is local (uv-style, not gitignored). `datasets/`, `results/`, `*.pth`,
`*.png` are gitignored.

## Suggested next steps

Ranked in `notes/audit_2026-09-06.md` §4, and unchanged by the cleanup:

1. ~~The random-init control for the operator decomposition~~ — **done**
   2026-09-12 from the curve's untrained dumps, no new GPU time
   (`notes/fable5.1_10-09_audit.md` §8.3).
2. **Fix the Z₂ smearing** (caveat 1) and add the α = 0.5 covariance case to
   `tests/test_glueball.py`. Changing the Z₂ inputs needs a retrain for a clean
   end-to-end statement; evaluating existing checkpoints on covariant inputs is
   the cheap robustness check first.
3. ~~Prune near-degenerate operators before the GEVP~~ — **done differently**
   2026-09-10: truncated whitening (`SFF_TRUNCATE=1`), not pruning, made `full`
   readable; pruning moves nothing (`notes/fable5.1_10-09_audit.md` §8.1).
4. **Dump per-config Ō from `z2_attention_correlator.py`** so the decomposition
   transports to the attention field without a GPU re-run.
5. ~~The matched-parameter L-CNN shootout~~ — **done** 2026-09-14, and it landed
   on the pre-registered **parity** (`notes/lcnn_shootout.md` §9). One arm is
   provisional: the ens0 60-epoch run is unusable and the row uses a sweep arm
   until a clean run exists (§9.1/§9.2).
6. **Where attention can win, given parity** — `notes/where_attention_can_win.md`.
   The flow-free topology study that used to sit here was **stopped 2026-09-15**
   on a measurement, and its code is deleted (§10 of that note). The lesson is
   the durable part: the target was the output of a classical smoother, and 48
   APE steps with one fitted scalar reproduce it at R² = 0.978 — free, untrained,
   and with more reach than any bounded receptive field. **Before proposing the
   next A/B, run it past the four criteria in §6 and then run §5's pre-flight:
   measure the best classical method at the architecture's own reach.** §8
   proposes the one experiment aimed at the mechanism that *has* been observed
   (boundedness, 0-of-14 vs 3-of-9), which costs a dozen short runs on data
   already on disk.
7. **Run the M1 probe** — `notes/m1_probe.md`. Built and pre-flighted
   2026-09-15; §7 of that note is the command sequence. `probe_batch.sh` part 0
   (the matched-DOF table + the pre-flight on both real ensembles, no GPU) is
   the next thing to run, and **part 1 must not start until part 0's logs have
   been read** — the pre-flight has already invalidated one target design once.

## Things to keep in mind

- **Do not silently broadcast across color axes.** A missed dagger or a wrong
  shortest-path step passes every Z₂ test and fails at the first non-abelian Ω.
- **Do not remove comments unless the code they describe is being deleted.**
- The two attention scripts share one estimator by import, and the two
  Wilson-loop scripts share one problem by construction. Both pairs must be
  changed together or the comparison silently stops being one.
