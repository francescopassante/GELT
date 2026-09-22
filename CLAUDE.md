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
- `notes/wilson_regression_1p1d.md` — **the reproduction of the L-CNN half of
  Fig. 3 of PRL 128, 032003**: 1+1D SU(2) Wilson-loop regression (1×1, 1×2, 2×2,
  4×4) on 8×8, on the authors' own L-CB, at the parameter counts and
  hyper-parameters the Supplemental Material prints. **One network per loop
  shape, four runs** — Table V's three sizes and SM §VI.A's ten seeds are a
  sweep whose minimum is the number in the panel, not three results, and the
  default takes the largest size at one seed (§8). **Their baseline-CNN half is
  deliberately not reproduced** (§1): 2 680 models over 264 architectures and
  four activations, and it says nothing about whether our L-CNN is theirs. Holds
  the β-ladder ambiguity (SM Table I says 11 couplings, SM §II says 10), the
  one-chain-per-configuration departure from their MC, and **§4, the divergence
  that forced `gelt/lcnn_exact.py`**. Wired and smoke-tested 2026-09-22; **the
  production ensemble is not generated and no reading has a value yet** (§7 is
  the pre-registration).
- `notes/where_attention_can_win.md` — **the architecture question's running
  record, and the first thing to read before proposing a GELT-vs-L-CNN task.**
  Both attempts and why each closed: the 0⁺⁺ tie (structural) and the flow-free
  topology study (**stopped 2026-09-15**, measured). It separates the two
  candidate mechanisms — *relative weighting* (M1, which **the probe's ablation
  has now shown to pay, 6/6 cells, on a constructed target** — §1.2; it has
  still never paid on a physics task) from *boundedness* (M2, observed three
  times) — states the **four criteria** a
  candidate task must now satisfy, and proposes the one experiment that targets
  M2 directly (§8). §5's pre-flight rule is the cheap gate that stopped topology for
  ~4 h instead of ~60–100: **measure the best classical method at the
  architecture's own reach before building any training code.**
  **§9 is attempt 4, and it is RUN and CLOSED (§9.9, 2026-09-20): a tie.**
  Vortex-cluster geometry in 3D Z₂ on the cached ensembles, the first
  candidate that is a physics observable and clears all four criteria.
  **ΔR²(GELT − L-CNN) = −0.073 ± 0.051 on V1 at β = 0.7520** (six seeds
  each, every arm at its own bracketed learning rate, 240 epochs,
  batch = 1) — no difference, t = 1.4. Both architectures clear the
  matched-depth classical bar [0.404, 0.577], which is the study's one
  unambiguously positive statement. **The significant reading is
  dispersion, and it goes the other way**: GELT's spread over
  initialisations is 6.2× the L-CNN's (sd 0.123 vs 0.020, F = 38 on
  (5,5) df, p < 0.002), below the classical bar in 3 of 6 seeds against
  0 of 6 — see the robustness caveat under the L-CNN heading below.
  **Qualified by attempt 5** (`notes/beta_transfer.md` §9.1): that holds
  *in distribution only* — at the three other cached couplings the ratio
  is 1.5× / 1.3× / 1.0×, GELT's spread barely moving and the L-CNN's
  created by the shift (17× at β = 0.7560). Quote it with the coupling
  named, not just the task.
  M1 (`gelt` − `frozen`) = +0.066 and M3 (`gelt` − `gelt_single`) =
  +0.018 at one seed, directional only. The verbatim readings for every
  run are tracked at `notes/z2_vortex_readout_2026-09-20.txt`, because
  the dumps they come from are gitignored and live only on the V100. **The pre-flight passes all
  five gates on all four cached ensembles** (2026-09-18;
  `scripts/z2_vortex_preflight.py`, `gelt/vortex_targets.py`,
  `tests/test_vortex_targets.py`). At the chosen primary coupling β = 0.7520 the
  best linear filter at Manhattan 8 reaches R² = 0.152 against 0.671 for the
  best *local* classical algorithm at **uncapped** BFS depth and 1.0 for the
  target, so the difficulty is routing, not density. 0.671 is **not** the bar
  a 4-layer network is asked to clear — that is the matched-depth interval
  [0.404, 0.577], the same BFS capped at 4 and 8 hops (§9.4). **The β ladder is fixed in §9.6: primary 0.7520,
  replication 0.7450** — the two that rank first and second on headroom, on a
  low linear ceiling and on a balanced cluster competition. Note the tension
  §9.4 records: the linear ceiling *rises* towards β_c (0.106 → 0.280), so the
  architecture question is cleanest where the physics is least critical. It also records a measured identity that changed the
  design: **in Z₂ GELT's path-averaged transport is a hard vortex mask**
  (`T_Δ² = (1 + P_enclosed)/2 ∈ {0,1}`) while the L-CNN's axis transport acts as
  the identity, which is a *third* input-dependent weighting (M3) and forces a
  2 × 2 design, `{softmax, frozen} × {average, single}`, rather than an A/B.
- `notes/beta_transfer.md` — **attempt 5, run and closed 2026-09-20 (§9):
  X-B fails as pre-registered.** The contrast favours GELT at 3 of 3 couplings
  (+0.007, +0.310, +0.013) but p = 0.485 / 0.240 / 0.240, none under the 0.05
  the claim required, so the adaptation hypothesis is **not supported**. Two
  findings survive: **§9.9's 6.2× dispersion result is a property of the
  training coupling** — one β away it is 1.5× / 1.3× / 1.0×, GELT's spread
  unchanged and the L-CNN's *created by the shift* — so quote it as "at the
  coupling both arms were trained on"; and **transfer failure is invisible in
  distribution** (GELT's one collapsing seed is its 2nd best operator at β₀,
  the L-CNN's three are ranked 3rd, 5th, 6th), so selecting on val loss does
  not select a transferable operator, for either arm. The M2-shaped count
  (1 of 6 seeds against 3 of 6) is the third appearance of that shape and is
  worth nothing alone — Fisher p = 0.545, the threshold is post-hoc, and the
  study's own pre-existing criterion reads 6 of 18 each, dead even. It does
  **validate §8's third stressor**. Readings verbatim at
  `notes/beta_transfer_readout_2026-09-20.txt`. The axis the four closed
  attempts never varied: every A/B here
  trained *and* tested at one coupling, which is precisely the setting in which
  an input-dependent reweighting over offsets has nothing to earn. It crosses
  §9.9's twelve V1 checkpoints with the four cached Z₂ ensembles — **no
  training**, forward passes only — and reads the degradation per seed, so
  §9.9's 6.2× dispersion cancels instead of drowning the comparison. Two
  columns, *raw* and *affine* (a two-parameter recalibration fitted at the new
  coupling, given to both arms): the primary reading is the affine one, because
  a GELT advantage that survives a recalibration is about **routing** and not
  about amplitude, and their difference is the amplitude piece, which is M2's
  fingerprint rather than M1's. Readings X-A…X-E are pre-registered in §4, the
  falsification hands the programme back to `where_attention_can_win.md` §8,
  and §6 names the confounds — regression to the mean first.
- `notes/m1_probe_status.md` — **the plain-language orientation for the M1
  probe: read it before `m1_probe.md`.** What is being tested and why, the
  arms and targets in one table each, the findings so far, what is running,
  what to run when it finishes, and the six things not to re-break.
- `notes/m1_probe.md` — **attempt 3, run and completed 2026-09-16 (132 runs).** The confound
  `where_attention_can_win.md` §1 missed: GELT and the matched L-CNN differ in
  *two* things (input-dependent offset weights **and** transport geometry), so
  no GELT-vs-L-CNN number has ever measured M1. The clean test is GELT against
  **GELT with the softmax frozen**, on three constructed per-site targets.
  Holds the arm table, the four-criteria audit (criterion 4 is knowingly
  violated — it is a mechanism assay, not a physics result), the pre-registered
  readings R-A…R-E, and §3.1, where the pre-flight **changed the design before
  any training code ran**. §3.2 is the production pre-flight (2026-09-15, both
  ensembles, all gates pass): T0 exactly 1.0000, T1 headroom 0.92, T2 0.38 — and
  the **radial linear filter equals the full-ball one to four decimals**, so the
  M1-free ceiling is a five-parameter object and nothing that separates the arms
  can be directional weighting.
  **§0 is the one-page readout of what it found**, from the 90-run grid plus
  parts 7 and 9: **M1 pays** — R-B = +0.144 on T2, 6/6 paired cells, p = 0.031,
  with capacity controlled by `frozen_matched` — GELT and the L-CNN tie on
  accuracy and differ 8× in dispersion; the softmax is worth ≈ 0.48 of R² at
  matched parameters but **not** for the reason proposed (§7.6 withdraws the
  non-negativity / normalisation / boundedness split: the T0 control says
  trainability); **path averaging in the transport is a null on accuracy**
  (§7.7) and is worth *less* on the mechanism target than on the calibration
  one, which makes GELT's 3.9× step cost a choice here; and **part of GELT's
  low dispersion is the transport rather than the attention** — the same network
  fed a single-path `T` falls below R² = 0.70 in 2 of 12 cells and the projected
  one in 3, against 0 of 12 for `gelt`. The verbatim readings for all 132 runs
  are tracked at `notes/m1_probe_readout_2026-09-16.txt`, because the dumps they
  come from are gitignored and live only on the V100.
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
(`notes/lcnn_shootout.md` §9–§9.2). **The robustness claim is
task-dependent, not architectural** — on the Z₂ vortex task it reverses,
GELT being 6.2× more dispersed over initialisations than the L-CNN
(`notes/where_attention_can_win.md` §9.9). Say "on the SU(2) glueball
task" wherever this is quoted. **And the reversal is itself
coupling-dependent** (`notes/beta_transfer.md` §9.1, 2026-09-20): one β
away from the training coupling the ratio is 1.5× / 1.3× / 1.0×, so the
Z₂ result is "GELT is the dispersed arm *in distribution*", not "on this
task". What survives all three measurements is narrower than either
sentence: a **failure-rate** asymmetry in the predicted direction
whenever the input distribution is stressed — 0-of-14 vs 3-of-9 (SU(2)),
6-of-6 vs 3-of-6 (`m1_probe.md` §0), 1-of-6 vs 3-of-6 seeds under
coupling transfer (Fisher p = 0.545) — directional three times,
significant none. `where_attention_can_win.md` §8 is the experiment that
would turn it into one measurement, and attempt 5 has now validated its
third stressor.

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
  - **`metropolis_sweep_multichain`** + **`_su2_proposal_paper`** — the batched
    sampler behind the 1+1D Wilson-loop datasets. A leading configuration axis is
    a batch of **independent chains**, each carrying its own β (a `(B,)` tensor),
    so a whole coupling ladder thermalises in one set of kernels and no `.item()`
    fires inside the hit loop. `_su2_proposal_paper` is Favoni et al.'s own kernel
    (SM §I): `V = exp(i Σ_a T^a X^a)` with `X^a = A η^a`, η standard normal — in
    closed form at nc = 2, the unit quaternion `(cos|X|/2, sin|X|/2·X̂)`. **Not**
    `_su2_proposal`, whose vector part is uniform on a cube. Pinned against the
    exact 2D result `I₂(β)/I₁(β)` at three couplings at once, so a β-broadcast
    bug shows up as the *ladder* being wrong rather than the sampler.
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
- **`vortex_targets.py`** — the 3D Z₂ vortex candidate's supervision
  (`notes/where_attention_can_win.md` §9). `vortex_field` → the gauge-invariant
  indicator `[Re Tr P/nc < 0]`; in 3D the negative plaquettes are dual links and
  Bianchi closes them into loops, which `closure_defect` checks as a hard gate
  (**Z₂ only** — the sign of a trace is not a centre element for SU(N)).
  `cluster_labels` is Shiloach–Vishkin on the dual graph, and **the hook must be
  applied to the tree's root**: hooking the boundary plaquette is `O(diameter)`
  and never converges at the production volume, where the percolating line is a
  few thousand dual links long. `build_targets` → **V1** (`log(1 + largest
  cluster touching x)` — global, the primary) and **V2** (the ball count — a
  convolution, exactly linear, the calibration arm). `local_component_size` is
  **not** a target: it is the classical local ceiling §5's rule requires be
  measured first. D = 3 only; in 4D the negative plaquettes form surfaces.
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
  **`conv_init_scale`** (default 1.0, so every existing caller is bit-identical)
  multiplies the L-Conv weight init. Unit scale per *layer* is not unit scale
  per *stack* at `nc = 1`: L-Act is `g(Re Tr W/nc)·W`, so at one colour the gate
  multiplies by its own argument instead of damping it and four layers reach
  1e21. The Z₂ probe arms use **0.2**, measured by
  `scripts/z2_init_gate.py` at the production volume (worst field at 4 layers
  over 4 seeds: 0.1 → 1.3, 0.2 → 2.6, 0.3 → 5.0, 0.4 → 8.4, **0.5 → 5.8e5**).
  The cliff is sharp and *seed-dependent*, so 0.2 rather than the largest
  passing value. **Read the field, never the output**: `build_arm`
  zero-initialises every head, so the output is 0 whatever the field is and a
  model one step from `inf` looks perfectly healthy.
  **`normalize_shifts`** (off by default) reparameterises ω as
  `g[i,j]·ω̂[i,j,s]` with `Σ_s |ω̂| = 1`, bounding the aggregation over offsets
  the way a softmax does while the per-channel magnitude stays free — the M2
  control arm of `notes/m1_probe.md` (R-D). It is the identity at
  initialisation, so the arm starts from the reference distribution.
- **`lcnn_reference.py`** — **the authors' own L-CNN**, not ours: their
  `LConvBilin` + `LActPoly` loaded verbatim out of `lge-cnn-master/` and wrapped
  so `LCNNRef` presents `gelt.lcnn.LCNN`'s I/O. Selected as `lcnn_ref` in
  `probe_common.ARMS` and as `GLUEBALL_ARCH=lcnn_ref`. Design record and the
  pre-registered readings: `notes/lcnn_reference_switch.md`. Three things it
  exists to get right: **their `kernel_size` is our `K + 1`** (their range is
  `[-(k-1), k-1]`, so passing K would halve the baseline's receptive field per
  layer with every other check still passing); their layout is
  `[B, N^D, N_C, nc, nc, 2]` with the links packed into the field tensor; and
  their layers are complex throughout, so a Z₂ arm is promoted and costs 2× the
  memory. `forward(W, U)` takes **raw links** — their block transports
  internally — and refuses an axis-transport tensor. `reference_dof_count`
  gives the parameter count from their shapes without building the model,
  because their kernel is quadratic in the input width and the matched width is
  therefore *not* ours (a per-layer channel list, their own `conv_ch` idiom).
  **Measured**: our `LCB` reproduces their kernel exactly (1.9e-15) once the
  L-Conv width reaches `2·c_in·(1+2DK)`, so ours is a low-rank member of their
  family — and the production widths are 8% and 1.6% of that.
  **`K` takes a per-layer sequence** (their deeper models grow the kernel with
  depth: L-CB(2,·), L-CB(2,·), L-CB(3,·), L-CB(3,·)), **`conv_impl`** picks the
  L-CB parametrisation (`"ref"` = as published, `"exact"` = `gelt/lcnn_exact.py`),
  and **`update_dims`** retargets a trained stack to another volume — which is
  the Letter's own generalisation claim and is a test, not a comment
  (a 2×2 tiling reproduces the tiled per-site output to 1e-10).
- **`lcnn_exact.py`** — **the L-CB the Letter *reports*, where the vendored code
  drifted.** Their published `LConvBilin` seeds the transported list with the
  field itself (`t_w = [w]`) and then appends the `D(k−1)` shifts, so the
  bilinear runs over `1 + D(k−1)` slots; PRL 128, 032003 Table V's parameter
  counts are reproduced **exactly, all ten architectures**, by `max(1, D(k−1))`
  slots — the transported copies alone, the local field kept only when there are
  none (`k = 1`). The extra slot is the same-site square `W_i · W_j` (the unit
  element already gives `W · 𝟙`), so the published class is a strictly larger
  function class at strictly more parameters: 47 vs 35 for `W^(1×2)` small,
  46 481 vs 39 905 for `W^(4×4)` large. Neither is treated as the correction of
  the other — `conv_impl="ref"` is the code as published, `"exact"` is the
  architecture as reported. `tests/test_lcnn_exact.py` pins the table, pins the
  excess, and shows the two are the **same function** once the added slots are
  zeroed (which tests the ordering of their `t_w` axis, not just its length).
  Nothing in `lge-cnn-master/` is modified; the class is a subclass of theirs.
  **Consequence elsewhere:** every "the authors' own L-CNN at N parameters" in
  `notes/lcnn_reference_switch.md`, `notes/lcnn_shootout.md` and
  `notes/m1_probe.md` quotes the *wider* variant. Those are matched to **our**
  L-CNN by parameter count and are unaffected as comparisons; what is not quite
  right is only the sentence "this is the paper's network".
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
- **`probe_common.py` / `probe_preflight.py` / `probe_transport_gate.py` /
  `train_probe.py` / `probe_readings.py` / `probe_batch.sh`** — the M1 probe
  (`notes/m1_probe.md`). `probe_common.py` holds ensemble, sample
  extraction, splits, inputs, standardisation, the `ARMS` and the R²
  sufficient statistics, so the arms cannot drift. Every entry point calls
  **`validate_argv()`** first: an unrecognised `--name=value` used to be ignored
  in silence, so a run would report its defaults while its command line claimed
  otherwise — it is now refused, with the known flags listed — the same discipline that
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
  the 20-epoch result. An arm's `transport` key is popped before the model is
  built (`arm_transport`) and fed to `probe_inputs` instead — `gelt_single` and
  `gelt_projected` are `gelt` to the byte with a different `T`
  (`notes/m1_probe.md` §4.4), which is why they are read against `gelt` and
  never against `lcnn`: the offset set is unchanged, so a single-path arm is
  not the L-CNN's transport. `probe_transport_gate.py` is their own pre-flight,
  a necessary-condition gate (do the two transports differ on production
  configurations at all?) rather than a ceiling.
  **`PROBE_GROUP=z2` switches the whole probe to the Z₂ vortex candidate**
  (`notes/where_attention_can_win.md` §9.5): group, cache key (β, not an
  ensemble seed), loader (a Z₂ configuration is *already* 3D, so it is fed
  whole with a length-1 slice axis and there is no timeslice extraction),
  targets V1/V2, and a **supervision mask** — ~90% of sites carry no vortex, so
  an unmasked loss would score "is there a vortex here" instead of the cluster
  size, and `standardize` takes its moments over the masked sites for the same
  reason. `R`, `LAYERS`, `LCNN_K`, `MLP_HIDDEN`, the splits and the R²
  statistics are shared by construction; the SU(2) path is bit-identical,
  checked against a worktree at the previous commit. Two things the switch
  forced: `GEMHSA` now takes a **per-axis lattice extent** (the Z₂ box is
  48 × 24 × 24; an int is still the cubic shorthand and builds bit-identical
  maps), and **the matched L-CNN has no usable forward pass in Z₂ at 4 layers**
  (§9.5.1 — 1.4e21 at initialisation, `inf` at the production volume, while
  GELT's field stays at 1: M2, visible before any training). Hence
  `conv_init_scale`, **0.2** on the Z₂ arms — 0.5 was the first value, chosen on
  an 8³ box, and `scripts/z2_init_gate.py` falsified it at the production volume
  on its first run.
- **`probe_curves.py`** — reads the dumps' stored `history` and answers the one
  question a sweep cannot be read without: **was the epoch budget the binding
  constraint rather than the learning rate?** A run whose best epoch is its last
  *and* whose val curve is still falling over its final quarter was cut off, and
  a sweep read on such runs picks the rate that converges fastest at that
  horizon rather than the best rate. Works for either group.
- **`probe_transfer.py` / `probe_transfer_readings.py`** — attempt 5
  (`notes/beta_transfer.md`). Evaluates trained probe checkpoints at a coupling
  they never saw; **one β per invocation**, because `probe_common.BETA` is the
  cache key and a second β in one process would be a second meaning for one
  name. Checkpoints are found by **glob, not by a constructed stem** — the
  learning rate is not in the stem and §9.9's campaign carries a run tag — and
  two checkpoints for one (arm, seed) is a refusal with both candidates listed,
  not a silent pick. Models are grouped by `(arch, transport)` so the transport
  is built once per configuration instead of once per checkpoint. Its first
  gate is the **anchor**: evaluated at the source coupling it must reproduce
  each source dump's own `r2`, which is a different route to the same number
  (R² is invariant under a common affine map of prediction and target), so a
  mismatch means the splits, the mask or the targets have drifted since the
  checkpoints were written. The reader refuses to print without it. Z₂ only —
  SU(2)'s second ensemble is a second chain at one fixed β, which is not a
  distribution shift.
- **`z2_init_gate.py`** — the Z₂ arms' initialisation gate, forward-only on
  cached configurations at the production volume: does the matched L-CNN's field
  survive 4 layers at each `conv_init_scale`? It exists because the first value
  was picked on an 8³ box, and the growth is multiplicative in depth with the
  statistic a maximum over sites, so a 54× smaller box is not a proxy. Refuses
  to run if the arms' configured scale is not in the scanned grid — otherwise
  the verdict would pass vacuously. `scripts/z2_dof_table.py` prints the
  matched-parameter table under whichever group is selected.
- **`z2_vortex_preflight.py`** — `probe_preflight.py`'s sibling for the Z₂
  vortex candidate (`notes/where_attention_can_win.md` §9.4), and the gate that
  must be read before any training code exists. Five gates (closure,
  non-degeneracy of the cluster competition, the transport-mask identity of
  §9.3, V2's exact linearity, headroom) and **two** classical ceilings, because
  this task has two: the best linear filter at the architecture's reach and the
  best *local connectivity* algorithm at the same reach. **The reading is the
  masked one** — only ~9% of sites carry a vortex, so an all-sites R² mostly
  scores "is there a vortex here" (0.65 against the masked 0.19). Env:
  `Z2V_BETAS`, `Z2V_N_CONFIGS`, `Z2V_LOCAL_CONFIGS`, `Z2V_REACH`, `Z2V_CHUNK`,
  `Z2V_SMOKE` (production *geometry* off a short fresh chain, under a minute —
  it samples nothing otherwise, the ensembles are `train_z2_glueball.py`'s by
  the identical cache key).
- **`wilson_regression_data.py` / `wilson_regression_common.py` /
  `wilson_regression.py` / `wilson_regression_figure.py` /
  `wilson_regression.sh`** — the Fig. 3 reproduction, L-CNN only
  (`notes/wilson_regression_1p1d.md`). `wilson_regression_common.py` holds the
  β ladder, SM Table V **verbatim in the table's own `L-CB(k, n_in, n_out)`
  notation**, the loader and the two MSE conventions, so nothing can drift
  between the generator, the trainer and the figure; the parameter count is
  checked against the printed table **at build time**, because a silently wider
  network is the one way this reproduction could "succeed" without reproducing
  anything. **The number to quote is `mse_avg`** — Fig. 3 plots one point per
  configuration, both sides averaged over the lattice first, which is their own
  `mse(global_average=True)`; `mse_site` is printed beside it because on an 8×8
  lattice the averaged one is 64× more forgiving, and the dumps keep the
  per-site predictions so either can be recomputed offline. The batch defaults
  to `WR_SIZES=best WR_SEEDS=1`, i.e. **four runs**. `WR_DATA_DIR` writes a
  smoke set somewhere the production set is not. Every entry point calls
  `validate_argv()` first, the same discipline as the probe.
  **`WR_ARCH=gelt`** runs this repo's architecture on the identical problem —
  same ensemble, splits, label, loss and optimiser, **39 569 real DOFs against
  the L-CNN's 39 905** (R=3 ↔ their k=4, 4 blocks ↔ their depth rule
  `⌈log₂ N²⌉`). Not part of the reproduction: it is the fifth GELT-vs-L-CNN
  comparison and the first on a *supervised per-site target with an exact
  answer* (`notes/wilson_regression_1p1d.md` §10, which knowingly violates
  criterion 4 of `where_attention_can_win.md` §6). `scripts/wilson_regression_init_gate.py`
  is its init gate, measured at **this** geometry — the field is flat at 1.0
  from `init_scale` 0.3 to 100, no cliff, unlike the L-CNN's. The open choice
  is the **zero-initialised head**: the gradient reaches the attention only
  after `fc2` moves, which the L-CNN has no analogue of, so `lr` and
  `mlp_zero_init` are bracketed by `WR_PARTS=gelt-sweep` rather than asserted.
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
- **`test_lcnn_reference.py`** — the authors' L-CNN as an arm: the layout
  round-trip bit-exact both ways *and* against their own `shift`; gauge
  invariance of the per-site readout (SU(2) complex128, Z₂ float64, stacked
  multi-level inputs); the `kernel_size = K + 1` convention and the support of
  one layer as integer set arithmetic; the parameter formula against the built
  model; checkpoint exactness; and the constructive reduction of their merged
  kernel to our L-Conv + L-Bilin, with the two bases matched numerically rather
  than by re-implementing their shift ordering.
- **`test_probe_targets.py`** — the M1 probe's premise: `f` is gauge invariant
  (SU(2), Z₂) and non-negative; the shell sums and the ball max against a
  brute-force enumeration of `[-R,R]^D` filtered by the L1 norm (independent of
  `l1_ball_offsets`, so a bug there cannot hide behind itself); **T0 is exactly
  a linear functional of the ball**; T2 recovers a hand-computed enclosing
  radius on a uniform field and is invariant under `f → λf` while T0/T1 are
  degree 1; the periodic-ball guard; and **both architectures reach the whole
  radius-4 target ball** (GELT after 2 of 4 layers, the L-CNN after 3) as
  integer set arithmetic — if that fails, R-C is measuring receptive field.
- **`test_vortex_targets.py`** — the Z₂ vortex candidate's premise: the cube-face
  enumeration checked **by physics** (per-cube vortex parity vanishes on gauge
  configurations, and does not on a hand-broken field); the dual graph against
  the one case known by hand (one flipped link ⇒ four vortex plaquettes in one
  loop); the vectorised labelling against a plain union-find, plus the long-line
  regression that the root hook exists for; non-cubic lattices and no merging
  across the configuration axis; **V2 is exactly the ball count** and V1 is not
  additive; the local arm is bounded by the global one, matches a brute-force
  BFS and **ignores structure outside its ball** — the honesty gate that makes
  it a ceiling; gauge invariance in Z₂ and SU(2); the D ≠ 3 guard.
- **`test_lcnn_exact.py`** — the Letter's L-CB and the Fig. 3 reproduction's
  premises: SM Table V's ten parameter counts; that the vendored class exceeds
  them by exactly the predicted slot; that `ref` and `exact` are the **same
  function** once the added slots are zeroed (1e-12, which tests the ordering of
  their `t_w` axis, not just its length); gauge invariance of the per-site
  readout; translation equivariance **and volume transfer** (a 2×2 tiling into
  16×16 reproduces the tiled output); the training hyper-parameters as SM §VI.A
  prints them; and `W^(1×1)` **constructively** exact in the 12-parameter
  network, so its MSE is a float32 floor and not an approximation error.
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
GLUEBALL_ARCH=lcnn_ref python scripts/train_glueball.py  # ...the authors' own
Z2GATE_ARM=lcnn_ref python scripts/z2_init_gate.py   # their init_w, at production volume
LCNN_DRY_RUN=1 bash scripts/lcnn_shootout.sh # the L-CNN batch: what would run
python scripts/bench_lcnn_reference.py       # our L-CNN block vs lge-cnn's
WR_DRY_RUN=1 bash scripts/wilson_regression.sh   # the PRL Fig. 3 batch: what would run
python scripts/wilson_regression_data.py     # the 1+1D SU(2) datasets (GPU)
WR_TARGET=W22 WR_SIZE=large python scripts/wilson_regression.py
python scripts/wilson_regression_figure.py   # the four panels + the MSE table (offline)
python scripts/operator_decomposition.py     # O = P + r (offline, seconds)
python scripts/input_architecture_curve.py   # A₀ vs input content (offline, seconds)
SFF_NOCACHE=1 python scripts/su2_fair_fight.py     # reproduce ΔA₀ offline
python scripts/z2_beta_scan.py               # Z₂ classical mass vs β
Z2G_R=6 Z2G_N_USE=800 python scripts/train_z2_glueball.py 0.756
python scripts/z2_attention_correlator.py    # the Z₂ attention table
python scripts/su2_attention_correlator.py   # the SU(2) row (~40 min)
PROBE_DRY_RUN=1 bash scripts/probe_batch.sh   # the M1 probe: what would run
python scripts/probe_preflight.py            # §5's gate (offline, seconds)
python scripts/probe_transport_gate.py       # the transport arms' gate (CPU, minutes)
PROBE_ARM=frozen PROBE_TARGET=T2 python scripts/train_probe.py
python scripts/probe_readings.py             # R-A…R-E (offline, seconds)
PROBE_GROUP=z2 PROBE_ARM=gelt PROBE_TARGET=V1 python scripts/train_probe.py
python scripts/z2_vortex_preflight.py        # the Z₂ vortex gate (offline, minutes)
python scripts/z2_init_gate.py               # the Z₂ arms' init gate (forward-only)
python scripts/probe_curves.py --group=z2    # did the runs converge? (offline)
PROBE_GROUP=z2 PROBE_Z2_BETA=0.7520 python scripts/probe_transfer.py  # the anchor gate
PROBE_GROUP=z2 PROBE_Z2_BETA=0.7450 python scripts/probe_transfer.py  # …and a transfer β
python scripts/probe_transfer_readings.py --group=z2   # X-A…X-E (offline, seconds)
PROBE_DRY_RUN=1 PROBE_PARTS=z-gate,z-sweep bash scripts/probe_batch.sh
Z2V_SMOKE=1 python scripts/z2_vortex_preflight.py  # …off a fresh short chain
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
   on a measurement, and its code is deleted (§11 of that note). The lesson is
   the durable part: the target was the output of a classical smoother, and 48
   APE steps with one fitted scalar reproduce it at R² = 0.978 — free, untrained,
   and with more reach than any bounded receptive field. **Before proposing the
   next A/B, run it past the four criteria in §6 and then run §5's pre-flight:
   measure the best classical method at the architecture's own reach.** §8
   proposes the one experiment aimed at the mechanism that *has* been observed
   (boundedness, 0-of-14 vs 3-of-9), which costs a dozen short runs on data
   already on disk. ~~**§9 is attempt 4**~~ — vortex-cluster geometry in 3D Z₂,
   **run and closed 2026-09-20 on a tie** (§9.9): ΔR²(GELT − L-CNN) =
   −0.073 ± 0.051 on V1 at β = 0.7520, six seeds each at bracketed per-arm
   rates, both architectures above the matched-depth classical bar, and the
   only significant separation is **dispersion, 6.2× against GELT** — which
   **attempt 5 has since shown to be a property of the training coupling**
   (`notes/beta_transfer.md` §9.1: 1.5× / 1.3× / 1.0× at the other three).
   ~~**Attempt 5**~~ — β-transfer of these same frozen checkpoints — is
   **run and closed 2026-09-20 on a null** (§9): X-B favours GELT at 3 of 3
   couplings and at p ≥ 0.24, so the claim is not made, and the falsification
   clause hands the programme to §8, whose third stressor attempt 5 has
   incidentally validated. Note that the β = 0.7450 **replication** below
   means a *retrain* there and is still not run — attempt 5 only evaluated
   0.7520-trained nets at that coupling. What is
   *not* run and is the cheap way to finish it: the β = 0.7450 replication,
   V2 for every arm but `lcnn` (W-A's comparison half), and six seeds for
   `frozen` / `gelt_single` / `frozen_single`, which would turn M1 = +0.066
   and M3 = +0.018 from directions into numbers.
7. ~~Run the M1 probe~~ — **done and complete** 2026-09-16, 132 runs, and it is
   the first of three attempts in which M1 was measured to pay
   (`notes/m1_probe.md` §0). Three cheap follow-ups, none needing new data:
   **six seeds per ensemble** (the binding constraint on R-B′ and on both
   dispersion readings, which are counts a sign test cannot resolve at n = 6),
   `signed_bounded` at 1e−1 (its rate is still unbracketed), and
   `PROBE_GATE_CONFIGS=64` on the GPU for the three transports' cost ratio.
   **The claim that needs re-wording elsewhere first**: GELT's robustness is
   attention *and* the shortest-path-averaged transport, not attention alone.

## Things to keep in mind

- **Do not silently broadcast across color axes.** A missed dagger or a wrong
  shortest-path step passes every Z₂ test and fails at the first non-abelian Ω.
- **Do not remove comments unless the code they describe is being deleted.**
- The two attention scripts share one estimator by import, and the two
  Wilson-loop scripts share one problem by construction. Both pairs must be
  changed together or the comparison silently stops being one.
