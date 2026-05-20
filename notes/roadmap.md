# Roadmap — Physics Results with the Gauge-Equivariant Graph Attention Network

This document maps a progressive sequence of results, from sanity checks on
Z₂ to potentially novel research directions. Each phase has:

- **Setup**: lattice, gauge group, β range, what configurations to generate.
- **Tasks**: what the network is trained to do.
- **Observables / comparisons**: what to measure and what to compare against
  (analytic / Monte Carlo / prior ML papers).
- **Pass criteria**: concrete numerical targets.
- **Estimated time** for a master's-level student already comfortable with
  PyTorch and lattice basics.
- **Pitfalls** specific to that phase.

The architecture is the **GELT** of `architecture.md` (gauge-equivariant
graph attention with multiplicative value path). All phases reuse the same
codebase; the gauge group, dimension, and head are the only things that
change.

For Monte Carlo data generation you can either write your own (Z₂ and U(1)
are a few hundred lines; pure SU(N_c) is a standard heat-bath / overrelaxation
pseudocode) or use existing tools: **JuliaQCD** (`Gaugefields.jl`,
`LatticeDiracOperators.jl`) for SLHMC, **lge-cnn** (gitlab.com/openpixi)
for L-CNN-compatible data, **Hipparchus / openQCD / Chroma** for serious
SU(3) production. For a master's thesis, a small custom Python MC for
Z₂/U(1) plus borrowed public SU(2)/SU(3) ensembles is the lightest
combination.

The thesis narrative is built around three claims:
1. The architecture works (Phases 0–2).
2. At matched parameter count it competes with or beats L-CNN, and the
   benefit scales with the physical correlation length (Phase 3, central
   novelty).
3. The architecture generalizes across β and across N_c (Phase 3 cross-β,
   Phase 4).

Phase 5 is outlook only — it lists post-thesis directions and PhD-tier
extensions.

---

## Phase 0 — Implementation validation on Z₂ in 2D
**Time: 2–3 weeks**

### 0.1 Setup
- Gauge group **Z₂**: links `U_{x,μ} ∈ {+1, −1}` represented as 1×1 complex
  matrices to keep the architecture's tensor shapes intact.
- 2D lattice, sizes `L = 4, 8, 16, 32`, periodic.
- Action: `S = −β Σ_p U_p` with plaquette `U_p = U_{x,0} U_{x+0̂,1} U_{x+1̂,0} U_{x,1}`.
- 2D Z₂ has **no phase transition** — this is intentional. The point is
  pure code validation, not physics.
- Generate `10^4` configurations per (L, β) at β ∈ {0.2, 0.4, 0.6, 0.8, 1.0}
  with single-site Metropolis. Cheap.

### 0.2 Tasks
1. **Plaquette regression**: predict `<U_p>` from a configuration. The label
   is just the configuration's plaquette mean; the network should achieve
   ≈ machine-precision MSE because the input *contains* the answer trivially.
2. **Wilson-loop regression**: predict `(1/N_c) Re Tr W^{(2×2)}` per site;
   then the configuration mean.
3. **Gauge-invariance unit tests**: random + adversarial Ω attacks (§7 of
   `architecture.md`).
4. **Identity-at-init / training-stability check**: with all attention
   blocks initialised so that they implement the identity on `W`, verify
   the loss on the trivial plaquette task is at noise floor and that
   small-LR fine-tuning does not destabilise. This catches transformer-init
   pathologies before they masquerade as physics bugs in later phases.

### 0.3 Pass criteria
- Plaquette MSE ≤ 10⁻¹⁰ (single precision).
- 2×2 Wilson loop MSE ≤ 10⁻⁸.
- Random Ω drift < 10⁻⁵; adversarial drift < 10⁻⁵.
- Forward pass on 32² in < 10 ms on a laptop GPU.

### 0.4 Pitfalls
- N_c = 1 makes the trace and the matrix coincide. Make sure your tensor
  shapes still carry the (1, 1) "color" axes — otherwise you'll have to
  refactor when moving to SU(2).
- Z₂ links commute with everything → many of your covariance unit tests
  will pass *even with wrong daggers*. Re-run all tests in Phase 1 with a
  non-abelian-style generic Ω.

---

## Phase 1 — 3D Z₂ gauge: first real physics
**Time: 3–4 weeks**

3D Z₂ pure gauge theory is exactly **dual to the 3D Ising model**
(Wegner 1971): the deconfinement transition of Z₂ gauge maps to the
ferromagnetic transition of Ising. This gives you analytical predictions
to test against, on a problem small enough to fit on a workstation.

### 1.1 Setup
- 3D Z₂ on `L³` lattices, `L = 8, 12, 16, 24, 32`.
- β-range bracketing the critical point `β_c ≈ 0.7613` (dual to Ising
  K_c = 0.2216544...). Sample β ∈ [0.5, 1.0] with 21 values.
- ~5×10³ decorrelated configurations per (L, β); use cluster-update Wolff
  on the dual Ising representation if Metropolis autocorrelation hurts.

### 1.2 Tasks
1. **Order-parameter proxy regression**: predict the **per-configuration
   't Hooft loop operator** — the product of links along a non-contractible
   loop with twisted boundary conditions — as a per-configuration scalar.
   The true vortex free energy `F_twist = −log(Z_twist/Z_periodic)` is a
   ratio of partition functions and has no per-configuration definition; it
   must be computed as a post-hoc MC measurement (two separate simulations)
   at each β and compared against the ensemble mean of the network's
   't Hooft-loop predictions.
2. **Phase classification**: train a binary classifier (confined /
   deconfined). Recover `β_c` from the inflection point of the network's
   confidence as a function of β.
3. **Wilson-loop area-law detection**: for each configuration predict the
   per-configuration Wilson loop value `W^{(R×T)}` (no ensemble average)
   at variable (R, T). After inference, compute `−log ⟨W^{(R×T)}⟩` by
   averaging the network's outputs over the ensemble at each β; fit
   `σ R T + c (R + T) + d` to that averaged curve to extract the **string
   tension σ(β)**. The label per training configuration is the
   per-configuration Wilson loop, not the ensemble average — the average is
   a downstream analysis step. Compare against the high-temperature
   expansion `σ ≈ −log tanh(β)` deep in the confined phase.
4. **Critical exponent extraction**: from the network's response on
   different volumes, do finite-size scaling on the order parameter to
   extract `ν` (correlation length exponent). 3D Ising universality class
   predicts `ν = 0.6299...` — a tight target. *Single exponent only* —
   multi-Δ bootstrap matching is left as PhD-tier outlook (Phase 5).

### 1.3 Observables to plot
- `<U_p>(β)` predicted vs. MC, all volumes overlaid.
- Σ-extracted vs. β with both ML prediction and MC fit.
- `β_c(L)` from network classifier vs. `1/L`; extrapolate to L → ∞.
- Data collapse: `M(β, L) · L^{β_exp/ν}` vs. `(β − β_c) L^{1/ν}`. The
  collapse quality is the test.

### 1.4 Pass criteria
- `β_c` extracted to within 0.5 % of 0.7613.
- `ν` to within 5 % of 0.6299.
- Cross-volume generalization: train on `L = 8, 12`; test on `L = 24, 32`
  with no MSE degradation.

### 1.5 Pitfalls
- 3D Z₂ duality only gives clean Ising mapping in **infinite volume**.
  Finite-size corrections are real and you must include them in any fit.
- Critical slowing down near β_c → ensure your MC autocorrelation is short
  (cluster-update Ising on the dual is much better than gauge-side
  Metropolis).
- Do **not** train data straddling the transition for regression tasks
  (very different physics on the two sides). Train per-phase; treat the
  phase boundary as the test.

---

## Phase 1.5 — Tooling: β-conditioning and a non-equivariant baseline
**Time: 2 weeks**

This is a short bridge phase. It does no new physics, but it produces two
artefacts that every subsequent phase needs:

1. A **β-conditioned GELT** so that one model can fit configurations across
   β (required for Phase 3's cross-β work and the attention-vs-ξ
   diagnostic).
2. A **non-equivariant transformer baseline** at matched parameter count.
   Without this, the comparison story in Phase 3 cannot distinguish
   "transformer wins" from "equivariance wins" — and that is the first
   question any ML referee will ask.

### 1.5.1 β-conditioning
- Extend the dataset format so each configuration carries its β as a scalar
  field (add to the `(X, T, y)` tuple → `(X, T, β, y)`).
- Inject β into the model. Cheapest option: **FiLM-style scale-and-shift**
  on each block's channel features, with β passed through a tiny MLP. This
  is gauge-invariant by construction (acts on scalar channels, not on
  matrix-valued features).
- Validate on the Phase 1 data: a single β-conditioned model should fit
  `<U_p>(β)` across all 21 β values as well as 21 per-β models do, within
  10 %.

### 1.5.2 Non-equivariant transformer baseline
- Build a standard vision-transformer-style baseline with the same depth /
  width / heads / attention range as the GELT, but **with no gauge
  equivariance**: replace adjoint transports with raw shifts, replace the
  matrix-bilinear value path with a standard linear projection. Same
  parameter budget.
- This is *not* `LatticeCNN`, which is convolutional. The point is to
  ablate the equivariance, not the architecture class.
- Validate on Phase 0 (Z₂ 2D): does it learn the plaquette regression at
  all? Expected: yes, slowly. Document the gap.

### 1.5.3 Pass criteria
- β-conditioned GELT matches 21 per-β GELTs within 10 % on Phase 1 plaquette
  regression.
- Non-equivariant baseline trains stably on Phase 0; gauge-invariance
  drift on random Ω is **large** (as expected — this is the point).

### 1.5.4 Why this matters
Every novel comparison in Phase 3 (matched-parameter shootout, attention
range vs ξ, cross-β generalization) is either weaker or impossible without
these two artefacts. Doing them now, on cheap Z₂ data, is much faster than
retrofitting them into Phase 3 under SU(2) compute pressure.

---

## Phase 2 — 4D U(1) compact: continuous abelian gauge
**Time: 3–4 weeks**

This is the smallest continuous gauge group, with a known **first-order
deconfinement transition** at `β_c ≈ 1.0111` (Lautrup-Nauenberg 1980;
later refined). It's the natural step before SU(N).

### 2.1 Setup
- Compact U(1): links `U_{x,μ} = e^{iθ_{x,μ}}`, θ ∈ (−π, π].
- 4D lattice, `L⁴` for L = 6, 8, 10, 12.
- Wilson action `S = −β Σ_p Re U_p`, β ∈ [0.85, 1.15].
- Heat-bath updates (Hattori-Nakajima for compact U(1)). Generate ~5×10³
  configurations per β in each phase; finer sampling near β_c.
- Beware of **metastability** at the first-order transition: thermalize
  separately from cold and hot starts and verify hysteresis.

### 2.2 Tasks
1. **Plaquette regression**: standard sanity check.
2. **Photon-mass / Coulomb-phase test**: in the deconfined phase the
   transverse plaquette correlator decays as `1/r²` (massless photon).
   The network predicts the per-site plaquette field `P_{0,1}(x)` for each
   configuration. The two-point function `⟨P_{0,1}(x) P_{0,1}(0)⟩(r)` is
   then computed by ensemble-averaging the product of network outputs at
   separation r — it is not a per-configuration regression target. Compare
   the resulting correlator against analytic lattice perturbation theory at
   large β.
3. **Monopole density**: in 4D compact U(1) the deconfinement is driven by
   monopole condensation (DeGrand-Toussaint construction). Train on the MC
   monopole density; test whether the network correctly identifies the
   transition without ever being shown the action — purely from
   configurations.
4. **Latent heat at the first-order transition**: from `<U_p>` jump.

### 2.3 Pass criteria
- Recover β_c to within 0.5 %.
- Distinguish the two phases on configurations from the metastable region
  (training on cold-started + hot-started separately) → the network's
  decision boundary should match the phase actually realized in each
  configuration.
- Monopole density prediction MSE within 1× of the L-CNN baseline if/when
  L-CNN is run on the same task. **Note**: L-CNN was not tested on U(1) in
  4D in the original paper — this is already a small novel data point.

### 2.4 Pitfalls
- The first-order transition in finite volume looks smooth — you need the
  L-dependence of the latent heat to argue for first-order.
- DeGrand-Toussaint monopole charges are integer-valued *labels*; use a
  classification head, not regression.

---

## Phase 3 — 4D SU(2): replicate L-CNN, beat it, and learn ξ from attention
**Time: 6–8 weeks (this is the thesis's central phase)**

This is where you reproduce the published L-CNN benchmarks, beat them at
matched parameter count, and produce the two GELT-specific diagnostics
that no convolutional baseline can match: **the optimal attention range
tracks the physical correlation length**, and **one model fits multiple β
through cross-β conditioning**.

### 3.1 Setup
- Pure SU(2), 4D, Wilson action.
- Lattices: `4 × 8³` (training), `4 × 16³, 8 × 16³, 8 × 24³` (test).
- β ∈ {1.5, 2.0, 2.3, 2.5, 2.7} (β_c ≈ 2.4 for finite-T deconfinement on
  N_t = 4; pick most volumes in the confined phase).
- Heat-bath + overrelaxation (Kennedy-Pendleton). 10³ thermalized
  configurations per (V, β); train/val/test = 80/10/10.

### 3.2 Tasks — replication of L-CNN paper
1. Wilson-loop regression: `W^{(1×2)}, W^{(2×2)}, W^{(4×4)}` (1+1D from
   the paper, but extend here to 4D).
2. **Topological-charge density** (plaquette discretization, Eq. 13 of
   review). Train at `4 × 8³`, test up to `8 × 16³`.
3. **Wilson-flowed inference**: take the `4 × 8³`-trained network and
   apply it to Wilson-flowed configurations on `8 × 24³` (Δτ = 0.005, 200
   flow steps with cooling). Recover near-integer global topological
   charge `Q_P(τ)`. Reproduce Fig. 5 of the L-CNN Letter.

### 3.3 Pass criteria — replication
- `W^{(2×2)}` MSE ≤ 1.1 × L-CNN-Medium reported (≈ 1.1 × 10⁻⁸).
- Topological-charge MSE ≤ 1.1 × L-CNN-Small reported (≈ 3 × 10⁻⁹).
- Volume generalization: MSE flat ±20 % across volumes.

### 3.4 Tasks — central novelty: GELT vs L-CNN
1. **Matched-parameter shootout**. For each Wilson loop task, train an
   L-CNN, a GELT, and the **non-equivariant transformer baseline from
   Phase 1.5** with the **same number of trainable parameters** (use the
   L-CNN sizes: 35 / 1305 / 13521 / 39905). Plot MSE vs. parameters. The
   triple comparison cleanly separates "transformer wins" from
   "equivariance wins."
2. **Long-range correlation regime**. Push β closer to the bulk-transition
   crossover (β ≈ 2.3) where the spatial correlation length grows.
   Hypothesis: at fixed parameter count, GELT pulls ahead of L-CNN as
   ξ/R_kernel increases, because attention with range R can see structure
   beyond the L-Conv receptive field. Quantify with `ΔMSE(ξ/R)`.
3. **Receptive-field sweep**. Train GELT at attention range R ∈ {1, 2, 3, 4};
   show how the optimal R correlates with the spatial correlation length
   measured on the same configurations. This is interpretable: the network
   is *learning to look as far as the physics demands*.
4. **Point-group equivariance ablation.** Train two GELT variants:
   (a) translation-equivariant only (default), (b) translation +
   hypercubic point group via tied `b_h(μ, k)` across μ-orbits and ±k
   (cf. `architecture.md` §12). Compare MSE at matched parameter count on
   rotation-symmetric targets (Wilson loops, topological charge). Expected:
   tying gives a free win; the size of the gap measures how much capacity
   the unconstrained network was burning on memorizing the point-group
   action. A clean, small, defensible result.

### 3.5 Tasks — attention range as a learned correlation length
*(promoted from the original Phase 6.2; this is the single result that
most cleanly says "this is what attention buys you over convolution.")*

**Question.** When trained on configurations at varying β, do the learned
attention scores `α_{x→y}` peak at offsets matching the physical
correlation length ξ(β)?

**Plan.**
- Use the β-conditioned GELT from Phase 1.5.1, trained jointly on the
  full β ∈ {1.5, 2.0, 2.3, 2.5, 2.7} ensemble for Wilson-loop regression.
- Diagnostic: for each β, average `|α_{x→y}|` over (x, head) at fixed
  offset `(μ, k)`. Plot vs. `k` to extract the attention-decay length
  `ℓ_att(β)`.
- Compare `ℓ_att(β)` against the MC-measured spatial correlation length
  `ξ(β)` from a two-point function on the same configurations.

**Pass criteria.**
- `ℓ_att(β)` and `ξ(β)` are monotonically related on β ∈ [1.5, 2.7].
- The ratio `ℓ_att / ξ` is constant within 30 % across β — i.e. the
  network is tracking, not memorising.
- Visualised as a single plot: this is the thesis-defence headline figure.

**Risk.** Low. The decay of α is interpretable by construction; the only
failure mode is that α is too sharply peaked at k = 1 (no useful
information). Mitigation: use larger R and a softer attention temperature
at init.

### 3.6 Tasks — cross-β generalization
*(promoted from the original Phase 6.1.)*

**Question.** Train the β-conditioned GELT on β values away from a
crossover region; test on held-out β within it. Does the model interpolate
physics, or memorise the training distribution?

**Plan.**
- Train on β ∈ {1.5, 2.0, 2.5, 2.7}; evaluate on β = 2.3 (held out).
- Compare per-β plaquette and Wilson-loop predictions against MC.
- Repeat the holdout pattern across the β grid; the worst-case held-out
  β tells you how much the model is interpolating vs extrapolating.

**Pass criteria.** Held-out-β MSE within 2× of trained-β MSE on the same
observables. Larger gaps are still publishable as a *measurement* of where
the network's prior fails.

**Why interesting.** Standard ML transferability tests use volume; very
few have cleanly tested across β at fixed N_c, and none for an equivariant
transformer. A positive result argues the network has learned the *gauge
action structure* rather than the configuration distribution. A partial
result still maps out the failure mode.

### 3.7 Pass criteria — overall
- At ≤ 10⁴ parameters and ξ/a ≥ 3, GELT beats L-CNN by ≥ 2× on Wilson-loop
  MSE. (If it doesn't, that itself is an interesting null result and you
  should publish it.)
- Optimal R tracks ξ/a within a factor of 2 across β ∈ [2.0, 2.5].
- `ℓ_att / ξ` constant within 30 % across β (§3.5).
- Held-out-β MSE within 2× of trained-β MSE (§3.6).

### 3.8 Pitfalls
- Topological charge labels are tiny (~10⁻⁴). Apply the L-CNN scale trick
  (multiply labels by 100; divide back at inference). Forgetting this is
  the most common L-CNN reproduction failure.
- The network has **no GAP**. Predict per-site, sum at the end if you need
  global Q. Do not collapse to a single output before the per-site Trace.
- For the matched-parameter comparison, *count complex parameters as 2
  reals*. Otherwise you'll silently double the GELT capacity.
- The §3.5 diagnostic requires the β-conditioned model from Phase 1.5;
  do not skip Phase 1.5.

---

## Phase 4 — SU(3) χ_t: continuum-limit check on borrowed ensembles
**Time: 3–4 weeks**

The same architecture with N_c = 3, on public pure-gauge configurations.
Scope is deliberately narrow: **only topological susceptibility** `χ_t`,
extending the Phase 3 topological-charge head to N_c = 3 and to multiple
lattice spacings. The point is to demonstrate **N_c-agnostic transferability
and continuum scaling**, not to do a full QCD spectroscopy program.

### 4.1 Setup
- SU(3), 4D, Wilson action, β ∈ {5.7, 6.0, 6.3} (lattice spacings
  a ≈ 0.17, 0.094, 0.06 fm).
- Lattices `8⁴, 12⁴, 16⁴`.
- Configurations: borrow from a public ensemble (ILDG / Gauge Connection /
  community releases) instead of generating from scratch — saves weeks.
- Apply gradient flow at inference time to suppress UV noise on `Q`
  (reuses Phase 3 Wilson-flow inference code).

### 4.2 Tasks
1. **Topological-charge density at N_c = 3**: extend the Phase 3 head;
   verify the gauge-invariance unit tests still pass with the worst-case-Ω
   adversarial search at machine ε in float64.
2. **Topological susceptibility** `χ_t = <Q²>/V`. Compare with ≈ (180 MeV)⁴
   in pure SU(3) (Del Debbio et al.).
3. **Continuum-limit scaling**. Plot `χ_t^{1/4}` vs. `a²`; verify standard
   `O(a²)` scaling and quote a continuum-extrapolated number with
   statistical and systematic errors.

### 4.3 Pass criteria
- Worst-case-Ω drift at machine ε in float64 on N_c = 3.
- `χ_t^{1/4}` within 10 % of published value on each ensemble.
- Continuum extrapolation linear in a² (no a-dependence in residuals
  beyond statistics).

### 4.4 Pitfalls
- N_c = 3 enlarges every per-site matrix from 2² = 4 to 3² = 9 complex
  numbers; memory and time go up by ≥ 5×. Plan ahead for batch sizes.
- Borrowed ensembles come with their own conventions (β scheme, action,
  flow definition). Document which you use before publishing any number.
- String tension, glueball spectroscopy, and full QCD scale-setting are
  **explicitly out of scope** for the master's. Each is a 4–6 week
  specialist project on its own and would dilute the thesis.

---

## Phase 5 — Outlook (post-thesis directions)
**Time: open-ended; not part of the master's arc**

The master's thesis stops at Phase 4. The directions below are PhD-tier
extensions; each is listed at the level of a paragraph, with timelines
deliberately omitted to avoid the implication that they're scoped for the
master's. They are organised roughly by risk.

### 5.1 Dynamical fermions and SLHMC

Train the network as a **surrogate effective fermion action** so that HMC
with a heavy-mass proposal accepts as if it were running at light mass
(Nagai-Tomiya 2103.11965, Nagai-Ohno-Tomiya 2501.16955 / CASK). This is
*the* direct head-to-head against CASK on their home turf.

The infrastructure cost is the load-bearing question: porting the
architecture to Julia (`Zygote.jl` + `LatticeDiracOperators.jl`) takes 3–6
weeks for a first port, not 1–2. A lighter version that stays inside the
PyTorch codebase is **Schwinger model (2D U(1) + staggered fermions)**:
same physics narrative (surrogate-action regression, SLHMC acceptance,
autocorrelation comparison) with no Julia port and no SU(2) overhead.

The result that would make this PhD-paper-worthy: GELT autocorrelation
≤ 0.7× CASK on a matched setup at matched parameter count.

### 5.2 Imaginary-θ topological susceptibility

Generate SU(N) ensembles at θ_I ∈ {0, 2, 4, 6} via reweighting or direct
simulation (`S → S + θ_I Q`); train the Phase 4 χ_t head per θ_I; fit
`<Q²>(θ_I) / V = χ_t (1 − b_2 θ_I² + …)` and analytically continue to real
θ. Builds directly on Phase 4 infrastructure; sign-problem-free.

The result that would make this PhD-paper-worthy: a measurement of `b_2`
at competitive statistics vs. plaquette discretization + Wilson flow,
ideally at finer a than the published ensembles cover.

### 5.3 PhD-tier directions (kept here as outlook only)

- **Trivializing / normalizing flows with GELT.** Stack of GELT + L-Exp
  as a flow `U → V_θ(U)` trained to target the Wilson action; diagnostic
  is topological tunneling rate at fine β. Compare with Albergo, Kanwar,
  Boyda et al. (2003.06413, 2008.05456, 2305.02402); Abbott et al.
  (2401.10874). SU(N) flows are notoriously hard past `2⁴`; a *partial*
  result (works at coarse β, fails at fine β) is still publishable.
- **Topological-sector sampling.** Train a GELT to predict the gradient
  of `|Q − Q_target|` w.r.t. U; use as a Langevin bias inside Metropolis
  with proper accept/reject. Detailed balance under a learned proposal is
  fragile (Bayer et al. 2306.04388).
- **Sign-problem alleviation via contour deformation.** Lefschetz-thimble
  parametrization by GELT for 1+1D U(1) with θ-term. Very high risk —
  most ML attempts haven't outperformed analytic constructions.
- **Continuous-time limit / learned Wilson flow.** Identify the GELT
  stack with a learned gauge-equivariant ODE on configuration space;
  benchmark against Wilson flow at scale-setting (`t_0`).
- **Variational ground-state wavefunctional.** `Ψ_θ(U) = exp(network(U))`
  with a Trace-MLP head; lattice analogue of neural quantum states
  (Carleo-Troyer 2017) for 2+1D Z₂ Hamiltonian.
- **Non-hypercubic lattices.** Extend the geometry to honeycomb / Kagome
  for condensed-matter applications (Z₂ spin liquids, Kitaev-like models).
  Implementation-heavy; scientifically clean.

These were each given a full sub-section with timelines and risk
discussions in the previous version of this roadmap; the consensus after
review was that listing them at that depth in a master's roadmap is
misleading. They live here for the thesis's outlook section and as a
research-question menu for a PhD application.

---

## Suggested thesis arc

A defensible master's-thesis arc with strong-but-realistic novelty:

```
months 1–2:  Phase 0 + Phase 1   (Z₂ implementation, 3D Z₂ critical point)
month  3:    Phase 1.5            (β-conditioning, non-equivariant baseline)
months 4–5:  Phase 2               (4D U(1), first continuous group)
months 6–8:  Phase 3               (SU(2): replication + §3.4 shootout
                                    + §3.5 attention-as-ξ + §3.6 cross-β)
month  9:    Phase 4               (SU(3) χ_t + continuum scaling)
months 10–12: writing + defense; Phase 5.1 / 5.2 as outlook section
```

This produces:
- Independent reimplementation of L-CNN results (defensive).
- A controlled three-way comparison: equivariant transformer (GELT) vs
  equivariant CNN (L-CNN) vs non-equivariant transformer (Phase 1.5),
  at matched parameter count. This is the comparison the field has been
  missing.
- Two GELT-specific results that no convolutional architecture can produce:
  attention range tracks correlation length (§3.5), and one β-conditioned
  model fits multiple β within statistical error (§3.6).
- An N_c-agnostic continuum-scaling check (Phase 4) demonstrating the
  architecture transfers from SU(2) to SU(3).
- A secondary critical-exponent result on 3D Z₂ (Phase 1).

---

## What to publish from the thesis

A single paper from the thesis would most naturally contain:

- **GELT architecture description** (1 figure, 2 pages of equations).
- **L-CNN replication on 1+1D Wilson loops + 3+1D topological charge**
  (1 table, 1 figure).
- **Matched-parameter three-way comparison (GELT vs L-CNN vs
  non-equivariant transformer)** at the L-CNN parameter ladder
  (1 figure — this is the central scaling claim).
- **Attention range vs correlation length** (1 figure — this is the
  thesis's defining interpretability result).
- **Cross-β generalization** (1 figure or short table).
- **Secondary check on a different gauge group**: either Phase 1 (3D Z₂
  critical exponents from ML) or Phase 4 (SU(3) χ_t continuum scaling).

That's a complete, defensible arXiv submission and a clean MSc thesis
structure. Phase 5 directions are then "outlook" pointing at the PhD.

---

## Reference reading order (per phase)

| Phase | First reads                                                          |
|-------|----------------------------------------------------------------------|
| 0     | Wegner 1971; Creutz *Quarks, Gluons, and Lattices* ch. 1–3           |
| 1     | Wegner 1971; Pelissetto-Vicari *Phys. Rep.* 368, 549 (Ising critical exponents) |
| 1.5   | Perez et al. 1709.07871 (FiLM); Vaswani et al. 1706.03762 (vanilla transformer) |
| 2     | Lautrup-Nauenberg 1980; DeGrand-Toussaint 1980 (monopoles)           |
| 3     | Favoni et al. 2012.12901; Müller et al. 2112.11239                   |
| 4     | Necco-Sommer hep-lat/0108008; Del Debbio et al. hep-lat/0407028; Lüscher 1006.4518 (Wilson flow) |
| 5.1   | Nagai-Tomiya 2103.11965; Nagai-Ohno-Tomiya 2501.16955                |
| 5.2   | Bonati et al. 1512.06746 (imaginary-θ); Panagopoulos–Vicari 1109.6815 |
| 5.3   | Albergo et al. 2003.06413, 2305.02402; Abbott et al. 2401.10874; Bayer et al. 2306.04388; Cristoforetti et al. 1205.3996; Carleo-Troyer 2017; Kitaev cond-mat/0506438 |
