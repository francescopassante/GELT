# Flow-free topological charge density — design record

**Status (2026-09-14): WP0, WP1 and WP2 built, and WP2's pre-flight has now
been run (§12.4–§12.5). Gate 0 relocated the primary rung, from t/a² = 2 to 4,
and cost the β = 2.3 row its lattice size. Everything from
WP3 on is still a proposal.** Nothing below §12 is a result about the *physics*; §12 is the build
log for the two work packages that landed, both offline, both CPU-seconds, both
gated by tests. The theory, the physics background and the argument for the task
are in `reports/topology/flow_free_topology.tex` (13 pp, compiles); this note is
the build order, the gates and the readings fixed in advance.

The question this answers: **the L-CNN shootout tied on 0⁺⁺ — is there a task
where GELT's attention is expected to win, and can it be run here?**

---

## 1. Why 0⁺⁺ tied (the short version)

Three properties of that task, each of which independently neutralises an
input-dependent kernel:

1. **Zero-momentum projected loss.** The Rayleigh quotient is built from
   `Ō(t) = Σ_x O(t,x)`. Site-averaging destroys site-to-site variation, which is
   the one thing attention has. `notes/attention_as_operator.md` §1 made this
   argument about ℓ_att; it applies verbatim to the operator.
2. **The optimum is linear in loop shapes.** Only 2.5% of ‖O_GELT‖² lies outside
   the 21-operator `full` span, and ~80% of that residual is rectangular loops
   (`notes/fable5.1_10-09_audit.md` §8.3). That *is* L-Conv.
3. **One β, one scale, the most symmetric channel.** A single fixed kernel is
   optimal by construction.

Parity was structural, and `reports/curve` §7 pre-registered it.

**One correction, measured 2026-09-14 and not in any other note.** Parity extends
to the noise-to-signal ratio `δC(Δ)/C(Δ)`, blocked jackknife, same 400 configs,
ens1: GELT 0.0322 / 0.0396 / 0.0720 / 0.1695 at Δ = 1,2,4,6 against L-CNN
0.0322 / 0.0399 / 0.0728 / 0.1676. Indistinguishable. So §9.2's top-1-share
finding is a **failure-rate** property across training runs (3 of 9 L-CNN arms
unusable, 0 of 14 GELT), **not** an error-bar advantage. Do not build a proposal
on "attention gives smaller errors".

## 2. The exact architectural difference

The loose version — "L-CNN is a polynomial, GELT is not" — is **wrong**. `LAct`
is `g(Re Tr W_c / nc) · W_c`, i.e. data-dependent gating, and L-Conv mixes
channels *before* it, so the gate argument is a learned linear combination of
transported channels: an L-CNN can gate on `ReLU(Re Tr[A − B])`, a difference
test between two scales held in different channels. GELT uses the *same* L-Act.

> **The only architectural difference is whether the weights over *offsets* are
> input-dependent.** Gating, bilinearity, receptive field, loop degree, inputs,
> loss, splits and estimator are matched.

The surviving mechanism is **normalisation**: softmax depends only on score
*differences* and is invariant under a common shift, so it is a scale-free ratio
across the neighbourhood, recomputed per site per configuration. `ReLU(Re Tr[A−B])`
thresholds an *absolute* magnitude and must be calibrated to an amplitude the
fixed weights cannot adapt.

**Selection rule.** Attention can separate where the optimal neighbourhood
weighting is *relative* rather than absolute. Three properties: per-site target;
sparse structures at configuration-dependent positions *and scales*;
amplitude-invariant identity.

## 3. The task

Learn `q̂(x)[U] ≈ q_t(x) ≡ q_clover[F_t(U)](x)` — the **gradient-flowed**
topological charge density — from **unflowed** thermalised SU(2) links, per site,
4D. Well posed because the flow's Green function is exponentially localised on
`√(8t)`; the constraint `√(8t)/a ≲ R_net` (Manhattan 8) is not a nuisance but a
measurement of how much flowed topology a bounded neighbourhood determines.

Scores on all three properties: per-site (not summed); instantons are sparse
lumps at configuration-dependent positions with a size distribution
(`n(ρ) ∝ ρ^{7/3}` at one loop for SU(2)); and the central operation — genuine
instanton vs lattice dislocation — is a **shape** test, scale-free, i.e. a ratio
of field strength at radius a, 2a, 3a, not a threshold on magnitude.

## 4. Prior art — the gap is real, the motivation is narrower than it looks

| work | input → target | status |
|---|---|---|
| Favoni et al., PRL 128 (2022) 032003 | flowed links → `q_plaq` of **the same config** | solved; local degree-2 algebraic identity, one L-Bilin layer spans it. Their Fig. 5 runs the model *along* a flow trajectory — the integers come from the flow, not the net. |
| Favoni et al., arXiv:2212.00832 | links → links, neural-ODE Wilson flow | toy models; a different problem |
| Matsumoto et al., PTEP 2021 023D01 (SU(3)) | q-density at **small** flow time t/a² ≤ 0.3 → integer Q at large t | >99%; scalar target, partial flow, non-equivariant nets |
| **this** | **unflowed links (τ=0) → flowed density field q_t(x)** | **open** |

**Consequence for the write-up:** "save the flow cost" is partly taken by the
PTEP result. Lead with the *density field* (instanton content, ⟨q(x)q(0)⟩, the
size distribution) and with the locality question, not with a cost argument.

## 5. A defect found in the audit — fix before anything else

`topological_charge_density` is the **naive, clover-free** plaquette density: all
six plaquettes based at the corner `x`, so it is not reflection-symmetric about
`x` and **is not parity-odd**, not even summed.

Verified directly. Build the reflection `x₁ → (−x₁) mod L` with the correct link
relabelling (`U'_ν(y) = U_ν(Py)` for ν≠1, `U'_1(y) = U_1(P(y+1̂))†`). It is an
exact symmetry of the Wilson action — `15478.81704264 → 15478.81704264` on a 6⁴
SU(2) config, all printed digits. Under the same map `Q: −1.3161 → +2.9888`: the
sign flips, the magnitude does not.

⇒ `definition="clover"` is **required**, and the test is
*"the clover charge is parity-odd to machine precision and the plaquette one is not"*.

## 6. What already exists (verified by running it at D=4, SU(2), L=6, complex128)

- `plaquette_tensor` → `(B,6,Λ,2,2)`; `topological_charge_density` → `(B,Λ)`;
  `build_transport_average` → `(B,40,Λ,2,2)` at R=2; `build_axis_transports` →
  `(B,4,2,Λ,2,2)`.
- Both models run and are gauge invariant to machine precision: **GELT 0.0,
  L-CNN 1.4e-14**, target itself 1.5e-16.
- **`LCNN` already takes `reduction="none"`** and returns `(B, *Λ)` — no per-site
  head to write for the baseline.
- `staple_sum(..., batched=True)` is exactly the flow drift, already vectorised.
- `data.build_plaquette_datasets` takes `target(configs, group) -> Tensor`, so the
  flowed density binds with `functools.partial`; `R` makes it emit `(X, T, y)`.
- `integrated_autocorrelation_time` sets `n_skip`.

`l1_ball_offsets(4,2) = 40` against the glueball's 24 → the transport stage
(already 62.8% of a GELT step) grows 1.67× per site.
**`d_qkv ≥ 2D = 8`** is a hard constraint in D=4 (caveat 2) — assert it.

## 7. Build order

| WP | what | gate |
|---|---|---|
| **0 ✅** | `topological_charge_density(..., definition="clover")` + the parity test | parity-odd to 1e-14 in complex128 — **met, 1.4e-17** (§12.1) |
| **1 ✅** | `gelt/flow.py`: `wilson_flow`, closed-form SU(2) exp + `[·]_TA`, batched; Lüscher RK3 with Euler as cross-check; `flow_trajectory` emits the 5-rung ladder in one pass | flow is gauge covariant, stays on the group, RK3 ≡ Euler(ε→0) — **all met** (§12.2) |
| **2 ⚙️** | `scripts/measure_topology.py`: ensembles, **Q(t) plateau check**, `τ_int(Q)`, classical arms (identity + best linear filter) | **Gate 0**: `t/a²=2` on a plateau at every β, Q near integer. If not at β=2.3, move the rung and record it — **written and smoke-verified, gate not yet read** (§12.3) |
| **3** | `scripts/train_topology.py`: per-site MSE, one `_build_model()`, `TOPO_ARCH=lcnn` switches architecture and nothing else | matched-budget test (real DOFs within 15%) passes in D=4 |
| **4** | trainings: 3 per-β + 1 mixed-β, × 2 architectures, + untrained controls (3 seeds, **median** fixed in advance) | — |
| **5** | `scripts/fit_topology.py`: M1–M6 offline, correlated `--vs` difference inside every jackknife sample | — |

## 8. Design decisions that are load-bearing

- **β ∈ {2.3, 2.4, 2.5} at L ∈ {8, 12, 16}** so the physical volume is ~constant
  (≈1.3 fm)⁴ while `a` halves — both architectures are translation-equivariant, so
  training at mixed sizes is legitimate (the L-CNN paper trains on 4×8³ and
  evaluates on 8×24³). **Re-derive the scale in-repo** from
  `rectangular_wilson_loop`; do not import `a√σ` from the literature into a
  physics claim.
- **Fixed `t/a²`, not fixed physical `t`.** This holds `r_sm/a` — and therefore the
  receptive-field requirement — constant across β, so a mixed-β model's
  degradation is attributable to *scale adaptivity* and not to a capacity wall
  that bites at one β only. Fixing `t` physically confounds the two.
- **Flow-time ladder `t/a² ∈ {0.5,1,2,4,8}`** (`r_sm/a = 2.0…8.0`), one trajectory,
  five targets. The top rung sits at the receptive-field boundary on purpose:
  it is the built-in capacity control.
- **The best-linear-filter arm is the most important control.** The
  least-squares-optimal Manhattan-8 isotropic filter on `q_clover[U]`, fitted on
  train. The gap between it and the best network *is* the nonlinear content of the
  flow. If it is not clearly beaten, **stop** — the task is a convolution.
- Splits contiguous and chain-ordered, 800/200/200, test at the far end of the
  chain (the glueball protocol; §9.1 of the shootout note is why it matters).

## 9. Readings, fixed in advance

- **R1 (primary, architecture).** Δ(1−R²), mixed-β GELT vs mixed-β L-CNN,
  correlated jackknife. GELT wins at >2σ; parity within 1σ.
- **R2 (control, adaptivity).** Same at fixed β. *Prediction: parity, or a
  clearly smaller margin than R1.* If single-β already separates, the effect is
  not scale adaptivity — re-stratify by measured ρ and say the control failed.
- **R3 (control, capacity).** Accuracy across the flow-time ladder. Both must
  collapse as `r_sm/a → 8`. A later collapse for GELT is receptive-field
  efficiency, not adaptivity, and must be reported as such.
- **R4 (physics, architecture-independent).** Does *either* net reproduce
  χ_top and the Q-histogram from τ=0 within the flowed reference's error? **This
  is the deliverable that does not depend on who wins.**
- **R5 (nulls).** Untrained arms must fail; identity and best-linear-filter must
  be clearly beaten; top-1 configuration share under the §9.2 gate.
- **M4 (stratified) is the reading to look at first** — the dislocation-rich tail
  (top percentile of |q_clover[U]|, top decile of configuration roughness) is
  where the mechanism predicts the separation.

## 10. Odds, honestly

**P(clear GELT win on R1) ≈ 50–55%**, revised down from 65% by this audit:

1. the surviving mechanism is one axis (§2), not four;
2. the raw→flowed map is *mostly* a smoothing, i.e. mostly a convolution — the
   adaptive part is a correction whose size WP2's linear-filter arm measures;
3. the base rate in this repo for "attention will help here" is currently 0 for 1.

**But the risk profile is better than that number.** Two separable deliverables:
*(a)* does a flow-free learned topological density work at all, and how much of
flowed topology is local — novel whoever wins, ~90%; *(b)* does input-dependent
offset weighting help — the coin flip, with a clean pre-registered A/B that makes
a null publishable. A second null after the shootout is genuinely informative:
it would say the mechanism does not pay on *local regression* either, which
sharply narrows where it could ever pay.

**So: frame it as a study of the flow-free topological density with a controlled
architecture A/B inside it, not as "GELT beats L-CNN at topology".**

## 11. Cost

~4–5 V100 nights + ~1 build week: ensembles ~1.5 nights, flow targets 1–2 h,
GELT trainings ~2 nights, L-CNN ~0.5 night, the rest offline.

Two known risks priced in `reports/topology/flow_free_topology.tex` §8:
**topological freezing at β = 2.5** (measure `τ_int(Q)`, not `τ_int(plaquette)`;
if it bites, the per-site readings M1/M4 survive and the aggregate M2/M3 are
quoted at β = 2.3, 2.4 only), and **GELT's transport cost in 4D** — for which
`notes/performance_audit.md` §5.1 (the adjoint SO(3) representation, predicted
1.8× end to end) is a better bet here than it was on the glueball task.


---

## 12. Build log

### 12.1 WP0 — the clover density (`gelt/lattice.py`, 2026-09-14)

`topological_charge_density(U, group, plaquettes=None, definition="clover")`,
with `"plaquette"` keeping the old naive density. **Clover is the default**:
nothing in the repository called the function outside `tests/test_lattice.py`, so
there was no back-compat surface to protect and no published number to revisit.

`F_{μν}(x) = (C_{μν}(x) − C_{μν}†(x))/8i`, with `C` the sum of the four leaves in
the (μ,ν) plane around `x`. **The basepoint is the whole subtlety.** The leaves
are cyclic rotations of the plaquettes at `x`, `x−μ̂`, `x−ν̂`, `x−μ̂−ν̂`, and a
cyclic rotation is a similarity transform by a link — invisible under a trace,
*not* invisible here, because `q_x` contracts `Tr[F_{μν}F_{ρσ}]` across two
different planes at the same site. So all four leaves are written as closed loops
starting and ending at `x`, and the `plaquettes=` shortcut cannot serve the
clover definition at all: passing it now raises rather than silently evaluating
`F` in the wrong colour frame.

Measured on a 6⁴ SU(2) config (complex128, seed 0), under the exact reflection
`x₁ → (−x₁) mod L`:

| | Q | Q′ | site-wise `max|q′(y) + q(Py)|` |
|---|---|---|---|
| plaquette | −1.316050 | **+2.988769** | 1.98e−01 |
| clover | −0.120544 | **+0.120544** | **1.39e−17** |

with the Wilson action preserved to all printed digits
(18428.51387259 → 18428.51387259, β = 2.4) — and the plaquette row reproduces
the audit's `−1.3161 → +2.9888` exactly.

**One trap worth recording, because the first attempt hit it.** The link along
the reflected axis obeys `U′_1(y) = U_1(P(y+1̂))†` and `P(y+1̂) = Py − 1̂`, so the
shift is *down* by one and it happens **before** the reflection. Shifting up, or
reflecting first, gives a map that still looks like a reflection and still
produces a finite answer — but it does not preserve the Wilson action
(off by 79 out of 18428 on the config above). That is why
`test_reflection_preserves_wilson_action` runs as its own test and not as an
assertion buried inside the parity test: the reflection map is the instrument,
and an instrument this easy to get plausibly wrong has to be calibrated
separately.

### 12.2 WP1 — the Wilson flow (`gelt/flow.py`, 2026-09-14)

`U_μ(x) ← exp(ε Z_μ(x)) U_μ(x)`, `Z_μ = −[U_μ(x) A_μ(x)]_TA`, the drift read
straight off `sampler.staple_sum(..., batched=True)` — so the flow inherits the
batching that made APE smearing 7.4× faster, with no new hot-path code.

**The normalisation is the load-bearing detail**, because `r_sm = √(8t)` and
therefore the receptive-field argument `√(8t)/a ≲ R` depend on it, and a stray β
or factor of two would be invisible in every other check (the flow would still
smooth, still be covariant, still stay on the group — it would just be at the
wrong time). Two independent confirmations:

1. *Analytic.* For `U_μ = exp(iθ_μ)` in the small-field limit the drift reduces
   to `θ̇_μ = Δ²θ_μ` plus a gauge term, i.e. the lattice heat equation with unit
   coefficient, whose 4D kernel has `⟨x²⟩ = 8t`. No β survives: it cancels
   against the `g₀²` in the flow equation.
2. *Measured.* A transverse plane wave (σ₃ phase on the direction-0 links,
   depending on `x₁` only, so its lattice divergence vanishes) must decay as
   `exp(−k̂²t)` with `k̂² = 4sin²(k/2)`. At L = 8, t = 1:

   | mode | measured | `exp(−k̂²t)` |
   |---|---|---|
   | n = 1 | 0.556668 | 0.556668 |
   | n = 2 | 0.135335 | 0.135335 |

   This is `test_linearised_flow_is_the_unit_heat_kernel`, and it is the test to
   look at if a flow time ever stops meaning what it should.

Integrators, on a 4⁴ SU(2) config flowed to t = 0.4, error against RK3 at
ε = 0.0025:

| ε | Euler | ratio | RK3 | ratio |
|---|---|---|---|---|
| 0.1 | 1.95e−01 | | 3.51e−03 | |
| 0.05 | 1.12e−01 | 1.74 | 4.11e−04 | 8.53 |
| 0.025 | 6.02e−02 | 1.86 | 5.65e−05 | 7.27 |

First order and third order respectively, as intended; at *matched cost*
(RK3 is three drift evaluations per step) RK3 at ε = 0.06 beats Euler at
ε = 0.02 by more than 10×. RK3 is the default.

Two API decisions worth their line:

- **The step is `t / ceil(t/eps)`, not `eps` with a ragged remainder.** A
  trajectory lands exactly on the requested `t`. The ladder is defined at
  specific flow times and the rungs have to be comparable across β, so "close to
  t" is not good enough — `t = 0.15` with `eps = 0.1` runs two steps of 0.075 and
  is bit-identical to asking for `eps = 0.075`.
- **Z₂ raises.** `[M]_TA` of a real 1×1 matrix is identically zero, so the flow
  on Z₂ is the identity map. Silently returning the input would be the worst
  possible failure mode for a smoother; `wilson_flow` refuses instead.

Gauge covariance measures 1.7e−15, the group is preserved to 1e−13 over a full
trajectory with no reprojection (`reunitarize_every` exists for long float32
runs and is off by default), and the action falls monotonically. 21 tests in
`tests/test_flow.py`, 5.4 s on CPU; the whole suite is 144 tests in 8 s.

### 12.3 WP2 — the classical half (`scripts/measure_topology.py`, 2026-09-14)

**Written and smoke-verified; not run.** Every number this script can produce is
still unmeasured — the gate verdict below is the *mechanism*, not a result.

Five phases, each caching its artifact and skipped if present:

| phase | what | artifact |
|---|---|---|
| `selftest` | the linear arm's estimator against the naive definition | — |
| `chain` | one chain at `n_skip = 1`, τ_int of the **flowed Q** and of the plaquette | `results/topology/chain_*.pt` |
| `gate0` | **the gate**: `Q(t)` to `t/a² = 16`, plateau + integer proximity, plus a step-size check | `gate0_*.{pt,png}` |
| `ensemble` | the production ensembles | `datasets/topo_ens_*.pt` |
| `targets` | `q_clov` at `t = 0` and the five-rung ladder, one trajectory per config | `datasets/topo_targets_*.pt` |
| `arms` | identity, identity×Z, best linear filter (free and isotropic) | `arms_*.pt` |

Default is `chain,gate0` — the pre-flight, which has to be read before a night
of sampling is spent. `TOPO_SMOKE=1` runs all five at L=4 in ~30 s on a CPU.

**Three decisions worth recording.**

*The gate can fail in three ways, not two.* A small volume carries no topology,
every `Q(t)` sits near zero, and "Q is near an integer" then passes trivially —
the smoke run at L=4 does exactly this. So the spread of Q across configurations
is part of the gate, and a third verdict **DEGENERATE** exists for it. Without
that, the cheapest possible run would have looked like the cleanest pass.

*τ_int is measured on the flowed Q, not on the plaquette.* Both are printed side
by side precisely so the gap is visible, and the phase refuses quietly to be
reassuring: if `2·τ_int(Q)` exceeds the configured `N_SKIP` it says the ensemble
is **not** decorrelated in the topological sector and to raise it before sampling.
Sector-change counts are printed for the same reason — that is what freezing at
β = 2.5 will look like if it bites.

*The best linear filter is fitted exactly, not by gradient descent.* For
`ŷ(x) = Σ_Δ w_Δ q(x+Δ)` on a periodic lattice, translation invariance is exact,
so the normal equations are `G_ij = A(Δ_j − Δ_i)`, `h_i = C(Δ_i)` with `A` the
autocorrelation of `q_clov[U]` and `C` its cross-correlation with the target —
both one FFT pair, and the fit is then a single linear solve. Applying the filter
is another FFT pair rather than the ~3.6k rolls a Manhattan-8 ball in 4D would
need. Two consequences:

- **The arm is the true least-squares optimum**, so "the network beat the best
  linear filter" cannot be an artifact of under-training the baseline. Both the
  free filter and the hypercubic-symmetrised one are reported; the free one is
  the stronger control and is the one the gate should be read against.
- **The conventions had to be pinned.** A sign, a conjugate or a shift in either
  FFT would still produce plausible numbers, so `arms` runs a self-test first:
  synthesise `y` with a known random filter and require that the fit recovers it.
  Measured: `max|FFT − naive rolls| = 1.4e−14`, `max|ŵ − w_true| = 1.3e−15`,
  `R² = 1.000000000000`.

One implementation trap, recorded because it silently destroys the arm: offsets
must be **de-duplicated modulo L**. A Manhattan-8 ball wraps an L = 8 torus
several times, and the same site entering the design matrix under two names makes
`G` singular — the fit would not error, it would just return garbage weights
through the ridge term.

### 12.4 The pre-flight, run (V100, 2026-09-14)

`chain` and `gate0` at all three rows. Two results, one of them a correction to
this script's own gate.

**No topological freezing, anywhere — and N_SKIP drops from 20 to 10.**

| row | τ_int(Q) | τ_int(plaq) | Q range | sector changes / 400 sweeps |
|---|---|---|---|---|
| β=2.3 L=8 | 0.55 (W=4) | 2.35 (W=15) | [−2.65, +3.50] | 303 |
| β=2.4 L=12 | 0.64 (W=5) | 1.35 (W=9) | [−5.15, +4.11] | 322 |
| β=2.5 L=16 | **2.76** (W=17) | 0.97 (W=6) | [−5.08, +4.25] | 266 |

The risk priced into §11 did not bite: the chain tunnels between sectors on most
sweeps at every β. But **the ordering inverts across the rows** — at β = 2.3 the
plaquette is the slower mode (2.35 vs 0.55) and at β = 2.5 the topology is
(2.76 vs 0.97). That is the topological slowdown appearing on schedule as `a`
falls; it is simply not yet large. τ_int(plaquette) would have said 2 sweeps at
β = 2.5, off by a factor of three — the reason the phase measures the flowed Q.
With τ_int(Q)'s own error at the worst row ≈ ±1.2, `N_SKIP = 10` keeps ≳3·τ_int,
and halves the sampling cost against the provisional 20.

**Gate 0's integrality half fails — and the first version of the gate hid it.**
The gate's criterion was `⟨|Q−[Q]|⟩ < 0.25`. **0.25 is the null**: any
distribution broad compared to the integer spacing gives 0.250 (uniform,
Gaussian at rms 0.85, Gaussian at rms 2.15 — all 0.250). So the threshold was
set exactly at "no integer structure whatever", and it returned PASS on the two
rows that measured 0.239 and 0.232. Those PASS verdicts are meaningless and are
retracted. The threshold is now 0.15, the null is printed beside the reading, and
the real criterion moved into `gatefit` (below).

The measured readings at the primary rung `t/a² = 2`:

| row | ⟨\|Q−[Q]\|⟩ | rms(Q) | per-config \|Q(3)−Q(1.5)\| | ⟨\|Q−[Q]\|⟩ at t=12 |
|---|---|---|---|---|
| β=2.3 L=8 | 0.239 | 0.847 | 0.226 | 0.103 |
| β=2.4 L=12 | 0.232 | 1.081 | 0.250 | 0.088 |
| β=2.5 L=16 | 0.298 | 2.152 | 0.257 | 0.142 |

So **at no row is Q integer-quantised at t/a² = 2**; integrality only begins to
appear at `t/a² ≳ 12`, i.e. `r_sm/a ≈ 9.8`, past the Manhattan-8 receptive field
and — at L = 8 and 12 — past the box itself.

What *does* hold up: the **distribution** of Q plateaus from `t/a² ≈ 2–3`
onward (rms 1.08 → 1.16 → 1.17 → 1.18 across t = 2, 3, 8, 16 at β = 2.4; 2.15 →
2.15 → 2.08 → 2.02 at β = 2.5), and the integrator is not the limit —
`|Q(ε=0.02) − Q(ε=0.005)| ≤ 7.2e−5` at every row, five orders below the effect.

**Two things this exposes about the design itself.**

1. *The smoothing radius must fit in the box, and at β = 2.3 it does not.* The
   ladder holds `r_sm/a` fixed across β by construction, but `L` grows with β to
   hold the physical volume — so the top rungs collide with the lattice at the
   coarsest row. At L = 8, `t/a² = 4` gives `r_sm/a = 5.7` and `t/a² = 8` gives
   `r_sm/a = 8.0 = L`: the flow has smoothed over the whole torus. Only
   `t/a² ≤ 2` is box-safe there (`r_sm ≤ L/2`). `gate0` now warns per rung. The
   choice is to restrict the β = 2.3 row to the lower rungs, or to enlarge it —
   which breaks the constant-physical-volume design.
2. *Non-integrality has two possible causes and the gate could not tell them
   apart.* Either the flow has not resolved the topology at that rung, or the
   lattice charge carries the usual multiplicative renormalisation
   `Q_latt ≃ Z(β,t)·Q` with `Z < 1` on a rough field, in which case the integer
   structure is real but sits at spacing Z. The new **`gatefit`** phase scans Z
   (offline, seconds, on the cached `gate0` dumps) and reports where the
   structure is sharpest, with `rms(Q/Z*)` beside it so the degenerate
   large-Z minimum — every Q driven toward zero, which is an integer — announces
   itself. Validated on synthetic data: it recovers a planted `Z = 0.8` to two
   digits and returns NONE on pure noise. `gate0` now also records the naive
   charge off the same trajectory, so clover-vs-plaquette is available as a
   discretisation scale.

**Cost, measured.** `gate0` to `t/a² = 16` on 16 configs: 1:19 (L=8), 6:36
(L=12), 20:47 (L=16). Scaled to the `targets` phase — 1200 configs to the top
rung `t/a² = 8` — that is ≈ 13 h at L=16, ≈ 4 h at L=12, ≈ 1 h at L=8, so ~18
GPU-hours of flowing, against the 1–2 h estimated in §11. **That estimate was
wrong by an order of magnitude.** Two levers before spending it: the top rung
costs half the trajectory on its own (dropping `t/a² = 8`, which is box-unsafe
at L=8 anyway, halves the bill), and `TOPO_CHUNK=64` should use the V100 far
better than the default 16 at these volumes.

**Next:** `TOPO_PHASE=gatefit python scripts/measure_topology.py` — seconds, no
GPU, reads the dumps already on disk. It decides whether the primary rung moves,
whether the β = 2.3 row survives as designed, and whether R4 ("reproduces the
integer Q-histogram") is available at all. Only after that is a night of
`ensemble` + `targets` worth spending. WP3–WP5 (`train_topology.py`, the
trainings, `fit_topology.py`) are untouched.


### 12.5 `gatefit`: the rung moves to t/a² = 4, and β = 2.3 loses its box

The Z-scan separates the two readings §12.4 could not. **The integer structure is
real and the charge is renormalised**, exactly as the alternative hypothesis
said: Z* converges to a stable value at large flow time on every row, and it
rises with β as it must —

| row | Z* at the three smoothest rungs | dev(Z*) there |
|---|---|---|
| β=2.3 L=8 | 0.86 | 0.055 → 0.020 |
| β=2.4 L=12 | 0.91 → 0.94 | 0.059 → 0.039 |
| β=2.5 L=16 | 0.92 → 0.93 | 0.068 → 0.043 |

Z → 1 as a → 0 is the continuum expectation, and a Z* near 1/2 or 2 would have
indicted the normalisation of `q_x` rather than the physics — that is what the
deliberately wide scan was for. It did not happen. **The clover charge is
behaving correctly**; what the naive reading of §12.4 called "no integer
structure" was a renormalisation of 10–15%.

**Where the structure actually sets in.** Applying the criterion — deviation
below the null by more than its own error, `r_sm ≤ L/2`, and Z* inside a
plausible band:

| row | usable rungs | the constraint that binds |
|---|---|---|
| β=2.3 L=8 | `[2]`, and marginal (Z* = 0.62, far from its converged 0.86) | the box: `r_sm ≤ L/2` ⇒ `t/a² ≤ 2` |
| β=2.4 L=12 | `[4]` | the box above, the null below |
| β=2.5 L=16 | `[3, 4, 6]` | the null below |

Consistent across rows: **topology is resolved at `r_sm/a ≈ 4.9–5.7`, i.e.
`t/a² = 3–4`** — in lattice units, the same everywhere, which is the fixed-`t/a²`
design vindicated at a rung the design set too low. So the primary rung moves
**2 → 4** and the ladder becomes `{1, 2, 4, 6, 8}`. `r_sm/a = 5.66` is still
inside Manhattan 8, but it eats the margin the design kept: the task is now
strictly harder for both architectures, and the well-posedness question
(§eq:wellposed) is correspondingly sharper.

**One filter bug found and fixed on the way.** The first `usable` criterion
accepted two aliasing minima — β=2.4 at `t=1` with Z* = 0.44 and β=2.5 at `t=8`
with Z* = 0.46. Dividing by a small Z inflates the spread until *some* alignment
with the integers appears; `rms(Q/Z*)` gives it away (2.17 against a raw 0.96;
4.59 against 2.09). Z is a renormalisation factor, so the band `[0.6, 1.1]` is
now part of the criterion and out-of-band minima are flagged in the table. With
the band, the three rows read cleanly and consistently.

**The β = 2.3 row cannot host the new primary rung.** At L = 8, `t/a² = 4` gives
`r_sm/a = 5.66 > L/2`: the flow wraps the torus. This is the collision §12.4
flagged, now binding — the ladder holds `r_sm/a` fixed across β while constant
physical volume shrinks L at coarse β, and the two requirements meet at
`L ≥ 12`. Three ways out, none free:

1. **Drop the row.** Two β (2.4, 2.5) at the design volume, primary `t/a² = 4`,
   no new cost. R2 (scale adaptivity) becomes a two-point control.
2. **Run β = 2.3 at L = 12.** Three rows, but its physical volume is then ≈5×
   the others — a confound in exactly the comparison R2 makes, and ~4 GPU-hours
   more flowing.
3. **Keep `t/a² = 2` as the primary rung everywhere** and accept a target whose
   topology is only partly resolved (dev(Z*) ≈ 0.17, Z* well short of converged).
   The architecture readings R1–R3 survive — they only need a well-defined field
   — but R4, the physics deliverable, does not.

The startup banner now refuses to be quiet about it: any row that cannot host the
primary rung says so before anything runs.


### 12.6 Decisions taken, and the batch size

**Two rows, not three.** β = 2.3 L = 8 is dropped (§12.5 option 1): at the new
primary rung it cannot fit the flow in its box, and enlarging it to L = 12 would
have made its physical volume ~5× the others — a confound in exactly the
comparison R2 makes. The run is **β = 2.4 L = 12 and β = 2.5 L = 16**, both at
the design volume, primary `t/a² = 4`, ladder `{1, 2, 4, 6, 8}`. R2 is therefore
a two-point control. It is a control, not the primary reading, and two honest
points beat three with a volume confound inside one of them.

**The batch size is computed, not guessed.** One "link field" is
`D · L⁴ · nc² · 8` bytes (8.0 MiB per configuration at L = 16). The peak live set
is ~13 of them per configuration: six for RK3 (`U`, `W1`, `W2`, three `Z`), two
for the staple temporaries inside each drift evaluation, five for the clover's
`F` dict and leaf temporaries. That is ~104 MiB/config at L = 16, so a 20 GiB
budget holds ~196 and the default caps at 128 for allocator headroom.
`TOPO_CHUNK` still overrides outright.

**A memory win found in the arithmetic.** `flow_trajectory` holds every ladder
snapshot to the end — five more link fields per configuration, 28% of the total.
`flowed_densities` now walks the segments itself and reduces each snapshot to a
density before flowing on. A density is 1/32 of a link field, so the snapshots
stop mattering: 144 → 104 MiB/config at L = 16, i.e. **38% more batch for
free**, and identical numbers (it is the same segmented walk).

**`probe`** times one chunk of the real targets work and extrapolates the row,
including peak device memory. Two minutes, and it is what should be read before
committing the night — both because the §12.4 cost estimate came from
extrapolating a differently-sized run, and because it is the only way to see
whether the larger chunk is actually being used.
