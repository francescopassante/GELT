# Where attention can win — and two places it cannot

**Status (2026-09-15): two attempts closed, a third built and pre-flighted.**
This note replaces `notes/flow_free_topology.md` (the design record) and
`notes/topology_go_nogo.md` (the decision dossier), both deleted along with the
study's code — see §10 for what went and where it lives. Everything below that
is labelled *measured* was measured; the proposal in §8 is a proposal.

**Attempt 3 is `notes/m1_probe.md`** — the ablation §1 below should have
implied and did not: GELT against GELT with the softmax frozen. See §1.1.

The question this note tracks: **is there a task where GELT's attention beats a
matched-parameter L-CNN, and can it be run here?**

---

## 1. The only architectural difference

The loose framing — "an L-CNN is a polynomial, GELT is not" — is wrong, and
getting it wrong wastes compute. `LAct` is `g(Re Tr W_c / nc) · W_c`, i.e.
input-dependent gating, and L-Conv mixes channels *before* it, so the gate's
argument is a learned linear combination of transported channels: an L-CNN can
realise `ReLU(Re Tr[A − B])`, a data-dependent difference test between two
scales held in different channels. GELT carries the *same* L-Act.

> **The only architectural difference is whether the weights over *offsets* are
> input-dependent.** Gating, bilinearity, receptive field, loop degree, inputs,
> loss, splits and estimator are matched by construction in both shootouts.

Two candidate mechanisms follow from that, and they are **not** the same bet:

- **(M1) relative weighting.** Softmax depends only on score *differences* and is
  invariant under a common shift, so α is a scale-free ratio across the
  neighbourhood, recomputed per site per configuration. `ReLU(Re Tr[A − B])`
  thresholds an *absolute* magnitude and must be calibrated to an amplitude
  fixed weights cannot adapt.
- **(M2) boundedness.** Softmax is a convex combination, weights in (0,1), on a
  score already normalised by `√(d_qkv·nc)`. L-Conv's weights are unbounded and
  an L-CB stack is a raw matrix polynomial (degree ≤ 16 after four layers), so a
  configuration in the tail of the input distribution is amplified without limit.

**M1 has now failed to find a task twice. M2 has already been observed three
times.** §7 and §8 are about taking that seriously.

### 1.1 …and the sentence above is wrong as stated (2026-09-15)

"The only architectural difference is whether the weights over offsets are
input-dependent" is true of the *mechanism*, false of the *comparison*. GELT and
the matched L-CNN differ in **two** things: input-dependent offset weights and
**transport geometry** — the shortest-path-averaged L1-ball against axis-aligned
link products. Every GELT-vs-L-CNN number in this repo therefore confounds M1
with geometry, §2's parity included: a tie is not evidence against M1 and a win
would not have been evidence for it.

The clean test is the ablation, not the baseline: **GELT against GELT with the
softmax frozen**, same transport, same bilinear value path, same everything,
α no longer reading the input. That is `notes/m1_probe.md`, built
2026-09-15. It also keeps M2 fixed while M1 varies — the frozen weights are a
softmax of free logits, so α stays a convex combination — which the L-CNN
comparison cannot do either.

This does not retract §2 or §4. It re-labels them: the 0⁺⁺ tie and the topology
stop are facts about *the architectures*, and §3's selection rule stands. What
they are not is a measurement of M1.

---

## 2. Attempt 1 — 0⁺⁺ glueball spectroscopy: parity (2026-09-14)

Full record in `notes/lcnn_shootout.md` §9. Same inputs, loss, splits, estimator
and parameter budget; only the block differs.

| | combined, two ensembles |
|---|---|
| ΔA₀(GELT − GEVP) | +0.077 ± 0.022 (3.6σ) |
| ΔA₀(L-CNN − GEVP) | +0.067 ± 0.020 (3.4σ) |
| **ΔA₀(GELT − L-CNN)** | **+0.012 ± 0.009 (1.3σ)** |
| Δm(GELT − L-CNN) | consistent with 0 |

Parity extends to the noise-to-signal ratio (§9.3): `δC(Δ)/C(Δ)` is
indistinguishable at every Δ. **Do not quote the shootout as an error-bar
advantage.**

**Why the tie was structural**, read against M1. Three properties of that task,
each of which independently neutralises an input-dependent kernel:

1. **Zero-momentum projected loss.** The Rayleigh quotient is built from
   `Ō(t) = Σ_x O(t,x)`. Site-averaging destroys site-to-site variation, which is
   the one thing attention has.
2. **The optimum is linear in loop shapes.** Only 2.5% of `‖O_GELT‖²` lies
   outside the 21-operator `full` span, and ~80% of that residual is rectangular
   loops (`notes/fable5.1_10-09_audit.md` §8.3). That *is* L-Conv.
3. **One β, one scale, the most symmetric channel.** A single fixed kernel is
   optimal by construction.

Under those three, softmax attention degenerates into a convolution with learned
per-offset weights — which is L-Conv. The tie was not bad luck.

---

## 3. The selection rule read off that tie

> **M1 can separate where the optimal neighbourhood weighting is *relative*
> rather than absolute.** Three properties: per-site target; sparse structures at
> configuration-dependent positions *and scales*; amplitude-invariant identity.

Flow-free topological charge density was selected by that rule, and it scores on
all three. It still failed — for a reason the rule did not cover. §5 and §6 are
the repair.

---

## 4. Attempt 2 — flow-free topological density: stopped (2026-09-15)

**The task.** Learn `q̂(x)[U] ≈ q_t(x) ≡ q_clover[F_t(U)](x)`, the
gradient-flowed topological charge density, from **unflowed** thermalised SU(2)
links, per site, 4D. Two deliverables: the physics (how much of flowed topology
a bounded neighbourhood determines) and the architecture A/B.

### 4.1 Everything that was built worked

| component | check | measured |
|---|---|---|
| clover charge, `gelt/lattice.py` | parity-odd site by site | 1.4e−17 |
| — its predecessor | the naive density is *not* | Q: −1.3161 → +2.9888 under a reflection preserving the action to all printed digits |
| Wilson flow, `gelt/flow.py` | gauge covariance | 1.7e−15 |
| | RK3 third order / Euler first | 8.53, 7.27 / 1.74, 1.86 |
| | flow-time normalisation vs `exp(−k̂²t)` | 0.556668 vs 0.556668 |
| linear-filter arm | FFT application vs naive rolls | 1.4e−14 |

No topological freezing at any row (τ_int(Q) = 0.55 / 0.64 / 2.76 at β = 2.3 /
2.4 / 2.5; 266–322 sector changes per 400 sweeps), so `n_skip = 10`.

### 4.2 The pre-flight cost two controls

`gate0` + `gatefit` found the charge renormalised (`Q_latt ≃ Z·Q`, Z → 1 as
`a → 0`, exactly the continuum expectation) and integer structure setting in at
`r_sm/a ≈ 4.9–5.7`, **not** at the design's `t/a² = 2`. The primary rung moved
**2 → 4**. Consequences: β = 2.3 could not host it (`r_sm/a = 5.66 > L/2` at
L = 8), so **R2 dropped from three β to two** — a lattice-spacing lever of 1.33×
where the design had 2.0×, i.e. **58% of the log-lever gone** — and the ladder's
top rungs became box-invalid at L = 12, making **R3 a single-row reading**. The
mechanism the selection rule predicts (scale adaptivity) was the one R2 tests.

### 4.3 The linear arm read 0.017, and it was the wrong instrument

β = 2.4, L = 12, 1200 configs, `t/a² = 4`, 800/200/200 contiguous chain-ordered:

| arm | R² | RMS ΔQ |
|---|---|---|
| identity | −102.0 | 1.689 |
| identity×Z (Z = 0.002) | 0.0000 | 1.677 |
| best linear filter, Manhattan 8, 3485 sites | **0.0174** | 1.634 |

Read as the design's stop condition ("if a network does not clearly beat the
best linear filter, stop") this says *not a convolution, unlimited headroom*.
Read as a reachability bound it says *the target is not there*. **Neither is
supported**, because the arm is handicapped in a way the networks are not:

> It gets **one scalar per site**. `q_clov` is a particular quadratic contraction
> of the field strength, so most of the information in the links is already gone
> before the filter sees it, and no convolution of the scalar can recover it. The
> flow does not work that way and neither do the networks: both smear the
> **links** and contract afterwards.

### 4.4 The alignment check: the pipeline is sound and the signal is real

| | rms Q | corr with Q(0) |
|---|---|---|
| t = 0 | 0.8503 | +1.0000 |
| t = 1 | 1.6109 | +0.3582 |
| t = 2 | 1.6898 | +0.3219 |
| t = 4 | 1.6973 | **+0.2981** |

with `corr Q(1)↔Q(2) = +0.9615` and `Q(2)↔Q(4) = +0.9708`.

**+0.30 on the lattice-summed charge against +0.02 per site** is a coherence
statement: the τ = 0 signal survives summing 20 736 sites (√V ≈ 144) but is
buried under white UV noise locally. It also explains `rms Q(0) = 0.85 <
rms Q(4) = 1.70` despite `q_clov[U]` being ~10× noisier *per site* — incoherent
noise sums incoherently, lumps sum coherently.

### 4.5 The smearing baseline — the measurement that stopped it

4D APE cooling of the **links** (α = 0.5, `directions=range(4)`), then the same
clover density, then the same `Z`-fit-on-train / R²-on-test protocol as `arms`:

| APE n | R² at t = 1 | R² at t = 2 | **R² at t = 4** |
|---|---|---|---|
| 0 | 0.0054 | 0.0009 | 0.0000 |
| 8 | 0.8764 | 0.3908 | 0.1490 |
| 16 | **0.8906** | 0.8469 | 0.4083 |
| 24 | 0.6145 | **0.9929** | 0.6344 |
| 32 | 0.4351 | 0.8783 | 0.8054 |
| 48 | 0.2660 | 0.6217 | **0.9783** |
| 64 | 0.1839 | 0.4606 | 0.8627 |

**The APE↔flow correspondence is exact.** One APE step with α = 0.5 and 6
staples in 4D is a flow step of `ε = α/6 = 0.0833`:

- t = 2 peaks at n = 24 → 24 × 0.0833 = **2.00**
- t = 4 peaks at n = 48 → 48 × 0.0833 = **4.00**

and `Z → 1` at each peak (0.9994 at t = 2, 0.9845 at t = 4): at the matched
level the classical density needs **no renormalisation**. Three independent
confirmations that the flow, the clover density and the targets are all correct
— and a useful standing fact for this repo: **APE cooling is a drop-in for
Wilson flow at ~8% of the cost** (48 APE steps × 4 directions against RK3's
600 drift evaluations × 4; the flow took 7411 s for 1200 configs at L = 12).

### 4.6 The squeeze, which is empty

APE n propagates information n sites: `n = 48` has rms smoothing radius 5.66
(matching `r_sm`) but **support 48**. The network has Manhattan 8 — a hard bound
no architecture moves. Read the classical curve at **matched reach**, n = 8:

| rung | topologically meaningful? | best classical R² | classical at matched reach n = 8 |
|---|---|---|---|
| t = 1 | no — Z\* unconverged | 0.891 (n = 16) | **0.876** |
| t = 2 | no — Z\* unconverged | 0.993 (n = 24) | 0.391 |
| **t = 4** | **yes** (primary rung) | 0.978 (n = 48) | **0.149** |

- **At t = 4** the target is topological, but a Manhattan-8 model reaches 15% of
  the variance where a longer classical iteration gets 98%. Both architectures
  compete for a residual they probably cannot reach, against a free method that
  beats them.
- **At t = 1** the receptive field suffices — but classical matched-reach already
  captures 0.876 of an achievable 0.891, so **98% of the headroom is gone**, and
  `gatefit` says the target is not topologically resolved there anyway.

**There is no rung where the target is topologically meaningful, the receptive
field is adequate, and headroom exists over free classical cooling.** Stop.

Total cost of reaching this verdict: ~4 h of V100 (2037 s sampling + 7411 s flow
+ ~20 min probe), against the ≥60–100 GPU-hours the full study needed.

---

## 5. Two lessons, generalised

**L1 — never define the target by applying a classical algorithm the baseline can
also run.** A target defined as *the output of a smoother* is, by construction,
well approximated by applying a similar smoother. Classical smoothers are cheap
and have effectively unbounded reach, so they are the ceiling, and the network is
reduced to competing in their residual. This killed the study independently of
anything about attention, and it would have killed an L-CNN-only version too.

**L2 — the bounded-receptive-field criterion.** A task is only a fair A/B if
there exists a radius reachable by the architecture at which (i) the target is
meaningful *and* (ii) substantial headroom remains over the best classical local
method **at that same radius**. Comparing against a classical method with more
reach is not a baseline, it is a handicap. The design compared `r_sm/a = 5.66`
against "Manhattan 8" and called the margin spent-but-adequate; the Manhattan
ball in 4D is a cross-polytope with Euclidean reach 8 along an axis, **4.0 along
a diagonal** and ~4.7 typical, and a 4D Gaussian of rms 5.66 puts only **41% of
its mass inside it** (76% at the design's original rung, 16% at the ladder top).

**The cheap pre-flight both lessons imply**, to run *before* building any
training code: **measure the best classical method's accuracy at the
architecture's own reach.** Here that was one 20-minute probe on already-cached
data, and it was decisive. It should be the first gate of every future A/B.

---

## 6. The revised selection rule

A task is a candidate only if it satisfies **all four**:

1. **Per-site target**, not zero-momentum projected — else site-to-site
   variation is integrated away before the loss sees it (§2.1).
2. **Sparse structures at configuration-dependent positions *and scales*, with an
   amplitude-invariant identity** — the M1 mechanism (§3).
3. **A bounded-receptive-field window**: headroom over the best classical local
   method *at the architecture's own reach* (L2).
4. **The target is not the output of a classical local algorithm** (L1).

Topology satisfied 1 and 2 and failed 3 and 4. The 0⁺⁺ satisfied 3 and 4 and
failed 1 and 2.

---

## 7. What is actually measured in GELT's favour

Not M1 — M1 has never been observed to pay. **M2 has, three times**
(`notes/lcnn_shootout.md` §9.2), share of `C(0)` carried by a single test
configuration:

| operator | top-1 share | × median |
|---|---|---|
| GELT, 12 untrained + 2 trained | 0.5 – 1.0% | 2.2 – 4.2 |
| L-CNN, trained ens1 + all 4 sweeps | 0.5 – 0.9% | 2.1 – 4.2 |
| **L-CNN trained ens0, 60 epochs** | **71.8%** | 1187 |
| **L-CNN untrained, seed 0, ens0** | **35.9%** | 1.2 × 10⁶ |
| **L-CNN untrained, seed 0, ens1** | **97.9%** | 3.2 × 10⁶ |

**0 of 14 GELT operators, 3 of 9 L-CNN ones**, with a mechanism (M2) that
predicts it and an existing gate in `fit_glueball_overlap.py` that fires on
exactly those rows. Config 371 is unremarkable to every classical operator
(rank 145/400 by variance at thin links) — it is the architecture, not the
configuration.

What this is **not**: a variance advantage. §9.3 measured `δC(Δ)/C(Δ)` for the
clean trained pair and found parity at every Δ. M2 is a **failure-rate**
property across training runs.

---

## 8. Proposal — make M2 a designed experiment

**Nothing below is implemented** — except the arm it needs, which now exists:
`LConv(normalize_shifts=True)` bounds an L-Conv's aggregation over offsets the
way a softmax does, and is `lcnn_norm` in `notes/m1_probe.md`'s reading R-D. It
is the one place GELT already demonstrably beats a matched L-CNN, and the
observation is currently an anecdote salvaged from runs that went wrong rather
than an experiment anyone designed.

**Claim to test.** Under input-distribution stress, a bounded convex aggregator
degrades gracefully where an unbounded matrix polynomial fails outright — at
equal parameters, inputs, loss, splits and estimator.

**Three stressors**, in increasing cost, all on the existing glueball pipeline
(`train_glueball.py` with `GLUEBALL_ARCH`, existing ensembles, existing gate):

1. **Low statistics.** `N_train ∈ {100, 400}`, 3 init seeds, both architectures —
   12 runs, **no new sampling**. Fewer configurations means a heavier empirical
   tail, which is exactly what M2 predicts L-CNN cannot bound.
2. **Volume transfer.** Train at L = 12, evaluate the frozen operator at L = 16.
   Both are translation-equivariant, so this is legitimate and needs one new
   ensemble, no retraining.
3. **Coupling transfer.** Train at β = 2.4, evaluate at β = 2.3 and 2.5. Two new
   ensembles.

**Readings, to be fixed in writing before the first run:**

- **Primary — failure rate.** Fraction of runs tripping the existing 10% top-1
  share gate, per architecture, pooled across stressors. Pre-registered because
  §9.2's 3/9 vs 0/14 is the prior and a second independent 0-vs-k is the claim.
- **Secondary — graceful degradation.** ΔA₀ against the classical GEVP as a
  function of stressor strength, correlated jackknife, median over seeds (the
  median rule is fixed in advance precisely because §9.2 showed a mean over seeds
  is meaningless when one arm blows up).
- **Null.** Untrained arms of both architectures under the same gate.

**Why this clears the revised rule.** Criterion 4 is satisfied — "failure rate
under distribution shift" is not any classical algorithm's output, so there is no
classical ceiling to be crushed by. Criteria 1–3 do not bind, because the
mechanism under test is M2, not M1: the 0⁺⁺ task's zero-momentum projection is
what made it a *fair* test of accuracy, and it stays fair for reliability.

**Why it is a good bet.** The effect has been observed three times with a
mechanism that predicts it; the marginal cost of (1) is a dozen short runs on
data already on disk; and a null is still publishable — it would say the
blow-ups were seed luck, which is worth knowing before the write-up leans on
§9.2.

**Run the §5 pre-flight first anyway**: before any of this, confirm the gate
fires on the archived dumps and that the top-1 statistic is stable under the
blocked jackknife. That is offline and costs minutes.

### The high-impact alternative, if there is appetite

`PLANS.md` §A.4 (**learned multigrid / preconditioner for the Dirac operator**)
is the best fit to the revised rule of anything in that document: near-zero modes
of the Dirac operator localise on topological objects at configuration-dependent
positions *and* scales (criterion 2, exactly), the target is a residual reduction
rather than a smoother's output (criterion 4), the comparison is against a
fixed-stencil prolongator at matched reach (criterion 3), and Lehner–Wettig have
already shown gauge-equivariant networks beat that baseline, so headroom is
demonstrated in the literature rather than assumed. The cost is that the repo has
no fermions and no U(1) sampler entry. It is a second thesis, not a chapter.

---

## 9. Consequences for `PLANS.md`

- **§A.3 (learned smearing) is devalued by §4.5 and should not be run as
  written.** Its target is APE/HYP/gradient flow, which is L1 exactly: 48 APE
  steps with one fitted scalar reproduce the flowed topological density at
  R² = 0.978 with zero parameters and zero training. Any learned map judged
  against a classical smoother is competing in a 2% residual. It survives only
  if re-scoped to a *downstream* metric no classical smoother optimises — and
  then criterion 1 has to be re-checked, because the obvious downstream metric is
  the zero-momentum glueball correlator that tied in §2.
- **§A.2 (equivariant normalizing flow) and §A.6 (learned collective topological
  updates) still measure `Q`**, which is why `topological_charge_density` is kept
  in `gelt/lattice.py` (§10). Both are M1 bets and both should be run past §6's
  four criteria before any code is written.
- **§A.5 (control variates) and §A.8 (flux tube from the attention map)** are
  unaffected. §A.8 is worth noting as a different kind of win: the attention map
  *is* a measurable lattice operator and an L-CNN has none, so it is a capability
  the baseline lacks rather than a number it loses.

---

## 10. What was deleted, and where it lives

Deleted 2026-09-15, not archived — recover with `git show <sha>^:<path>` against
the commit that removed them:

| path | what it was |
|---|---|
| `notes/flow_free_topology.md` | the design record, build order, pre-registered readings R1–R5, build log §12 |
| `notes/topology_go_nogo.md` | the decision dossier — every pre-flight number with its provenance |
| `reports/topology/` | `flow_free_topology.tex`, 13 pp of theory and design |
| `scripts/measure_topology.py` | the five-phase classical half: chain, gate0, gatefit, ensemble, targets, arms |
| `scripts/probe_smearing_baseline.py` | the smearing baseline of §4.5 |
| `gelt/flow.py`, `tests/test_flow.py` | Wilson flow (RK3 + Euler, closed-form SU(2) exponential), 21 tests |

**Kept**: `topological_charge_density` / `topological_charge` in
`gelt/lattice.py` and their parity tests in `tests/test_lattice.py`. They predate
this study, they fixed a real defect (the naive corner-based density is not
parity-odd — CLAUDE.md caveat 7), and two surviving `PLANS.md` entries measure
`Q`. If a future plan needs the Wilson flow back, `gelt/flow.py` is one
`git checkout` away and its test suite is 5.4 s.

Every number in §4 was produced by the deleted code before it was deleted. The
artifacts on the V100 (`datasets/topo_*`, `results/topology/*`) are gitignored
and are the only copies; keep or delete them as you like, but this note is the
record.
