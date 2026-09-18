# Where attention can win — and two places it cannot

**Status (2026-09-18): two attempts closed, a third run to completion, a
fourth proposed and pre-flighted.** This note replaces
`notes/flow_free_topology.md` (the design record) and
`notes/topology_go_nogo.md` (the decision dossier), both deleted along with the
study's code — see §11 for what went and where it lives. Everything below that
is labelled *measured* was measured; §8 and §9 are proposals.

**Attempt 3 is `notes/m1_probe.md`** — the ablation §1 below should have
implied and did not: GELT against GELT with the softmax frozen. See §1.1, and
§1.2 for what it found.

**Attempt 4 is §9** — vortex-cluster geometry in 3D Z₂, the first candidate that
is a physics observable *and* clears §6's four criteria. Proposed and
pre-flighted 2026-09-18; not built.

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
times.** §7 and §8 are about taking that seriously. — **Superseded in part by
§1.2: on a constructed target, with the transport confound removed by ablation
rather than by a baseline, M1 pays at 6/6 cells (2026-09-16).** The sentence
stands for *physics* tasks, which is what §7 and §8 are about.

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

### 1.2 The ablation ran, and M1 paid (2026-09-16)

`notes/m1_probe.md` §0 is the readout; the five things this note has to absorb:

1. **M1 has been observed to pay, for the first time in three attempts.** On the
   probe's primary constructed target, GELT beats the frozen-softmax ablation in
   **6 of 6** paired `(ensemble, seed)` cells, median ΔR² = +0.144, sign-test
   p = 0.031 — the floor at n = 6 — and the capacity-matched frozen arm is no
   closer, so it is the mechanism and not the parameters. Every count of "M1 has
   failed to find a task" in this note (§1, §7) predates that and should be read
   as *"on a physics task"*: the probe's targets are constructed and knowingly
   violate criterion 4 of §6.
2. **The clean test had to be an ablation, and that is the durable lesson.**
   Every one of the two earlier attempts compared GELT against the L-CNN, which
   §1.1 shows can never isolate M1. The comparison that worked varies one thing
   inside one architecture. Any future M1 bet should be posed the same way
   *before* it is posed as a baseline shootout.
3. **M2 gained no direct support.** The probe's M2 control (`lcnn_norm`) is a
   null on accuracy (R-D, 3/6) and only 2.15× on dispersion; everything M2 has
   is still a dispersion or failure count, never a significance test. Point 5 is
   worse news for it than that.
4. **The geometry half of §1.1's confound is now partly measured, and it splits
   into two answers.** Feeding GELT a single-path transport instead of the
   shortest-path-averaged one changes *accuracy* not at all on the probe's
   target (R-I −0.037, 2/6; R-I′ +0.024, 3/6) — and on the calibration target
   the average is slightly ahead, so calibrated it is worth **less** where the
   mechanism lives. But the same swap costs the arm its **consistency**: over
   the 12 cells each arm ran, `gelt` never falls below R² = 0.70 and the two
   single-path variants do twice and three times. This does not turn any
   GELT-vs-L-CNN number into a clean comparison — the offset *set* still differs
   — but it narrows what the unattributed part of the 0⁺⁺ tie can be, and it
   does something sharper to §7 below, see point 5.
5. **§7's M2 reading needs re-wording, and this note is where that starts.**
   Every observation filed under "boundedness" is a *dispersion or failure-count*
   observation: 0-of-14 against 3-of-9 in the shootout, and the probe's own
   0.049 against 0.415. Point 4 shows the same kind of advantage appearing with
   α held completely fixed and only `T` changed. So low dispersion is **not** by
   itself evidence for M2 — GELT has at least two mechanisms that produce it,
   and no measurement here separates their contributions. The M2 case now rests
   on `lcnn_norm`, which is a null on accuracy (R-D) and only 2.15× on
   dispersion. **M2 is weaker evidence than this note has been treating it as.**

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

Not M1 — M1 had never been observed to pay when this was written; §1.2 is the
first time it has, on a constructed target. **M2 has, three times**
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

> **And what it is not, as of 2026-09-16: evidence that the mechanism is
> boundedness.** §1.2 point 5 — the M1 probe produced the same failure-rate
> signature by holding α fixed and changing only the *transport*
> (`notes/m1_probe.md` §7.7). A table of failure counts cannot distinguish the
> two, and this one does not. The rows above stand as a fact about the two
> architectures; the attribution to M2 does not follow from them alone, and §8's
> proposal is what would earn it.
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

## 9. Proposal 2 — vortex-cluster geometry in 3D Z₂ (2026-09-18)

**Status: proposed, pre-flighted, no training code.** The supervision, the
gate and their tests are built (§9.5); nothing has been trained and no arm has
been run. Everything labelled *measured* below was measured on 2026-09-18 on
freshly sampled Z₂ configurations at production geometry. The pre-flight is §5's gate run before any training code exists,
and it did its job twice — once on the task (§9.4) and once on the design
(§9.3), which it changed.

This is the answer to the question §8 leaves open: §8 targets M2 on the existing
0⁺⁺ pipeline and buys *reliability*, not accuracy. The question the thesis
actually has to answer is whether there is a task where GELT's attention wins on
**accuracy** against a matched L-CNN. §1.2 says M1 pays on a constructed target;
§6 criterion 4 says a constructed target is not a physics result. What follows is
the first candidate that is a physics observable and still scores on all four
criteria.

### 9.1 Why the standard observables cannot separate the two architectures

Worth writing down once, because it is what closed attempt 1 and what rules out
most of the obvious replacements. Every local gauge-invariant function of the
links is a function of Wilson loops, and both architectures generate loops of
degree ≤ 16 over the same Manhattan-8 receptive field (the four matched axes,
`notes/lcnn_shootout.md` §9).
**Expressiveness is matched by construction**; what differs is what is *easy*.
L-Conv makes functions that are linear in transported loops easy; the softmax
makes selection and routing easy. Every standard lattice observable — any J^PC
channel, any correlator, any smeared density — is a sum of loops, i.e. sits in
the half of the space the L-CNN is built for. The 0⁺⁺ tie was structural, and it
would repeat on every observable of that shape.

The quantities that are *not* of that shape are the ones defined by a global
competition or a global topological property: percolating clusters, minimal
surfaces, vortex networks, the low modes of the Dirac operator. There the
optimal local predictor has to decide which of several comparable candidates in
its neighbourhood wins — and that is selection, not a linear functional.

### 9.2 The task

In Z₂ a negative plaquette is pierced by a vortex line. In 3D the plaquettes are
dual to links, and the Bianchi identity — the product of the six plaquettes
bounding a cube is 1 identically, since every link appears twice — forces an
even number of negative faces per cube. So **the negative plaquettes form closed
loops on the dual lattice**, and their percolation is the confinement mechanism
of the 3D Z₂ theory. The four cached ensembles all sit in the confining phase
approaching β_c ≈ 0.7614 from below.

| the cached Z₂ 3D ensembles | |
|---|---|
| lattice | 48 × 24 × 24, isotropic, time is axis 0 |
| β | 0.7450, 0.7520, 0.7560, 0.7585 |
| configurations per β | 2000 (`datasets/z2_configs_L24_Lt48_b<β>_N2000.pt`) |
| sampler | exact Z₂ heat-bath, 500 thermalisation + 200 decorrelation sweeps |
| classical ξ | 2.05 … 5.28 (`results/attention/z2_beta_scan.pt`) |

Two targets, both defined by the vortex objects and never by GELT's own
primitive:

- **V1, primary — `log(1 + |C(x)|)`**, with `|C(x)|` the size of the largest
  vortex cluster touching any of the three plaquettes based at `x`, and 0 where
  none of them is negative. Cluster size is a **global** connected-component
  property: no local algorithm computes it, so §6's criterion 4 is satisfied in
  the strong sense, and the physics question is real — *how locally visible is
  the percolating vortex network?*
- **V2, calibration — the number of vortex plaquettes in the L1 ball of radius
  4.** A convolution of the indicator field with a unit kernel, so a linear
  filter reproduces it exactly; it plays the role T0 plays in the M1 probe, and
  its purpose is to catch a readout artifact before it is read as a mechanism.
  **Corrected while the code was written**: this was first written here as a
  BFS-truncated line length. A calibration arm has to be *exactly* linear —
  that is the whole of its job — and a truncated length is not. The truncated
  length is the **classical local arm** instead (the ceiling §9.4 measures the
  architectures against), and it is deliberately not a target.

Nothing here is smeared: the inputs are thin plaquettes, exactly as
`probe_inputs` builds them, so **CLAUDE.md caveat 1 (the Z₂ APE defect at
α = 0.5) does not bite.** That is not a small thing — it is the reason this task
can be run in Z₂ at all.

### 9.3 A measured structural fact, and the design it forces

*Measured 2026-09-18, locally, on random links — it is an identity, not a
statistic.*

In Z₂ the group is scalar, so the adjoint action of a transport is `T W T† =
T² W`. For **the L-CNN this is the identity**: `build_axis_transports` returns
single link products, which are exactly ±1, so `T² ≡ 1` and an L-Conv in Z₂
reduces to a plain real convolution over axis offsets. For **GELT it does not**.
`build_transport_average` divides by the multinomial path count, so at `R = 2` in
`D = 3` the 24 offsets split into 12 axis offsets (one shortest path, `T = ±1`)
and 12 diagonal ones (two shortest paths, `T ∈ {−1, 0, +1}`), and on the
diagonals

> **`T_Δ(x)² = (1 + P_{μν}(x + b))/2 ∈ {0, 1}`, with `b_μ = min(Δ_μ, 0)`** —
> verified to exactly zero error on all 12 diagonal offsets.

That is the `T·T† = (𝟙 + Re W)/2` of `build_transport_average`'s own docstring,
made binary by the group. **GELT's transport in Z₂ is a hard vortex mask**: a
neighbour is switched off exactly when a vortex line pierces the square between
`x` and `x + Δ`. Measured on a sampled β = 0.7585 configuration it fires on
**4.0% of the diagonal offsets and 2.0% of all 24** — sparse, and precisely on
the objects this task is about.

Three consequences, and the third is the one that matters:

1. **This is not an information advantage.** The L-CNN reads the same plaquette
   field as input and every masked plaquette is inside its receptive field. It
   is an *inductive bias*: GELT gets the vortex indicator as a multiplicative
   gate on its aggregation, the L-CNN would have to learn it.
2. **It is a third input-dependent offset weighting**, and the note's M1/M2
   vocabulary does not cover it. Call it **M3 — input-dependent masking via
   multi-path transport coherence.** It is input-dependent like M1 but it is not
   the softmax, and §1.2 point 7 already found that GELT's consistency is partly
   transport rather than attention. In Z₂ M3 is not a small correction, it is a
   binary gate; a GELT win here would be confounded with it unless the design
   says so up front.
3. **So the design is a 2 × 2, not an A/B**: `{softmax, frozen} × {average,
   single}`. `single` is the mask removed (its transports are ±1, hence trivial),
   `frozen` is M1 removed, and both are already arms in `probe_common.ARMS`.
   This factorial costs four arms instead of two and is the only way the result
   can be attributed.

`gelt_projected` is **not** an available arm here: `build_transport_average`
raises for Z₂ under `mode="projected"`, because projecting the vanishing
transports back onto the group needs the arbitrary `0 → +1` tie-break that is
caveat 1's defect in another place. That guard is in the library already.

### 9.4 The pre-flight — measured 2026-09-18

§5's rule is to measure the best classical method **at the architecture's own
reach** before writing any training code. It is implemented as
`scripts/z2_vortex_preflight.py`, `scripts/probe_preflight.py`'s sibling, and
it measures *two* classical arms because this task has two:

- the **M1-free ceiling** — the best linear filter over the same reach, one
  fixed weight per (plane, offset). That is what "input-independent offset
  weighting" means with the nonlinearity stripped away.
- the **local classical ceiling** — the connected component of the vortex line
  through the site, computed by BFS **truncated to the same ball**. Not a
  linear filter, and the strongest thing the baseline family could learn at
  that reach. §5's L1 is about being crushed by a method with *more* reach;
  this one has exactly the architecture's, so beating it is a real claim and
  failing to beat it is a real stop.

**The reading is the masked one.** Only ~9% of sites carry a vortex at all; the
rest have V1 exactly 0 and a linear filter predicts them from the local count.
An all-sites R² therefore mostly scores *"is there a vortex here"*, which is not
the question. Every number below restricts to vortex-carrying sites, and the
linear arms are refitted there so the comparison is a ceiling and not a
transplanted predictor. This was not obvious before the code ran — the all-sites
column reads 0.65 where the masked one reads 0.19.

Measured at β = 0.7585 on the **production geometry** (48 × 24 × 24, 40
configurations, reach Manhattan 8 = 4 layers × R 2 = 4 layers × K 2, blocked
jackknife over configurations):

| method at Manhattan 8 | R² (vortex sites) | AUC for "in the largest cluster" |
|---|---|---|
| linear filter on the radial shells — the **M1-free ceiling** | **+0.186 ± 0.037** | 0.701 |
| the full ball at radius 4 (direction adds almost nothing) | +0.217 ± 0.044 | 0.718 |
| **+ local component size** (BFS truncated at reach 8) | **+0.736 ± 0.033** | 0.885 |
| the target itself | 1.000 | 1.000 |

and the five gates it runs, all of which pass:

- **G1 closure.** The per-cube vortex parity is identically zero — Bianchi, and
  the check that validates the cube-face enumeration every index downstream is
  built from. A hard stop, not a warning.
- **G2 non-degeneracy.** `p_neg = 0.0385`; the largest cluster holds **38%** of
  the vortices on average, with a runner-up of the same order. "Which cluster is
  the biggest" is a genuine competition between comparable candidates — if the
  giant cluster had swallowed 98% of the vortices the target would have been a
  constant and the candidate dead.
- **G3 the transport asymmetry.** Re-verified on a production configuration:
  the path-averaged transport masks **4.32%** of its diagonal offsets against a
  vortex density of **4.32%** — the same number, which is §9.3's identity — and
  the single-path transport is ±1 throughout, so its adjoint action is trivial.
- **G4 calibration.** A linear ball filter reproduces V2 at **R² = 1.000000**,
  so W-A is a usable reading.
- **G5 headroom.** 0.814 over the M1-free linear filter and **0.264 over the
  strongest local classical arm**. Connectivity is worth **+0.550** of R² over
  density at the same reach: the task's difficulty is concentrated in routing,
  which is the operation a fixed kernel must spend its budget enumerating.

An earlier pass, scored per *vortex plaquette* on 24³ boxes rather than per site
on the production geometry, gave 0.205 / 0.758 / 0.242 for the same three rows.
The agreement across the two scorings is the reason the shape is believed.

**What the production run must still confirm**: these 40 configurations come
from a short chain (200 + 20 sweeps against the cache's 500 + 200), and the
run on the four cached ensembles is what fixes the β ladder of §9.6. The
command is in CLAUDE.md's *Running* block; it needs no GPU and takes minutes.

### 9.5 What exists, and the arms

**Built 2026-09-18** — the supervision and the gate, not the training:

| file | what it is |
|---|---|
| `gelt/vortex_targets.py` | the vortex indicator, the dual-graph connected components (Shiloach–Vishkin with a root hook), V1, V2, and the classical local arm |
| `scripts/z2_vortex_preflight.py` | §9.4's five gates and two ceilings, offline on the cached ensembles; `Z2V_SMOKE=1` runs the whole thing at production geometry off a short freshly sampled chain, in under a minute |
| `tests/test_vortex_targets.py` | 20 tests: closure by Bianchi, the four-plaquette loop around a flipped link, label propagation against a plain union-find, the long-line regression, V2's exact linearity, the local arm's locality, gauge invariance |

Two things the code changed about the design, both recorded where they happened:
V2's definition (§9.2) and the masked reading (§9.4). A third is a plain bug
worth remembering: hooking the *boundary plaquette* rather than its tree's root
makes the labelling `O(diameter)`, and at the production volume the percolating
line is a few thousand dual links long — it was still 223 components after 20
rounds and never converged. With the root hook it converges in 6.

`probe_common.py` already holds every one of the arms. The adapter this needs is a
`PROBE_GROUP=z2` switch **inside** `probe_common.py`, not a sibling module — the
whole point of that file is that the arms cannot drift, and a second copy of the
ensemble/splits/estimator layer is how a matched comparison silently stops being
one. What the switch changes: the group (`Z2`, `nc = 1`, real dtype), the cache
key and the loading path (a Z₂ configuration is *already* 3D, so one
configuration is one sample and there is no timeslice extraction), and the
target module. `R`, `LAYERS`, `LCNN_K`, `MLP_HIDDEN`, the splits, the
standardisation and the R² sufficient statistics are untouched.

| arm | M1 | M3 (vortex mask) | role |
|---|---|---|---|
| `gelt` | ✓ | ✓ | the architecture |
| `frozen` | ✗ | ✓ | M1 removed, nested (fewer parameters) |
| `frozen_matched` | ✗ | ✓ | M1 removed at matched capacity |
| `gelt_single` | ✓ | ✗ | M3 removed — in Z₂ the transport becomes the identity |
| `frozen_single` | ✗ | ✗ | the 2 × 2's fourth cell; **the one new arm** |
| `lcnn` | ✗ | ✗ | the thesis-relevant baseline |
| `lcnn_norm` | ✗ | ✗ | M2's control, rides along |

Worth saying plainly, because it makes the 2 × 2 sharper than it looks: in Z₂ a
single-path transport is ±1 and its adjoint action is `T² W = W`, so
`gelt_single` is not "GELT with a worse transport" — it is **GELT with no
transport at all**, a pure attention-over-offsets network on the raw plaquette
field. The same is true of the L-CNN's transport, which is why the `lcnn` and
`frozen_single` arms are close in spirit and the difference between them is
softmax-versus-fixed-kernel aggregation and nothing else.

Cost is the reason this is worth doing at all: `nc = 1`, 24 offsets, 27 648
sites, so a transport is 2.6 MB per configuration and a run is minutes rather
than the hours the SU(2) probe costs.

### 9.6 Pre-registered readings

Fixed here, before any training code exists, in the form §4 of
`notes/m1_probe.md` uses: median over paired `(ensemble, seed)` cells with an
exact sign test, and **six seeds** from the start — the M1 probe's own closing
recommendation is that n = 6 is the binding constraint on every count it could
not resolve.

- **W-A — calibration.** ΔR²(`gelt` − `lcnn`) and (`gelt` − `frozen`) on **V2**.
  Both must be within noise. V2 is the local classical algorithm written down as
  a target, so an architecture gap *there* means the instrument is measuring
  optimisation or capacity, not mechanism. This is the reading that would stop
  the study, and it is the role T0 earned in the M1 probe.
- **W-B — primary, M1.** ΔR²(`gelt` − `frozen`) on **V1**, and
  (`gelt` − `frozen_matched`) as the capacity control. This is the only clean M1
  reading (§1.1: a GELT-vs-L-CNN number never is).
- **W-C — primary, M3.** ΔR²(`gelt` − `gelt_single`) on **V1**. In Z₂ this is
  the vortex mask's entire contribution, isolated, with α and every parameter
  identical.
- **W-D — the thesis reading.** ΔR²(`gelt` − `lcnn`) on **V1**. What an examiner
  asks. It is *not* attributable on its own; W-B and W-C are what decompose it,
  and the 2 × 2's fourth cell is what makes the decomposition additive.
- **W-E — the classical gate.** Every trained arm against the local-BFS ceiling
  of §9.4 (`gelt.vortex_targets.local_component_size`), re-measured on the same
  configurations. An arm below it has not beaten the classical local method and
  should not be reported as an architecture result whatever W-D says. As of the
  pre-flight that bar is **R² = 0.736 ± 0.033** on vortex-carrying sites.
- **W-F — null.** Both architectures frozen at initialisation, as R-E.
- **Dispersion**, as a count and not a test: the spread over the six cells per
  arm. §1.2 point 7 is the reason it is listed separately from accuracy.

A **β ladder** is the one free axis, and it moves the task. At production
geometry the pre-flight reads, at the two couplings run so far:

| β | p_neg | largest-cluster share | R² linear | R² local | headroom |
|---|---|---|---|---|---|
| 0.7450 | 0.053 | 0.511 | +0.107 | +0.723 | +0.277 |
| 0.7585 | 0.039 | 0.384 | +0.186 | +0.736 | +0.264 |

so the gas thins and the percolating cluster's share falls as β → β_c, while the
headroom barely moves. Run the pre-flight on all four cached ensembles and fix
the two couplings in writing before any training.

### 9.7 The four criteria, and the fifth

Against §6, and against the criterion T1's withdrawal added:

1. **Per-site target** — yes, V1 is per site by construction.
2. **Sparse structures at configuration-dependent positions and scales** — yes,
   measured: `p_neg ≈ 0.03 … 0.04`, and cluster size *is* the scale. V1 is invariant
   under `f → λf` trivially, because the plaquette signs are already ±1; the
   amplitude-invariance clause is satisfied in a degenerate way rather than an
   interesting one, which is worth saying plainly.
3. **Headroom at the architecture's own reach** — yes, measured on the
   production geometry: 0.264 of R² over the strongest local classical method,
   0.814 over the M1-free linear one.
4. **Not the output of a classical local algorithm** — yes. Connected-component
   membership is global; the best local version of it reaches R² = 0.736, and
   that is the arm to beat, not the target.
5. **Not GELT's own primitive restated.** This is the criterion T1 failed
   (`notes/m1_probe.md` §7.5: softmax *is* a soft argmax, so "max over the ball"
   was a tautology) and it is the one this candidate has to answer most
   carefully. V1 is *not* a max over the ball of a local scalar: it is a global
   connected-component size, and a single soft-argmax layer does not compute it —
   what is needed is iterated routing along a line, four layers deep. The honest
   residual risk is that the *per-site reduction* in V1's definition ("the
   largest cluster touching any of the three plaquettes at `x`") is a max over
   three objects, and three is small. If a reviewer finds that circular, the
   masked-loss formulation — supervise only on vortex plaquettes, no per-site max
   — removes it at the cost of a masked head. **The pre-flight gives that
   variant a second, independent motivation**: 91% of sites carry no vortex and
   their V1 is exactly 0, so an unmasked loss spends most of its gradient on "is
   there a vortex here" (§9.4). Masked supervision is the recommended primary,
   the unmasked form the secondary.

**Criterion 4 is satisfied and that is the whole point**: this is the first
candidate in this note that is a physics observable *and* clears the rule the
topology study died on.

### 9.8 What would falsify it, and what the honest prior is

Three ways this fails, each of which the design surfaces rather than hides:

- **W-A fails.** The calibration target separates the arms, so the instrument is
  not measuring mechanism. Stop — that is what a calibration reading is for, and
  `notes/m1_probe.md` R-A is the precedent for trusting it.
- **W-D wins but W-B is null and W-C carries it.** Then the win is M3 — the
  path-averaged transport — and not attention. That is a *real* result and a
  publishable one, but it is a different sentence, and CLAUDE.md's "what
  attention buys" paragraph would have to say transport. §1.2 point 7 already
  pushes in that direction and this would settle it.
- **Everything ties.** Then the honest conclusion is that the L-CNN learns
  connectivity as well as attention selects it, and the architecture question is
  closed negative on physics tasks — which, after four attempts, is itself the
  thesis's answer and should be written as one.

**The prior, stated without a number.** M1 has paid exactly once, on a
constructed target, and the argument that it should pay here rests on an analogy
with T1 — *which was withdrawn as circular*. What is new and load-bearing is not
that analogy: it is §9.4's measured +0.550 gap between local density and local
connectivity at the architecture's own reach, which says the task's difficulty
is concentrated in routing, and
§9.3's measured fact that GELT carries a vortex gate in Z₂ that the L-CNN does
not. One caution against the session transcript this proposal came from
(`2026-09-18-fable_audit.txt`, tracked at the repo root): its argument was that
the L-CNN fails
at selection because the field is *dense*, and that fixed kernels do fine on
sparse objects. This field is sparse. The counter-argument is
that the task is not detection but comparison of extended objects, and a cluster
size is not a local template — but that is an argument, and W-D is the
measurement.

---

## 10. Consequences for `PLANS.md`

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
  in `gelt/lattice.py` (§11). Both are M1 bets and both should be run past §6's
  four criteria before any code is written.
- **§A.5 (control variates) and §A.8 (flux tube from the attention map)** are
  unaffected. §A.8 is worth noting as a different kind of win: the attention map
  *is* a measurable lattice operator and an L-CNN has none, so it is a capability
  the baseline lacks rather than a number it loses.

---

## 11. What was deleted, and where it lives

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
