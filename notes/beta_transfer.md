# Attempt 5 — β-transfer: the axis four attempts never varied

**Status (2026-09-20): run and closed. X-B fails as pre-registered — §9.**
§§1–8 are the design and the readings as they were fixed in writing *before*
the first evaluation, which is what §5 of `notes/where_attention_can_win.md`
exists to enforce; they are left untouched. **§9 is the result.** The verbatim
output of every phase is tracked at `notes/beta_transfer_readout_2026-09-20.txt`,
because `results/z2_vortex/` is gitignored and lives only on the V100.

Read `notes/where_attention_can_win.md` §9.9 first — this study consumes its
checkpoints and inherits its caveats.

---

## 1. The design choice that made M1 invisible

Four attempts, four ties. The tempting reading is "attention has no advantage".
The record does not support that, and the reason is a property of the
*benchmarks* rather than of the architectures:

> Every A/B in this repo trains and tests **at one coupling, one volume, one
> input distribution**, with each arm at its own bracketed learning rate, and
> scores a **mean over a well-behaved test set**.

That design is deliberate and it is right for an accuracy comparison — it is
what the SU(2) straw-man audit forced. But it removes both of GELT's candidate
mechanisms from view:

- **M2 is bought back by tuning.** The matched L-CNN has no usable forward pass
  in Z₂ at four layers — 1.4e21 at initialisation, `inf` at the production
  volume, against GELT's field sitting at 1 (§9.5.1). `scripts/z2_init_gate.py`
  exists *only* because of that, and the arm then needs a learning rate an order
  of magnitude below GELT's. The comparison at each arm's own optimum is the
  fair one; what it silently prices out is the cost of finding that optimum.
- **M1 has nothing to earn.** A softmax renormalises the weighting over offsets
  **per configuration**. If the configurations never change distribution, a
  fixed kernel calibrated once on that distribution is not at a disadvantage.
  The one place M1 has been measured to pay — `notes/m1_probe.md` R-B, +0.144,
  6/6 — is a target that is *exactly invariant under `f → λf`*, i.e. a target
  that manufactures the distribution change the physics tasks never had.

So this is not a new task. It is the same task, the same checkpoints, and **the
one axis that has never been varied**.

---

## 2. The mechanism, stated so it can be wrong

At a new coupling the vortex gas thins and the cluster-size competition changes
(§9.4, measured: `p_neg` 0.054 → 0.032 and the largest cluster's share
0.587 → 0.212 across the four cached ensembles). The optimal routing moves with
it. Then:

- GELT's aggregation over offsets is a convex combination — `Σ_δ α_δ = 1` by
  construction, at every configuration and every coupling. Its *relative*
  weighting is recomputed from the input.
- The L-CNN's is a free sum over ~13 offsets (`2·D·K + 1` at D = 3, K = 2) with
  unbounded `ω` fitted once, at one coupling.

**Prediction.** Off-coupling, the L-CNN loses more than GELT, and it loses it
in two separable pieces: an *amplitude* piece (its effective gain over offsets
is miscalibrated — M2 wearing transfer's clothes) and a *shape* piece (it routes
to the wrong neighbours — M1). The two readings below are built to separate
them, because only the second is an attention claim.

**What would make the prediction plainly false.** Both arms carry a degree-16
matrix polynomial in the field, so an overall rescaling of the input hurts both;
if the shift at these couplings is mostly an overall rescaling, both arms
degrade together and nothing separates. That is the honest null, and §4's
falsification names it.

---

## 3. What is run — and what it costs

**Nothing is trained.** This is the cheapest experiment the programme has
produced: the twelve V1 checkpoints of §9.9 (6 seeds × `gelt`, `lcnn`, each at
its own bracketed rate, 240 epochs, `batch = 1`) already exist, and so do four
cached ensembles at β ∈ {0.7450, **0.7520**, 0.7560, 0.7585}, β_c = 0.7614.
The study crosses them. Forward passes only.

| | |
|---|---|
| source coupling | β₀ = 0.7520 (§9.6's primary, where the checkpoints were trained) |
| transfer couplings | 0.7450 (away from β_c), 0.7560 and 0.7585 (towards it) |
| arms | `gelt`, `lcnn` — the two W-D compares, the only two with six seeds |
| target | V1, `log(1 + largest cluster touching x)`, masked to vortex-carrying sites |
| splits | `probe_common.splits()` unchanged: the fit half of the affine reading is β′'s *train* split, the reading is β′'s *test* split |
| code | `scripts/probe_transfer.py` (one coupling per invocation), `scripts/probe_transfer_readings.py` (offline) |

**Two readings per checkpoint, and the difference between them is the point.**

- **raw** — the operator as it stands. Its output is in the units its own run
  standardised to, so it is mapped back through that run's `(μ, σ)` and scored
  against the raw target at β′.
- **affine** — the same prediction after a **two-parameter** recalibration (one
  slope, one offset) fitted on β′'s train split, available identically to both
  arms.

The affine column is the **primary** one, and deliberately the kinder one for
the arm with a scale problem: an operator that transfers the right shape and the
wrong amplitude recovers there. A GELT advantage that survives the
recalibration is about *routing*. The gap between the columns is the amplitude
piece, and it is reported separately as X-E.

**The statistic is a difference of differences.** §9.9 measured GELT's spread
over initialisations at 6.2× the L-CNN's on this task — an order of magnitude
above either arm's jackknife error. A raw comparison at β′ would drown in it.
Each seed is its own control: `Δ(β′) = R²(β′) − R²(β₀)` for that checkpoint,
and the arms are compared on their Δ. A uniform inflation cancels, the
leave-one-out repair included.

**The leave-one-out repair is applied to every row, not to the rows that look
bad.** §9.9's two pathological seeds (R² of −67 and −259 from one held-out
configuration each, with healthy validation curves) established the repair; a
rule applied uniformly to both arms is a statistic, and one applied where it
helps is not. Both columns are dumped either way.

---

## 4. Pre-registered readings

- **X-A — the anchor gate.** Evaluated at β₀, the *raw* reading recomputes a
  number the source dumps already hold: R² is invariant under a common affine
  map of prediction and target, so the two must agree to rounding. They are
  computed by different routes — the source through the standardised target,
  this one through physical units — so agreement says the splits, the mask and
  the target construction have not drifted since the checkpoints were written.
  `probe_transfer_readings.py` **refuses to print anything** if the anchor dump
  is missing or failing. Run the anchor first.
- **X-B — primary.** Median over seeds of `Δ(β′)` on the affine column, per arm;
  the contrast is `median Δ(gelt) − median Δ(lcnn)` at each β′, with an exact
  two-sided Mann–Whitney on the two sets of six. **A positive contrast at two of
  three couplings, with p < 0.05 at at least one, is the claim.**
- **X-B′ — the pre-registered sensitivity check.** X-B restricted to seeds whose
  anchor R² clears the matched-depth classical bar at β₀ — **0.404**, the lower
  end of §9.9's own W-E interval [0.404, 0.577], reproduced by the production
  pre-flight of 2026-09-20 at +0.4038 ± 0.0407. (This note first wrote 0.440,
  which is §9.4's inset from the hop-cap correction on different statistics and
  is not the number §9.9 used; corrected before any transfer number existed.)
  It exists because of the confound in §6, it is fixed here rather than chosen
  after seeing the transfer, and it is tied to a number that already existed.
- **X-C — the harsher reading.** X-B on the raw column.
- **X-D — dispersion.** Spread over initialisations of `Δ`, per arm, and the
  ratio. §9.9 found 6.2× against GELT *at* the anchor; whether transfer widens
  or narrows that is a second, independent reading of the same runs.
- **X-E — the amplitude piece.** `affine − raw` per arm per coupling. A large
  gap is an operator whose shape transfers and whose amplitude does not — M2's
  fingerprint, not M1's. Reported at the anchor too, where it is the baseline.

**Context, not a reading:** the matched-depth classical bar at each β′ (BFS
capped at 4 hops, §9.4), measured on all four ensembles 2026-09-20:

| β | linear (M1-free) | **matched depth, 4 hops** | 8 hops | unlimited |
|---|---|---|---|---|
| 0.7450 | +0.106 | **+0.450** | +0.668 | +0.747 |
| **0.7520** (anchor) | +0.152 | **+0.404** | +0.577 | +0.671 |
| 0.7560 | +0.156 | **+0.515** | +0.694 | +0.806 |
| 0.7585 | +0.280 | **+0.500** | +0.668 | +0.762 |

**The bar is higher at all three transfer couplings than at the anchor**
(0.45, 0.51, 0.50 against 0.40), so an operator that merely holds its R²
off-coupling has still lost ground against the classical method. That moves the
absolute difficulty of the task between couplings, and it moves it for both
arms — which is the reason X-B is a difference of differences and not a
comparison of levels. The readings script reads these from the pre-flight
dumps; where a dump predates the hop cap it says so rather than displaying the
uncapped number as if it were the bar, because §9.4 established that an
uncapped BFS is not a ceiling a four-layer network can be asked to clear.

**Falsification.** If X-B is consistent with zero at all three couplings, the
adaptation hypothesis is dead for this task, and the remaining route is M2 on a
**tail-dominated metric** — `notes/where_attention_can_win.md` §8, which is
designed, pre-registered and still unrun. That outcome is worth having: it
would say that what the softmax buys is not distributional robustness either,
and it would close the last cheap M1 bet this repo can make.

---

## 5. The four criteria (`where_attention_can_win.md` §6)

1. **Per-site target** — V1 is, unchanged from §9.
2. **Sparse structures at configuration-dependent positions and scales, with an
   amplitude-invariant identity** — V1 is, and the β shift is what makes the
   *distribution* of those positions and scales move. This is the criterion the
   study is built to exercise rather than merely satisfy.
3. **A bounded-receptive-field window** — inherited: both architectures cleared
   the matched-depth bar at β₀ (§9.9, W-E). It must be re-checked at each β′,
   which is what the pre-flight re-run is for.
4. **The target is not the output of a classical local algorithm** — V1 is a
   *global* connected-component size; the classical arm at matched depth is a
   bounded-hop BFS, which is a bar and not a ceiling. Unchanged from §9.

A fifth, specific to this study: **the comparison must not hand either arm a
free recalibration before the raw reading is taken.** The target is kept raw and
the source run's own `(μ, σ)` are used, which is why `probe_transfer.py` does
not call `standardize` at the evaluation coupling.

---

## 6. The confounds, named before the numbers exist

- **Regression to the mean.** Seed 5 of `gelt` scored 0.295 at the anchor where
  its siblings scored 0.55–0.61. It has room to move up that a healthy seed does
  not, and `Δ` rewards it for that. X-B′ is the stated response; the per-seed
  table is printed above every summary for the same reason, because at n = 6 the
  table is more informative than the p-value. This repo has already retracted
  one result that was one lucky seed of six (`m1_probe.md` §7.5).
- **The L-CNN's two tuned knobs.** `conv_init_scale = 0.2` and a rate of 3e-4
  were both found at β₀. Off-coupling they are frozen along with the weights,
  which is correct — they are part of the trained operator — but it means X-B
  measures *the transfer of a tuned arm*, not the transfer of an arm tuned at
  each β. The fairer-to-the-L-CNN version is phase 3, and it costs training.
- **Volume and target scale.** V1 is `log(1 + size)` and the cluster-size
  distribution changes with β, so the target's own variance moves between
  couplings. That is the stressor, not a defect — but it is why the readings are
  R² (scale-free by construction) and differences of R², never MSE.
- **One target.** V2, the exactly-linear calibration arm, has checkpoints for
  `lcnn` only (§9.9). Without it the study has no control saying the effect is
  routing rather than a generic transfer property of one architecture. That is
  phase 2 and it is the first thing to run if X-B is positive.

---

## 7. Phases, in the order they should be run

| phase | what | cost |
|---|---|---|
| **1** | X-A … X-E on the existing twelve checkpoints, four couplings | forward passes only, ~an hour |
| **1b** | `z2_vortex_preflight.py` at the three transfer couplings, for the matched-depth bar | offline, minutes, no GPU |
| **2** | the calibration control: V2 at β₀ for `gelt` (6 seeds) and `lcnn` if absent, then transfer them | ~12 short trainings |
| **3** | the M1 attribution: `frozen` at six seeds at β₀, then transfer. `gelt` − `frozen` off-coupling is the mechanism claim with the transport confound removed by ablation rather than by baseline (§1.1) | ~6 trainings |

Phase 3 is the one that would turn a positive X-B into a statement about
*attention* rather than about *GELT*. Phase 1 is what says whether phase 3 is
worth the GPU time.

---

## 8. The honest prior

Four attempts have tied, and the one mechanism observed to pay did so on a
constructed target. The prior for X-B is not high. What makes this worth an
hour of forward passes rather than another week of training is the ratio: it
consumes work already done, it varies the one axis the programme has held fixed
throughout, and both of its outcomes are publishable — a positive X-B is the
first physics-task win attention has had, and a null closes the last cheap M1
bet and hands the programme to §8's M2 experiment with the ground cleared.


---

## 9. The result — measured 2026-09-20

**X-B fails on the half that was pre-registered as the claim.** §4 fixed it as
*"a positive contrast at two of three couplings, with p < 0.05 at at least
one"*. The contrast is positive at **three** of three — +0.0067, +0.3101,
+0.0133 — and p is 0.485, 0.240, 0.240. **No p is below 0.05, so the claim is
not made.** X-C, the raw column, is the same shape and weaker: +0.0049,
+0.0045, +0.0339 at p ≥ 0.24.

The falsification clause of §4 therefore fires, and this section is written to
that clause rather than around it.

### 9.1 X-D is the one clean pre-registered finding, and it qualifies §9.9

> **The dispersion asymmetry that closed attempt 4 does not survive leaving the
> training coupling.**

| β | sd(gelt) | sd(lcnn) | ratio |
|---|---|---|---|
| **0.7520 (anchor, §9.9)** | 0.1223 | 0.0203 | **6.2× against GELT** |
| 0.7450 | 0.1128 | 0.0749 | 1.5× |
| 0.7560 | 0.2704 | 0.3480 | 1.3× (against the L-CNN) |
| 0.7585 | 0.2642 | 0.2562 | 1.0× |

GELT's spread is roughly what it was; the **L-CNN's is created by the shift**,
rising 3.7× at 0.7450 and 17× at 0.7560. `where_attention_can_win.md` §9.9
states the 6.2× as the study's one significant separation and this note's §1
repeats it. It has to be read with a qualifier now: *at the coupling both arms
were trained on*. One coupling away it is gone, and at 0.7560 it points the
other way.

### 9.2 What the per-seed tables show, and why no statistic here captures it

The means hide the structure. Off-coupling the runs are bimodal — a seed either
holds its R² to within a few hundredths or falls to ≈ 0 — and the **failures
are one seed for GELT and three for the L-CNN**:

| | cells with Δ < −0.2 | seeds affected |
|---|---|---|
| `gelt` | 3 of 18 | **1 of 6** — seed 0, at all three couplings |
| `lcnn` | 5 of 18 | **3 of 6** — seeds 1, 3, 4 |

That is the third independent appearance of the shape M2 predicts
(`lcnn_shootout.md` §9.2's 0-of-14 vs 3-of-9; `m1_probe.md` §0's all-six-above
0.81 against three-of-six below 0.25). **It is also not significant and not
pre-registered**: Fisher exact on 1/6 against 3/6 is **p = 0.545**, and §4 fixed
no collapse criterion, so the −0.2 threshold was chosen after seeing the table.

**And the study's own pre-existing criterion disagrees with it.** Counting cells
at or below the matched-depth classical bar — W-E's criterion, already in use,
printed by the reader — gives **6 of 18 for each arm**. Dead even. The two
criteria differ because GELT's below-bar cells are one *chronically mediocre*
seed (5, which scores ≈ 0.3 everywhere including the anchor, and transfers
perfectly well) plus one collapsing seed, while all six of the L-CNN's are
collapses. Which criterion is right is exactly the thing that should have been
fixed in advance and was not.

So: directional, in the predicted direction, for the third time, and worth
nothing on its own.

### 9.3 The finding neither arm can spin, and the most useful one here

> **Transfer failure is invisible at the training coupling.**

| collapsing run | its rank at the anchor |
|---|---|
| `gelt` seed 0 | **2nd of 6** |
| `lcnn` seed 4 | 3rd of 6 |
| `lcnn` seed 1 | 5th of 6 |
| `lcnn` seed 3 | 6th of 6 |

GELT's one collapsing seed is its *second best* operator at β₀, and the L-CNN's
best seed (0, 0.6467) transfers fine everywhere. Selecting on in-distribution
score — which is what every model-selection rule in this repo does, val loss
included — does not select a transferable operator. For a scan across couplings
that is a practical statement independent of which architecture wins, and it
holds for both.

### 9.4 X-B′ was aimed at the wrong confound

The pre-registered sensitivity check drops seeds whose anchor R² is below the
matched-depth bar. It removed `gelt` seed 5 — the chronically mediocre one,
which transfers fine — and kept seed 0, which is healthy at the anchor and
collapses at all three transfer couplings. The contrasts barely move
(+0.0082, +0.3121, +0.0148). Regression to the mean was the confound §6 named;
the one that actually bit is an initialisation that is fine in distribution and
not out of it, which §6 did not anticipate.

### 9.5 X-E does not isolate an amplitude piece, and the reason is instructive

`affine − raw` is +0.006/+0.011 at 0.7450, **−0.161/−0.364** at 0.7560 and
+0.100/+0.045 at 0.7585. A negative value means a recalibration fitted on the
new coupling's train split makes the *test* R² worse, which is what happens when
the prediction being recalibrated is near-garbage — the collapsed runs dominate
the mean. The reading was designed on the assumption that transfer degrades
gracefully; it does not, it degrades bimodally, and a mean over a bimodal
sample measures neither mode. **What X-E does say** is that at the anchor the
affine fit is the identity to three decimals (a ∈ [1.005, 1.029]), which is the
independent check that the physical-units mapping is right.

### 9.6 What this closes, and what it hands on

**Closed.** The adaptation hypothesis — that an input-dependent reweighting over
offsets buys accuracy when the input distribution moves — is **not supported on
this task**. Five attempts; M1 has now failed to produce a physics-task win in
every one of them, and the one place it pays remains a constructed target.

**Handed on, and this is the part worth acting on.** §4's falsification points
at `where_attention_can_win.md` §8 — M2 on a tail-dominated metric — and this
run has incidentally *validated §8's third stressor*. §8 proposes coupling
transfer as its most expensive stressor and had never run one; it is now run, on
Z₂ instead of SU(2), and it produced catastrophic failures in the predicted
direction at 8 of 36 cells. The stressor works. What it lacks is **statistics**:
1/6 against 3/6 is p = 0.545, and a failure-rate claim at that effect size needs
either more seeds or the pooling across stressors that §8 already specifies.

**What should not be run**: phases 2 and 3 of §7 as written. Phase 3 (`frozen`
at six seeds, then transferred) was there to attribute a positive X-B to
attention rather than to GELT. There is no positive X-B to attribute, and
spending ~6 trainings to decompose a null is the mistake §5's pre-flight rule
exists to prevent. Phase 2 (the V2 calibration control) is only worth running if
§8 is run and wants a matched control.

### 9.7 The honest scope

One target, one source coupling, three transfer couplings, two arms, six seeds,
and **no retraining anywhere** — every operator is frozen at the weights §9.9
selected. A version in which each arm is retuned at each coupling would answer a
different and more favourable question for both, and is not this. The cost of
the whole study was about an hour of forward passes on work already done, which
is the only reason a null of this size is worth having written down.
