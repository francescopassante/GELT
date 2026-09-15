# The M1 probe — does input-dependent offset weighting pay?

**Status (2026-09-15): built, pre-flight passed on both production ensembles,
grid not yet run.** Everything below
labelled *measured* was measured; §4's readings are pre-registered and §5's
sentence is written before the first production run, on purpose.

This is attempt 3 in the programme `notes/where_attention_can_win.md` tracks.
Attempts 1 and 2 (the 0⁺⁺ tie, the flow-free topology study) are closed there;
read §1–§6 of that note first. What follows is the experiment §7 said was
missing, aimed at the mechanism that has *never* been observed to pay.

---

## 1. Why the existing comparisons cannot answer the question

`notes/where_attention_can_win.md` §1 isolates the one architectural difference
between GELT and a matched-parameter L-CNN as *whether the weights over offsets
are input-dependent*, and splits the consequences into two mechanisms:

- **M1 — relative weighting.** α is a softmax, so it depends only on score
  *differences*: a scale-free ratio across the neighbourhood, recomputed per
  site per configuration.
- **M2 — boundedness.** α is a convex combination, weights in (0,1). L-Conv's
  weights are unbounded and an L-CB stack is a raw matrix polynomial.

**That framing is right and the experiment built on it was not.** GELT and the
L-CNN differ in *two* things, not one: input-dependent offset weights **and**
transport geometry — GELT's shortest-path-averaged L1-ball against the L-CNN's
axis-aligned link products. Every GELT-vs-L-CNN number in this repo therefore
confounds M1 with geometry, the 0⁺⁺ parity included. A tie is not evidence
against M1; a win would not be evidence for it.

> **The clean test is GELT against GELT with the softmax frozen.** Same
> transport, same bilinear value path, same RoPE geometry, same L-Act, same
> channel mix, same optimiser, same data. Only α stops depending on the input.
> It is an ablation inside one architecture, not a new model.

---

## 2. WP-A — the ablation arm

`GEMHSA(alpha_mode="frozen")` in `gelt/blocks.py`. α becomes
`softmax(alpha_logits)` over a learned `(H, n_offsets)` table, read from no
input at all. Three things about it are deliberate:

**(i) It is still a softmax.** The frozen weights are the softmax of free
logits, not free weights. If the arm used unconstrained weights it would remove
input-dependence *and* boundedness at once, and M1 would be re-confounded with
M2 — the exact defect this experiment exists to fix. As built, α stays a convex
combination over offsets in both arms, so M2 is held fixed and M1 alone varies.
`tests/test_blocks.py::test_frozen_alpha_does_not_depend_on_the_input` pins
both halves: α is identical for two different fields on the same links, is
non-negative and sums to one over the offset axis, and the softmax arm on the
same data is *not* input-independent. The contrast is the test.

**(ii) The score path is deleted, not disabled.** Q_s, K and `rope_freq` are
never allocated. An arm carrying weights that receive exactly zero gradient
would report a parameter budget it does not use, which is the opposite of what
a matched-parameter ablation needs.
`test_frozen_mode_allocates_no_score_path` pins that and that nothing is left
without a gradient.

**(iii) The value path is shared as code, not as equations.** `attend` and
`attend_frozen` both call `GEMHSA.value_path`. The ablation is only a test of
offset weighting if everything downstream of α is the same code.

Equivariance is unchanged and is the gate: each `Q_v†·Ṽ_Δ` term is equivariant
and a fixed convex combination of equivariant terms is too. The existing
`tests/test_blocks.py` suite carries the frozen arm through the same cases as
the softmax one — SU(2)/SU(3)/Z₂ oracle agreement to 1e-12 on outputs *and*
input gradients, SU(2) complex128 equivariance at both gates, Z₂ float64, and
the full-stack invariant readout.

**The M2 control** is `LConv(normalize_shifts=True)` in `gelt/lcnn.py`:
`ω = g[i,j]·ω̂[i,j,s]` with `Σ_s |ω̂| = 1`, so the aggregation over offsets is
bounded the way a softmax is while the per-channel magnitude stays free — which
is also what GELT does (α sums to one; V and the channel mix carry the scale).
The reparameterisation is the identity at initialisation, so the arm starts from
the reference distribution and only its trajectory differs
(`tests/test_lcnn.py::test_normalize_shifts_bounds_the_offset_axis_and_preserves_init`).

### The arms, and what "matched" means

| arm | block | width | real DOFs | ratio |
|---|---|---|---|---|
| `gelt` | softmax attention | d_model 16, d_qkv 6 | 15405 | 1.000 |
| `frozen` | frozen α | d_model 16, d_qkv 6 | 9257 | 0.601 |
| `frozen_matched` | frozen α | d_model 16, **d_qkv 10** | 14505 | 0.942 |
| `lcnn` | L-CB stack | K 2, c_hidden 6 | 14801 | 0.961 |
| `lcnn_norm` | L-CB, bounded ω | K 2, c_hidden 6 | 15077 | 0.979 |

`frozen` is deliberately the **nested** ablation — strictly fewer parameters,
because deleting the score path deletes weights. A tie there is the stronger
statement: GELT has 66% more parameters *and* the mechanism, and still does not
separate. `frozen_matched` is the disambiguator for the other outcome, and it
widens `d_qkv` rather than `d_model` so the residual stream, the channel mix
geometry and the layer count are all untouched — the minimum surgery that buys
back the budget.

---

## 3. WP-B — three targets, ordered by how much M1 they demand

`gelt/probe_targets.py`. All three are reductions of one gauge-invariant
scalar, the local action density `f(x) = 1 − Re Tr P̄(x)/nc`, over the L1-ball
of Manhattan radius **R = 4**, on 3D timeslices of the SU(2) glueball ensembles
already on disk. Nothing here samples.

| | reduction | what it demands |
|---|---|---|
| **T0** | `Σ_Δ c_{\|Δ\|₁} f(x+Δ)`, `c_r = (−1)^r e^{−r/2}` | nothing — it *is* a convolution. The calibration arm. |
| **T1** | `max_{\|Δ\|₁≤4} f(x+Δ)` | the winning offset varies per site — *position*. |
| **T2** | the θ = ½ enclosing radius of `f⁴`'s mass | a scale-free ratio test at variable radius — *position and scale*, amplitude-invariant. |

Against the four criteria of `notes/where_attention_can_win.md` §6:

1. **Per-site target** — yes, by construction. No zero-momentum projection.
2. **Configuration-dependent positions and scales, amplitude-invariant
   identity** — T1 has the position half, T2 both: `f → λf` leaves T2 exactly
   unchanged (`test_T2_is_scale_free_and_T0_T1_are_degree_one`). That is the
   criterion isolated, and it is why T2 is the primary target.
3. **Headroom at the architecture's own reach** — §3.1.
4. **The target is not the output of a classical local algorithm** — **this one
   is violated, knowingly.** T0/T1/T2 *are* defined by local algorithms. The
   reason it does not bite: criterion 4 exists so the network is not left
   competing in the 2% residual of a free classical method with unbounded reach
   (the topology study's death, §5 L1 of that note). Here the algorithm is
   bounded to radius 4 *by definition*, both architectures reach radius 8, and
   **no reading is of the form "the network beats classical X"** — every one is
   a difference between two arms trained on identical data at matched budget.
   The honest label for this study is a **mechanism assay, not a physics
   result**, and §6 says what that costs.

The bounded-receptive-field claim is arithmetic, not prose, and is a test
(`test_both_architectures_reach_the_whole_target_ball`): GELT at R = 2 covers
the radius-4 ball after **2** of its 4 layers, the L-CNN at K = 2 after **3**.
Neither arm is geometry-limited. If that ever fails, R-C is measuring receptive
field.

### 3.1 The pre-flight, and what it changed — *measured*

`notes/where_attention_can_win.md` §5 makes this the first gate of every A/B:
**measure the best classical method at the architecture's own reach before
building any training code.** `scripts/probe_preflight.py` is that gate, and
its instrument is the **best linear filter over the same radius-4 ball** — which
is exactly the M1-free family, one weight per offset, fixed for every site and
every configuration. Its R² is the ceiling; `1 − R²` is the headroom
input-dependent weighting has to work in. (It is a *lower* bound on what the
fixed-weight *architectures* reach, since `frozen` and the L-CNN are deep and
nonlinear — which is the right direction for a gate: a target the linear filter
saturates cannot be separated by anything downstream of it.)

**It fired on the first run.** T2 as originally designed — the enclosing radius
of `f` itself — came back at **R² = 0.978, 2% headroom**: the topology failure
again, in a new costume. The cause is not subtle once measured. The outer shell
of the ball holds 66 sites, so the cumulative profile concentrates, `r_first` is
4 at 82% of sites, the `min{r}` selection never actually *selects*, and what
survives is

    r* = 4 + (θS₄ − S₃)/(S₄ − S₃),

a ratio of two linear functionals, which linearises about its mean.

The repair is to sparsify the mass onto the largest few sites of the ball by
raising f to a power before summing shells, under a rule fixed in advance —
**the smallest p whose linear ceiling is below 0.80** — smallest because
polynomial degree is a cost the architectures pay for something that is not the
mechanism under test:

| p | 1 | 2 | 3 | **4** | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| best linear R² | .978 | .939 | .864 | **.774** | .679 | .588 | .429 |
| `r_first` split (r=3 / r=4) | .18/.82 | .31/.69 | .37/.63 | **.40/.60** | .42/.58 | .43/.57 | .45/.55 |

`SCALE_POWER = 4`: 22 points of headroom against a ΔR² resolution of order
0.01, a near-even shell split, and degree 4 — two of the four bilinear layers —
so both architectures can build it and still have two layers to aggregate with.
`f⁴` is still gauge invariant, still supported on the ball, and still exactly
scale-free.

Gates, fixed in `probe_preflight.py`: a non-calibration target needs linear
R² ≤ **0.90** and relative spread ≥ **1e−3**; T0 needs R² ≥ **0.999**.

The table above was measured locally on *hot random links* (a synthetic cache
under seed 99). The linearisation argument that forced `SCALE_POWER` is
ensemble-independent — it is about summing 66 sites — but the numbers were not
the production ones, so part 0 re-ran the gate on the real ensembles.

### 3.2 The production pre-flight — *measured 2026-09-15, V100*

β = 2.4, ξ = 3.0, L = 12, Lt = 24, 200 configs × 6 strided timeslices,
140/20/40 contiguous chain-ordered.

| target | std/\|mean\| | R² ball, ens0 | ens1 | radial only | headroom |
|---|---|---|---|---|---|
| T0 | 2.66e−1 | **+1.0000** | **+1.0000** | +1.0000 | 0.0000 |
| T1 | 1.01e−1 | +0.0785 ± 0.0040 | +0.0888 ± 0.0052 | +0.0785 / +0.0887 | **0.92 / 0.91** |
| T2 | 4.53e−2 | +0.6206 ± 0.0026 | +0.6200 ± 0.0019 | +0.6206 / +0.6200 | **0.379 / 0.380** |

**All three clear, on both ensembles.** Three readings worth keeping:

1. **T0 is exactly 1.0000 on real data**, which is the pipeline validation — it
   must be, or R-A is not a calibration and a linear R² < 1 on T0 would be a
   fact about the target rather than about the fit.
2. **The radial filter equals the full-ball filter to four decimals**, on every
   target and both ensembles. The 129-offset convolution gains *nothing* from
   directional structure — the optimal linear filter over this ball is radially
   symmetric, so the M1-free family is effectively five parameters, not 130.
   That sharpens every downstream reading: **whatever separates the arms, it is
   not which directions they weight.** It also means the linear ceiling is a
   cheap and stable object, which the 0.0019–0.0052 errors and the ens0/ens1
   agreement confirm.
3. **T2's headroom is 0.38, not the 0.23 the synthetic estimate promised** —
   the real ensemble is 0.15 *less* linear at p = 4 than hot links are.

**`SCALE_POWER` stays at 4, and that is a decision, not an oversight.** Applied
to this table, "the smallest p whose linear ceiling is below 0.80" would plausibly
have selected a lower p — the production ceilings sit ~0.15 under the synthetic
ones at matched p, so p = 2 may well clear 0.80 here. Re-deriving it now would be
re-tuning the target after seeing the production data, which is the one thing
pre-registration exists to prevent, and it would buy nothing: the gate that
governs go/no-go is R² ≤ 0.90, T2 passes it with 0.38 of headroom against a ΔR²
resolution of order 0.01, and both architectures reach degree 16. The cost of
p = 4 over p = 2 is one bilinear layer of the four, paid symmetrically by every
arm. The scan on production data is worth running *for the write-up*, after the
readings are in; it must not move p before them.

---

## 4. WP-C — the readings, fixed before the first run

Three init seeds per arm, two ensembles (the same ens0/ens1 every other claim
in this repo uses). Every ΔR² is a **correlated** blocked jackknife: both arms
lose the same configurations in every sample, so the shared ensemble
fluctuation cancels instead of being counted twice. Seeds combine by **median**,
not mean — `notes/lcnn_shootout.md` §9.2 showed a mean over seeds is meaningless
the moment one arm blows up, and the spread is reported alongside so a bimodal
set cannot hide behind its median. Ensembles combine by inverse-variance
weighting.

- **R-A — calibration.** ΔR²(`gelt` − `lcnn`) and ΔR²(`gelt` − `frozen`) on
  **T0**. Both must be within 1σ of zero. T0 is a convolution: if GELT wins it,
  the arms differ in *capacity or optimisation*, not in the mechanism, and
  **every other reading is void**.
- **R-B — primary.** ΔR²(`gelt` − `frozen`) on **T1** and **T2**. This is M1
  with everything else held fixed. **> 2σ on T2 confirms M1; within 1σ on both
  falsifies it at this depth and width.**
- **R-B′ — capacity or mechanism.** ΔR²(`gelt` − `frozen_matched`). Read **only
  if R-B favours GELT at > 2σ**, when the 66% parameter gap becomes a live
  alternative explanation. It is `probe_batch.sh` part 5 and is not in the
  default parts for that reason.
- **R-C — architecture.** ΔR²(`gelt` − `lcnn`) on T1/T2 — the thesis-relevant
  number, read *after* R-B and interpreted through it, because it carries the
  transport confound §1 is about.
- **R-D — M2 control.** ΔR²(`lcnn_norm` − `lcnn`). If bounding the offset
  weights closes a gap R-C opened, the effect was boundedness, not relative
  weighting.
- **R-E — null.** The same arms with the equivariant stack frozen at
  initialisation and only the readout head trained. Note this is *not* "no
  training": every arm's head is zero-initialised, so a wholly untrained network
  predicts exactly the training mean and scores R² = 0 by construction, which
  would measure nothing. The random-features floor is the informative null.

### 4.1 R-F — divergence, added post-hoc

**Not pre-registered.** It was added after part 1's sweep (§8) showed both
L-CNN arms blowing up at a learning rate at which no GELT-family arm did. The
statistic: a run whose *best* validation loss exceeds 10× the trivial
predictor's, which on a standardised target is an absolute threshold (predicting
the training mean scores exactly 1.0), recorded per run as `diverged` and
counted per architecture family by `probe_readings.py`.

It is confirmatory rather than exploratory only because
`notes/lcnn_shootout.md` §9.2's **3-of-9 vs 0-of-14** is the prior and M2 is the
mechanism that predicts it. It is a **count over a non-uniform mix of learning
rates, never a probability**, and it must not be quoted as one: the sweep
deliberately visits rates chosen to fail. What it can say is whether the two
families fail at the same places.

---

## 5. The pre-registered sentence

> If R-B is within 1σ on both T1 and T2, then **input-dependent offset
> weighting does not pay at matched parameters, at this depth and width, on a
> task built to demand it** — and the thesis says so, in those words, with the
> 0⁺⁺ parity and the topology stop cited as the two prior attempts.

The converse is equally usable, and is what makes this worth running: a > 2σ
R-B on T2 that survives R-A and R-B′ is the first *isolated* demonstration that
attention's offset weighting buys something, on a task whose defining property
(scale-free ratio at variable radius) then becomes the criterion for
recognising the physics task that contains it.

---

## 6. The honest limit

A negative result here does not prove M1 never pays on any physics task. It
proves it does not pay on a task built to demand it maximally, at this depth
and width — strong evidence, not a theorem. And §3's criterion-4 violation is
real: these are constructed targets, not observables, so a positive result is a
statement about a mechanism and still needs a physics task to land in.

**The other limit is optimisation.** Every reading is at a fixed epoch budget,
so in principle a ΔR² could be a difference in convergence *speed* rather than
in what an arm can represent. Three things bound that: each arm gets its own
learning rate from part 1's sweep, each run is a complete cosine schedule
annealed to zero within its own budget (so no arm is cut off mid-descent
relative to another), and the reported number is always the val-selected
checkpoint, never the last epoch. It is bounded, not eliminated — a
near-threshold R-B should be re-read at double the budget before it is believed.

What the study cannot be accused of is the failure mode of the last two
attempts. There is no free classical method with more reach to be crushed by,
because the target is defined inside the architecture's own ball; and both
outcomes are publishable before the first GPU-hour is spent.

---

## 7. Cost and how to run

Small model, 200 configs × 6 strided timeslices, ~20 epochs, 25 offsets in 3D.
The `frozen` arm is also the *cheaper* one — with no score there is no K, so it
moves one offset-expanded tensor per layer where `gelt` moves two.

`PROBE_DRY_RUN=1` counts the grid: **15 sweep runs** (part 1) and **72 grid
runs** (parts 2–4: 18 on T0, 48 on T1/T2, 6 nulls), plus **12** more if R-B
triggers part 5. At the measured mix — GELT ~10 min, `frozen` ~7, the L-CNN
arms ~3 — that is roughly **45 min + 7 h**: one night, with part 5 a second
short session only if it is earned.

```bash
PROBE_DRY_RUN=1 bash scripts/probe_batch.sh          # what would run, no GPU
PROBE_PARTS=0,1 bash scripts/probe_batch.sh          # gates + LR sweep (~1 h)
# read logs/probe_preflight_ens*.log and the sweep's val curves, then:
PROBE_PARTS=2,3,4 PROBE_LR_GELT=… PROBE_LR_FROZEN=… PROBE_LR_LCNN=… \
    bash scripts/probe_batch.sh                      # the grid (~7 h)
python scripts/probe_readings.py                     # the table, offline
```

Part 1 exists because GELT's 3e-3 was tuned for the Rayleigh loss on the
glueball task; a losing arm at another arm's learning rate is uninterpretable,
which is `notes/lcnn_shootout.md` part 1's lesson applied in advance.

Trim to T0 + T2 (`PROBE_PARTS=2` then part 3 with `T2` only) for half a night —
T2 is the primary target and T1 is the position-only supporting reading.

---

## 8. Build log

### The part-1 sweep — *measured 2026-09-15, V100*

T2 / ens0 / seed 0, 6 epochs, each a complete cosine schedule annealed to zero
at its own rate. Best validation MSE on the standardised target (the trivial
predictor scores 1.0):

| arm | 1e−2 | 3e−3 | 1e−3 |
|---|---|---|---|
| `gelt` | **0.642** | 0.732 | 0.810 |
| `frozen` | **0.798** | 0.863 | 0.904 |
| `frozen_matched` | **0.821** | 0.949 ‡ | 0.893 |
| `lcnn` | **3.2 × 10²⁰** | 0.989 | **0.712** |
| `lcnn_norm` | **6.7 × 10¹⁷** | 0.988 | **0.985** |

‡ best at epoch 1 — it got worse after the first epoch and never recovered.

Extended to 3e−2 and 3e−4 (`PROBE_SWEEP_LRS`), which is what settled it:

| arm | 3e−2 | 1e−2 | 3e−3 | 1e−3 | 3e−4 | 1e−4 | chosen |
|---|---|---|---|---|---|---|---|
| `gelt` | 1.008 ‡ | **0.642** | 0.732 | 0.810 | 0.980 | — | **1e−2**, interior |
| `frozen` | 0.901 | **0.798** | 0.863 | 0.904 | 0.981 | — | **1e−2**, interior |
| `frozen_matched` | 0.979 ‡ | **0.821** | 0.949 ‡ | 0.893 | 0.981 | — | **1e−2**, interior |
| `lcnn` | *crashed* | 3.2e20 | 0.989 | **0.712** | 0.950 | — | **1e−3**, interior |
| `lcnn_norm` | inf † | 6.7e17 | 0.988 | 0.985 | **0.951** | 0.987 | **3e−4**, interior |

† diverged outright: not one epoch improved on the initialisation. Both L-CNN
arms crashed here on the first pass, which was a defect in `train_probe.py` and
not in them — a run whose val loss is nan from epoch one never improves on
`inf`, so no checkpoint was written and the reload at the end died on a missing
file. Those are exactly R-F's most extreme points, so the crash was deleting the
evidence of divergence instead of recording it. Fixed; both cells now score
their final weights and record `diverged=True`.

**All five arms have interior optima.** 1e−2 for the three GELT-family arms,
1e−3 for `lcnn`, 3e−4 for `lcnn_norm`. One risk stands and one reading falls
out.

**The risk: `lcnn_norm` barely learns at any rate.** Its best is 0.951 against a
trivial predictor's 1.0, where the unbounded `lcnn` reaches 0.712 on the same
budget. Bounding every `(c_out, c_in)` pair's profile over offsets is a genuine
restriction and some of that gap is the arm's nature, which is the point of a
control. But **if it is still far behind at the full budget, R-D is
uninterpretable** — "the bounded arm is worse everywhere" is not evidence about
M2 — and this note says so before the number exists.

**The reading: R-F's ledger, at rates both families visited.** Five rates
(3e−2 … 3e−4) were run for every arm:

| family | runs | diverged |
|---|---|---|
| GELT (`gelt`, `frozen`, `frozen_matched`) | 15 | **0** |
| L-CNN (`lcnn`, `lcnn_norm`) | 10 | **4** |

The mix of rates is uniform across families here, which is the one condition
under which §4.1's count is more than a count. Two things sharpen it:

1. **GELT's optimum is 1e−2 — the rate at which both L-CNN arms diverge.** The
   GELT family's best operating point sits inside a region the L-CNN family
   cannot enter at all. That is a cleaner statement of M2 than an accuracy
   difference, and it is the fourth independent sighting.
2. **`normalize_shifts` did not prevent divergence, only shrank it** — 6.7e17
   against 3.2e20 at 1e−2, and both arms alike at 3e−2. Bounding the L-Conv's
   aggregation over offsets is therefore *not* what bounds an L-CNN: the L-CB
   stack's bilinear degree growth is the dominant amplifier, and it is untouched
   by anything R-D varies. This is an informative negative about the control
   itself, and it means R-F, not R-D, is where the M2 evidence in this study
   lives.

### The training-budget gate

Six epochs converges nothing and twenty may not either, so before the grid one
run per **family** — `gelt` at 1e−2, `frozen` at 1e−2, `lcnn` at 1e−3 — is taken
to **forty** epochs on T2/ens0/seed0 under a `_budget` tag and its val curve
read. Three rather than one because the tuned rates differ by an order of
magnitude between families: if `gelt` has converged at forty epochs and `lcnn`
at 1e−3 has not, then R-C is a statement about trainability rather than about
what either architecture can represent, and the same question applies to
`frozen`, whose 6-epoch val is 0.798 against `gelt`'s 0.642 — a gap that is
either the M1 signal or a convergence difference, and the curve is what tells
them apart. `PROBE_EPOCHS` is then set where the *slowest* of the three
flattens. The tag matters: an untagged 40-epoch run
would occupy the `gelt`/T2/ens0/seed0 grid cell, which part 3 would then skip,
leaving one cell at a different budget from the other 71 — a silent,
uninterpretable inhomogeneity. `PROBE_EPOCHS` for the whole grid is set from
where that curve flattens, and §3.2's linear ceiling of **0.62** is the number
it has to clear: below it, every arm is losing to a five-parameter radial
convolution and every ΔR² is a statement about optimisation.

**Three readings, and two of them stop the grid from starting.**

1. **Neither family's optimum is bracketed.** Every GELT-family arm is monotone
   in the learning rate and best at the *top* edge of the grid; both L-CNN arms
   are best at the *bottom* edge. An unbracketed optimum is the same defect part
   1 exists to prevent — "a losing arm at another arm's learning rate is
   uninterpretable" — so the sweep is extended (`PROBE_SWEEP_LRS`) with 3e−2 and
   3e−4 before anything else runs. Running all five arms at both is a few
   minutes of waste and documents where the blow-up boundary sits.
2. **Six epochs does not converge, and the grid's twenty may not either.** Every
   run but one improved right up to its final epoch, and GELT's 0.642 is
   R²_val ≈ 0.36 against a linear ceiling of **0.62** (§3.2). *If the arms land
   below a five-parameter radial convolution at twenty epochs, every ΔR² between
   them is a statement about optimisation, not about the mechanism.* One
   20-epoch GELT run on T2/ens0/seed0 at the chosen rate is the gate — it is a
   grid cell, so the grid skips it afterwards and it costs nothing. §3.2's
   ceiling is the number it has to clear.
3. **Both L-CNN arms diverged at 1e−2 and no GELT-family arm did** (0 of 3
   against 2 of 2, at the rate where each family's own optimum sits or sits just
   past). `lcnn_norm` blew up **480× less** than `lcnn` — bounding the
   aggregation over offsets bounds the damage without preventing it, which is
   what it should do, since it bounds neither the L-Bilin degree growth nor the
   head. This is M2 turning up uninvited for the fourth time. It is *not* the
   pre-registered R-D reading, which is an accuracy difference; it became R-F
   (§4.1), post-hoc and labelled.



- **2026-09-15** — WP-A landed (`alpha_mode` in `GEMHSA`/`GELT`,
  `normalize_shifts` in `LConv`/`LCB`/`LCNN`), 9 new tests in
  `tests/test_blocks.py`, 2 in `tests/test_lcnn.py`. WP-B landed
  (`gelt/probe_targets.py`, 11 tests). Pre-flight, trainer, batch driver and
  readings written and smoke-tested end to end on a synthetic ensemble.
  **The pre-flight changed the design before any training code ran** (§3.1) —
  which is the §5 rule working exactly as intended, at a cost of one afternoon
  instead of one night.
- **2026-09-15, the smoke test, and why part 1 is not optional.** All three
  arms train end to end. On the synthetic cache (14 train configs × 1
  timeslice, 25 CPU epochs, T0, **one shared LR of 1e-2**) they landed at
  R² = 0.815 (`frozen`), 0.395 (`gelt`), 0.034 (`lcnn`). Those are not
  readings — wrong ensemble, no per-arm LR, one seed, a hundredth of the data —
  but the *spread* is the point: at a single learning rate the three arms are
  scattered across almost the whole range, and R-A would have "failed" for a
  reason with nothing to do with capacity. Read part 1's val curves.
- **2026-09-15, part 0 on the V100 — all gates pass on both ensembles** (§3.2).
  T0 exactly 1.0000, T1 headroom 0.92/0.91, T2 headroom 0.379/0.380, ens0 and
  ens1 agreeing inside their errors. The full-ball and radial linear filters are
  indistinguishable, so the M1-free ceiling is a five-parameter object. Nothing
  was retuned on the strength of it.
- **2026-09-15, part 1 — the sweep does not clear the grid to start** (§8).
  Neither family's optimum was bracketed, six epochs converges nothing, and both
  L-CNN arms diverged where no GELT arm did.
- **2026-09-15, part 1 extended** (§8). Four of five arms now have interior
  optima: 1e−2 for all three GELT-family arms, 1e−3 for `lcnn`. `lcnn_norm`
  remains at an edge and barely learns at any rate, which is a live risk to
  R-D's interpretability and is written down as one. The 3e−2 crashes were a
  missing-checkpoint bug in `train_probe.py`, now fixed — it was discarding
  precisely the runs R-F counts.
- **2026-09-15, part 1 settled.** `lcnn_norm` brackets at 3e−4 (1e−3 → 0.985,
  3e−4 → 0.951, 1e−4 → 0.987); the re-run 3e−2 cells recorded as diverged
  instead of crashing. Final rates: 1e−2 / 1e−2 / 1e−2 / 1e−3 / 3e−4. R-F's
  ledger at matched rates is **0 of 15 against 4 of 10**, and GELT's optimum is
  the rate where both L-CNN arms die.
- Next: three 40-epoch `_budget` runs (`gelt`, `frozen`, `lcnn`) against §3.2's
  0.62 ceiling, then parts 2–4 at whatever budget the slowest of them needs.

### Reproducing the smoke test

The synthetic cache is hot random links, not physics, and is deleted rather
than left in `datasets/` under a name that looks like an ensemble. To rebuild
it:

```python
import torch
from gelt import SU, random_links
cfgs = torch.stack([random_links(L=12, D=4, gaugegroup=SU(2),
                                 dtype=torch.complex64, Lt=24) for _ in range(20)])
torch.save(cfgs, "datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000_seed99.pt")
```

then `PROBE_ENSEMBLE_SEED=99 PROBE_N_CONFIGS=20 PROBE_N_SLICES=3
PROBE_JACK_BLOCK=1 python scripts/probe_preflight.py`, and `train_probe.py`
with `PROBE_DEVICE=cpu PROBE_BATCH=4` on top of those.
