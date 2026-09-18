# The M1 probe — plain-language status

**Start here.** `notes/m1_probe.md` is the full design record and is long; this
is the orientation. Written 2026-09-15, while the main grid is running.

---

## 1. What we are testing, in one paragraph

GELT's attention decides *how much to weight each neighbouring site* by looking
at the data — the softmax over offsets. The L-CNN baseline instead uses fixed
learned weights. For three attempts we have tried to find a task where that
difference pays, and failed twice. The reason we kept failing is that **GELT and
the L-CNN differ in two things at once**: the data-dependent weights *and* the
shape of the parallel transport (GELT averages over all shortest paths in a
diamond-shaped ball; the L-CNN only steps along axes). So no comparison between
them has ever isolated the attention.

**The fix: compare GELT against GELT with the softmax frozen.** Same transport,
same everything, but the offset weights stop looking at the input. That is an
ablation inside one architecture, so nothing is confounded. We call the
mechanism under test **M1** ("does data-dependent offset weighting pay?"). A
second mechanism, **M2**, is that GELT's weights are bounded (they sum to 1)
while the L-CNN's are not.

## 2. The three targets

Not physics — they are constructed per-site tasks built to demand M1 by
different amounts. All three are computed from the local action density `f(x)`
inside a ball of radius 4 around each site, which both networks can see.

| | what it is | why |
|---|---|---|
| **T0** | a fixed weighted sum over the ball | it *is* a convolution, so attention is worth nothing. **The calibration arm.** |
| **T1** | the max of `f` over the ball | which neighbour wins varies per site |
| **T2** | the radius enclosing half the "mass" of `f⁴` | a scale-free ratio test. **The primary target.** |

## 3. The five arms

| arm | what it is | params | learning rate |
|---|---|---|---|
| `gelt` | GELT as built | 15405 | 1e−2 |
| `frozen` | GELT, softmax frozen | 9257 | 1e−2 |
| `frozen_matched` | same, widened back to GELT's budget | 14505 | 1e−2 |
| `lcnn` | matched-parameter L-CNN | 14801 | 1e−3 |
| `lcnn_norm` | L-CNN with bounded offset weights | 15077 | 3e−4 |
| `signed` | GELT, softmax dropped entirely | 15405 | 3e−3 |
| `signed_bounded` | softmax → tanh (bounded, not normalised) | 15405 | 3e−2 |
| `signed_l1` | softmax → signed weights that still sum to 1 in size | 15405 | 3e−3 |
| `gelt_single` | GELT fed one-path transport | 15405 | 1e−2 |
| `gelt_projected` | GELT fed the averaged transport, put back on the group | 15405 | 1e−2 |

Arms 6–8 were added late, to answer section 4's last bullet (see 5b); the last
two later still, for the transport half of the R-C confound (see 5c).

## 4. What we have found  — **the 90-run grid finished 2026-09-16**

Six independent runs per cell (3 seeds × 2 ensembles). Headlines first:

- **Attention pays.** GELT beats the frozen-attention ablation in **all six**
  paired runs on T2, by 0.144 in the median. Restoring the ablation's missing
  parameters does not close it (5 of 6, +0.162 — same direction, larger gap, but
  on its own not statistically significant at six runs). This is the first time
  in three attempts that the mechanism has been shown to pay.
- **GELT does *not* beat the L-CNN on accuracy** — 4 of 6, statistically
  nothing. The earlier single-seed result saying the L-CNN won by 0.12 was one
  lucky initialisation and is withdrawn.
- **What GELT wins on is consistency.** Its six T2 runs land between 0.81 and
  0.93. The L-CNN's land between **0.007 and 0.965**. GELT's *worst* run beats
  the L-CNN's median, and the L-CNN's spread is eight times GELT's — same data,
  same budget, same tuned learning rate, only the starting weights differ.
- **Bounding the L-CNN's weights shrinks its spread** (0.96 → 0.65, worst run
  0.007 → 0.235) without changing its median — consistent with the robustness
  mechanism, but with six runs it is *suggestive only*, not established.
- **T1 was a bad target and is withdrawn.** GELT scores 0.996 and everything
  else 0.02–0.06, but that is circular: softmax attention *is* a soft maximum,
  so "the max over the ball" is GELT's own operation restated as a task. All
  claims rest on T2.

Two post-hoc parts ran after the grid and each changed a conclusion:

- **The softmax is worth 0.48 of R² at identical parameters** — but *not*
  because of non-negativity, and the three-way split published on 2026-09-16 is
  withdrawn. See 5b.
- **Path averaging in the transport buys no accuracy** (a null on T2, and worth
  *more* on the calibration target than on the mechanism one) — but it is where
  part of GELT's consistency advantage comes from. See 5c.

The one-page version with every number and p-value is `notes/m1_probe.md` §0;
the full working, error bars and the two estimator repairs needed before any of
it was quotable are in §7.5 of the same file.

## 4b. The earlier single-seed numbers (superseded)

**This section was written before the grid and is kept only so the change is
visible.** It is one seed on one ensemble.

- **The attention pays, a lot.** On T2: `gelt` 0.822, `frozen` 0.613 (R², higher
  is better). Gap **+0.21**.
- **And it is not just parameters.** `frozen_matched` has 94% of GELT's
  parameters and scores 0.634 — restoring 57% more parameters bought only a
  tenth of the gap. **So ~90% of the gap is the mechanism.** This is the first
  time M1 has been observed to pay in three attempts.
- **But the L-CNN beats GELT** on T2: 0.943 vs 0.822. Unexpected. About half of
  that is a *generic* advantage it also shows on T0 (the easy target where
  attention is worthless), so it is real but partly not about architecture.
- **The L-CNN is much more fragile.** It blew up at learning rates where GELT
  did not, 4 times out of 10 against 0 out of 15. And when GELT did have one
  wild epoch it recovered; the L-CNN never did.
- **Likely reason GELT loses on T2**, written down before its controls ran:
  GELT's weights are non-negative and sum to 1, so a layer can only *average*
  over neighbours, never *subtract* one from another. T2 is built from exactly
  such subtractions. If that holds, the boundedness that makes GELT robust is
  also what costs it accuracy here — a trade-off, not a defeat.

## 5. What has run — **the probe is finished**

**132 runs, 2026-09-16.** Nothing is pending. The verbatim readings behind every
number in this note are tracked at `notes/m1_probe_readout_2026-09-16.txt` —
worth knowing, because `results/m1_probe/` and `logs/` are gitignored and live
only on the V100.

| part | what | runs | state |
|---|---|---|---|
| 0, 1 | gates + LR sweep | — | done (§8 of `m1_probe.md`) |
| 2, 3, 4 | the grid: 5 arms × 3 targets × 3 seeds × 2 ensembles + 6 nulls | 90 | **done**, nothing failed |
| 6, 7 | the signed-α arms, T0 + T2, 3 seeds, ens0 only | 18 | **done** — and it withdrew a result, see 5b |
| 8, 9 | the transport arms, T0 + T2, 3 seeds, 2 ensembles | 24 | **done** — a null, plus an unplanned finding, see 5c |

The whole grid, for the record — every phase skips itself if its dump exists, so
these are re-runnable as written:

```bash
PROBE_PARTS=2,3,4 PROBE_EPOCHS=40 PROBE_LR_GELT=1e-2 PROBE_LR_FROZEN=1e-2 \
  PROBE_LR_LCNN=1e-3 PROBE_LR_LCNN_NORM=3e-4 bash scripts/probe_batch.sh
PROBE_PARTS=6 PROBE_EPOCHS=40 bash scripts/probe_batch.sh   # then read the sweep
PROBE_PARTS=7 PROBE_EPOCHS=40 PROBE_LR_SIGNED=3e-3 \
  PROBE_LR_SIGNED_BOUNDED=3e-2 PROBE_LR_SIGNED_L1=3e-3 bash scripts/probe_batch.sh
python scripts/probe_transport_gate.py                       # CPU, minutes
PROBE_PARTS=8,9 PROBE_EPOCHS=40 PROBE_LR_GELT=1e-2 bash scripts/probe_batch.sh
python scripts/probe_readings.py                             # offline, seconds
```

**If you come back to this, three cheap things would strengthen it** and none
needs new data: six seeds per ensemble instead of three (the binding constraint
on R-B′ and on *both* dispersion readings), `signed_bounded` at 1e−1 (its rate
is still unbracketed), and `PROBE_GATE_CONFIGS=64` on the GPU for the three
transports' cost ratio.

## 5b. The side experiment (second GPU, ~6 h)

GELT's weights are non-negative and sum to 1, so a layer can only *average*
neighbours — it can never subtract one from another. T2 is built from exactly
such subtractions, which is the best guess for why the L-CNN wins it. The two
new arms drop that constraint while keeping everything else, **at exactly the
same parameter count as `gelt`**, so there is no capacity argument to make.

```bash
CUDA_VISIBLE_DEVICES=1 PROBE_PARTS=6 PROBE_EPOCHS=40 bash scripts/probe_batch.sh
grep -H "best epoch\|R² =" logs/probe_sweep40_*.log     # pick the rates
CUDA_VISIBLE_DEVICES=1 PROBE_PARTS=7 PROBE_EPOCHS=40 \
  PROBE_LR_SIGNED=3e-3 PROBE_LR_SIGNED_BOUNDED=3e-2 PROBE_LR_SIGNED_L1=3e-3 \
  bash scripts/probe_batch.sh
```

It shares no output files with the main grid, so the two can run at once.

**Result (2026-09-16): the first guess was wrong, and informatively.**
`signed` scores 0.23 and `signed_bounded` 0.32, against `gelt`'s 0.82. Removing
the softmax does not recover the L-CNN's edge — it wrecks GELT. The reason
looked at first like a second job nobody had counted: the softmax **normalises**
— its weights always sum to 1, so each layer has a fixed "gain" no matter how
big the raw scores are — and dropping it makes the numbers explode (`signed`
diverges at the highest rate) or makes the layer quietly contribute nothing
(`signed_bounded` trains smoothly to a bad answer). Neither arm tested *sign*
alone; both tested "sign **and** no normalisation" at once.

**`signed_l1` is the repair, and it has now run: 0.38.** Negative weights
allowed, but still summing to 1 in absolute size — the softmax's own gain, with
only the sign freed. It scores **0.3819** against `gelt`'s **0.8221** on the
same cell. Two things follow.

- **The sign question is answered, and the answer is no.** Letting the weights
  go negative does not recover the L-CNN's edge; it costs **0.44 of R²**. The
  hypothesis that started this side experiment is dead.
- **The 0.50 was misattributed.** Putting the normalisation back recovers only
  0.06 of it. Laying the three arms out as a ladder — each step removing exactly
  one property — gives **non-negativity 0.440, normalisation 0.061,
  boundedness 0.089**. So the study's largest single number keeps its size and
  changes its name: it is what **non-negativity** is worth, not normalisation.

And it is not a capacity argument, which is what makes it interesting: the
weights a softmax can produce are a *subset* of the ones `signed_l1` can. The
arm with the strictly larger repertoire loses by 0.44. What the constraint buys
is not what the network *can* express but how easily it finds it — the same
message as the main grid's dispersion result, in another currency.

**Part 7 has now run — three seeds, and the T0 control. The headline promoted,
the ladder withdrawn.** (Full working: `notes/m1_probe.md` §7.6.)

- **Promoted.** Removing the softmax costs **0.48 of R²** at identical parameter
  count, in 3 of 3 seeds, every cell in [−0.61, −0.44]. The 0.44 was not a lucky
  seed. And it cannot be a capacity argument, because the weights a softmax can
  produce are a *subset* of the ones `signed_l1` can — so what the constraint
  buys is how easily the network finds a good answer, not what it can express.
- **Withdrawn: the ladder, and the name on the big number.** Part 7 also ran
  T0 — the target that is a plain weighted sum, where the sign of the weights
  cannot possibly matter — and the signed arm is **worse there than on T2**:
  −0.658 against −0.480. So the arms are not paying for dropping non-negativity,
  they are simply harder to train at all, and most of the damage shows up on the
  target where the mechanism is inert. "Non-negativity is worth 0.44,
  normalisation 0.061, boundedness 0.089" is **not supportable** and is
  withdrawn. The two smaller rungs are inside the noise, and one of them changed
  sign once three seeds existed.
- This was foreseen in writing before the run — §4.3 of `m1_probe.md` states the
  condition and part 7 exists to test it — which is the only reason the walk-back
  is one paragraph instead of a re-analysis.
- **`signed_bounded` collapsed in all three T0 runs** (never beat the trivial
  predictor), so anything computed through that arm is reading a collapsed run.
  Its learning rate is still unbracketed at the top of the sweep grid; the 1e−1
  run is still the first thing to do if anyone returns to this.

## 5c. The transport side experiment — **run 2026-09-16, complete**

R-C — GELT against the matched L-CNN — has never been a clean comparison,
because the two differ in *two* things: the attention (which R-B isolated) and
the **parallel transport**. GELT averages over every shortest path in a diamond
of radius 2; the L-CNN walks along axes. The averaging has been optional in the
code since the architecture was built (`mode="single"` takes one path), and
nobody had ever fed it to an arm.

Two new arms do, at **exactly `gelt`'s parameter count** — they are the same
network fed a different transport:

| arm | transport | keeps rotation symmetry | stays in the group |
|---|---|---|---|
| `gelt` | averaged over all shortest paths | yes | **no** |
| `gelt_projected` | the average, mapped back onto SU(2) | yes | yes |
| `gelt_single` | one canonical path | **no** | yes |

**What this can and cannot say.** It measures the *path averaging*, which is
half the transport difference. It does **not** make GELT's transport into the
L-CNN's: a single-path arm still reaches every diagonal offset, just along one
route. So the reading is against `gelt`, never against the L-CNN — it narrows
what the unexplained part of R-C can be, and does not resolve R-C.

Either answer is worth having. If the averaging pays, part of what looked like
"attention" in R-C is geometry. If it does not, the cheaper single-path DP is
available, and the transport is 63% of a GELT step — the 3.9× cost against the
L-CNN becomes a choice rather than a fact about the architecture.

**Run the CPU gate first**, before any GPU time — the arms are the same network
fed the same shape, so if the two transports turn out to be numerically close
the six cells would buy a tautology:

```bash
python scripts/probe_transport_gate.py          # minutes, no GPU, no sampling
PROBE_PARTS=8 PROBE_EPOCHS=40 PROBE_LR_GELT=1e-2 bash scripts/probe_batch.sh
PROBE_PARTS=9 PROBE_EPOCHS=40 PROBE_LR_GELT=1e-2 bash scripts/probe_batch.sh
```

**The gate passed by thirty times its threshold** (the two transports differ by
0.60 in relative norm over the two-path offsets, against a 0.02 bar), so the
arms were worth training.

**Result (2026-09-16, complete — all 24 runs).** Full working:
`notes/m1_probe.md` §7.7.

- **Averaging over all shortest paths buys no accuracy.** `gelt` − `gelt_single`
  is −0.037 on T2 (2 of 6 cells favour the average), `gelt` − `gelt_projected`
  +0.024 (3 of 6). Both are nothing. On this target, at this radius, the cheap
  single-path transport is as good.
- **And the calibration target settles which way to read that.** On T0 — the
  plain convolution — the average *is* ahead, 5 of 6 cells and 4 of 6. So the
  single-path arms carry a small generic handicap, and on the target built to
  need input-dependent weighting they make it back and then some. Averaging is
  worth **less** where the mechanism lives, not more. Which is the opposite of
  the reason the architecture does it.
- **That is a useful null, not a disappointment.** The transport is 63% of a
  GELT step, so the 3.9× cost against the L-CNN is a *choice* here rather than a
  fact about the architecture. It does **not** carry over to the glueball task,
  where the loop content of the average is the physics argument for it and the
  operator is the product rather than a regression target.
- **But the averaged transport is what keeps the runs trainable, and that is
  the finding.** Count the cells where an arm falls below R² = 0.70, over all
  12 it ran (2 targets × 2 ensembles × 3 seeds): `gelt` **0**, `gelt_single`
  **2**, `gelt_projected` **3**. Same network to the byte, same parameters, same
  rate — only `T` differs. The bad cells are the *same* cells in both arms
  (`(ens0, seed 2)` on T2, `(ens1, seed 0)` on T0), which is what an
  initialisation × data pathology looks like when one variant suppresses it and
  the others do not.
- **Why that matters.** Consistency is the one axis GELT has ever won on — here,
  and in the L-CNN shootout. Part of it is the **geometry**, not the attention:
  same α, same everything, and the consistency goes away. Anything in the thesis
  that reads GELT's robustness as an attention property has to say "attention
  *and* the shortest-path-averaged transport".
- **Still a count, not a test.** 0-of-12 against 2- and 3-of-12 at n = 6 cannot
  be resolved by a sign test. Six seeds would do it, and these arms need no new
  data.

## 6. Reading the results

```bash
python scripts/probe_readings.py
```

Offline, seconds, no GPU. Prints every reading with proper error bars. The
output as of the final run is tracked at
`notes/m1_probe_readout_2026-09-16.txt`, so the numbers are readable without the
V100. The names
(R-A … R-F) are defined in `notes/m1_probe.md` §4; the short version:

- **R-A** — the sanity check on T0. If the arms differ much *there*, the other
  readings mean less.
- **R-B** — `gelt` vs `frozen`. **The headline.** Does attention pay?
- **R-B′** — `gelt` vs `frozen_matched`. Is R-B just parameters?
- **R-C** — `gelt` vs `lcnn`. The thesis-relevant number, but confounded.
- **R-D / R-F** — the two M2 readings: accuracy, and failure rate.
- **R-G / R-H** — the side experiment: is the L-CNN's win the sign constraint
  (**no** — R-G is −0.48 over three seeds). R-H tried to split that number into
  normalisation and boundedness; **its rungs are withdrawn** — the T0 control
  says the arms are just harder to train (5b).
- **R-E** — the untrained-network floor.
- **R-I / R-I′** — the transport arms: does averaging over shortest paths pay
  (**no** on T2 — both nulls; slightly, on the T0 calibration target), and is it
  the averaging or the fact that the average is not a group element (neither —
  both arms behave the same way).

## 7. Things fixed along the way — do not re-break them

1. **Early stopping is off on purpose.** The learning-rate schedule anneals to
   zero, so the last epochs are where it converges.
2. **The learning rates were tuned per arm at 40 epochs**, not at 6 — tuning at
   a short horizon picks rates that blow up at a long one.
3. **`lcnn_norm` has its own learning-rate knob**; it is not `lcnn`'s.
4. **T2 uses `f⁴`, not `f`.** With plain `f` a simple linear filter already
   scored 0.978 and the task was worthless.
5. **Runs tagged with `--run-tag` are excluded from the readings** — they are
   tuning runs, not results. (So part 8's rate check does not appear in the
   table; it only exists in `logs/probe_sweep40_*.log` on the V100.)
6. **Every new arm runs T0 as well as T2.** T0 is a plain convolution, so an arm
   that is down *there* is down for reasons the probe is not measuring. Skipping
   it is what produced the retracted ladder in 5b, and running it is what caught
   it.

## 8. Where the detail is

`notes/m1_probe.md` — the full record, including every number above with its
error bar, the pre-registered readings, and §4.2, which explains why one
criterion had to be repaired after the fact. §7.5 is the grid, §7.6 part 7 and
§7.7 part 9. `notes/where_attention_can_win.md` §1.1 is the confound that
motivated the whole thing.

`notes/m1_probe_readout_2026-09-16.txt` — the verbatim `probe_readings.py`
output for all 132 runs: every run's R², the per-cell spreads, every reading
with its error bars, the residual ratios and the failure ledger. Tracked in the
repo on purpose, since the dumps it was computed from are gitignored and exist
only on the V100.
