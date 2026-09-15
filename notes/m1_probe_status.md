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
| `signed` | GELT, softmax dropped — weights may go negative | 15405 | sweeping |
| `signed_bounded` | same but bounded (tanh instead of softmax) | 15405 | sweeping |

The last two were added late, to answer section 4's last bullet. See section 5b.

## 4. What we have found so far

**All of section 4 is one seed on one ensemble.** It is what the grid is for.

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

## 5. What is running right now

```
PROBE_PARTS=2,3,4 PROBE_EPOCHS=40 PROBE_LR_GELT=1e-2 PROBE_LR_FROZEN=1e-2 \
  PROBE_LR_LCNN=1e-3 PROBE_LR_LCNN_NORM=3e-4 bash scripts/probe_batch.sh
```

**90 training runs, ~15 hours**, started 2026-09-15 evening. Every combination
of arm × target × 3 initialisation seeds × 2 ensembles, plus 6 control runs with
the network frozen at initialisation. It writes one small file per run to
`results/m1_probe/` and logs to `logs/`. It skips anything already done, so it
is safe to kill and restart.

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
  PROBE_LR_SIGNED=… PROBE_LR_SIGNED_BOUNDED=… bash scripts/probe_batch.sh
```

It shares no output files with the main grid, so the two can run at once. What
it answers:

- if `signed_bounded` ≈ `lcnn` (0.94), the L-CNN's win was the *sign* constraint
  and nothing else — a one-line change to GELT closes it;
- if `signed` is better than `signed_bounded` but fails more often at high
  learning rates, then boundedness is a price GELT pays for robustness, measured
  inside one architecture instead of across two.

## 6. When it finishes

```bash
python scripts/probe_readings.py
```

Offline, seconds, no GPU. Prints every reading with proper error bars. The names
(R-A … R-F) are defined in `notes/m1_probe.md` §4; the short version:

- **R-A** — the sanity check on T0. If the arms differ much *there*, the other
  readings mean less.
- **R-B** — `gelt` vs `frozen`. **The headline.** Does attention pay?
- **R-B′** — `gelt` vs `frozen_matched`. Is R-B just parameters?
- **R-C** — `gelt` vs `lcnn`. The thesis-relevant number, but confounded.
- **R-D / R-F** — the two M2 readings: accuracy, and failure rate.
- **R-G / R-H** — the side experiment: is the L-CNN's win the sign constraint,
  and what does boundedness cost inside GELT.
- **R-E** — the untrained-network floor.

## 7. Things fixed along the way — do not re-break them

1. **Early stopping is off on purpose.** The learning-rate schedule anneals to
   zero, so the last epochs are where it converges.
2. **The learning rates were tuned per arm at 40 epochs**, not at 6 — tuning at
   a short horizon picks rates that blow up at a long one.
3. **`lcnn_norm` has its own learning-rate knob**; it is not `lcnn`'s.
4. **T2 uses `f⁴`, not `f`.** With plain `f` a simple linear filter already
   scored 0.978 and the task was worthless.
5. **Runs tagged with `--run-tag` are excluded from the readings** — they are
   tuning runs, not results.

## 8. Where the detail is

`notes/m1_probe.md` — the full record, including every number above with its
error bar, the pre-registered readings, and §4.2, which explains why one
criterion had to be repaired after the fact. `notes/where_attention_can_win.md`
§1.1 is the confound that motivated the whole thing.
