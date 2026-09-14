# Flow-free topology — go/no-go dossier

**Purpose.** Everything measured about `notes/flow_free_topology.md`'s study, so
that a reader who was not present can decide whether to spend the compute. This
document makes no recommendation. It separates, explicitly, what was *measured*
from what was *extrapolated* from what was *assumed*, and §9 gives the command
that reproduces each number.

**Date of all measurements: 2026-09-14.** Hardware: one 32 GB V100 (the flow and
sampler numbers) and a CPU (the test suite).

---

## 1. The claim being tested

Learn `q̂(x)[U] ≈ q_t(x)`, the **gradient-flowed** topological charge density, from
**unflowed** thermalised SU(2) links, per site, in 4D — and use that task as a
controlled A/B between GELT (attention: input-dependent weights over offsets) and
a matched-parameter L-CNN (fixed weights over offsets). Everything else — inputs,
loss, splits, estimator, parameter budget — is matched by construction.

Two separable deliverables, per the design:

- **(a) the physics.** Does a flow-free learned topological density work at all,
  and how much of flowed topology is determined by a bounded neighbourhood?
  Novel whoever wins.
- **(b) the architecture.** Does input-dependent offset weighting help? A
  pre-registered A/B in which a null is publishable.

The design's own prior odds: **~50–55% for (b)**, **~90% for (a)**. Those are the
author's estimates from `notes/flow_free_topology.md` §10, not measurements.
Relevant base rate: the one previous head-to-head in this repo, the 0⁺⁺ glueball
shootout, **tied** (ΔA₀ = +0.012 ± 0.009, 1.3σ).

---

## 2. What is built, and verified to machine precision

All three work packages are written and committed. The test suite is **144 tests,
8 s on CPU**.

| component | check | measured |
|---|---|---|
| clover charge (`gelt/lattice.py`) | parity-odd site by site | **1.4e−17** |
| — its predecessor | the naive density is *not* | Q: −1.3161 → **+2.9888** under a reflection that preserves the action to all printed digits |
| Wilson flow (`gelt/flow.py`) | gauge covariance | **1.7e−15** |
| | stays on the group, no reprojection, full trajectory | **1e−13** |
| | RK3 third order (error ratio per halving) | 8.53, 7.27 |
| | Euler first order | 1.74, 1.86 |
| | **flow-time normalisation**: linearised decay vs `exp(−k̂²t)` | 0.556668 vs 0.556668; 0.135335 vs 0.135335 |
| linear-filter arm (`scripts/measure_topology.py`) | FFT application vs naive rolls | **1.4e−14** |
| | recovery of a known generating filter | ŵ: **1.3e−15**, R² = 1.000000000000 |

The normalisation check is the one that licenses `r_sm/a = √(8 t/a²)`, on which
the entire receptive-field argument rests. It is exact to six digits.

**Caveat on scope:** none of this verifies that the *task* is learnable. It
verifies that the target, the smoother and the classical baseline are what they
claim to be.

---

## 3. What was measured on real configurations

Ensembles: SU(2), 4D, heat-bath + 4×overrelaxation. Gate statistics are **16
configurations** per row; autocorrelation from **400-sweep** chains at n_skip = 1.

### 3.1 Autocorrelation — positive, and it saves compute

| row | τ_int(Q), flowed | τ_int(plaquette) | Q range | integer-sector changes / 400 sweeps |
|---|---|---|---|---|
| β=2.3 L=8 | 0.55 | 2.35 | [−2.65, +3.50] | 303 |
| β=2.4 L=12 | 0.64 | 1.35 | [−5.15, +4.11] | 322 |
| β=2.5 L=16 | **2.76** | 0.97 | [−5.08, +4.25] | 266 |

**No topological freezing at any row** — the risk the design priced in did not
materialise at these volumes. The chain changes sector on most sweeps.

Two facts worth carrying: the ordering **inverts** (the plaquette is the slower
mode at β=2.3, the topology at β=2.5), which is the topological slowdown arriving
on schedule as `a` falls; and τ_int(plaquette) would have prescribed n_skip = 2
at β=2.5, **off by a factor of three** — the reason the measurement is done on Q.
Production `n_skip = 10` (≳3·τ_int at the worst row, given τ_int's own ±1.2 error
there), halved from the provisional 20.

### 3.2 Gate 0 — the primary rung had to move

At the design's primary rung `t/a² = 2`:

| row | ⟨\|Q−[Q]\|⟩ | rms(Q) | per-config \|Q(3)−Q(1.5)\| |
|---|---|---|---|
| β=2.3 L=8 | 0.239 | 0.847 | 0.226 |
| β=2.4 L=12 | 0.232 | 1.081 | 0.250 |
| β=2.5 L=16 | 0.298 | 2.152 | 0.257 |

**⟨|Q−[Q]|⟩ = 0.250 is the null** — any distribution broad compared to the
integer spacing gives it (checked: uniform 0.251, Gaussian at rms 0.85 → 0.250,
at rms 2.15 → 0.250). So at `t/a² = 2` there is **no integer structure at any
row**. An earlier version of the gate used 0.25 as a *threshold* and returned
PASS on two rows; those verdicts were retracted.

The Z-scan separates the two possible causes:

| row | Z* at the three smoothest rungs | dev(Z*) there |
|---|---|---|
| β=2.3 L=8 | 0.86 | 0.055 → 0.020 |
| β=2.4 L=12 | 0.91 → 0.94 | 0.059 → 0.039 |
| β=2.5 L=16 | 0.92 → 0.93 | 0.068 → 0.043 |

**The integer structure is real and the charge is renormalised**, `Q_latt ≃ Z·Q`
with Z rising toward 1 as `a` falls — the continuum expectation. A Z* near 1/2 or
2 would have indicted the implementation of `q_x`; the scan was deliberately wide
and that did not happen.

Where structure sets in, applying: deviation below the null by more than its own
error, `r_sm ≤ L/2`, and Z* within `[0.6, 1.1]`:

| row | usable rungs | binding constraint |
|---|---|---|
| β=2.3 L=8 | `[2]`, marginal (Z* = 0.62, far from its converged 0.86) | the box |
| β=2.4 L=12 | `[4]` | box above, null below |
| β=2.5 L=16 | `[3, 4, 6]` | null below |

Consistent across rows: **topology resolves at `r_sm/a ≈ 4.9–5.7`, i.e.
`t/a² = 3–4`** — the same in lattice units everywhere, which vindicates the
fixed-`t/a²` design at a rung the design set too low. Primary rung moved **2 → 4**.

Integrator adequacy: `|Q(ε=0.02) − Q(ε=0.005)| ≤ 7.2e−5` at every row, five
orders of magnitude below the effects being measured.

### 3.3 Three consequences that degrade the design's controls

1. **β = 2.3 is dropped.** At L = 8 the new primary rung needs `r_sm/a = 5.66 >
   L/2 = 4` — the flow wraps the torus. Running it at L = 12 instead would make
   its physical volume ≈5× the others, a confound inside exactly the comparison
   R2 makes. **R2 (scale adaptivity) is now a two-point control** (β = 2.4, 2.5).
   Per the design's own selection rule, scale adaptivity is the *mechanism most
   likely to separate the architectures*, so the weakest-powered control is the
   one aimed at the most probable effect.
2. **The ladder's top rungs are invalid at L = 12.** `r_sm ≤ L/2` ⇒ `t/a² ≤ 4.5`
   there, so of `{1, 2, 4, 6, 8}` only `{1, 2, 4}` are usable at β = 2.4. The
   full ladder exists only at L = 16. **R3 (capacity, read across the ladder) is
   effectively a single-row reading.**
3. **The receptive-field margin is largely spent.** The network is R = 2 per
   layer × 4 layers = Manhattan 8. The design put the target at `r_sm/a = 4.0`
   inside that; it now sits at **5.66**. The task is harder for both
   architectures equally — it does not bias the A/B — but it raises the chance
   that *neither* net reaches the target, which would make R1 uninformative
   rather than a null.

---

## 4. What it costs

### 4.1 Measured: the flow (the `targets` phase)

Timed directly, RK3 step extrapolated over the 400-step ladder:

| row | s/config | h for 1200 configs | MiB/config (measured) |
|---|---|---|---|
| β=2.4 L=12 | 12.28 | **4.1** | 30 |
| β=2.5 L=16 | 38.83 | **12.9** | 95 |
| | | **17.0 total** | |

**Batching does not help.** Throughput is flat across batch 8 → 128: ms/step
scales linearly with batch to under 1% (246.2 × 16 = 3939 predicted vs 3903
measured at L=12; 774.1 × 16 = 12386 vs 12340 at L=16). The device is already
compute-bound at batch 8. The memory model is accurate to ~10% (30 measured vs 33
predicted at L=12; 95 vs 104 at L=16), so the auto chunk of 128 needs ≈12 GiB.

Note this is **10× the design's estimate** of 1–2 h for the targets.

Two levers, neither yet verified:

- **Cap the L = 12 ladder at `t/a² = 4`** (rungs 6 and 8 are box-invalid there
  anyway, §3.3): 400 → 200 steps, so 4.1 → 2.05 h. Total **15.0 h**. No loss
  beyond what the box already forbids.
- **ε = 0.02 → 0.04** halves both rows (**≈7.5 h**). RK3 error scales `h³`, so
  the Q-residual would rise from 7.2e−5 to ≈6e−4 at `t/a²=2` and ≈2e−3 at
  `t/a²=8` — still three orders below anything measured. **Unverified for the
  density field**: the step-size check so far was on the lattice-summed Q, and
  the target is the per-site `q_t(x)`, where errors do not average down. This is
  cheap to check and has not been checked.

### 4.2 Measured: the gate runs already spent

`gate0` to `t/a² = 16` on 16 configs: 1:19 (L=8), 6:36 (L=12), 20:47 (L=16).

### 4.3 Not measured

- **Ensemble sampling.** 500 thermalisation + 1200 × 10 = 12 500 sweeps per row.
  No timing was recorded. This is a real gap in the budget.
- **Training, either architecture.** The only anchor is the glueball
  configuration: **5.04 s/step** for GELT on this V100, and **1.29 s/step** for
  the matched L-CNN (GELT costs 3.9×). The topology task is 4D rather than
  per-timeslice 3D, and `l1_ball_offsets(4, 2) = 40` against the glueball's 24,
  so the transport — already 62.8% of a GELT step — grows 1.67× per site, on
  20 736 (L=12) or 65 536 (L=16) sites. A per-step figure has **not** been
  measured and the design's "~2 V100 nights for GELT trainings" is an assumption.
- **Disk**, computed not measured: configurations 3.2 GB (L=12) + 10 GB (L=16);
  targets 0.5 GB + 1.6 GB.

---

## 5. What is not known, and decides the science

1. **Is the task a convolution?** The best linear filter — the
   least-squares-optimal Manhattan-R convolution on `q_clov[U]`, solved exactly
   (not fitted), so it cannot be dismissed as an under-trained baseline — has
   **not been run**, because it needs the targets. The design states plainly: if
   a network does not clearly beat it, **stop**. This is the single most
   informative unknown and it arrives with the first `targets` run.
2. **Can any network reach the target at all?** Unknown. `r_sm/a = 5.66` against
   a Manhattan-8 reach is the well-posedness question, and it is now tighter than
   designed.
3. **R1, the architecture reading.** Prior 50–55%, base rate 0-for-1.

---

## 6. The case for running it, stated plainly

- The infrastructure is verified to machine precision at every layer that could
  silently corrupt the result: the target's parity, the smoother's covariance and
  normalisation, and the classical baseline's estimator.
- The physics question — how much of flowed topology is determined by a bounded
  neighbourhood of the unflowed field — is answered by the run **whoever wins**,
  and the literature table in the design document shows it is open (the L-CNN
  paper regresses the naive density of the *same* configuration; Matsumoto et al.
  predict the integer Q from a *partially flowed* density; neither predicts the
  flowed density field from τ = 0).
- Sampling is cheap and honest: no topological freezing, `n_skip = 10`.
- The charge renormalisation `Z(β)` came out physically correct, which is
  independent evidence that the target is what it claims to be.
- The A/B is pre-registered with readings fixed in advance, so a null is a
  result rather than a disappointment.

## 7. The case against, stated plainly

- **The compute estimate in the design was wrong by 10×** on the one component
  that has now been measured (flow: 17 h against 1–2 h), and the two largest
  remaining components — sampling and training — are still unmeasured. A budget
  with that track record should be treated as a lower bound.
- **Two of the three controls have been degraded** by constraints discovered
  after the design: R2 is now two points, R3 is effectively one row. The
  mechanism the selection rule predicts (scale adaptivity) is the one R2 tests.
- **The receptive-field margin is mostly gone** (5.66 of 8), raising the
  probability of an uninformative outcome — neither net learns the target — as
  distinct from a clean null.
- **The base rate is 0 for 1.** The previous matched A/B in this repo tied, and
  that tie extended to the noise-to-signal ratio; the only asymmetry found was
  robustness across training runs, not accuracy.
- **The primary rung is not where the design put it**, and the reason (no
  integer structure at `t/a² = 2`) was only visible after building the whole
  measurement stack. Further such discoveries are possible in WP3–WP5, which do
  not exist yet.

## 8. What would decide it cheaply, before committing the full budget

In increasing cost:

1. **Verify ε = 0.04 on the density field** (minutes). Halves the flow bill if it
   holds. Unverified today.
2. **Measure one training step** for both architectures at L = 12 and L = 16
   (minutes, once WP3 exists). This is the largest unquantified item in the
   budget.
3. **Run `targets` + `arms` at β = 2.4, L = 12 only** (≈2 h with the ladder cap,
   plus unmeasured sampling). This produces the best-linear-filter R² — the
   design's own stop condition — for one row, at ~12% of the flow budget. If the
   filter is not clearly beatable, nothing further is worth running.

Ordering (3) before the L = 16 row converts a 17-hour commitment into a 2-hour
one with the same stop/go information.

## 9. Provenance — how to re-derive every number here

```bash
pytest tests                                            # §2, 144 tests
TOPO_PHASE=chain  python scripts/measure_topology.py    # §3.1
TOPO_PHASE=gate0  python scripts/measure_topology.py    # §3.2 (this is the ~30 min part)
TOPO_PHASE=gatefit python scripts/measure_topology.py   # §3.2 Z-scan, offline, seconds
TOPO_PHASE=probe  python scripts/measure_topology.py    # §4.1, ~2 min, no sampler
TOPO_SMOKE=1 python scripts/measure_topology.py         # all phases, L=4, ~30 s, CPU
```

Design and pre-registered readings: `notes/flow_free_topology.md` (§7 build
order, §9 readings R1–R5, §12 build log with the failures). Theory:
`reports/topology/flow_free_topology.tex`. The architecture baseline this study
follows from: `notes/lcnn_shootout.md` §9.

**Known defects in this document's own instruments, already fixed, recorded
because they show the failure mode:** the gate's integrality threshold was
originally set at the null value (0.25) and returned PASS on structureless data;
the `usable` filter originally accepted two aliasing minima at Z* ≈ 0.45; and the
first `probe` timed 8 configurations regardless of the batch size, reproducing
the old estimate instead of testing it. All three produced plausible numbers
while measuring nothing.
