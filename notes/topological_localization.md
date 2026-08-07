# Topological localization of attention — design record

The conference abstract (submitted, talk ~3 weeks out from 2026-08-02) promises
two interpretability investigations:

> *We also investigate the interpretability of the attention filters, studying
> whether attention **localizes in topologically rich regions** and whether the
> **attention range correlates with physical correlation length**.*

This note records what each clause needs, what was decided, and why. Clause 2
is **closed as not measurable on the available lattices** (§6). Clause 1 is the
active program (§1–§5).

Note that the abstract says *investigate whether*, not *demonstrate that*. A
measured negative is a deliverable; a manufactured positive is not.

---

## 1. Why this needs a different model from the glueball operator

The Run-5 variational operator (`train_glueball.py`) cannot answer clause 1,
for a structural reason rather than a practical one:

- It is a **per-timeslice 3D** operator by design — the transfer-matrix bound
  that makes the Rayleigh loss a mass estimate requires `Ō(t)` to depend on
  timeslice `t` alone (see `glueball_spectroscopy.md` § Audit).
- Topological charge density is **intrinsically 4D**: `q_x ∝ ε_{μνρσ}
  Tr[F_{μν} F_{ρσ}]` needs four distinct directions, and
  `lattice.topological_charge_density` raises on `D ≠ 4`.

So clause 1 runs on the **4D q(x) regression model** (`train_gelt.py`'s task),
not the glueball operator. The two studies share the attention-readout
machinery and nothing else.

What the glueball study *did* deliver on the localization theme is a
methodological rehearsal: `ℓ_att(x)` anticorrelates with the smeared
spatial-plaquette density at Spearman −0.564 ± 0.002 (config-level errors),
with a 5.4σ head-ablation intervention behind it. That is localization on a
physical field — but the field is **action density, not topological charge**,
and it must never be presented as the latter.

---

## 2. Ground truth: cooled q(x)

The naive plaquette charge on thin links is useless as ground truth — it is
dominated by UV fluctuation, and "topologically rich region" is not a
statement one can make about it. Cooling (`gelt/topology.py`) is APE smearing
over **all four** directions, iterated, which descends towards the nearest
classical solution: the fluctuation is stripped, the instanton content (a
local minimum of the action, not a fluctuation about one) survives.

**Cooling deliberately touches the time links.** That voids the
transfer-matrix interpretation and must never be used for spectroscopy — hence
it is a separate module from `glueball.ape_smear`, whose default stays
spatial-only. `ape_smear` gained a `directions` argument; the default is
unchanged and the spectroscopy path is byte-identical.

### Pre-flight result (`scripts/check_cooling.py`, SU(2) 8⁴ β=2.4 N=16)

| n_cool | action (β=1) | ⟨\|Q − round Q\|⟩ | ⟨\|Q\|⟩ |
|---|---|---|---|
| 0  | 9115 | 0.207 | 1.043 |
| 10 | 160  | 0.209 | 0.547 |
| 20 | 67   | 0.216 | 0.518 |
| 35 | 35   | 0.122 | 0.516 |
| 40 | 30   | 0.122 | 0.511 |

**Decision: `n_cool = 35`.** The action descends monotonically, ⟨|Q|⟩ plateaus
from ~10 steps onward (stable content, not dislocations — dislocations would
vanish), and the integer deviation bottoms out by 35.

**Q plateaus near 0.9 and 1.7, not 1 and 2.** This is the known multiplicative
renormalization of the *naive* (clover-free) charge, `Q_naive = Z · Q` with
Z < 1 drifting towards 1 as cooling smooths. It is **not** a bug and **not**
noise.

**Why Z is harmless here:** the localization statistic is a *rank* correlation
(Spearman), which is invariant under any monotonic rescaling. Z would matter
if we quoted Q as a topological charge; we do not. If a future version needs
integer-valued Q, the fix is the clover-improved charge operator, which does
not exist in the codebase.

---

## 3. Lattice and ensemble choices

| choice | value | why |
|---|---|---|
| dimension | 4D | q(x) is undefined otherwise |
| volume | 8⁴ | needs room for a lump; the `train_gelt.py` debug default L=4 (256 sites) is far too small for "region" to mean anything |
| β | 2.4 | scaling window for SU(2); the debug default β=1 is strong coupling with no topology in it |
| anisotropy | **none** (ξ=1) | anisotropy exists to resolve a temporal correlator. There is no correlator here — q(x) is a static per-site field — so the complication buys nothing and the coarse `a_s` it forces is actively harmful |
| sampler | heat-bath + OR | `mcmc_ensemble`'s registry default is **Metropolis** for SU(2); at β=2.4 in 4D its decorrelation is poor. Must be passed explicitly as `sweep_fn` |
| N | 400 | 400 × 4096 = 1.6M per-site labels, ample. The binding constraint is not statistics |
| R | 2 | R=1 gives only nearest neighbours, so ℓ_att ∈ [0,1] and the range has nowhere to go. See the R=2 saturation lesson in §5 |

### Measured costs (M2 CPU; the V100 is faster)

- sampling: 0.107 s/sweep → **3.9 min** for N=400
- cooling: 0.04 s/config/step → **8 min** for 400 configs × 35 steps
- 4D transport: **1.0 MB/config** at R=1, **5.2 MB** at R=2, 16.8 MB at R=3

The "4D transport memory wall" noted in `CLAUDE.md` is about the 12³×24
glueball lattice, **not** an 8⁴ topology lattice. Neither study is blocked by
it.

---

## 4. Training target: the circularity trade-off

Two candidates, and the choice is not obvious:

**(A) Train on naive q(x), validate against cooled |q(x)|.** *Chosen, first.*
The validation field was never a training target, so a correlation between
attention and cooled topological structure is a genuine emergent finding. Near
zero new code — it is `train_gelt.py`'s existing task.
*Risk:* naive q(x) is **quadratic in the plaquettes at x alone**, so zero
receptive field is needed and the attention may collapse to pure
self-attention (`train_gelt.py`'s own comment notes one bilinear value path
suffices). Then ℓ_att → 0 everywhere and there is nothing to localize.

**(B) Train on cooled q(x) from thin links.** Genuinely non-local — the
network must learn an effective cooling, which reaches over a neighbourhood —
so the attention is guaranteed meaningful.
*Cost:* the ground-truth field **is** the training target, so "attention sits
where the target is large" is partly circular. Blunt it by asking mechanistic
questions instead (does the attention range match the instanton size?).

**(C) Train on total Q per config.** Non-circular *and* requires range, but
400 configs means 400 labels against 1.6M for per-site targets. Too few.
Rejected.

**Plan: run (A), inspect ℓ_att, pivot to (B) if it collapses.** The collapse
check is one training run and mirrors the gate structure that worked for
clause 2 — let the cheap measurement decide before paying for the expensive
one.

### Result of (A) — 2026-08-07, checkpoint `best_gelt_topo_L8_b2.4_R2.pth`

Training reached **R² = 1.0000** in a handful of epochs (150 epochs, 2 h 07,
but `|fc2|` drifted only 0.058 → 0.048 after the first few). That is not
overfitting or a bug: `q_x` is a single matrix bilinear in the plaquettes at
`x`, GELT's value path *is* matrix-bilinear, so the network reproduced an
algebraic identity it already had the exact form for.

The readout (`scripts/topology_attention.py`, N=32 held out, seed 11):

| | ℓ_att | self-α | ablation ΔMSE |
|---|---|---|---|
| L1 h0 | 0.365 ± 0.173 | 0.774 | 0.00098 |
| L1 h1 | **1.620** ± 0.053 | 0.083 | 0.00017 |
| L2 h0 / h1 | 0.007 / 0.011 | 0.994 / 0.991 | 0.011 / 0.014 |
| L3 h0 | **0.580** ± 0.219 | 0.644 | 0.00043 |
| L3 h1 | 0.006 | 0.996 | 0.034 |
| L4 h0 | 0.006 | 0.995 | **0.241** |
| L4 h1 | 0.001 | 0.999 | **0.348** |

**Not a total collapse — three heads kept genuine range** (L1h1 puts 8% of its
mass on the site and 72% at |Δx|=2). **But range and usefulness are
anticorrelated:** the two heads that carry the task are pure self-attention
(ΔMSE 0.241 and 0.348 on a unit-variance target, i.e. R² 1.0 → ~0.65), while
the long-range head costs 0.00017 to delete — 2000× less.

**Localization: null.** Spearman(ℓ_att(x), |q_cool(x)|) = −0.011 … −0.004
with config-level errors ~0.003, against −0.564 for the glueball operator's
action-density correlation. Not a weak signal — nothing. It is *not* an
artifact of collapsed heads having no variance to correlate: L1h0 and L3h0
have site-to-site spreads of 47% and 38% of their means and still return
−0.011 and −0.008.

**Conclusion: a network given a target that needs no context uses no context,
and the context it does gather is not topological.** The ablation is what makes
this a measurement rather than an impression. This is a legitimate result for
the talk in its own right, and it converts the pivot to target (B) from a
speculation into an evidence-backed decision.

Ensemble note: 12 of 32 held-out configs had |Q| ≥ 0.5 (⟨|Q|⟩ = 0.42), with
plateaus near 0.85 and 1.65 — the Z ≈ 0.85 renormalization of §2, consistent
with true |Q| = 1 and 2. So roughly a third of configurations carry a lump at
all; see risk 2 in §7.

---

## 5. Attention readout — inherited conventions

Reuse `scripts/visualize_glueball_attention.py`'s machinery, and with it the
statistical rules that study established the hard way:

- **The independent unit is the configuration, not the site or the slice.**
  Per-site Spearman errors that divide by √N_sites understate; resample
  configurations.
- **Interventions need error bars.** The head-ablation Δloss must be a
  delete-one jackknife over configs of the *correlated* difference (ablated
  and intact scored on the same configs). Without it, the first pass read
  +0.127 for the load-bearing head where 32 configs and a jackknife give
  +0.0787 ± 0.0145 — a shift of >3σ of the final error, and four of eight
  cells spuriously negative.
- **Separate the static from the dynamic.** The config-averaged attention
  kernel is exactly what a fixed convolution could supply; the claim lives in
  the site-to-site residual around it, so report the spread beside every mean.
- **Watch for ceiling effects.** At R=2 six of eight glueball heads sat at
  ℓ_att = 2.000 ± 0.000 — saturated, hence uninformative. Check the radial
  profile before interpreting any range statistic.
- **Check the axis split.** Glueball heads turned out locked to a single
  spatial axis at 100% of sites, meaning that operator is a cubic-group scalar
  only approximately. Expect to have to check the same here.

---

## 6. Clause 2 (attention range vs correlation length): closed, not measurable

`scripts/beta_scan.py` measured the classical GEVP mass across
β ∈ {2.1, 2.3, 2.4, 2.5, 2.7} on the 12³×24 ξ=3 family (β=2.4 reproduced the
established anchor 0.333 ± 0.011, validating the scan). Converting to the
spatial correlation length `ξ_s = 1/(ξ·m·a_t)`:

| β | 2.1 | 2.3 | 2.4 | 2.5 | 2.7 |
|---|---|---|---|---|---|
| ξ_s | 0.57 | 0.83 | 1.00 | 1.10 | 1.00 |

**ξ_s never exceeds 1.1 lattice spacings.** Attention offsets are integers, so
the entire variation the range would have to track lives inside the first
offset shell. Training one operator per β would return a flat scatter whose
flatness measured the lattice spacing, not the attention.

**The box is not the problem — the spacing is.** L/ξ_s ≈ 12 correlation
lengths, which is the regime where finite-volume effects are *absent*. At ξ=3
the spatial coupling is β_s = β/ξ ≈ 0.8 (strong coupling), so `a_s` is coarse
and the whole glueball fits inside one grid cell. Plenty of cells; the object
occupies one.

**Correction on the record:** the β=2.7 turn-up in m·a_t was first attributed
to finite volume. That is wrong — with L/ξ_s ≈ 12 nothing is being squeezed.
The likely causes are (i) excited-state contamination at *fixed* Δ=2, since a_t
shrinks with β and the same Δ is a shorter physical time in which to filter
excited states, and (ii) the conversion uses the **bare** anisotropy while the
renormalized one differs and drifts with β (`validate_anisotropy.py` exists to
measure exactly this). Both make β=2.7 an upper bound on the mass, hence a
*lower* bound on ξ_s — the true window may be wider than the tabulated ×1.9,
though it would have to be far wider to change the conclusion. Settle it by
checking whether m_eff(Δ) is still falling at Δ=2 in `datasets/beta_scan.pt`.

**What reaching ξ_s ≫ 1 would take:** a finer `a_s`, via either a larger volume
at larger β (cost ∝ L³) or a milder anisotropy — since β_s = β/ξ, lowering ξ
from 3 toward 1.5 at fixed β doubles β_s and should push ξ_s past one spacing
while keeping m·a_t resolvable in time. Both are ensemble campaigns, not
analyses. Neither is attempted before the talk.

Written up in `glueball_report/glueball_spectroscopy.tex` § "Attention range
versus correlation length".

---

## 7. Open risks

1. **Attention may collapse to self-attention** under target (A) — see §4. The
   pivot to (B) is planned, not hypothetical.
2. **8⁴ at β=2.4 may be marginal for topology.** ⟨|Q|⟩ ≈ 0.5 on the pre-flight
   ensemble means many configs sit in the Q = 0 sector, which carries no lumps
   to localize on. If the localization sample is dominated by |Q| = 0
   configurations, condition the statistics on |Q| ≥ 1 — and report how many
   configurations that leaves.
3. **The naive charge is not integer-valued** (§2). Fine for rank statistics,
   not for any statement about Q itself.
4. **Nothing here is continuum physics**, exactly as for the glueball result.
   One volume, one β, coarse spacing.
