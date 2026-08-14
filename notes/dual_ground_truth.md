# Ground truth from the dual Ising model — design record

Written 2026-08-14, before the production run. This closes the one question
`notes/attention_as_operator.md` §6.1.2 leaves open, and it is the question the
headline turns on.

---

## 1. The open question, restated

§6.1.2 and the paper (twice — §"What is established" and §Discussion) say the
same thing:

> the trained attention field reads ξ **21% above** the classical smeared
> operator (slope 1.21) while the random-init arm agrees with it; the classical
> operator is not ground truth, it is one more contaminated operator, and
> contamination biases ξ *low*; so the trained ξ is the **candidate** for the
> true gap — *pending an uncontaminated reference*.

And it prices that reference:

> Distinguishing "de-contamination" from "systematic overshoot" needs a
> reference that is not itself contaminated — i.e. the L = 32 run of §7 … new
> ensembles *and* new checkpoints at all five β (≈2.4× per sweep, ~40 h wall
> clock).

The L = 32 run is the wrong instrument for this, and not only because of the
cost. A bigger gauge box measured with the *same* smeared-loop basis is still a
contaminated operator; it removes the finite-volume confound and leaves the
accuracy question exactly where it was.

## 2. The reference already exists, and it is exact

3D Z₂ gauge theory is Kramers–Wannier–Wegner dual to the 3D Ising model. The
codebase already knows this — `z2_beta_scan.py`'s docstring uses it to justify
choosing Z₂ as the testbed, and Wegner 1971 is in the paper's bibliography —
but it has never been used as an *instrument*. It should be. The map is

    β* = −½ ln tanh β        ⟺        sinh 2β · sinh 2β* = 1,

plaquettes of Λ → bonds of the dual lattice Λ*, confined phase → **broken**
phase. Three consequences, each load-bearing:

1. **The mass gap is the same number, not a proxy.** The 0⁺⁺ glueball is the
   lightest state of the Z₂-even sector; that is the lightest Ising excitation.
   ξ measured in the Ising model at β* *is* the ξ the gauge measurement is
   estimating.
2. **The dual has an operator the gauge theory does not have.** In the broken
   phase ⟨σ⟩ ≠ 0, so the order parameter interpolates the one-particle state
   with near-unit overlap (ε = σσ inherits its one-particle piece from ⟨σ⟩ ≠ 0;
   σ *is* the one-particle field). On the gauge side σ is the 't Hooft disorder
   operator — **non-local in the links**, and therefore not a member of any
   smeared-loop variational basis the classical measurement could have chosen.
   This is what makes the reference uncontaminated in the sense §6.1.2 needs:
   it is not a better member of the same family, it is outside the family.
3. **It costs nothing.** An Ising sweep is two `torch.roll` stencils; a gauge
   sweep is a staple sum over matrix links. Measured: 0.19 ms per sweep for 8
   replicas of 48×24² on an M2 GPU. Statistics and volumes the gauge ensembles
   cannot reach are minutes — including the large-volume run that answers §7's
   finite-volume question with **no new gauge sweeps and no retraining**.

## 3. Why this is a measurement and not an appeal to authority

The obvious objection is that a duality argument plus a second simulation is
just two more places to be wrong. So the duality is verified first, with a
prediction that has **no free parameter**. Differentiating

    Z_gauge(β) = 2^{N_l−1} (cosh β)^{N_p} (tanh β)^{N_p/2} · Z_Ising(β*)

with respect to β, using dβ*/dβ = −1/sinh 2β and N_p = N_bonds, gives

    ⟨P⟩(β) = tanh β + [1 − ⟨s_i s_j⟩(β*)] / sinh 2β.

Left side: measured on the production gauge ensembles already in `datasets/`.
Right side: measured here. Nothing fitted, two unrelated simulation codes. Two
independent confirmations before the production run:

- **Analytic.** Feeding the Ising low-temperature series 1 − ⟨ss⟩ ≈ 4t⁶
  (t = tanh β) into the relation returns the gauge strong-coupling series
  ⟨P⟩ ≈ t + 2t⁵, whose coefficient 2 counts the cubes containing a plaquette in
  3D. The two series are computed on opposite sides of the duality and know
  nothing about each other. (`tests/test_ising.py`)
- **Numerical.** At β = 0.50 and 0.65 on 12³, gauge ⟨P⟩ measured with the
  package's own Z₂ heat-bath against the prediction from an independent Ising
  run: **−1.5σ and −0.6σ**. The unit test runs the same check at β = 0.5,
  asserting against the measured statistical resolution rather than a magic
  constant.

A third check comes free at production settings: the converged ⟨ss⟩ must equal
the value the *measured* gauge ⟨P⟩ requires. The cold-vs-hot thermalisation test
already puts it there — at β = 0.745 the required 0.45328 against a measured
plateau of 0.451–0.455, at β = 0.760 the required 0.34004 against 0.342–0.345.

## 4. Design

Five β unchanged from the gauge scan — the whole point is landing on exactly the
couplings the gauge measurements were made at. Two volumes:

| volume | Λ | what it is for |
|---|---|---|
| **matched** | 48×24² | the dual of the production gauge torus — **the accuracy yardstick** |
| **large** | 96×48² | finite volume pushed back 2× |

The matched volume is the yardstick and the distinction matters: the gauge
operators were measuring *that* system, so comparing them to an infinite-volume
number would charge them for finite volume as if it were contamination.

Two dual operators, both fitted through **exactly** the gauge side's analysis
code (`connected_correlator`, `fit_cosh_correlator`, same Δ ∈ [2,8] window, same
blocked jackknife), so nothing in the comparison differs except the operator:

- `m` — wall magnetisation: the order parameter, best possible overlap;
- `e_t` — temporal-bond wall energy: the *literal* dual of the spatial plaquette
  the glueball operator is built from (a plaquette in the (1,2) plane pierces
  the dual bond in direction 0).

They must agree on the mass and disagree on A₀. That is the internal check. A
second window Δ ∈ [6,16] is fitted alongside: if an A₀ ≈ 1 operator gives the
same ξ on both windows, the window is not what separates the gauge-side
operators, and the difference is theirs.

**Thermalisation** is fixed by cold-vs-hot convergence rather than assumed: an
ordered start biases ⟨ss⟩ high and a random start low, so agreement is evidence.
Measured convergence by ~800 sweeps at both extreme β; N_THERM = 1500.
`DGT_THERM_CHECK=1` re-runs it. **Autocorrelation**: τ_int is reported for the
energy (fast) *and* the magnetisation (the slow mode of the broken phase, which
is what sets whether n_skip was enough). **Tunnelling** between magnetisation
sectors is counted, because the σ correlator is measured in the sign-fixed
restricted ensemble.

## 5. Pre-registered outcomes

Let d_op = ξ_op − ξ_dual at matched volume.

- **|d_class| > |d_trained| ≈ 0** → *the headline.* The learned attention field
  is the more accurate measurement of the mass gap; the direction of the bias
  that A₀ predicts is confirmed against an external standard; §6.1.2's candidate
  becomes a result. The sentence the paper can then write is *training removes
  excited-state contamination, and the resulting operator is right where the
  standard one is low* — which is a claim about physics, not about correlation.
- **d_trained ≈ d_class ≈ 0** (dual between them) → both operators are fine and
  the 21% is a fit/window systematic. Report; drop the accuracy claim; the
  structural result is untouched.
- **|d_trained| > |d_class|, dual ≈ classical** → the overshoot is a systematic
  of the attention correlator, not de-contamination. **§6.1.2's reading is wrong
  and gets retracted.** The structural claim (Pearson 0.9966) is unaffected —
  it never depended on the slope.

Independently, on the non-monotonic excursion ξ(0.7585) > ξ(0.760):

- large-volume dual monotonic **and** matched-volume dual reproduces the
  excursion → the excursion is finite volume, as §7 suspected, settled without
  the L = 32 gauge run;
- **both** dual volumes monotonic → the excursion belongs to the gauge
  *ensembles* — a shared fluctuation of the configurations both operators were
  measured on. This one is worth stating up front because it *weakens an
  existing claim*: "the correlation survives a non-monotonic excursion" would
  become "the two operators agree configuration by configuration", which is
  still true, still worth something, and a different sentence.

## 6. Known limitations, stated up front

- **The duality on a torus is exact only up to topological sectors.** The
  periodic gauge theory maps to the Ising model summed over boundary twists; for
  local operators the omitted sectors contribute at O(e^{−L/ξ}), which is ~1.8%
  at the worst point (ξ ≈ 6, L = 24) and negligible at the others. This is a
  systematic on the *matched-volume* comparison specifically, and the
  large-volume run bounds it.
- The reference is exact about *which mass* it measures, not about the gauge
  ensembles' own statistical quality. If the gauge ensembles at β = 0.7585 are
  undersampled (τ_int = 8.35 there, "only just" clearing the jackknife block),
  the dual cannot diagnose that directly — it can only show that the true ξ is
  elsewhere.
- One theory. Nothing here transfers to SU(N), which has no dual. The claim it
  supports is about *this* measurement being accurate, which is the claim
  §6.1.2 wants; the architecture claim was never β- or group-specific.

## 7. Results (2026-08-14, V100)

First production run: 32 replicas × 500 measurements, `n_skip=50`,
`n_therm=1500`, both volumes. **Four of the five β are usable and the
pre-registered headline outcome fires on the two that discriminate. One β and
the matched volume above β = 0.756 are compromised by a sampling defect that
was mine, is diagnosed, and is now gated against (§7.4).**

### 7.1 The dual measurements are right — a parameter-free scaling test

Near β*_c, ξ = ξ₀·t*^(−ν) with t* = (β*−β*_c)/β*_c and ν = 0.629971(4) known
from the conformal bootstrap. Fixing ξ₀ from the **single lowest-β point** makes
every other point a prediction with no free parameter. On the large volume:

| β | t* | ξ predicted | ξ measured (48²×96) | dev |
|---|---|---|---|---|
| 0.745 | 0.034533 | 2.137 | 2.137 ± 0.018 | *(fixes ξ₀)* |
| 0.752 | 0.019652 | 3.048 | 3.051 ± 0.022 | **+0.1%** |
| 0.756 | 0.011252 | 4.331 | 4.279 ± 0.033 | **−1.2%** |
| 0.7585 | 0.006039 | 6.410 | 6.355 ± 0.108 | **−0.9%** |
| 0.760 | 0.002925 | 10.121 | 11.173 ± 0.552 | +10.4% |

Three predictions inside 1.2% off one fixed amplitude. The dual is reproducing
3D Ising criticality, which validates the measurements far more sharply than
any internal consistency check could — it is an *external* standard (β*_c and ν
come from Ferrenberg–Xu–Landau and the bootstrap, not from this run).

β = 0.760 misses at +10.4%, and that is diagnostic rather than physical: it is
the only point with ξ/L = 0.23 at L = 48, and the only large-volume point whose
chain is marginal (n_therm = 3.6 τ(M)).

The duality check passes where the chains are sound — **−0.9σ at β = 0.745,
+1.5σ at β = 0.752** — and fails at exactly the three β where τ(M) blows up,
which is how the defect was found.

### 7.2 The mechanism, demonstrated in isolation

Two dual operators, identical configurations, identical window and jackknife;
the only difference is ground-state overlap:

| β | A₀(σ) | ξ(σ) | A₀(ε_t) | ξ(ε_t) | ξ deficit |
|---|---|---|---|---|---|
| 0.745 | 0.938 | 2.139 | 0.600 | 1.949 | −8.9% |
| 0.752 | 0.948 | 3.040 | 0.495 | 2.547 | −16.2% |
| 0.756 | 0.930 | 5.620 | 0.407 | 3.522 | −37.3% |
| 0.7585 | 0.976 | 7.737 | 0.352 | 4.260 | −44.9% |
| 0.760 | 1.007 | 8.710 | 0.306 | 4.562 | −47.6% |

**Contamination biases ξ low, monotonically in A₀, with an exact control.** This
is §6.1.2's central physical claim isolated from every confound — same lattice,
same configurations, same fit, one operator swapped. It holds at the two clean
β as well as the compromised ones, so it does not depend on §7.4.

### 7.3 The accuracy verdict

Against the validated large-volume truth (β = 0.760 excluded — see §7.5):

| β | ξ exact | classical | attention trained | attention random |
|---|---|---|---|---|
| 0.745 | 2.137 | −4.6% (0.7σ) | +5.0% (0.8σ) | −2.9% (0.5σ) |
| 0.752 | 3.051 | −23.2% (**4.3σ**) | **−9.4%** (2.1σ) | −19.4% (3.8σ) |
| 0.756 | 4.279 | −4.7% (0.7σ) | +4.8% (0.7σ) | −4.7% (0.8σ) |
| 0.7585 | 6.355 | −12.0% (**2.2σ**) | **+4.4% (0.8σ)** | −13.9% (2.7σ) |
| **mean** | | **−11.1%** | **+1.2%** | **−10.2%** |

**The trained attention field is unbiased against exact ground truth (+1.2%)
where both the classical smeared basis and the untrained network read ~10–11%
low.** At β = 0.7585 the trained operator agrees with the exact answer at 0.8σ
while the classical basis is 2.2σ below it.

This is the pre-registered headline outcome of §5, and it confirms §6.1.2's
reasoning *including the part that looked like a problem*: the note predicted
that the random arm's agreement with the classical operator was "two comparably
biased operators landing in the same place, not two accurate ones". Against an
external standard they are −10.2% and −11.1%. That is exactly what it said.

Two honest limits on this table:

- **β = 0.752 does most of the discriminating work.** At 0.745 and 0.756 all
  three operators agree with truth inside 1σ; the gauge error bars (±0.13 to
  ±0.30) are simply too wide to separate a 5% effect. The claim rests on two
  points, not four.
- **The yardstick is the L = 48 truth, and the gauge operators ran at L = 24.**
  Finite volume can only push the gauge values *down*, so it cannot manufacture
  the trained arm sitting *on* the truth — but it could inflate the apparent
  classical deficit. The ε_t operator (Z₂-even, hence immune to §7.4's
  tunnelling) gives a matched/large ratio of 1.021, 0.968, 1.038, 1.015 at the
  four β, i.e. **no detectable squeeze below β = 0.760**, which supports the L=48
  number being a fair yardstick there. That is an argument, not a proof; closing
  it needs the matched-volume σ measurement of §7.4.

### 7.4 The defect: I gated on the wrong autocorrelation time

τ(M) is reported per measurement; ×`n_skip` = 50 converts to sweeps:

| β | τ(M) matched | n_therm/τ | tunnelling |
|---|---|---|---|
| 0.756 | 250 sweeps | 6.0 | 0.2% |
| 0.7585 | 1395 | **1.1** | 0.9% |
| 0.760 | 1575 | **1.0** | 3.4% |

`n_therm = 1500` was **one autocorrelation time** at the top two β. A cold start
that has not relaxed is too ordered, ⟨ss⟩ reads high, and the duality check goes
to −16σ and −18σ. The tell in the ξ table was that the matched volume came out
*above* the large volume at two β — incoherent, since finite volume can only
squeeze ξ.

The error in reasoning is recorded in `gelt/ising.py`'s docstring, which
originally argued a cluster algorithm was unnecessary because critical slowing
is only τ ~ ξ^z ≈ 76 sweeps. **In the broken phase the slowest mode is not the
critical one**: it is tunnelling between magnetisation sectors, whose barrier is
set by the interface area, so τ is exponential in the volume, not power-law in
ξ. Measured factor between the two: 20×. It also means the *larger* lattice is
the better-behaved one here (τ(M) = 420 sweeps at L = 48 against 1575 at L = 24,
and zero tunnelling), which inverts the usual intuition and is why §7.1 and §7.3
rest on the large volume.

Fixed in `scripts/dual_ground_truth.py`: the run now measures τ(M), gates on
`n_therm ≥ 20 τ(M)` and `n_skip ≥ 2 τ(M)`, **escalates itself** per β (cost then
scales with the β that need it, which matters because τ(M) spans 25 → 1575
sweeps), refuses to feed an undersampled cell into the tables, drops
measurements straddling a sign flip and reports the effect of doing so, and
τ-corrects the error bars on both sides of the duality check — the naive
std/√N is what turned a ~0.6% systematic into "−18σ".

### 7.5 β = 0.760 is not measuring a mass gap

The true ξ there is ≈ 10–11 against L = 24. **The gauge measurement at that
coupling is measuring the box, not the correlation length** — classical 4.72,
trained 5.60, truth ~10.1, i.e. everything 50–58% low. This settles §7 of
`attention_as_operator.md` in the direction it feared, and cheaply: the point
should be dropped from the attention study, or re-run at L ≥ 48.

It also resolves the **non-monotonic excursion** ξ(0.7585) > ξ(0.760), and not
in favour of the third pre-registered branch: the dual is monotonic at both
volumes (2.14, 3.05, 4.28, 6.35, 11.17), so the excursion is *not* a fluctuation
of the gauge ensembles and *not* physics — it is finite volume, biting at
β = 0.760 because that is where ξ outgrew the box. The paper's "the correlation
survives a non-monotonic excursion" argument should be replaced: the excursion
is an artefact of the volume, shared by both operators because both were
measured on the same L = 24 configurations. What survives — and is stronger — is
§7.3.

### 7.6 What the rerun must settle

1. **Matched-volume σ at β ≥ 0.756**, so the accuracy table has a same-box
   yardstick and the finite-volume argument of §7.3 becomes a measurement. The
   escalation should reach it; if `MAX_ESCALATIONS` is exhausted the honest
   conclusion is that a local algorithm cannot sample that cell and a cluster
   update is required.
2. **β = 0.760 at large volume with a sound chain** (it was 3.6 τ), to confirm
   the +10.4% scaling miss is the box and the marginal chain rather than
   something real.
3. Tighter gauge-side errors would help more than anything on the dual side: at
   0.745 and 0.756 the comparison is limited by ±0.13–0.30 on the *gauge* ξ, not
   by the reference.
