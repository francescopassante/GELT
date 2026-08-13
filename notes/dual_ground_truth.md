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

## 7. Results

*(filled in after the production run — `scripts/dual_ground_truth.py`, writes
`results/dual/dual_ground_truth.{png,pt}`)*
