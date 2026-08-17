# The attention map as a lattice operator — design record

Written 2026-08-10, before the run. The rescue attempt for clause 2 of the
conference abstract ("whether the attention range correlates with physical
correlation length"), which `notes/topological_localization.md` §6 and §6.1
closed **negative twice** — once in SU(2) (ξ_s ≤ 1.1 spacings, unaskable) and
once in 3D Z₂ (ξ reaches 5.3, but ℓ_att sat on its uniform value at both R=6
and R=12).

This note argues those two failures share a single cause, that the cause is a
property of the *statistic* rather than of the physics, and that the right
statistic is measurable on artifacts that already exist.

---

## 1. The diagnosis: we measured the kernel, not the field

Both attempts reduced the attention to

    ℓ_att = Σ_Δ |Δ|₁ · ᾱ(Δ),      ᾱ = attention averaged over sites and configs,

i.e. the **first radial moment of the mean kernel**. That number has two
structural defects, and §6.1 records hitting both:

- it is **bounded above by R** (the SU(2) R=2 study: six of eight heads at
  ℓ_att = 2.000 ± 0.000 — censored, not measured);
- it is **centred by the ball geometry** (the Z₂ study: ℓ_uniform =
  Σ_d(4d·d)/n_off = 4.28 at R=6 and 8.31 at R=12, and every measurement landed
  on those numbers).

Both defects come from the same place: `ᾱ` is an *average over the lattice*, so
it is a property of the learned kernel — a design quantity, of the same kind as
a convolution's filter width. Averaging is exactly the operation that throws
away the one thing attention has and convolution does not: **site-to-site,
configuration-to-configuration variation**.

The R=12 readout (`z2_attention_readout.pt`) already shows the discarded signal
is large. Offset entropy relative to its uniform ceiling, per head, at
β = 0.745: 0.87, 0.92, 0.93, 0.97, **0.42**, 0.73, 0.97, **0.72**. Several heads
are strongly structured — while their ℓ_att sits at 8.3 against a uniform 8.31.
The attention is doing something; the radial first moment is blind to it.

## 2. The claim

Write the attention weight of layer ℓ, head h at query site x as
`α^{(ℓ,h)}_{x→x+Δ}(U)`. The score that produces it is `Re Tr[Q†K̃]` with K
adjoint-transported to x, so under a gauge transformation Ω the score at x is
**invariant**, and so is α. Therefore any reduction

    A(x) = f(α_{x→·})        e.g.  α_{x→x},  Σ_Δ|Δ|α,  −Σ_Δ α log α

is a **gauge-invariant local scalar field built from the links** — a lattice
operator in the ordinary sense, with all the machinery that implies.

In particular, for the per-timeslice Z₂ operator (`train_z2_glueball.py`) the
network sees one timeslice at a time, so `A(t, x)` depends on slice `t` alone
and the zero-momentum correlator

    C_A(Δt) = ⟨ δĀ(t+Δt) δĀ(t) ⟩,    Ā(t) = Σ_x A(t,x),  δĀ = Ā − ⟨Ā⟩

has an exact transfer-matrix decomposition `Σ_n |⟨0|Â|n⟩|² e^{−m_n Δt}`. Hence

> **the attention field has a mass**, and that mass is the theory's mass gap —
> so `ξ_A = 1/m_A` should equal the ξ measured classically, with **no ceiling at
> R and no geometric offset**, because C_A is a correlator of a *fluctuation*
> (mean zero by construction) across the lattice, not a moment of a kernel.

Three consequences, in increasing order of interest:

1. **ξ_A tracks ξ(β).** The length the failed statistic was supposed to find,
   found — in the fluctuations of the attention rather than in their width.
2. **This measurement does not exist for a convolution.** A fixed kernel has
   δ ≡ 0 identically, so C_A ≡ 0. The existence of a signal *is* the
   content-dependence result, and it is quantitative (the fluctuation-to-mean
   ratio is the effect size).
3. **A₀, the ground-state overlap of the attention field**, says whether
   *training* taught the routing to look at the physics, or whether the
   equivariant construction alone is responsible. That is what the random-init
   control arm is for.

## 3. What makes it a measurement rather than a tautology

Any local gauge-invariant scalar has a correlator decaying at the gap — a
random-init network's attention field included. So `ξ_A ≈ ξ` on its own
establishes point 2 above (attention maps in an *equivariant* network are
physical fields, which is precisely `notes/explainability.md`'s premise made
quantitative) but **not** that anything was learned. The design therefore has
three arms:

| arm | what it isolates |
|---|---|
| trained network, **matched** β (diagonal of the 5×5) | the headline ξ_A vs ξ |
| trained network, **mismatched** β (off-diagonal) | content-dependence vs memorisation: ξ_A must follow the *evaluation* ensemble, not the training β |
| **random-init** network, all β | what equivariance gives before learning — the baseline for A₀ |

The off-diagonal arm also kills the confound that closed §6.1: *"with operator
quality varying from 1.18× to 2.10× the classical mass — and degrading along
the very axis being plotted — the five points are not comparable."* One fixed
network evaluated across five ensembles has no such gradient. Operator quality
is a property of the row; the physics being probed is a property of the column.

## 4. Statistical conventions (inherited, deliberately)

Mirrors `scripts/z2_beta_scan.py` exactly so the two numbers are comparable:
GEVP-project the multi-channel basis onto its ground-state vector (`t0=1`,
`td=2`), cosh-fit the projected correlator over Δ ∈ [2, 8], and take errors from
a **blocked** jackknife over configurations with block 20 (τ_int reaches 6.3
near β_c, so single deletion would understate). The classical smeared-plaquette
basis is re-measured **on the same configurations**, so ξ_A − ξ_class can be
quoted as a correlated difference rather than two independent numbers.

From `notes/topological_localization.md` §5, still binding:

- the independent unit is the **configuration**;
- **quote every statistic against its null** — the mistake that produced two
  false starts. Here the nulls are: the uniform value for ℓ_att (already
  established), C_A ≡ 0 for a static kernel, and an explicit
  time-shuffled arm that must return no mass.

## 5. Data — all of it already exists

Five Z₂ ensembles at 24²×48, N=2000, `n_skip=200`, with classical
ξ = 2.05 → 5.28 (`datasets/z2_beta_scan.pt`), and five trained checkpoints
`best_z2_glueball_b<β>.pth` at R=12. Training used only `configs[:400]`, so
**configs[400:] are unseen by every checkpoint** — the evaluation slice is
clean for all 25 cells of the matrix, including the diagonal.

No training, no sampling. The cost is forward passes, and the expensive part
per configuration (APE smearing + the R=12 transport) is shared across all six
networks.

## 6. What each outcome means

- **ξ_A tracks ξ across the column, flat across the row** → clause 2 delivered,
  with a methodological lesson (read the field, not the kernel) that explains
  both prior negatives.
- **ξ_A tracks ξ but the random arm does too, with equal A₀** → the honest
  result is "equivariant attention maps are physical fields"; the *learning*
  claim is not supported and must not be made.
- **No signal at all (C_A consistent with 0)** → the learned attention is
  effectively a fixed convolution on this task. That is a strong negative for
  the architecture's premise and worth reporting plainly; §6.1's entropy
  numbers make it unlikely but it is the honest null.

## 6.1 Result (2026-08-11) — the attention field has a mass

`ZAC_R=6 ZAC_N_USE=800 ZAC_CROSS=0 ZAC_N_EVAL=1200`, i.e. all 1200
configurations no checkpoint has seen, on the retrained R=6 operators
(`z2_attention_rows_R6.pt`, all five passing the corrected gate). ~4 min per
ensemble. Numbers below are the **best single attention channel**
(`attention_single`); see the defects in §6.2 for why the multi-channel GEVP arm
is not the one quoted.

Classical column below is the **repaired** reference (§6.2 defect 1): measured on
the same 1200 configurations, with the GEVP falling back to its best single
smearing level at every β. The N=2000 scan's numbers are kept beside it because
they disagree at the top two β, and that disagreement is itself a result.

| β | ξ classical (same configs) | **ξ_A trained** | ξ_A random init | ξ_out | ξ scan (N=2000, superseded) |
|---|---|---|---|---|---|
| 0.745 | 2.04 ± 0.14 | 2.24 ± 0.13 | 2.08 | 2.33 | 2.05 ± 0.16 |
| 0.752 | 2.34 ± 0.16 | 2.76 ± 0.13 | 2.46 | 2.92 | 2.61 ± 0.31 |
| 0.756 | 4.08 ± 0.27 | 4.48 ± 0.30 | 4.08 | 4.63 | 4.14 ± 0.26 |
| 0.7585 | **5.59 ± 0.34** | **6.64 ± 0.32** | 5.47 | 6.65 | 4.59 ± 1.31 |
| 0.760 | **4.71 ± 0.34** | **5.60 ± 0.37** | 4.80 | 5.81 | 5.28 ± 0.82 |

**Pearson(ξ_A, ξ_classical) = 0.9966**, slope 1.21, over ×2.7 of dynamic range,
at 5–7% precision on both sides. The claim of §2 holds: the attention map is a
lattice operator and its correlator decays with the mass gap.

**The correlation survives a non-monotonic excursion.** ξ(0.7585) > ξ(0.760) in
the classical measurement *and* in the attention, on the same configurations,
from independent operators. This is much stronger evidence than tracking a
smooth trend would be — two quantities that both rise with β correlate for
trivial reasons; two that both turn around at the same place do not. The old
scan's opposite ordering came from its two worst-determined points measured with
the broken GEVP, and is superseded.

The excursion is a property of the **ensembles**, not of the attention. Both top
points sit at ξ ≳ L/4 = 6, the scan's own finite-volume guard, so the natural
reading is that β=0.760 is squeezed hardest. Settling it means L = 32, not more
statistics — see §7.

**Training separates from initialisation at every β.** ΔA₀ is a blocked
jackknife of the *difference* on shared configurations (both arms see identical
configs, so the naive √(σ²+σ²) discards the cancellation — the same argument
that turned a naive +0.127 head ablation into +0.0787 ± 0.0145 in the glueball
study):

| β | δA/A trained | random | ratio | **ΔA₀ (correlated)** | Δξ |
|---|---|---|---|---|---|
| 0.745 | 0.091 | 0.0067 | ×13 | +0.144 ± 0.017 (**8.4σ**) | +0.17 ± 0.04 |
| 0.752 | 0.140 | 0.0060 | ×23 | +0.110 ± 0.020 (**5.4σ**) | +0.31 ± 0.07 |
| 0.756 | 0.246 | 0.0054 | ×46 | +0.204 ± 0.012 (**16.9σ**) | +0.41 ± 0.13 |
| 0.7585 | 0.113 | 0.0049 | ×23 | +0.267 ± 0.011 (**23.5σ**) | +0.27 ± 0.20 |
| 0.760 | 0.130 | 0.0046 | ×28 | +0.205 ± 0.012 (**16.9σ**) | +0.80 ± 0.19 |

Δξ > 0 everywhere: the trained attention field is also *less* contaminated by
excited states than the untrained one, not merely better normalised.

τ_int of the selected channel along the chain: 0.63, 1.13, 2.65, **8.35**, 5.85.
The jackknife block of 20 clears 2·τ_int at every β, but only just at β=0.7585.

The random network's site-to-site fluctuation is ≈0.005 independently of β;
training raises it by one to one-and-a-half orders of magnitude.

**The showpiece is β = 0.760**, the point closest to β_c. Correlator profiles
C(Δ)/C(0):

```
classical smeared basis : +1.000 +0.007 +0.009 -0.002 +0.006 …   dead
trained attention field : +1.000 +0.496 +0.394 +0.329 +0.276 …   clean exponential
random attention field  : +1.000 -0.002 +0.006 -0.005 +0.004 …   noise
shuffled null           : +1.000 -0.004 +0.004 +0.005 +0.002 …   flat
```

The attention field measures the correlation length at a β where the classical
operator has no signal and an untrained network has none either.

### The caveat that fixes the framing

**The random-init arm also tracks ξ** (Pearson 0.94). This is exactly what §3
predicted — any local gauge-invariant scalar decays at the gap — and it is not
a disappointment, it is the correct partition of the two claims:

- *structural*: attention maps in a gauge-**equivariant** network are bona fide
  lattice operators, with a mass, an overlap and a spectral decomposition.
  Established by ξ_A ≈ ξ, and true already at initialisation;
- *learning*: training makes them **good** operators. Established by A₀
  (+0.11 to +0.27, up to 14.7σ) and δA/A (×13 to ×46), not by ξ_A.

"The network discovers the correlation length" is **not** a supportable claim
and must not be made.

Note also that the trained arm is *not* the one closer to the classical ξ — the
random arm is. That is expected once A₀ is read as excited-state contamination;
see §6.1.2, which also says why "closer to classical" is the wrong accuracy
target here.

Known confound on δA/A: α is a softmax output, so a network with larger score
magnitudes has a sharper — hence more variable — attention regardless of what
it learned. δA/A is therefore descriptive; the load-bearing learning statistic
is A₀, which a sharp-but-untrained attention would not raise (it would build a
UV-dominated field with *poor* ground-state overlap). An entropy-matched random
arm would close this properly.

## 6.1.1 The cross-β control matrix (2026-08-11)

`ZAC_CROSS=1 ZAC_N_EVAL=600`, five trained networks plus the random arm on all
five ensembles. One cell (`train@0.756` on β=0.752) returned A₀ = 5.76 with
ξ = 0.39 — a fit that ran to its grid edge — and is masked throughout; the
`A0_MAX` guard now reports such cells as unresolved.

**The null passes.** With `_scramble_configs`, 26 of 30 cells do not fit at all
(nothing to fit), and the four that do give |A₀| ≤ 0.0066.

**ξ_A is 11.9× more sensitive to the ensemble than to the training β:** spread
within a row (across evaluation ensembles) 1.405, spread within a column
(across training β) 0.118. The column means correlate with the classical ξ at
**Pearson 0.9983**. The attention reads the configuration in front of it, and a
fixed kernel has no analogue of this measurement at all.

**The transfer test: no memorisation.** If a network had learned its own
ensemble rather than physics, the diagonal would dominate its column. It does
not:

| β | A₀ diagonal | A₀ off-diagonal | advantage | A₀ random |
|---|---|---|---|---|
| 0.745 | 0.855 | 0.770 | +0.085 | 0.723 |
| 0.752 | 0.774 | 0.815 | **−0.041** | 0.656 |
| 0.756 | 0.697 | 0.687 | +0.010 | 0.484 |
| 0.7585 | 0.764 | 0.696 | +0.069 | 0.495 |
| 0.760 | 0.660 | 0.640 | +0.020 | 0.423 |

Mean diagonal advantage **+0.029**, inconsistent in sign, against a mean
trained − random of **+0.167**: the learning effect is **6× the β-matching
effect**. Every trained network beats the random arm on every ensemble
(worst-trained 0.747 / 0.774 / 0.636 / 0.659 / 0.571 against random
0.723 / 0.656 / 0.484 / 0.495 / 0.423).

What the matrix *does* separate is network quality: ensemble-averaged A₀ per row
is 0.734 / 0.677 / 0.702 / **0.775** / 0.726 against 0.556 for random.
`train@0.7585` is the best operator on three of five ensembles including ones it
never saw — a globally better operator, not a β-specialist.

**So the learning claim is about transferable physics.** The routing did not
memorise a correlation length; it learned something that works at every ξ in the
scan.

## 6.1.2 Why the trained ξ_A overshoots — and why "closer to classical" is the wrong target (2026-08-10)

The natural inference from §6.1 is: the trained attention has the better
ground-state overlap, so its ξ_A should sit *closer* to the classical ξ than the
random arm's does. **The data say the opposite**, and it is worth recording why,
because the wrong reading here would invert the conclusion.

| β | ξ classical | ξ_A trained | ξ_A random | |trained − class| | |random − class| |
|---|---|---|---|---|---|
| 0.745 | 2.04 | 2.24 | 2.08 | 0.20 | **0.04** |
| 0.752 | 2.34 | 2.76 | 2.46 | 0.42 | **0.12** |
| 0.756 | 4.08 | 4.48 | 4.08 | 0.40 | **0.00** |
| 0.7585 | 5.59 | 6.64 | 5.47 | 1.05 | **0.12** |
| 0.760 | 4.71 | 5.60 | 4.80 | 0.89 | **0.09** |

The *random* arm is the one that agrees with the classical operator, at every β.
The trained arm overshoots systematically — that is the slope 1.21 of the
Pearson fit, and the Δξ column of §6.1 (+0.17 … +0.80) restated against the
classical reference instead of against the random one.

This is not a contradiction, because **the classical operator is not ground
truth.** It is one more operator with its own excited-state contamination, and
contamination has a *signed* effect: excited states add faster-decaying
exponentials, so a contaminated correlator decays faster than the true gap and
reads ξ **low**. A₀ measures how much of an operator's own C(Δ) is the ground
state — so higher A₀ means less of that downward bias, and the operator with the
highest A₀ should read the *largest* ξ. It does. The ordering
ξ_trained > ξ_random ≈ ξ_classical is exactly the ordering the A₀ column
predicts, once you stop treating "classical" as the target.

So the natural reading of the whole result is:

- the classical smeared basis is badly contaminated at these β — at β = 0.760 it
  has essentially no signal at all (C(1)/C(0) ≈ 0.007, the "dead" profile in
  §6.1);
- the random attention field is contaminated to a *similar* degree, and its
  agreement with the classical ξ is two comparably-biased operators landing in
  the same place, not two accurate ones;
- training removes contamination — that is what A₀ = +0.11 … +0.27 *is* — and
  the trained operator's larger ξ is therefore the candidate for the value
  closest to the true gap.

**Stated as a candidate, not as a result.** We have no independent ground truth
for ξ at these β: the N=2000 scan is superseded by the GEVP defect (§6.2), and
both top points sit at ξ ≳ L/4 where finite volume bites (§7). A 21% systematic
in the attention correlator fit would look identical. Distinguishing
"de-contamination" from "systematic overshoot" needs a reference that is not
itself contaminated — i.e. the L = 32 run of §7, or a classical basis good
enough at β = 0.760 to have a signal at all.

Consequences for how this is written up:

- Do **not** quote |ξ_A − ξ_classical| as an accuracy metric, and do not present
  the random arm's smaller residual as the random arm being *better*. Agreement
  with a contaminated reference is not accuracy.
- The load-bearing statements stay the two of §6.1: the *correlation* (Pearson
  0.9966, through the non-monotonic excursion) for the structural claim, and
  A₀ for the learning claim. The slope is a separate, open quantity.
- "Learning = removing excited-state contamination" is the physically honest
  gloss of the A₀ result, and it is stronger than the correlation framing: it
  says what the training bought in operator terms, which is exactly what a
  variational-operator audience asks for.

## 6.2 Defects, and how each was resolved

1. **The GEVP was selecting a near-null direction — at every β.** First
   suspicion was our own `_prune` cutting the four nested APE levels to two;
   fixing that (prune only above the cap) moved the classical masses by
   **4×10⁻¹⁵**. That non-result *is* the diagnosis: m, A₀ and C(Δ)/C(0) are all
   invariant under rescaling the projected operator, so identical values mean
   the same direction was selected either way. Basis size was never the issue.

   The real cause is `gevp_ground_vector`'s eigenvalue floor of `1e-12`
   relative to the largest — twelve orders of magnitude, i.e. no regularisation.
   A nested-smearing C(t0) has three or four meaningful orders, and a small
   signal divided by a floored noise eigenvalue produces a spuriously large
   generalized eigenvalue pointing into noise: a near-cancelling combination
   whose C(0) is almost pure noise while C(Δ>0) still carries the physical
   decay. Symptom: C(1)/C(0) = 0.039 / 0.020 / 0.007 at the three largest β with
   36% / 87% / unresolved errors — *correct masses, useless error bars*.

   Fixed with `GEVP_EPS = 1e-4` plus a **variational self-check**: v₀ maximises
   the Rayleigh quotient over the span of the basis, so the projection can never
   interpolate worse than any single member; when it does, fall back to that
   member. The decision is taken once on the full sample and held fixed across
   jackknife replicas. Result: all five β resolve at 6–7%, C(1)/C(0) = 0.39–0.44
   uniformly, and **the fallback fires at every β** — the four-level variational
   basis never beats its own best member here. `z2_beta_scan.py` ran the same
   pathological GEVP and survived on N=2000 statistics; its top two points are
   the casualties.

2. **The multi-channel attention GEVP is unstable** — three NaN cells and one
   absurd one (ξ = 4.85 ± 89, A₀ = 0.048 ± 0.852) out of ten. Now moot: with the
   self-check it falls back to the single best channel at every β, so
   `attention` and `attention_single` coincide and the tables quote one number.
   Informative in itself — the 24 attention channels are redundant enough that a
   variational combination never helps.

3. **ΔA₀ errors were uncorrelated.** Fixed with `_corr_delta`, a blocked
   jackknife of the difference on shared configurations. The two weak points
   went from 2.4σ to 8.4σ and 5.4σ; nothing else moved.

4. **β = 0.7585 breaks monotonicity.** Resolved as *not* a defect of the
   measurement: the repaired classical operator reproduces the same excursion on
   the same configurations. See §6.1.

5. **A time-shuffle is not a null.** The null's A₀ rose from 0.005–0.014 to
   0.043–0.230 when the GEVP fallback landed. First suspicion was selection bias
   (the null re-selecting its own best-of-six; that bias is real and measures
   7.4× on pure noise, and it was removed by reusing the channel index chosen on
   the real data). It was not the cause: removing it changed the null by
   **0.000**.

   The correct explanation is structural. A permutation of timeslices preserves
   each configuration's mean over t, and `connected_correlator` subtracts only
   the **global** mean, so the residual retains

       Var_config(Ō_c) = C(0) · (2ξ − 1)/Nt

   as a *flat pedestal at every Δ* — which is exactly what the profiles show
   (+1.000 +0.236 +0.229 +0.231 +0.234 … at β = 0.7585: no decay at all). The
   measured plateaux 0.062 / 0.079 / 0.138 / 0.231 / 0.159 track (2ξ−1)/Nt at
   **Pearson 0.98**, including the non-monotonic excursion. The "null" was
   measuring the finite-Nt zero mode — physics, not noise — and could never have
   gone to zero.

   Fixed with `_scramble_configs`: each timeslice is drawn from an
   independently permuted configuration, which destroys the temporal
   correlation *and* the zero mode together. Validated on a synthetic operator
   of known mass (ξ = 5.56 injected): the real arm returns ξ = 5.65 with
   A₀ = 1.000, the time-shuffle plateau lands at 0.213 against a predicted
   0.215, and the config-scramble returns ±0.001 with no fittable mass.

   The time-shuffle is retained as a **zero-mode consistency check** — its
   plateau must equal (2ξ−1)/Nt — which is now a second, independent way the
   data confirm ξ. ξ, A₀ and ΔA₀ were never affected: they all come from the
   best-single arm on unshuffled data.

## 7. Known limitations, stated up front

Two of these were resolved by the dual ground truth on 2026-08-14; they are
struck rather than deleted so the reasoning that produced them stays traceable.
See §8.

- Z₂ in 3D, one volume, one lattice family — a statement about the *method*,
  not about SU(N), exactly as in §6.1. **Still stands.**
- The R=12 operators are of uneven quality (`m_net/m_class` 1.18 → 2.10). The
  cross-evaluation design makes this a nuisance parameter rather than a
  confound, but the diagonal alone would still inherit it. **Still stands**
  (the quoted results are the retrained R=6 set).
- ~~The effective exponent from ξ ~ (β_c − β)^(−ν) on these five points is
  **≈ 0.39**, not the 3D Ising ν = 0.63: ξ ≤ 5.3 at L = 24 is not the
  asymptotic scaling regime. The exponent may be quoted only as a
  *consistency* check between the attention and the classical scan on the same
  points — never as a measurement of ν. With the ξ(0.7585) > ξ(0.760) excursion
  now confirmed on both operators, it should probably not be quoted at all
  until the volume question is settled.~~ **Superseded — see §8.1.** The
  diagnosis was wrong: it was not "not the asymptotic regime", it was one
  finite-volume point at the end of the lever arm. Drop it and the fit returns
  0.622 / 0.656 against the dual's 0.626.
- ~~**Finite volume at the top two β is the open physics question.** ξ = 5.6 and
  4.7 (classical) against L/4 = 6 puts both points at or past
  `z2_beta_scan.py`'s own guard, and the non-monotonicity is what a
  volume-squeezed pair looks like. The test is L = 24 → 32 — but GELT is built
  per-L, so that means new ensembles *and* new checkpoints at all five β
  (≈2.4× per sweep, ~40 h wall clock), not one point. Worth it only if the
  excursion is to be reported as physics rather than flagged as unresolved.~~
  **Resolved — see §8.** The test was not L = 24 → 32 and cost no gauge sweeps
  at all: the dual settled it, β = 0.760 was outside the box (true ξ ≈ 10, not
  the 4.7 the contaminated operator reported — which is *why* the L/4 guard
  passed a point it should have rejected), and β = 0.7585 shows no detectable
  squeeze in the tunnelling-immune ε_t operator.

## 8. Amendment (2026-08-14): β = 0.760 is dropped

The dual Ising ground truth (`notes/dual_ground_truth.md`) settled §7's open
questions, and the answer is that **β = 0.7600 was never measuring a mass gap**:
its true ξ ≈ 10 against L = 24, so classical (4.72), trained (5.60) and random
(4.80) are all 50–58% low together, measuring the box. It is dropped from
`BETAS` in `z2_attention_correlator.py` and `z2_beta_scan.py`, and the operator
trained on that ensemble goes with it (`net_names` iterates `BETAS`, so the
`train@0.76` row of the transfer matrix disappears too — correct, since its
training data was the compromised ensemble).

Recomputed offline from the saved dumps (`ZAC_REPLOT`; no per-β number changes,
the ensembles were always measured independently):

| statistic | five β (published) | four β |
|---|---|---|
| Pearson(ξ_A, ξ_classical) | 0.9966 | **0.9965** |
| Pearson(ξ_A, **exact truth**) | 0.7440 | **0.9946** |
| dynamic range in ξ | ×2.96 | **×2.96** |
| monotonic in β | no | **yes** |
| mean deviation from truth | −9.0% | **+1.2%** |
| cross-matrix ensemble/training ratio | 11.9× | **12.3×** |

**Nothing is lost.** β = 0.7585 already carried the largest clean ξ (true value
6.36, not the 5.59 the contaminated classical operator reported), so the range
is unchanged. What is gained is that ξ_A now tracks the *exact* correlation
length at 0.9946 rather than tracking a contaminated operator at 0.9966 — a
better sentence, and one that no longer needs the excursion to carry it.

### 8.1 §7's ν caveat was β = 0.760, and it is withdrawn

§7 said the effective exponent from these points is "≈ 0.39, not the 3D Ising
ν = 0.63", and the paper's Limitations repeats it as 0.35–0.48. **That was one
bad point at the end of the lever arm.** The same weighted log-log fit, on the
same code, with β = 0.760 removed:

| arm | ν (five β) | ν (four β) |
|---|---|---|
| classical | 0.426 | **0.622** |
| attention, trained | 0.480 | **0.656** |
| attention, random | 0.420 | 0.592 |
| **dual ground truth** | 0.631 | **0.626** |

The dual arm gives 0.631 either way — it was never fooled, because at L = 48 it
can hold ξ = 11 — which is what identifies the finite volume as the cause rather
than "not the asymptotic regime".

Quote this as *agreement with the dual arm fitted identically on the same four
couplings* (0.622 and 0.656 against 0.626), **not** as a measurement of ν: four
couplings, a diagonally-weighted line, and corrections to scaling at t* ≈ 0.035
are all real. But the standing claim that the scan cannot see 3D Ising scaling
is false and should be removed from the paper.

### 8.2 What the paper must change

1. Drop β = 0.760 from every table and figure of §"The attention map as a
   lattice operator".
2. **Delete the non-monotonic-excursion argument.** The dual is monotonic at
   both volumes; the excursion was finite volume, inherited by both operators
   because both were measured on the same L = 24 configurations. §7.3 of
   `dual_ground_truth.md` replaces it and is a stronger argument.
3. Replace the ν limitation with §8.1.
4. Add the accuracy result: against exact ground truth the trained attention
   field is unbiased (+1.2%) where the classical basis and the untrained network
   read ~11% and ~10% low.
5. `attention_as_operator.md` §6.1's headline framing survives unchanged; only
   the fifth row of its table goes.


## 9. Transporting the result to SU(2) (queued 2026-08-16)

`scripts/su2_attention_correlator.py` runs the §6.1 measurement on anisotropic
SU(2) in 3+1D — i.e. reproduces the paper's Table 5 (`tab:z2`) on the group the
thesis is actually about. Read this before touching that script.

### 9.1 Why SU(2) is measurable after all

`topological_localization.md` §6 closed the *range* question on SU(2) with
"ξ_s ≤ 1.1 spatial spacings, smaller than the integer grid the attention lives
on". That is the correct verdict for ℓ_att and the wrong number for ξ_A. The
attention field is a **per-timeslice** operator and its correlator runs in
**time**, on a lattice deliberately made fine in that direction (a_t = a_s/ξ,
ξ_bare = 3) precisely so the 0⁺⁺ mass would resolve. `results/attention/beta_scan.pt`
records m·a_t = 0.581 → 0.303 over β ∈ [2.1, 2.7], i.e.

    ξ_t = 1/(m·a_t) = 1.72, 2.49, 3.00, 3.31, 3.01   at 3–8% precision.

So there is a genuine correlator with a **×1.92** dynamic range. It is smaller
than Z₂'s ×2.74 and that should be said when the Pearson is quoted — five points
over a factor of two is a weaker lever arm than five points over a factor of
three, and two of the five (β = 2.4 and 2.7) are degenerate within errors, so
the fit really has four distinct lengths.

The §6.1 ceiling argument does not apply here: C_A correlates a *fluctuation*
about the mean, so nothing bounds ξ_A by R the way the ball geometry bounded
ℓ_att. R = 2 in the SU(2) operator is not an obstruction to measuring ξ_t ≈ 3.

### 9.2 β = 2.7 is kept, and it is not the Z₂ β = 0.760 mistake

β = 2.7 is **non-monotonic**: it reads the same mass as β = 2.4. §8 dropped Z₂'s
β = 0.760 for looking similar, so the distinction matters. There the cause was
diagnosed exactly: true ξ ≈ 10 against L = 24, i.e. the operator was measuring
the box. Here ξ_s ≈ 1.0 against L = 12 is **L/ξ_s = 12** — the box is enormous
compared to the correlation length, and the same is true at every coupling in
the scan. The non-monotonicity is coarse-a_s strong coupling plus a bare
anisotropy whose renormalisation drifts with β (the caveat `CLAUDE.md` already
records: β_s = β/ξ = 0.8 is not continuum physics), not finite volume.

That makes it worth keeping rather than merely harmless. ξ_A is compared against
the classical ξ measured **on the same configurations**, so monotonicity in β is
irrelevant to the correlation, and a point where the two operators must agree
that ξ went *down* when β went *up* is a stronger test than a monotone one. This
is the argument §8.2 required the paper to delete for Z₂; SU(2) can carry it
legitimately, because here the finite-volume alternative is excluded by
L/ξ_s = 12 rather than merely unmeasured.

### 9.3 Scope: one row, at β = 2.4

The default run is a **single coupling**. Z₂ had one checkpoint per β; SU(2) has
exactly one trained coupling, and a fresh operator costs ≈15 h of V100 (47 min
sampling + ~14 h training, `logs/overnight.log`), so a five-row table is a
three-day batch.

β = 2.4 is the choice because it is that trained coupling, which makes the row a
genuine **diagonal** cell — network and ensemble at the same β, as every row of
Table 5 is. Any other β would be off-diagonal, and would additionally be
answering a question (does ξ_A follow the ensemble rather than the training β?)
that only the full matrix makes worth asking. β = 2.4 is also the anchor of the
§6.2 result and carries ξ_t = 3.00, near the top of the range this lattice
family reaches.

A single row delivers everything Table 5 puts in a row — ξ_class, ξ_A trained,
ξ_A random, the three A₀, the correlated ΔA₀ — plus the detail the paper prints
beside the table: the C(Δ)/C(0) profile of every arm, the time-shuffle zero-mode
check, the config-scramble null, and the site-level δA/A. What it cannot support
is the *aggregates*: Pearson(ξ_A, ξ_class), the slope, the dynamic range, and
§6.1.1's row/column control. `report()` declines to print those below three
couplings rather than emitting a NaN that reads like a failed measurement.

`SAC_BETAS=2.1,2.3,2.4,2.5,2.7` runs the scan later without any retraining: the
trained arm is then the β = 2.4 operator evaluated everywhere — matched in its
own row, off-diagonal in the other four — which §6.1.1 licenses, since it
measures (spread across evaluation ensembles)/(spread across training β) = 12.0
with a mean diagonal advantage in A₀ of +0.038 against a mean
trained-minus-random of +0.156. That is the weaker of the two positions and
therefore the one to report; off-diagonal arms tracking ξ would be §6.1.1's
content-dependence result reproduced on a non-abelian group, not a weakness.

Both β = 2.4 checkpoints are run as separate arms in either mode — Run 5 (seed-0
ensemble) and the replication (`_ens1`, seed-1 ensemble). They were trained on
*different* ensembles, so agreement between them is §6.2's replication statement
transported to the attention field, and it is available from the single row.

### 9.4 Nothing measured here was ever trained on

The evaluation ensemble is sampled fresh at seed 11 — none of 0, 1, 2, the
seeds every checkpoint's training ensemble used — and at N = 1600 rather than
2000, so the cache key cannot collide with a training ensemble either. Every
configuration is unseen by every network at **every** β, the matched one
included, so no train/val/test slicing is needed and the diagonal cell is as
clean as the off-diagonal ones. (The Z₂ study had to carve `configs[N_USE:]` out
of the training ensembles; here re-sampling is ~40 min at N = 1600, which is
cheaper than the argument about whether the slice is clean.)

### 9.5 The estimator is imported, not reimplemented

`su2_attention_correlator.py` imports `_jack`, `_fit`, `_project`,
`_gevp_is_sane`, `_window`, `_corr_delta`, `_shuffle_time`, `_scramble_configs`
and the constants from `z2_attention_correlator.py`. GEVP t0 = 1 / td = 2, the
largest positive window inside Δ ∈ [2,8] fixed once on the full sample, the cosh
fit and Morningstar–Peardon A₀, blocked jackknife with block 20, the
config-scramble null and the time-shuffle zero-mode check are therefore
*identical by construction* rather than by claim — "same conventions as Table 5"
becomes a fact about the call graph. Only the inputs differ: the classical
comparator is SU(2)'s APE ladder (0, 2, 4, 6), the one `train_glueball.py` and
`measure_glueball.py` build their GEVP anchor from, not Z₂'s (0, 4, 8, 16).

The one code change on the SU(2) side is that `train_glueball.network_obar` was
split into `config_inputs` (4D config → per-timeslice 3D `W`, `T`) plus the
forward, so the analysis pays the per-configuration cost — the whole cost, and
none of it weight-dependent — once and shares it across the three networks. The
split is bit-exact: `network_obar == config_inputs ∘ model` was checked on the
Run-5 checkpoint.

Two traps found while wiring it, both silent if missed:

* the reduction weight `dist` must be **real**. Z₂'s `MODEL_DTYPE` is float32 so
  copying that line works there; SU(2)'s is complex64, and since the score is
  `Re Tr[Q†K̃]` the softmax α is real, a complex weight would promote the whole
  `ell` reduction to complex and it would flow to the GEVP unnoticed.
* the zero-momentum projection sums **three** spatial axes, not two. The Z₂
  slice is 2D; SU(2)'s is 12³.

### 9.5.1 ΔA₀ now compares the operators the table quotes (2026-08-17)

The first SU(2) table printed `A₀ tr./rnd. = 0.90/0.76` beside
`ΔA₀ = +0.286(56)` — a factor of two apart, and the reader is invited to
subtract. Neither number was wrong; they were **different operators**. The `A₀`
column is `entry["attention"]`, the multi-channel GEVP arm, while `_corr_delta`
was fed `best_series`, the single best channel chosen by `_jack_best_single`.

This was invisible in Z₂ because §6.2's variational self-check **fell back to a
single channel at every β**, so `attention` and `attention_single` coincided
(verified in `z2_attention_correlator_diag_R6.pt`: identical A₀ to six decimals
at all five β, `gevp_fell_back = True` throughout) and the columns subtracted by
accident. On SU(2) the fallback does not fire uniformly — 24 attention channels
of a *random* network are redundant in a different way than a trained one's, and
a variational combination of them buys the random arm ≈ +0.15 in A₀ that its
best single channel does not have. That is a real property of the GEVP (a
variational optimum fitted on the same data helps a bad operator most), not a
bug, but it must not be split across two columns of one row.

Fix: `_jack` now records `basis_idx`, the rows it actually settled on after
pruning and after the fallback; `_resolved_series` replays that choice; and
`_corr_delta` takes both arms' resolved bases plus the windows `_jack` fixed on
the full sample, so v₀ is recomputed per replica for each arm exactly as in the
quoted number and **ΔA₀ is the difference of the two quoted A₀ by
construction**. `_delta_consistency` asserts that identity at every call site
(including the offline `write_tex`, which would otherwise regenerate a stale
dump's mismatched row silently). The single-channel difference is kept as
`delta_vs_random_single` — it is the conditioning-free floor, and it is what
§6.2 above reports.

**The Z₂ numbers in the paper are unchanged**: with the fallback firing at every
β the resolved basis *is* the single channel, so the old and new definitions
agree there identically.

### 9.6 What would falsify the transported claim

Same pre-registration discipline as §6. The claim survives if ξ_A tracks the
classical ξ_t across the scan and the trained arms carry more ground-state
overlap than the random one. It fails if:

* **ξ_A does not resolve.** With ξ_t ≈ 2–3 on N_t = 24 the fit window Δ ∈ [2,8]
  spans 2–4 correlation lengths, which is where the signal-to-noise falls off
  exponentially; if the window collapses to its 4-point minimum at every β, the
  answer is "more configurations", not "no mass".
* **ξ_A is flat in β.** Then the attention is reading the architecture, not the
  configuration, and §6.1.1's control did not transport to SU(2). *Not testable
  from the single-β row* — it needs `SAC_BETAS` set to the scan.
* **ΔA₀ ≤ 0.** The structural claim (equivariant attention maps are lattice
  operators with a mass) would still stand — it stands for the random arm in Z₂
  too — but the *learning* claim would be Z₂-specific. At β = 2.4 this arm is
  fully diagonal, so a null result there is clean and cannot be blamed on the
  off-diagonal design; in the scan, four of the five ΔA₀ cells compare a network
  trained at β = 2.4 against a random one on an ensemble neither has seen.
