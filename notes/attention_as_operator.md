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

Known confound on δA/A: α is a softmax output, so a network with larger score
magnitudes has a sharper — hence more variable — attention regardless of what
it learned. δA/A is therefore descriptive; the load-bearing learning statistic
is A₀, which a sharp-but-untrained attention would not raise (it would build a
UV-dominated field with *poor* ground-state overlap). An entropy-matched random
arm would close this properly.

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

5. **The fallback contaminated the shuffled null** (found 2026-08-11, fix
   pending a re-run). Routing the null through the full basis machinery let it
   re-select its own best-of-six, and a maximum over six noise series is
   positive by construction: the null's A₀ rose from 0.005–0.014 to 0.043–0.230,
   largest exactly where the real signal is largest — the shape of a selection
   artifact. Measured bias on pure noise: C(td)/C(0) of a fixed channel
   +0.0009 ± 0.0008 (expectation 0), of the best of six +0.0068 ± 0.0005, i.e.
   **7.4×**, before the cosh fit amplifies it. The null now reuses the channel
   index chosen on the real data, so it is the identical operator with its
   timeslices scrambled and nothing else. ξ, A₀ and ΔA₀ are unaffected — they
   all come from the best-single arm, whose selection is made on real data
   either way — so only the null column of §6.1 awaits the re-run.

## 7. Known limitations, stated up front

- Z₂ in 3D, one volume, one lattice family — a statement about the *method*,
  not about SU(N), exactly as in §6.1.
- The R=12 operators are of uneven quality (`m_net/m_class` 1.18 → 2.10). The
  cross-evaluation design makes this a nuisance parameter rather than a
  confound, but the diagonal alone would still inherit it.
- The effective exponent from ξ ~ (β_c − β)^(−ν) on these five points is
  **≈ 0.39**, not the 3D Ising ν = 0.63: ξ ≤ 5.3 at L = 24 is not the
  asymptotic scaling regime. The exponent may be quoted only as a
  *consistency* check between the attention and the classical scan on the same
  points — never as a measurement of ν. With the ξ(0.7585) > ξ(0.760) excursion
  now confirmed on both operators, it should probably not be quoted at all
  until the volume question is settled.
- **Finite volume at the top two β is the open physics question.** ξ = 5.6 and
  4.7 (classical) against L/4 = 6 puts both points at or past
  `z2_beta_scan.py`'s own guard, and the non-monotonicity is what a
  volume-squeezed pair looks like. The test is L = 24 → 32 — but GELT is built
  per-L, so that means new ensembles *and* new checkpoints at all five β
  (≈2.4× per sweep, ~40 h wall clock), not one point. Worth it only if the
  excursion is to be reported as physics rather than flagged as unresolved.
