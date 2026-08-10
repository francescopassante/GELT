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

| β | ξ (classical scan, N=2000) | **ξ_A trained** | ξ_A random init | ξ_out |
|---|---|---|---|---|
| 0.745 | 2.05 ± 0.16 | 2.24 ± 0.13 | 2.08 | 2.33 |
| 0.752 | 2.61 ± 0.31 | 2.76 ± 0.13 | 2.46 | 2.92 |
| 0.756 | 4.14 ± 0.26 | 4.48 ± 0.30 | 4.08 | 4.63 |
| 0.7585 | 4.59 ± 1.31 | 6.64 ± 0.32 | 5.47 | 6.65 |
| 0.760 | 5.28 ± 0.82 | 5.60 ± 0.37 | 4.80 | 5.81 |

**Pearson(ξ_A, ξ_scan) = 0.92**, slope 1.25, over ×2.5 of dynamic range, at
5–7% precision. The claim of §2 holds: the attention map is a lattice operator
and its correlator decays with the mass gap.

**The null is clean.** The time-shuffled arm returns A₀ = 0.005 ± 0.003 with a
correlator flat at ±0.005 out to Δ=9, at every β and for both networks.

**Training separates from initialisation at every β:**

| β | δA/A trained | random | ratio | ΔA₀ (uncorrelated errors) |
|---|---|---|---|---|
| 0.745 | 0.091 | 0.0067 | ×13 | +0.144 ± 0.059 (2.4σ) |
| 0.752 | 0.140 | 0.0060 | ×23 | +0.110 ± 0.046 (2.4σ) |
| 0.756 | 0.246 | 0.0054 | ×46 | +0.204 ± 0.023 (9.0σ) |
| 0.7585 | 0.113 | 0.0049 | ×23 | +0.267 ± 0.018 (14.7σ) |
| 0.760 | 0.130 | 0.0046 | ×28 | +0.205 ± 0.020 (10.1σ) |

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

## 6.2 Defects in the first pass

1. **The classical reference was broken by our own pruning.** `_prune` cut the
   four nested APE smearing levels to `n_ops = 2`: they correlate above 0.99
   *by design*, and extracting the small non-collinear part is precisely what
   the GEVP is for. Consequence: C(1)/C(0) of 0.039 / 0.020 / 0.007 at
   β = 0.752 / 0.7585 / 0.760 and an unresolved fit at the last. Fix: prune only
   when the basis exceeds the cap, so a ≤6-operator basis is passed through
   untouched, exactly as `z2_beta_scan.py` does.
2. **The multi-channel attention GEVP is unstable** — three NaN cells and one
   absurd one (ξ = 4.85 ± 89, A₀ = 0.048 ± 0.852) out of ten. The
   single-channel arm resolved in all ten and is what the tables above quote.
3. **ΔA₀ errors are uncorrelated** although both arms run on identical
   configurations, so the quoted significances are conservative — a shared-block
   jackknife of the difference is the right statement, and matters at the two
   2.4σ points.
4. **β = 0.7585 breaks monotonicity** (ξ_A = 6.64 ± 0.32 against 5.60 ± 0.37 at
   β = 0.760). That ensemble is the scan's worst point (±28%, τ_int = 6.25), so
   suspect the ensemble before the measurement — but it is unresolved.

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
  points — never as a measurement of ν.
