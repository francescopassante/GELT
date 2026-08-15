# The attention field has quantum numbers — design record

Written 2026-08-15, before the run. Everything below the "Result" heading is
empty on purpose; the pre-registered outcomes in §6 are what the run is allowed
to conclude.

`notes/attention_as_operator.md` established that the attention map of a
gauge-equivariant network is a **local gauge-invariant scalar lattice
operator**: it has a mass (ξ_A tracks the exact dual ξ at Pearson 0.9946 over
×2.96), and training makes it a *better* operator (correlated ΔA₀ = +0.11 …
+0.27, 5.4σ … 23.5σ; unbiased at +1.2% against exact truth where the classical
basis reads −8.9%).

That closes "does it have a mass". It leaves open the next question a
spectroscopist asks: **what operator is it?** An operator is identified by its
quantum numbers, and the ones available on a 2D spatial slice are the irreps of
the lattice rotation group D₄.

---

## 1. The observation that motivates this

The word *scalar* in "local gauge-invariant scalar field" was assumed, never
measured. It is false.

Rotate a configuration by 90°, run the network, rotate the output field back.
For a genuine scalar operator — every classical smeared-loop operator, by
construction — the difference is *identically zero*. Measured on the β = 0.7585
production ensemble (2 unseen configurations × 48 timeslices), as a fraction of
the field's own fluctuation ‖f − ⟨f⟩‖:

| network | ‖f(RU) − Rf(U)‖ / ‖f − ⟨f⟩‖ |
|---|---|
| trained (`best_z2_glueball_b0.7585_R6`) | **0.671** |
| random init, seed 20260810 | 0.002 |
| random init, seed 7 | 0.006 |

Two to three orders of magnitude, and it is *training* that produces it. The
random-init network is a rotationally symmetric operator to float32 noise —
which is what it should be: the L₁ ball is D₄-closed and near-uniform attention
inherits the ball's symmetry. Training drives the operator to where its
symmetry-breaking part is two-thirds the size of the operator itself.

This is the third setting in which the network's anisotropy shows up
(`notes/topological_localization.md` §6.1: six of eight SU(2) heads locked to a
globally fixed spatial axis in 100.0% of 1.3M site-slice samples, with an
explicit A₁⁺⁺ projection flagged there as "the open next step"). It is the first
time it is a **number with an exact null**, and the first time it is attributed
to training rather than observed at the endpoint.

## 2. Why it matters — the correlator decomposes by channel

The ensemble is D₄-invariant, so by Schur the channels do not interfere:

    C_Ō(Δt) = Σ_ρ C_ρ(Δt),      ρ ∈ {A₁, A₂, B₁, B₂, E}

with A₁ the scalar (0⁺⁺, the target), B₁/B₂ the spin-2 channels (2⁺⁺), and E
spin-1. Two consequences, both testable:

1. **The published A₀ and ξ_A were measured on a contaminated operator.** Every
   number in `attention_as_operator.md` §6.1 comes from the *unprojected* field
   Ō = Σ_x f(x), whose correlator carries the spin-2 tower on top of the 0⁺⁺.
   Projecting onto A₁ removes it exactly. A₀ should rise.
2. **There is a mechanism for why training did not do this itself.**
   `LOSS_DELTAS = (1, 2)` is a *short-distance* ratio, so contamination by a
   state 2–3× heavier costs the loss almost nothing while buying whatever the
   anisotropy buys. The loss cannot see the defect it creates.

And the wrong-channel content is not waste — it is a second operator. The B
channel interpolates the 2⁺⁺ glueball, and after projection the A₁ ground state
contributes **exactly zero** to it by symmetry, which is the leakage problem
that normally makes higher-spin channels unmeasurable.

## 3. The construction

Let O_w be the operator "run the network, reduce the attention row of one head
with weight w(Δ)":  O_w[U](x) = Σ_Δ w(Δ) α_{x→x+Δ}[U].

For g ∈ D₄ define the pulled-back field  f_g(x) = O_w[gU](gx)  — run the network
on the rotated configuration and rotate the answer back. Then

    P_ρ O_w = (1/8) Σ_g χ_ρ(g) f_g

for the four one-dimensional irreps, and the E part is the remainder
f_id − Σ_{1D} P_ρ. Each projected field is a bona fide lattice operator with
definite quantum numbers, and its zero-momentum correlator couples only to
states in that channel.

Note what this does *not* need: the reduction weight w never has to be rotated,
because the projector acts on the operator as a whole. It also does not need
the network to be equivariant — that is the point.

**Cost is 8 forward passes, not 8 configurations' worth of work.** The expensive
half per configuration is APE smearing plus the R = 6 transport, and both
transform exactly under D₄ without being rebuilt:

    W'(x)      = (rot90 / flip) W          — the plaquette field is cell-anchored
    T'_{gΔ}(x) = (rot90 / flip + roll 1) T — the transport is site-anchored

Both laws were verified to **0.0e+00** against rebuilding `config_inputs` from
genuinely rotated links, for both generators; the script asserts them for all
eight group elements before measuring anything.

## 4. What is measured

Per β ∈ {0.7450, 0.7520, 0.7560, 0.7585}, per arm ∈ {trained at that β, random
init}:

- **the irrep variance split** of every field — 40 attention channels (4 layers ×
  2 heads × 5 reductions) plus the network's own output operator. Fractions
  v_{A₁}, v_{A₂}, v_{B₁}, v_{B₂}, v_E summing to 1. *This number is guaranteed:
  it needs no fit, and its null is an exact zero rather than a geometric
  centre* — which is precisely what ℓ_att lacked, twice;
- **A₀ and ξ of the A₁-projected field against the unprojected one**, as a
  blocked-jackknife correlated difference on shared configurations (block 20,
  Δ ∈ [2,8], `fit_cosh_correlator` — the conventions of §4 of
  `attention_as_operator.md`, unchanged, so the projected numbers are directly
  comparable to the published ones);
- **the mass in the B channel**, on a shorter window (heavier state), quoted as
  the ratio m_B/m_{A₁} against the dual ground truth from `dual_spin2.py`;
- **the kernel-level split** — the same decomposition applied to ᾱ(Δ), the
  attention averaged over sites and configurations. Free, and it separates a
  *fixed anisotropic filter* (anisotropy in the weights, a spurion, fixable by
  projection) from a *per-configuration axis choice* (anisotropy in the
  fluctuations, which would mean the operator locally adapts to structure — a
  much stronger and more interesting finding).

Two reductions are added to the existing three for this study: the existing
`self`, `ell`, `ent` all use rotationally invariant weights, while
`q1 = Σ_Δ (Δx²−Δy²)/|Δ|₁² α` and `q2 = Σ_Δ 2ΔxΔy/|Δ|₁² α` are the natural
spin-2 interpolators. They are not needed for the headline (an equivariance
defect shows up in the projection of an isotropic reduction just as well) but
they give the B channel its best chance of a resolvable mass.

## 5. Self-checks, all of which must pass before anything is read off

The failure mode of this program has twice been a statistic that could not have
come out any other way. Every check below has an answer known in advance:

1. **transport/plaquette push laws** — `push_g(W, T)` equals `config_inputs(gU)`
   to 0.0e+00 for all eight g. Asserted at start-up.
2. **the classical positive control** — the input plaquette channels `W[:, c]`
   are scalar fields by construction, so their non-A₁ variance fraction must be
   **0 to machine precision** through the identical projector. This validates
   the whole pipeline end-to-end on a field whose answer is known.
3. **the random arm** — its non-A₁ fraction must be ≈ 0 (§1 measured the raw
   version at 0.002–0.006). If it is not, the transformation laws or the
   projector are wrong and nothing else in the run means anything.
4. ~~**`P_{A₁} O_{q1} ≈ 0` on the random arm** — a B-type reduction of an
   equivariant network has no scalar part.~~ **Wrong as stated — see §5.1.** The
   random arm is not an equivariant network. The check that replaces it is the
   bottom row of the attribution table: with RoPE disabled *and* a trivial
   transport the non-A₁ fraction is **exactly 0**, and that is the strongest
   validation the projector has.
5. **Schur** — Var(f_id) ≈ Σ_ρ Var(P_ρ f) on the ensemble. Cross-channel terms
   vanish only if the ensemble really is D₄-symmetric.

### 5.1 The random arm is not the equivariant reference (found while building this)

Self-check 3 passes at the level of the network's **output** field: the random
arm reads A₁ = 1.0000 with a breaking of 0.002. Self-check 4 fails badly — its
*attention rows* are ~15% non-scalar. Both are true, and the resolution changes
what this study is measuring.

The output field is nearly scalar at initialisation only because `init_scale`
makes the attention path a small perturbation on the residual `W`, which is
exactly scalar. (This is also, in retrospect, why §6.1.2 of
`attention_as_operator.md` found the random arm agreeing with the classical
operator: at init the operator very nearly *is* the classical operator.) The
attention itself is anisotropic from the start, and there are two architectural
reasons, both isolated by ablation in `attribution()`:

- **the anchoring mismatch.** `W` is the plaquette *based at* site x — the loop
  x → x+μ̂ → x+μ̂+ν̂ → x+ν̂ → x. Under a 90° rotation that loop maps to the one
  whose base corner is `Rx − μ̂`: as an array indexed by base corner, the
  plaquette field rotates about a **cell centre**. `T_Δ(x)` transports between
  lattice **sites** and rotates about a site. The two index maps differ by one
  lattice unit — measured, not assumed: `rot90` with no roll reproduces
  `config_inputs(gU)` for W, and `rot90` *plus a roll of 1* reproduces it for T,
  both to 0.0e+00. The block pairs `W[x]` with `T[·, x]` at equal array index,
  so the composite is not D₄-covariant even though each input separately is.
  Feeding a trivial transport removes exactly this term.
- **RoPE.** `pair_axis = [p % D]` ties each channel pair to a fixed lattice axis
  and `logspace(0, -1, n_pairs)` gives the axes different frequencies, so the
  score acquires a Δ-dependent phase no rotation preserves. Zeroing `rope_freq`
  removes this term.

Remove both and the block is equivariant to float32 noise — on the smoke
configuration the non-A₁ fraction goes to **exactly 0.0000**, with the two terms
contributing 0.159 (RoPE) and 0.031 (anchoring) separately. Which dominates at
R = 6 on production ensembles is for the run to say.

Consequences:

- the **measurement** is unaffected. The projector is exact, the channels still
  do not interfere, and the A₁-projected operator is still a genuine scalar;
- the **attribution** changes. "Training breaks the symmetry" is too strong as a
  bare statement: there is an architectural floor, and the trained/random
  contrast at the output (0.671 vs 0.002) measures training *amplifying the
  attention path* as much as training producing new anisotropy. The three arms
  (`norope`, `random`, `trained`) plus the ablation table are what separate
  them, and that separation is now a headline of this study rather than a
  footnote;
- there is a **concrete architectural finding** that owes nothing to the
  spectroscopy: the GELT block as implemented is not rotation-equivariant, for
  two identifiable reasons, one of which (the anchoring) is a genuine design
  question rather than a hyperparameter. `blocks_bias`'s per-offset `b_h` is a
  third such term by construction, so this is not specific to the RoPE variant.

## 6. Pre-registered outcomes

- **A₁ projection raises A₀ (correlated, > 3σ)** → the published A₀/ξ_A were
  measured on a spin-contaminated operator and the projected values supersede
  them; "training buys ground-state overlap in the scalar channel while
  injecting spin-2 contamination the short-distance loss cannot see" is
  established, and the fix — a rotation-equivariant GELT — is motivated by
  measurement rather than by taste.
- **A₁ projection leaves A₀ unchanged** → the symmetry breaking is real
  (§1 is not in doubt) but orthogonal to the zero-momentum correlator: the
  network broke the symmetry in a direction that costs the operator nothing.
  §6.1's headline stands unchanged and this study reports one number — the
  variance split — plus a null.
- **A₁ projection *lowers* A₀** → a bug. Self-checks 1–5 are the debugging path;
  do not report it as physics.
- **B channel resolves and m_B/m_{A₁} matches the dual's spin-2 ratio** → bonus
  spectroscopy: the 2⁺⁺ glueball is extractable from a neural network's internal
  attention weights, validated against exact ground truth.
- **B channel is noise** → the learned anisotropy is UV and does not couple to
  physical spin-2 states. Reportable as stated; it makes the "defect" reading
  of §1 the whole story.

The one thing this run **cannot** conclude, whatever it returns, is anything
about the correlation length. That axis is closed (`attention_as_operator.md`
§8) and re-opening it is how the two false starts happened.

## 7. Scope

Deliberately kept outside `gelt/`: `scripts/z2_rotation_irreps.py` (gauge side),
`scripts/dual_spin2.py` (ground truth), `results/rotation/` (artifacts). Nothing
in the library or in any existing script is touched, so if §6's second or fifth
outcome lands, the whole study deletes in one command and leaves no trace.

## 8. Notes from the build

- **Cost.** Rotating configurations would have meant re-running APE smearing and
  the transport DP for all eight group elements. Rotating `(W, T)` instead makes
  the group action a relabelling, so the expensive half is paid once and the
  study costs 8 forward passes per configuration rather than 8 configurations'
  worth of work. The two push laws are asserted against the honest rebuild at
  start-up, so the shortcut cannot silently drift.
- **A translation control was decisive** while debugging §5.1. `model(roll(W),
  roll(T))` vs `roll(model(W, T))` is exact (0.00e+00), which proved the
  comparison machinery was right and forced the conclusion that the block itself
  is not rotation-covariant. Keep it if this file is ever revisited.
- **The dual spin-2 operator needs smearing.** A thin bond-energy difference is
  almost pure contact term: `C(1)/C(0) ≈ 0.04` and nothing resolvable beyond.
  Spatially smearing the spin field first (the Ising analogue of APE, never in
  time) is what makes the B channel fittable at all — the same lesson as
  `glueball_spectroscopy.md` §7, in a model with no gauge field in it.
- **`b2` (the xy partner) is noisier than `b1`** and did not resolve at smoke
  statistics. If it does not resolve in production either, that is reportable as
  it stands; the two are independent handles on the same channel.

---

## Result

*(not yet run)*
