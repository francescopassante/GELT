# Glueball spectroscopy — GELT as a learned variational operator

A plan for moving GELT from per-configuration regression toward **glueball
spectroscopy**, with GELT trained to **maximize overlap with the glueball
ground state**. The central claim: this is the cleanest, most principled use
of the architecture — the loss value at convergence *is* the physics answer
(the glueball mass), the trained network *is* the optimal interpolating
operator, and the attention map becomes a measurement of what spatial loop
structure that operator uses. It lands directly on the thesis spine in
`notes/explainability.md` ("the attention map is a measurement"), applied to a
real, hard observable instead of a toy target.

## Status & findings (updated 2026-06-29)

> **Verdict: a 0⁺⁺ mass IS now resolvable.** On an *anisotropic* lattice the
> multi-level GEVP gives a clean plateau at **m·a_t ≈ 0.33** (see "Run 3" below).
> This is the go decision: the classical ground-truth `m_G` that GELT will be
> validated against now exists, so deliverable §6.2 (`train_glueball.py`) is
> unblocked. The isotropic lattice could *not* resolve it; the anisotropy was the
> missing ingredient.

### What is built (code inventory)

- `gelt/glueball.py` — classical baseline (§6.1): `glueball_operator`, spatial
  APE smearing (`ape_smear`), connected vacuum-subtracted `connected_correlator`,
  `effective_mass`, `jackknife_effective_mass`; and the **multi-level GEVP**:
  `smearing_operator_basis`, `connected_correlator_matrix`, `gevp_eigenvalues`
  (robust eigh-whitening with an eigenvalue floor, *not* Cholesky — low statistics
  can make C(t₀) indefinite), `gevp_effective_mass`, `jackknife_gevp_effective_mass`.
- `gelt/sampler.py` — SU(2) heat-bath + overrelaxation; `integrated_autocorrelation_time`;
  **anisotropy**: `staple_sum` and every sweep take `xi`, `time_axis` (ξ folded
  into the staple, β stays the overall scale, `ξ=1` bit-exact); `mcmc_ensemble`
  / `haar_ensemble` take `Lt` for a non-cubic lattice.
- `gelt/lattice.py` — `action(..., xi, time_axis)` (β_t=β·ξ temporal, β_s=β/ξ
  spatial, tree-level); `random_links(..., Lt=)` non-cubic `Lt × L^(D-1)` (time=axis 0).
- Scripts: `check_glueball_autocorrelation.py` (τ_int → n_skip), `validate_anisotropy.py`,
  `measure_glueball.py` (caches the ensemble under `datasets/`; cache key includes ξ, Lt).
- Tests: `tests/test_glueball.py` (gauge covariance/invariance, synthetic + two-state
  GEVP mass recovery), `tests/test_sampler.py` + `tests/test_lattice.py` (anisotropy:
  ξ=1 regression, anisotropic-action conservation/invariance, non-cubic shape).

### The process, run by run

**Run 0 — isotropic, single smeared operator (`L=12 β=2.4 N=2000`, HB+OR).**
Code checks pass (synthetic mass recovered, ⟨O⟩ rises with smearing). Physics
**marginal**: the smeared `m_eff(Δ)` shows only a weak `m·a ≈ 0.8` at Δ=1–2 and
slides into noise (negative `m_eff`, huge bars) by Δ≈3; `C(Δ)` hits its O(1) noise
floor by Δ≈3. Textbook heavy-0⁺⁺ problem (§7): the signal dies in ~3 slices and
more statistics barely helps (Lepage). *Lever is operator overlap, not N.*

**Run 1 — isotropic + multi-level GEVP (same ensemble).** The GEVP ground state
is flatter/lower than the single operators but **still does not plateau** — the
isotropic lattice simply has no late-Δ signal for any basis to exploit. Conclusion:
the operator basis is not the bottleneck; the *lattice* is.

**Run 2 — anisotropy validation (`validate_anisotropy.py`).** Confirms the
anisotropic implementation is faithful and acts as intended:
- ξ=1 regression: ⟨P⟩ = 0.4329 ± 0.0017 vs exact I₂/I₁ = 0.4331 → **Δ = 0.1σ**
  (the refactor changed nothing at ξ=1).
- ξ-scan (4D): plaquette splits cleanly — `⟨P_st⟩` rises and `⟨P_ss⟩` falls with ξ
  (β_t > β_s), coinciding at ξ=1.
- Renormalized anisotropy from a Creutz-ratio ratio: **ξ_bare=3 → ξ_R ≈ 3.32**
  (the tree-level ≠ renormalized mismatch, made visible; no auto-tuning).

**Run 3 — anisotropic glueball (`L=12 Lt=24 D=4 β=2.4 ξ=3.0 N=2000`, HB+OR, acc 1.00).**
*This resolves the mass.*
- `C(Δ)` now decays cleanly over **~10–12 time slices** before the periodic
  turnaround at Δ = Lt/2 = 12 (vs dying by Δ≈3 in the isotropic run).
- The GEVP ground state (levels [0,2,4,6], t₀=1) **plateaus** with tight bars:
  - `m_eff(Δ=1): m·a_t = 0.365 ± 0.008`  (→ m·a_s = ξ·m·a_t = 1.096)
  - `m_eff(Δ=2): m·a_t = 0.333 ± 0.011`  (→ m·a_s = 0.999)
  a small descent then flattening — the expected approach to the plateau from
  above (variational upper bound). The GEVP plateaus where the thin/smeared single
  operators still do not.
- **Plateau value: m·a_t ≈ 0.33.**

### Caveat on the physical number

`m·a_t ≈ 0.33` (and hence the plateau) is the trustworthy, method-validating
result. The reported `m·a_s = ξ·m·a_t ≈ 1.0` is **not continuum physics**: with
β_s = β/ξ = 2.4/3 = 0.8 the *spatial* lattice is deep in strong coupling (coarse
a_s), so `m·a_s ≈ 1` carries large discretization artifacts. A publishable `m_G`
would need anisotropy **tuning** (raise β with ξ to keep β_s in the scaling
window so a_s is fixed/known) and a continuum extrapolation — deliberately left
as future work. The point established here is that *the pipeline resolves a
plateau*, which is exactly the go/no-go that was open.

### Next step

Deliverable §6.2: **`train_glueball.py`** — `GELT(reduction="none")` on the
Rayleigh loss `−C(1)/C(0)`, evaluated by jackknife, **on the cached anisotropic
ensemble** (`datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000.pt`), with its
`m_eff(Δ)` compared against this classical GEVP plateau (`m·a_t ≈ 0.33`) and the
L-CNN baseline. The win to look for: GELT plateaus at least as low, and earlier
in Δ, than the hand-built GEVP basis (a *learned* variational operator).

> **Amended by the audit below (2026-07-01):** the plan as stated in §2/§3 has
> a flaw — GELT must be run **per-timeslice in 3D** (spatial links only), not on
> the full 4D config, or the variational principle is void and the loss is
> gameable. See "Audit" for the full list of blockers and fixes.

> **Executed (2026-07-03):** the first training run is done — GELT learns a
> genuine 0⁺⁺ operator (m_eff(Δ=3) = 0.310 ± 0.045, consistent with the 0.33
> anchor) but loses to the classical GEVP on overlap. See "Run 4" below for
> the full diagnosis and the revised next steps (smeared-input channels,
> GELT-in-the-GEVP-basis).

## Audit (2026-07-01) — what's broken, what to change, thesis-worth

A code + plan audit of this document and everything it touches
(`gelt/glueball.py`, `gelt/sampler.py`, `gelt/blocks_rope.py`,
`scripts/measure_glueball.py`, `scripts/check_glueball_autocorrelation.py`).
Bottom line: the classical baseline code is sound, but §6.2 as written in §2/§3
has one conceptual flaw that would silently invalidate the result, plus two
concrete code blockers. All are fixed by the same move: **run GELT as a 3D
network on the spatial links of each timeslice.**

### Critical — will prevent (or silently invalidate) §6.2 as planned

1. **Temporal receptive field breaks the variational principle** *(supersedes
   §2's "no architecture change needed" and the §3 loop as written).* The
   Rayleigh argument in §1 requires Ō(t) to be a functional of the fields on
   **timeslice t only** — the transfer-matrix condition, which §7 already
   states for smearing ("never in time") but never applies to the network
   itself. As designed, GELT ingests the full 4D config: 3 of its 6 plaquette
   input channels are temporal, and the L1-ball transport + attention step in
   time, so O(t) depends on links at t ± R·depth. Then C(1) is not a
   transfer-matrix element, the upper bound is void, and the loss is
   *gameable*: the optimizer's easiest way to push C(1)/C(0) → 1 is an output
   slowly varying in t (limit: per-config constant ⇒ C(Δ) = C(0) ∀Δ ⇒ "m" → 0).
   The network would spuriously "beat" the classical GEVP and §5's win
   criterion would be satisfied for an unphysical reason.
   **Fix:** feed each timeslice's spatial links as a `D=3` config
   (`GELT(D=3, L=12)`, input channels = the 3 spatial plaquette planes, 3D
   L1-ball transport), batch over (config × timeslice), reshape to `(B, Nt)`
   for the correlator. This makes O(t) a legitimate single-timeslice operator —
   exactly the domain the classical operator uses. (Equivalent alternative if a
   single 4D codepath is ever wanted: mask the offset list to Δt = 0 and drop
   temporal plaquettes from the input — same physics as masking, but the
   per-timeslice batching is simpler and far cheaper.)
   *Within* the 3D operator, spatial loop-building persists in full — the
   multiplicative value path grows arbitrarily large spatial loops with depth,
   and hand-tuned smearing is likewise purely spatial, so the "learned
   smearing" analogy of §7 gets cleaner, not weaker. The *absence* of temporal
   loop content is not a limitation; it is the definition of a valid
   interpolating operator.
2. **GEMHSA is cubic-only.** `blocks_rope.py` builds `_nbr_idx` with a single
   `torch.arange(L)` and `% L` on *every* axis — the model literally cannot
   ingest the 24 × 12³ anisotropic ensemble (wrong neighbor wrap on the time
   axis). Moot under fix 1: each timeslice is a cubic 12³ lattice, and Lt only
   ever appears as a batch dimension. (Caveat for the future: running the *4D*
   network on non-cubic configs — masked-offset variant, or regression targets
   on anisotropic ensembles — needs per-axis extents in `_nbr_idx`.)
3. **Zero gradient at initialization.** `GELT` defaults to
   `mlp_zero_init=True`, so the output is identically 0 at init. C(0) and C(1)
   are both quadratic in the output, so the gradient of −C(1)/(C(0)+ε) at
   O ≡ 0 is **exactly zero** — training never leaves the saddle.
   `train_glueball.py` must pass `mlp_zero_init=False`.
4. **Memory at 4D scale.** At R=2, D=4 on 24 × 12³ the transport alone is
   ~40 offsets × 41k sites × nc² complex64 ≈ 53 MB *per config* (≈ 100 GB to
   precompute the ensemble the `data.py` way), before the much larger
   per-layer K/V intermediates — the `fable_audit.md` offset-chunking gate
   hitting §6.2 head-on. The 3D per-timeslice restriction shrinks it ~70×
   (24 offsets × 1728 sites); computing T per batch on the fly becomes cheap.

### Moderate — weaknesses in the baseline GELT will be judged against

- **The "plateau at m·a_t ≈ 0.33" rests on two points that disagree.**
  0.365 ± 0.008 (Δ=1) vs 0.333 ± 0.011 (Δ=2) differ by ~2σ — a *descent*, not
  a plateau; the asymptotic mass could sit below 0.33. Before using it as the
  validation anchor, add GEVP points at Δ=3–4 (or a two-state fit). Note also
  that jackknife bands at neighboring Δ are strongly correlated, so eyeballing
  overlap overstates consistency.
- **N_SKIP=5 was calibrated on the wrong ensemble.**
  `check_glueball_autocorrelation.py` measures τ_int on an *isotropic* L=8
  lattice; production runs at ξ=3 (β_t=7.2 / β_s=0.8), Lt=24 — different
  dynamics on both axes. Residual autocorrelation makes the leave-one-out
  jackknife (which assumes independent configs) underestimate every error bar.
  Fixes: re-run the pre-flight with `xi=XI, Lt=LT`, and/or a blocked jackknife.
- **The GEVP eigenvalue floor amplifies noise instead of removing it.**
  `gevp_eigenvalues` floors near-zero modes of C(t₀) at `eps·s_max` then
  whitens — a floored direction gets weight 1/√(1e-12), i.e. pure noise blown
  up ×10⁶ inside the whitened matrix. Standard treatment: *truncate* (project
  out) those directions. Probably harmless for 4 well-separated smearing
  levels; a landmine for the larger learned bases of deliverable 3.
- **Train/held-out split is mandatory, not optional.** A network optimizing the
  *empirical* C(1)/C(0) on 2000 configs overfits the noise; the variational
  bound holds in expectation, not on the training sample, so a trained
  operator can spuriously violate it on its own training set. §4.1's passing
  mention of "an independent ensemble" must become structural in
  `train_glueball.py`: hard split, all reported masses jackknifed on held-out
  configs only.
- Smaller items: `measure_glueball.py` prints `m·a_s = ξ·m·a_t` with the
  *bare* ξ=3 although Run 2 measured ξ_R ≈ 3.32 (the physics caveat above
  covers it, but the printed conversion is the wrong one);
  `effective_mass` ignores time periodicity (fine at Δ=1–2, biased near Lt/2 —
  a cosh mass would clean the plot's tail); `glueball_operator` with R ≠ T is
  not a rotational scalar (only the (μ,ν) orientation is summed, not (ν,μ)) —
  latent, since only R=T=1 is used; and the cached ensemble this document
  points to is absent from `datasets/` locally, so §6.2 starts with an
  expensive re-sample.

### Checked and sound (no action)

The anisotropic staple weighting is consistent with the anisotropic action
(ξ / 1/ξ per plane, β the single scale); the weighted staple stays quaternionic
(real coefficients), so the heat-bath decomposition `A = k·V` remains valid at
ξ ≠ 1. The APE dagger convention is right, the update genuinely simultaneous
(staples from the previous iteration), spatial-only and correctly unweighted
(`xi=1`). The Creutz w₀ sampler, the overrelaxation reflection + closed-form
reprojection, the correlator normalizations, and the incremental smearing basis
are all correct, and the test suite covers the right invariances.

### What survives from the 4D program

Everything. The regression results (topological charge density, time-spanning
R×T Wilson loops) keep the **4D** network: those are supervised tasks with no
transfer matrix in sight, an operator with temporal support is fine (for q(x),
necessary — 3 of the 6 F_{μν} planes are temporal), and the loop-building
argument — successive layers composing transported values through
`α · Q_v† · Ṽ`, including temporal loops — stands exactly as demonstrated. The
3D restriction is a per-task choice of input domain, not a change to GELT. Two
regimes of one architecture: 4D where the task is "represent this known
observable", 3D-per-slice where the task is "be a variational operator" and
the transfer-matrix formalism dictates the domain. Stating explicitly *why*
the receptive field must be spatial there (the §7 smearing rule applied to the
network itself) is a physics-aware design decision, not a retreat.

### Is this thesis-worth?

**Yes — comfortably, provided fix 1 is made.** The framing is unusually clean
for an ML-for-LGT thesis: the loss is a variational principle, so the converged
number is falsifiable against an independent classical measurement built
in-house, and "the trained network *is* the interpolating operator" gives the
interpretability chapter (attention as measurement, `notes/explainability.md`)
a real observable instead of a toy regression. Nothing in the published
gauge-equivariant-network line (L-CNN, the covariant ResNet, CASK's
attention-for-smearing) does Rayleigh-quotient-trained glueball operators, so
the combination appears novel; even if a similar preprint exists, the
matched-baseline shootout + interpretability program stands on its own at the
master's level.

Two honest risks, to manage in the writing rather than the code:

- **The classical multi-level GEVP is a strong, nearly free baseline** — a
  well-smeared 4-operator basis is close to optimal for the ground state, so
  GELT's single-operator margin ("plateaus a bit earlier/lower") may be thin.
  The defensible wins: (a) the learned operator needs no hand-tuned smearing
  schedule; (b) the deliverable-3 extension (network emits a basis → learned
  GEVP), where a hand-built basis can't compete and the novelty is strongest.
- **No continuum physics** (β_s = 0.8 is deep strong coupling). The thesis
  claim must stay "the method resolves the variational ground state and
  matches the classical answer on the same ensemble" — which the caveat above
  already states; keep it that disciplined and the coarse lattice is a
  non-issue.

### Revised §6.2 checklist for `train_glueball.py`

1. `GELT(D=3, L=12, reduction="none", mlp_zero_init=False)` on per-timeslice
   spatial links (3 spatial-plaquette input channels), batched over
   (config × timeslice), reshaped to `(B, Nt)`.
2. 3D L1-ball transport computed on the fly per batch (no precomputed-T
   dataset; it is cheap at 12³).
3. Rayleigh loss `−C(1)/(C(0)+ε)` with batch-estimated VEV subtraction (§3),
   large batches, C(0) monitored against collapse (§4).
4. Hard train/held-out split; every reported mass from a (blocked) jackknife
   on held-out configs only.
5. Validate against a *strengthened* classical anchor: GEVP with Δ=3–4 points
   (or a two-state fit) and an anisotropic re-run of the τ_int pre-flight.
6. Comparison curves: GELT vs thin/smeared single operators vs classical GEVP
   vs L-CNN (matched parameters, same 3D per-timeslice domain).

## Run 4 (2026-07-03) — first §6.2 training run: diagnosis & next steps

First execution of `scripts/train_glueball.py` as specified by the audit
checklist above (V100, cached anisotropic ensemble `L=12 Lt=24 β=2.4 ξ=3.0
N=2000`). Setup: per-timeslice `GELT(D=3, R=2, layers=4, nhead=2, d_qkv=6,
d_model=16)` (~5k params), multi-Δ Rayleigh loss `−mean[C(1)/C(0), C(2)/C(0)]`,
AdamW (lr 3e-3, wd 1e-3, cosine), batch 6 configs (= 144 timeslices),
per-layer grad checkpointing, contiguous 70/10/20 split, blocked jackknife
(block 10) on the untouched test split.

**Result: GELT learns a genuine 0⁺⁺ interpolating operator but loses to the
classical GEVP on overlap.** Early stop at epoch 22 (val minimum ≈ epoch 12);
best val Rayleigh loss **−0.433** vs the variational optimum
`−(e^{−0.33} + e^{−0.66})/2 ≈ −0.62`. On test:

| operator | m_eff(Δ=1) | m_eff(Δ=2) | m_eff(Δ=3) | m_eff(Δ=4) |
|---|---|---|---|---|
| GELT (learned) | 0.527 ± 0.028 | 0.441 ± 0.042 | **0.310 ± 0.045** | ~0.5, large bars |
| classical GEVP | 0.398 ± 0.016 | 0.357 ± 0.025 | 0.333 ± 0.036 | 0.284 ± 0.050 |

(GELT's Δ=0→1 Rayleigh mass: 0.705. Thin operator at Δ=1: ≈ 0.8.)
relevant figure: glueball_gelt_from_raw_plaquettes.png

### What worked

- **The pipeline is sound end-to-end.** Training is stable (no
  constant-operator collapse; the C(0) floor never approached), the val-selected
  checkpoint + test-only reporting worked as designed, and the Ctrl-C/early-stop
  → eval → `glueball_gelt.png` path delivered the full comparison.
- **The learned operator couples to the true ground state.** Its `m_eff(Δ)`
  descends 0.705 → 0.527 → 0.441 → 0.310, and the Δ=3 point is fully consistent
  with the GEVP plateau — trained from *thin* spatial plaquettes, no hand-built
  smearing. It clearly beats the thin classical operator (≈ 0.8 at Δ=1).
- **The classical anchor is strengthened** (audit "Moderate" item): the GEVP now
  shows Δ=3 = 0.333 ± 0.036 confirming the plateau at **m·a_t ≈ 0.33** (the Δ=4
  point drifts low but with ~2× the error — noise, not a contradiction).

### What failed, and why

- **Overlap gap.** At Δ=1 GELT sits at 0.527 vs the GEVP's 0.398 (and APE×6's
  ≈ 0.40): the learned operator carries excited-state contamination the smeared
  basis has projected out. It also has **no convincing plateau of its own** —
  after the Δ=3 dip the points bounce back to ~0.5 with growing bars, so the
  Δ=3 agreement is "consistent with the anchor", not yet a demonstrated plateau.
- **Overfitting.** Val loss bottoms at ≈ epoch 12 and rises while train keeps
  falling (−0.55 at stop): the network optimizes the *empirical* correlator
  noise of ~1400 train configs. AdamW wd=1e-3 delayed but did not stop it.
- **Diagnosis: depth budget, again.** Four bilinear layers reach loop degree
  ≤ 16 in the plaquettes (~64 links); APE×6's iterated, branched staple content
  has degree ~3⁶ in the links within the same radius-6 support. GELT cannot
  rebuild an iterated smearing schedule from thin links at affordable depth —
  the same lesson as depth 3 (which lost outright), now at depth 4 with a
  smaller margin. Capacity along *depth* is the binding constraint, not width.
- *(En-route failures, already fixed in code, recorded for the thesis write-up:
  batch 8 OOMs on the 32 GiB V100 even with checkpointing → batch 6; the
  chained-ratio loss `C(Δ)/C(Δ−1)` put the signed, noisy C(1) in a denominator —
  clamped ratios stayed finite but their gradients were garbage in direction,
  pumping the operator scale to C(0) ~ 5e8 → all denominators moved to C(0),
  loss in float64, non-finite-batch skip guards.)*

### Next steps (ranked)

1. **Smeared-input channels — the physics fix for the overlap gap.** Feed
   plaquettes at several cumulative APE levels (e.g. [0, 2, 4, 6], matching the
   GEVP basis) stacked on the channel axis, transport built from the
   least-smeared level. Smearing is spatial-only, so the per-timeslice
   transfer-matrix bound is untouched; GELT stops having to *re-derive* deep
   staple content and instead learns a spatially-resolved, nonlinear
   generalization of the GEVP (which can only take one global linear
   combination of the levels). Needs an `in_channels` override in `GELT` (the
   input width is currently hardcoded to `D(D−1)/2`), and a per-levels
   checkpoint name (the ChannelLift shape changes, so old checkpoints are
   incompatible with `RESUME`).
2. **GELT as a 5th GEVP operator — eval-only, uses the existing checkpoint.**
   Append the learned Ō(t) to the classical 4-operator basis and redo the
   blocked-jackknife GEVP on test. GELT doesn't need to beat the GEVP alone: if
   the enlarged basis plateaus lower/earlier, the learned operator contributes
   overlap the smearing ladder lacks — a positive result with no retraining
   (this is exactly the "learned basis" defensible win of the audit's
   thesis-worth section). Add alongside it the ladder diagnostic
   `corr(GELT Ō, APE×k Ō)` on test fluctuations, to locate how much effective
   smearing was learned (also feeds the interpretability chapter). Note
   `EPOCHS = 0` with `RESUME = True` already acts as an eval-only mode.
3. **Against the overfit:** more configs — sampling is cheap next to a training
   run, and the val turn-up at ~1400 train configs says statistics, not just
   regularization, is short; alternatively/additionally raise weight decay. The
   val-selected checkpoint already keeps the *reported* numbers honest.
4. **Deprioritized:** R = 3 (offset count 24 → 62, transport and K/V memory
   ~2.6×, forces batch 2–3 and worsens the batch-VEV estimate in the loss);
   more width/heads (the depth-3 → 4 experiment showed loop *degree* binds, not
   width). Depth 5–6 is plausible (checkpointing makes it mostly compute) but
   strictly worse ROI than step 1, which buys degree-~3⁶ content for free.

### Run 4b (2026-07-03) — step 2 executed (eval-only, thin-input checkpoint)

Eval-only rerun (`EPOCHS=0`, `RESUME=True`, `INPUT_SMEAR_LEVELS=(0,)`) of the
Run 4 checkpoint with the new diagnostics:

- **Ladder correlation — GELT ≈ APE×2.** `|corr(GELT Ō, APE×k Ō)|` on test:
  0.681 (thin), **0.867 (×2)**, 0.713 (×4), 0.589 (×6). (Signs are all
  negative — irrelevant: the readout's overall sign is arbitrary and the
  correlator is quadratic in Ō.) The depth-4 network trained on thin links
  learned ~two APE steps' worth of staple content — the depth-budget diagnosis,
  now quantitative.
- **Enlarged GEVP — null.** Classical + GELT 5-op basis:
  0.393 ± 0.017 / 0.355 ± 0.029 / 0.335 ± 0.035 / 0.286 ± 0.059 at Δ=1–4,
  identical to the classical 4-op basis within a fraction of the error bars
  (the purple and red curves overlap in `glueball_gelt.png` — that overlap *is*
  the result). The ~25% of GELT's fluctuation variance that is linearly
  independent of the ladder carries **no additional ground-state overlap**:
  it is excited-state admixture and noise, so the thin-input GELT is
  variationally redundant with the hand-built smearing basis.

Consequence: the "learned basis" reframe does not rescue the *thin-input*
operator — next step 1 (smeared-input channels) now carries the full weight.
This null is the clean baseline for it: if the [0,2,4,6]-input GELT's enlarged
GEVP separates from the classical curve, the gain is attributable to what the
network computes *on top of* the smearing, not to the smearing itself.

### Run 5 (2026-07-03, preliminary) — smeared-input GELT: the win criterion, in one epoch

First run with `INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`. The run itself crashed
into a *new* pathology — the smeared channels are smooth and site-coherent, the
attention sums add constructively, the bilinear value path squares that gain per
layer, and the Rayleigh loss's scale-flat direction (C(Δ)/C(0) is invariant
under Ō → λŌ) let the operator scale run to C(0) ~ 1e73 by epoch 2 (float32
overflow → val = nan → **no checkpoint can ever be saved**, since nan never
improves best_val_loss). Fixed by a variationally neutral **(log C(0))² scale
pin** in the loss (`SCALE_REG`) plus a loud non-finite-val warning. But epoch 1
completed cleanly with **val Rayleigh −0.6015** — within ~0.008 of the
single-mass loss floor at m ≈ 0.35 — and its checkpoint evaluates to:

| operator | m_eff(Δ=1) | m_eff(Δ=2) | m_eff(Δ=3) | m_eff(Δ=4) |
|---|---|---|---|---|
| GELT (smeared input, 1 epoch) | **0.379 ± 0.017** | 0.343 ± 0.023 | 0.362 ± 0.034 | — |
| classical GEVP | 0.398 ± 0.016 | 0.357 ± 0.025 | 0.333 ± 0.036 | 0.284 ± 0.050 |
| GEVP + GELT (5-op) | 0.379 ± 0.017 | 0.343 ± 0.023 | 0.362 ± 0.034 | 0.341 ± 0.046 |

- **The §5 win criterion is met (preliminarily):** GELT is flat from Δ=1
  (plateau ≈ 0.35) and sits *below* the GEVP at Δ=1–2 — less excited-state
  contamination, earlier plateau onset. (Rayleigh Δ=0→1 mass: 0.411, vs 0.705
  for the thin-input operator.)
- **The enlarged GEVP collapses onto GELT:** its ground state at Δ=1–3 equals
  GELT's own m_eff to all printed digits — offered the classical ladder plus
  GELT, the variational solver picks essentially pure GELT; the hand-built
  basis adds nothing beyond the learned operator. Against the Run 4b null this
  is cleanly attributable to the network (both methods see the same smearing
  levels; GELT mixes them nonlinearly and site-by-site, the GEVP only
  globally/linearly).
- **Ladder correlations:** r = 0.342 / 0.754 / 0.927 / 0.957 for ×0/×2/×4/×6 —
  the operator lives at the deep end of the ladder; the ~8% unexplained
  variance is what buys the improvement.

Caveats before this is a thesis claim: (i) one epoch, pre-scale-pin checkpoint
— the warm-started retrain must reproduce/refine it *(resolved: see final
results below)*; (ii) the Δ=1 GELT−GEVP difference is ~1σ under *independent*
errors, but both are measured on the same test configs — add a blocked
jackknife of the **difference** m_GELT − m_GEVP (correlated errors cancel) to
state the significance honestly *(added; see below)*; (iii) on-the-fly
smearing costs ~55% step time (1863 s/epoch vs ~1200) — cache smeared links if
this direction gets iterated heavily.

### Run 5 final (2026-07-03) — §6.2 delivered: the learned operator beats the GEVP

Warm-started retrain with the scale pin: stable throughout (C(0) held at O(1),
no skipped batches, no overfit turn-up — val flat on the floor the whole run),
best val Rayleigh **−0.6185** at ~epoch 13, Ctrl-C'd at ~epoch 18 after 5
no-improve epochs. Decoding the loss: −0.6185 ↔ **m ≈ 0.329** — the Rayleigh
quotient *saturates the transfer-matrix bound at the anchor mass*. "The
converged loss is the glueball mass" (§1) is realized. On test:

| operator | m_eff(Δ=1) | m_eff(Δ=2) | m_eff(Δ=3) | m_eff(Δ=4) |
|---|---|---|---|---|
| GELT (learned) | **0.370 ± 0.016** | **0.333 ± 0.020** | 0.345 ± 0.029 | — |
| classical GEVP | 0.398 ± 0.016 | 0.357 ± 0.025 | 0.333 ± 0.036 | 0.284 ± 0.050 |
| GEVP + GELT (5-op) | 0.366 ± 0.017 | 0.333 ± 0.020 | 0.347 ± 0.030 | 0.319 ± 0.048 |

**Correlated-difference jackknife (the significance test), m_GELT − m_GEVP:**
Δ=1: **−0.028 ± 0.007 (3.9σ)**; Δ=2: −0.024 ± 0.014 (1.8σ); Δ=3: +0.012 ±
0.021; Δ=4: +0.051 ± 0.038 — GELT is *significantly below* the classical GEVP
where contamination lives (small Δ) and identical on the plateau, i.e. a
better operator measuring the same physics.

- **The §5 win criterion is met with significance.** GELT is flat at ≈
  0.33–0.37 from Δ=1 (in the figure it hugs the anchor through Δ≈6) while the
  GEVP is still descending at Δ=1–2. Even GELT's most-contaminated Δ=0→1
  Rayleigh mass (0.392) sits below the GEVP's m_eff(Δ=1).
- **The enlarged GEVP still collapses onto (essentially pure) GELT** — the
  classical ladder adds ~0.004 at Δ=1, within noise: one learned operator
  variationally subsumes the hand-built 4-level basis.
- **Ladder correlations:** |r| = 0.446 / 0.758 / 0.870 / 0.905 for ×0/×2/×4/×6.
  Training grew the ladder-independent variance from ~8% (epoch 1) to ~18%
  *while improving* — the network's own content, not residual noise, drives
  the win. What that content looks like is now the interpretability chapter's
  first real question (`_last_alpha` on this checkpoint).
- Curiosity, not a bug: val < train in the training curves — the 6-config
  batch Rayleigh estimator is noise-biased toward 0 (batch-estimated VEV and
  C(Δ) over 144 slices), the 200-config val estimate is cleaner.

Remaining honesty caveats for the write-up: one ensemble, one (β, ξ), coarse
a_s (no continuum claim — as for the classical result); the 3.9σ statement is
about *operator quality* (excited-state contamination at Δ=1), not about the
mass value, on which the two methods agree. Still to do from the §6.2 list:
the matched-parameter **L-CNN baseline** on the same per-timeslice task, and
ideally a replication on a freshly sampled ensemble.

### Presentation layer (2026-07-04) — cosh fits + ground-state overlap A₀

How the Run-5 result should be *reported* — the LGT-standard presentation,
not new physics. The m_eff(Δ) point cloud is how results are plotted, but a
mass is **quoted from a fit**, and the operator-quality claim is quoted as a
**ground-state overlap**, not a σ-count on an m_eff difference.
`scripts/fit_glueball_overlap.py` (offline, CPU-trivial; input = the
test-split Ō arrays `train_glueball.py` now dumps to
`datasets/…_test_obars.pt` — set `EVAL_ONLY = True` there to reproduce the
dump from the existing checkpoint in one GPU eval pass):

- **Cosh fit**: `C(Δ) ≈ A·[e^{−mΔ} + e^{−m(Nt−Δ)}]` over one *shared* window
  (default Δ ∈ [2, 7]; comparability beats per-operator optimality) for GELT,
  the **projected GEVP operator** (fixed ground eigenvector v₀ at (t0, t0+1)
  — `gevp_ground_vector`, the optimal single operator in the classical span,
  the apples-to-apples comparator for a single learned operator), and APE×6.
  σ_Δ from the blocked jackknife as diagonal χ² weights
  (`fit_cosh_correlator`: profiled-A grid fit, no scipy); quoted errors from
  redoing the *whole* fit (including v₀) inside every delete-block jackknife
  sample.
- **Overlap**: `A₀ = A·(1 + e^{−m·Nt})/C(0)` — the fraction of the operator's
  spectral weight on the ground state, Morningstar–Peardon's operator-quality
  metric, and *exactly what the Rayleigh loss maximises*: the training
  objective and the reported quality metric coincide. The headline becomes
  "A₀(GELT) vs A₀(GEVP)" with the correlated same-configs jackknife of the
  differences (Δm consistent with 0 = same physics; ΔA₀ > 0 = better
  operator) as the significance statement — replacing the bare
  "3.9σ at Δ=1" phrasing.
- **Figure** `glueball_overlap.png`: left, m_eff(Δ) with fitted-mass bands
  over the window (the standard plateau + fit-band presentation); right,
  ρ(Δ) = [C(Δ)/C(0)]/cosh_ref(Δ) — a pure ground-state operator is flat at
  its A₀, excited-state contamination is the small-Δ excess: one panel that
  *shows* why GELT wins.

Status: code + tests in place; validated end-to-end on a synthetic
two-state ensemble (periodic Gaussian fields with known couplings: recovers
m = 0.322 ± 0.025 vs true 0.33, A₀ = 0.955 ± 0.050 vs true 0.99 for the
GELT-like operator, 0.826 ± 0.053 vs 0.764 for the APE×6-like one, and
Δm = 0.000 ± 0.002 where the GEVP can reach the exact optimum).

**Real numbers (2026-07-04, Run-5 checkpoint, 400 test configs, window
Δ ∈ [2,7]):** GELT m·a_t = 0.332 ± 0.027, **A₀ = 0.903 ± 0.047** (χ²/dof
0.05); GEVP-projected 0.340 ± 0.030, A₀ = 0.837 ± 0.056; APE×6
A₀ = 0.805 ± 0.052. Correlated differences: Δm = −0.008 ± 0.014 (0.6σ, same
physics), **ΔA₀ = +0.066 ± 0.031 (2.1σ)**. The GEVP gains only ~0.03 of
overlap over its best member; GELT gains ~0.10 over the whole basis. The
2.1σ is complementary to (not weaker than) the 3.9σ: the shared fit window
starts at Δ = 2, so ΔA₀ never uses the Δ = 1 point where GELT's advantage is
sharpest. Written up in `glueball_report/glueball_spectroscopy.tex` §
"Quoting the result like a lattice paper" (Table 3 / Figure 5).

### Replication batch (2026-07-04, queued) — two fresh ensembles

`scripts/overnight_replication.sh` (~24 h unattended on the V100) closes the
"one ensemble" caveat via independent from-scratch phases, driven by the new
env overrides in `train_glueball.py` (`GLUEBALL_ENSEMBLE_SEED` seeds the
sampler and gets its own cache file; `GLUEBALL_INIT_SEED` seeds init/batch
order independently; `GLUEBALL_RESUME` / `GLUEBALL_EVAL_ONLY` toggle the
flags; non-default seeds get a `RUN_TAG` suffix on checkpoint / dump / plot
so Run-5 artifacts are never clobbered):

1. **`replication_ens1`** — sample a fresh seed-1 ensemble (same L, β, ξ, N;
   the sampling is the long pole) and train GELT from scratch on it: an
   independent replication of both the m_eff and A₀ statements.
2. **`replication_ens2`** — same with seed 2 (bonus): three fully
   independent A₀ measurements (Run 5 + ens1 + ens2) combine to ~3.5σ on
   the overlap headline. If the clock runs out mid-phase, phase 1 is
   already on disk.

Init-seed robustness runs were considered and dropped as low-value: a
converged loss that saturates the transfer-matrix floor is already
init-blind evidence — any init reaching the variational bound has found the
same optimum (and the one init-sensitive failure mode, the zero-init saddle,
is excluded by `mlp_zero_init=False`).

Each phase logs to `logs/<phase>.log` (`python -u` + `TQDM_MININTERVAL=30`,
so the sampler sweep bar and the epoch/batch bars tick every ~30 s under
`tail -f` — a stuck run is visible as a stale log) and leaves a test-Ō dump
in `datasets/*_test_obars.pt`; analyse each with
`scripts/fit_glueball_overlap.py <dump>` (CPU, offline). Note each fresh
ensemble's classical anchor is recomputed on its own test split
automatically — check the GEVP mass lands near 0.33 there too before
comparing the learned numbers.

## 0. Where we are vs. what spectroscopy needs

Everything built so far is **per-configuration regression toward a known
scalar function** — action, `Q`, Wilson loops, with a label `y` to fit.
Glueball spectroscopy is a different object: there is **no label**. You
measure a zero-momentum-projected temporal correlator and extract a mass from
its exponential decay. The lightest channel is the scalar **0⁺⁺**, built from
spatial Wilson loops.

The gap is three things the codebase does not yet have:

1. **A real 3+1D ensemble with a trustworthy time axis.** 4D SU(N) targets and
   a Metropolis sampler exist, but spectroscopy needs many well-decorrelated
   configs at a coupling where a mass is resolvable. Metropolis critical
   slowing fights this — heat-bath + overrelaxation (`notes/sampling.md`) is
   the prerequisite long pole.
2. **Connected correlator measurement + plateau fitting** — does not exist (the
   only `C(t)` in the repo is the plaquette *autocorrelation* in
   `validate_sampler.py`, a sampler diagnostic, not a physics correlator).
3. **Operator construction & smearing** — glueball signals are notoriously
   noisy; no cooling/smearing exists yet (also flagged in `fable_audit.md`).

## 1. The variational principle (why this is doable and clean)

A glueball operator is a gauge-invariant scalar field `O(x, t)`. Project to
zero momentum by summing over the **spatial** slice at fixed time:

```
Ō(t) = Σ_{x⃗} O(x⃗, t)        (sum over spatial axes only — keep the time axis)
```

Form the **vacuum-subtracted** connected correlator, averaged over time
origins `t₀` for statistics (time-translation invariance):

```
C(Δ) = ⟨ Ō(t₀+Δ) Ō(t₀) ⟩ − ⟨Ō⟩²  =  Σ_{n>0} |⟨0|Ō|n⟩|² e^{−m_n Δ}
```

The effective mass `m_eff(Δ) = log[C(Δ)/C(Δ+1)] → m_G` as `Δ→∞`, and
`m_eff(Δ) ≥ m_G` for all `Δ` (variational upper bound under reflection
positivity). The punchline:

```
R = C(1) / C(0) = ⟨Ō T̂ Ō⟩_c / ⟨Ō Ō⟩_c
```

is a **Rayleigh quotient of the transfer matrix** `T̂ = e^{−Ĥ}` in the state
`Ō|0⟩` (vacuum removed). Over *all* operators its maximum is `e^{−m_G}`, the
largest non-vacuum eigenvalue — the lightest glueball. The maximizer is the
optimal-overlap operator. Hence the loss:

> **Loss = −C(1)/C(0)** (minimize), and at the optimum `m_G = −log R`.

The loss value *is* the mass; the trained network *is* the operator.
Unsupervised, variational (a rigorous upper bound on `m_G`), and the answer
falls out of the converged loss.

Two structural gifts:

- The quotient is **scale-invariant** (`O → λO` cancels), so the output need
  not be normalized.
- Because the scalar 0⁺⁺ shares the vacuum's quantum numbers, after
  subtraction the lightest surviving state *is* the 0⁺⁺ glueball — so for the
  ground state the cubic-group / J^PC projection can be **deferred**. It is
  only needed for excited states or other channels.

## 2. Architecture change (minimal)

GELT already emits the right object. With `reduction="none"`,
`GELT.forward` returns a per-site gauge-invariant scalar field
`O(x)` of shape `(B, *Λ)` (`blocks_rope.py:668-670`) — an operator density.
No new head is needed. Instantiate `GELT(..., reduction="none")` and do the
zero-momentum projection in the training loop.

## 3. Training loop (Rayleigh loss)

```python
# batch U: B configs of a 4D ensemble; axis order (B, D, *Λ, nc, nc),
#          Nt = Λ[time axis], periodic in time.
O    = model(W, T)                     # (B, *Λ)  per-site invariant field
Obar = O.sum(dim=spatial_axes_only)    # (B, Nt)  zero-momentum proj per timeslice

# vacuum subtraction + time-origin averaging, estimated over the batch
mu   = Obar.mean()                     # ⟨Ō⟩ — 0⁺⁺ has a NONZERO VEV; must subtract
d    = Obar - mu
C0   = (d * d).mean()                              # connected variance
C1   = (d.roll(-1, dims=1) * d).mean()             # one-step, summed over all t₀
R    = C1 / (C0 + eps)
loss = -R                                          # ⇔ minimize m_eff(0→1)
```

At convergence `m_G ≈ −log(R)`. Cross-check by computing `m_eff(Δ) =
log[C(Δ)/C(Δ+1)]` at larger `Δ` and confirming it plateaus to the same value.

## 4. Pitfalls — where it is actually hard

1. **Ratio-estimator bias.** `C(1)/C(0)` is nonlinear in batch-estimated
   means → systematically biased gradient on small batches. Use large
   batches, accumulate, and validate the final number with a **jackknife**
   over an independent ensemble. This is the single biggest engineering risk.
2. **Vacuum subtraction is essential and noisy.** Unlike `Q` (zero VEV), the
   0⁺⁺ operator has nonzero `⟨Ō⟩`; without subtraction you measure the vacuum,
   not the glueball — and `⟨Ō⟩` is itself a noisy batch estimate inside the
   loss.
3. **Constant-operator collapse.** If `O(x)` becomes config-independent,
   `C(0)→0` and `R` diverges (`0/0`). Guard with `eps`; monitor `C(0)`; add a
   variance floor / penalty if it drifts toward zero.
4. **Ensemble requirement.** Needs well-thermalized, decorrelated 4D **SU(2)**
   configs at a coupling where a mass is resolvable. Metropolis critical
   slowing hurts; heat-bath + overrelaxation is the prerequisite long pole.
   Glueball SNR is the field's classic hard problem — but the `R = C(1)/C(0)`
   anchor lives at **small** Δ where SNR is *best*, which is exactly why this
   formulation is the tractable one.

## 5. Validation (the proof, and the thesis result)

Measure the *same* `m_eff(Δ)` curve for a **plain plaquette operator** (and an
APE-smeared loop) on the same ensemble, classically. Two things must hold, and
together they are the proof:

- GELT's plateau value **agrees** with the classical asymptotic `m_G`
  (meaningful because the variational bound makes it an upper bound — agreement
  is not luck).
- GELT's `m_eff(Δ)` plateaus **earlier / lower at small Δ** than the
  plaquette's — i.e. it found a better operator (higher ground-state overlap).
  That earlier plateau *is* the win, and the L-CNN baseline can be made to
  compete on the same quantity (matched-parameter shootout).

Thesis payoff: the trained operator's **attention map shows what spatial loop
structure the optimal glueball operator attends to** — "attention as
measurement" on a real observable.

## 6. Deliverables, in order

1. **✅ Classical 0⁺⁺ correlator + `m_eff` extraction** — *built* (reuses
   `rectangular_wilson_loop`): timeslice-summed scalar operator, spatial APE
   smearing, connected vacuum-subtracted `C(Δ)`, `m_eff(Δ)` + jackknife on an
   SU(2) heat-bath ensemble. **Outcome:** on the *isotropic* lattice the
   single-operator `m_eff` did *not* plateau (weak m·a ≈ 0.8, drowns by Δ ≈ 3).
1b. **✅ Multi-level smearing GEVP (classical)** — *built*: a variational basis of
   operators at several APE levels (`smearing_operator_basis`), correlator matrix
   (`connected_correlator_matrix`), robust GEVP solver (`gevp_eigenvalues`),
   per-state `m_eff` + jackknife (Morningstar–Peardon). Isotropically still
   marginal — the lattice, not the basis, was the bottleneck.
1c. **✅ Anisotropic lattice** — *built and validated* (§8): finer `a_t = a_s/ξ`.
   **This resolved the mass:** on `L=12 Lt=24 β=2.4 ξ=3.0 N=2000` the GEVP ground
   state **plateaus at m·a_t ≈ 0.33** (Run 3 in the Status block). That is the
   ground-truth `m_G` (in temporal units) that anchors deliverable 2.
2. **`train_glueball.py`** — `GELT(reduction="none")` + Rayleigh loss +
   jackknife eval; compare GELT vs. plaquette vs. L-CNN `m_eff` curves. *Now
   unblocked* — validate against the classical GEVP plateau `m·a_t ≈ 0.33` on the
   cached anisotropic ensemble.
3. **(extension) Multi-operator GEVP inside GELT** — network emits a *vector* of
   operators → generalized eigenproblem → excited states / other J^PC channels
   (needs the cubic-group projection deferred in §1). The learned analogue of
   deliverable 1b's hand-built smearing basis.

## 7. Smearing — the crucial enabler, and what it means for GELT

Smearing is **not optional** for the classical baseline; it is the single
technique that makes glueball spectroscopy work at all, and §5's passing
mention of an "APE-smeared loop" undersold it. The reason is signal-to-noise.

**Why thin-link operators fail.** The connected correlator's signal decays as
`C(Δ) ~ e^{−m_G Δ}`, but its statistical variance is set by vacuum fluctuations
and is roughly **Δ-independent**, so the relative error grows like
`e^{+m_G Δ}/√N` (the Lepage argument; this is the §4 SNR pitfall sharpened).
The 0⁺⁺ is heavy (`m_G·a` is order 1 on typical lattices), so the signal is
gone within a couple of time slices. You *must* read the `m_eff(Δ)` plateau at
**small Δ** — there is no late-Δ signal to wait for. A thin plaquette operator
overlaps poorly onto the ground state (dominated by UV fluctuation, couples to
high-lying states and lattice artifacts), so its `m_eff(Δ)` starts far too high
and descends only slowly — past the point where the signal has drowned. You
never reach the plateau.

**What smearing does.** Spatial smearing (APE or stout: replace each spatial
link by a projected sum of itself and its staples, iterated; reuse
`staple_sum`) builds **spatially extended** operators whose size matches the
physical glueball wavefunction. That raises ground-state overlap so the plateau
appears at small Δ where SNR is still alive. The modern glueball spectrum
(Morningstar–Peardon and successors) rests on a *variational basis* of loops at
several smearing/blocking levels solved by GEVP. Smear **spatial links only** —
never in time, or you corrupt the transfer-matrix / spectral interpretation in
§1.

**So is it needed for GELT?** The nuance — and it is a thesis selling point:

- **GELT is, in part, a *learned* smearing.** Its L1-ball transport-averaging
  plus stacked attention is a gauge-covariant, multi-scale, content-dependent
  smearing, and the variational loss (§1) optimizes overlap *directly* —
  exactly what hand-tuned smearing approximates. Honest framing: "the network
  learns its own glueball operator (its own smearing) instead of us tuning APE
  steps by hand." A cleaner story than bolting smearing on.
- **But not a free lunch — receptive field is the GELT analogue of smearing
  level.** GELT can only build an operator as extended as `R` (L1-ball radius)
  × depth allows. If `R`×depth is smaller than the physical glueball size, no
  training reaches the plateau — the same failure as an under-smeared classical
  operator. The receptive-field budget *is* the "how much smearing" knob, and
  it is the same memory gate as offset-chunked attention in `fable_audit.md`.
- **Pre-smeared input is a sensible warm start.** GELT's `W` and `T` are built
  from **thin**, UV-noisy links. Feeding it stout-smeared links instead
  (smearing is gauge-covariant preprocessing; stout is differentiable, so it
  can even be folded in as trainable layers, CASK-style — see
  `papers_review.md`) hands the network a cleaner, better-conditioned input so
  it need not learn UV-smoothing from scratch. De-risks the optimization
  without touching the variational principle.
- **Multi-scale basis ↔ multi-operator GEVP.** The classical "several smearing
  levels" basis maps onto the §6 deliverable-3 GELT extension: emit a *vector*
  of operators at different effective sizes (different layers/heads) and solve
  the GEVP — the learned analogue of a multi-smearing-level variational basis.

**Smearing ≠ cooling.** Distinct from the topological-charge cooling in
`fable_audit.md`: cooling is many sweeps that *flow the config toward classical
solutions* to expose topology (it changes the physics). Operator smearing is a
few APE/stout steps tuned purely to maximize ground-state *overlap* (it leaves
the ensemble alone). Both are built from staples, but used for different ends —
keep them separate.

## 8. Prerequisite long poles (independent of the network)

- **✅ Heat-bath + overrelaxation SU(2) sampler** (`notes/sampling.md`): without
  it there is no usable 4D ensemble. Biggest single cost. *Done.*
- **✅ Spatial APE/stout smearing** (reusing `staple_sum`): the crucial enabler
  for the classical baseline and operator overlap — see §7 for the full role
  (including why GELT only partly replaces it). *Done (APE).*
- **✅ Anisotropic lattice** (finer `a_t`): the field's primary tool for resolving
  the heavy 0⁺⁺ — without it the signal dies in ~3 slices on an isotropic lattice.
  *Done (tree-level ξ, non-cubic `Lt`, renormalized-ξ diagnostic);* nonperturbative
  anisotropy *tuning* (β_s-matching) remains future work.

## 9. Verdict

Doable and well-posed: the math is clean, the architecture already emits the
right object, and the loss is a one-screen change. The honest costs are
exactly two — the **heat-bath sampler** (no ensemble without it) and the
**ratio-estimator bias/noise** of the loss. Neither is a showstopper; both are
real work. Build the classical baseline (§6.1) before involving the network.
