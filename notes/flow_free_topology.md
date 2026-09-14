# Flow-free topological charge density — design record

**Status (2026-09-14): proposed, nothing built.** Nothing below is a result.
The theory, the physics background and the argument for the task are in
`reports/topology/flow_free_topology.tex` (13 pp, compiles); this note is the
build order, the gates and the readings fixed in advance.

The question this answers: **the L-CNN shootout tied on 0⁺⁺ — is there a task
where GELT's attention is expected to win, and can it be run here?**

---

## 1. Why 0⁺⁺ tied (the short version)

Three properties of that task, each of which independently neutralises an
input-dependent kernel:

1. **Zero-momentum projected loss.** The Rayleigh quotient is built from
   `Ō(t) = Σ_x O(t,x)`. Site-averaging destroys site-to-site variation, which is
   the one thing attention has. `notes/attention_as_operator.md` §1 made this
   argument about ℓ_att; it applies verbatim to the operator.
2. **The optimum is linear in loop shapes.** Only 2.5% of ‖O_GELT‖² lies outside
   the 21-operator `full` span, and ~80% of that residual is rectangular loops
   (`notes/fable5.1_10-09_audit.md` §8.3). That *is* L-Conv.
3. **One β, one scale, the most symmetric channel.** A single fixed kernel is
   optimal by construction.

Parity was structural, and `reports/curve` §7 pre-registered it.

**One correction, measured 2026-09-14 and not in any other note.** Parity extends
to the noise-to-signal ratio `δC(Δ)/C(Δ)`, blocked jackknife, same 400 configs,
ens1: GELT 0.0322 / 0.0396 / 0.0720 / 0.1695 at Δ = 1,2,4,6 against L-CNN
0.0322 / 0.0399 / 0.0728 / 0.1676. Indistinguishable. So §9.2's top-1-share
finding is a **failure-rate** property across training runs (3 of 9 L-CNN arms
unusable, 0 of 14 GELT), **not** an error-bar advantage. Do not build a proposal
on "attention gives smaller errors".

## 2. The exact architectural difference

The loose version — "L-CNN is a polynomial, GELT is not" — is **wrong**. `LAct`
is `g(Re Tr W_c / nc) · W_c`, i.e. data-dependent gating, and L-Conv mixes
channels *before* it, so the gate argument is a learned linear combination of
transported channels: an L-CNN can gate on `ReLU(Re Tr[A − B])`, a difference
test between two scales held in different channels. GELT uses the *same* L-Act.

> **The only architectural difference is whether the weights over *offsets* are
> input-dependent.** Gating, bilinearity, receptive field, loop degree, inputs,
> loss, splits and estimator are matched.

The surviving mechanism is **normalisation**: softmax depends only on score
*differences* and is invariant under a common shift, so it is a scale-free ratio
across the neighbourhood, recomputed per site per configuration. `ReLU(Re Tr[A−B])`
thresholds an *absolute* magnitude and must be calibrated to an amplitude the
fixed weights cannot adapt.

**Selection rule.** Attention can separate where the optimal neighbourhood
weighting is *relative* rather than absolute. Three properties: per-site target;
sparse structures at configuration-dependent positions *and scales*;
amplitude-invariant identity.

## 3. The task

Learn `q̂(x)[U] ≈ q_t(x) ≡ q_clover[F_t(U)](x)` — the **gradient-flowed**
topological charge density — from **unflowed** thermalised SU(2) links, per site,
4D. Well posed because the flow's Green function is exponentially localised on
`√(8t)`; the constraint `√(8t)/a ≲ R_net` (Manhattan 8) is not a nuisance but a
measurement of how much flowed topology a bounded neighbourhood determines.

Scores on all three properties: per-site (not summed); instantons are sparse
lumps at configuration-dependent positions with a size distribution
(`n(ρ) ∝ ρ^{7/3}` at one loop for SU(2)); and the central operation — genuine
instanton vs lattice dislocation — is a **shape** test, scale-free, i.e. a ratio
of field strength at radius a, 2a, 3a, not a threshold on magnitude.

## 4. Prior art — the gap is real, the motivation is narrower than it looks

| work | input → target | status |
|---|---|---|
| Favoni et al., PRL 128 (2022) 032003 | flowed links → `q_plaq` of **the same config** | solved; local degree-2 algebraic identity, one L-Bilin layer spans it. Their Fig. 5 runs the model *along* a flow trajectory — the integers come from the flow, not the net. |
| Favoni et al., arXiv:2212.00832 | links → links, neural-ODE Wilson flow | toy models; a different problem |
| Matsumoto et al., PTEP 2021 023D01 (SU(3)) | q-density at **small** flow time t/a² ≤ 0.3 → integer Q at large t | >99%; scalar target, partial flow, non-equivariant nets |
| **this** | **unflowed links (τ=0) → flowed density field q_t(x)** | **open** |

**Consequence for the write-up:** "save the flow cost" is partly taken by the
PTEP result. Lead with the *density field* (instanton content, ⟨q(x)q(0)⟩, the
size distribution) and with the locality question, not with a cost argument.

## 5. A defect found in the audit — fix before anything else

`topological_charge_density` is the **naive, clover-free** plaquette density: all
six plaquettes based at the corner `x`, so it is not reflection-symmetric about
`x` and **is not parity-odd**, not even summed.

Verified directly. Build the reflection `x₁ → (−x₁) mod L` with the correct link
relabelling (`U'_ν(y) = U_ν(Py)` for ν≠1, `U'_1(y) = U_1(P(y+1̂))†`). It is an
exact symmetry of the Wilson action — `15478.81704264 → 15478.81704264` on a 6⁴
SU(2) config, all printed digits. Under the same map `Q: −1.3161 → +2.9888`: the
sign flips, the magnitude does not.

⇒ `definition="clover"` is **required**, and the test is
*"the clover charge is parity-odd to machine precision and the plaquette one is not"*.

## 6. What already exists (verified by running it at D=4, SU(2), L=6, complex128)

- `plaquette_tensor` → `(B,6,Λ,2,2)`; `topological_charge_density` → `(B,Λ)`;
  `build_transport_average` → `(B,40,Λ,2,2)` at R=2; `build_axis_transports` →
  `(B,4,2,Λ,2,2)`.
- Both models run and are gauge invariant to machine precision: **GELT 0.0,
  L-CNN 1.4e-14**, target itself 1.5e-16.
- **`LCNN` already takes `reduction="none"`** and returns `(B, *Λ)` — no per-site
  head to write for the baseline.
- `staple_sum(..., batched=True)` is exactly the flow drift, already vectorised.
- `data.build_plaquette_datasets` takes `target(configs, group) -> Tensor`, so the
  flowed density binds with `functools.partial`; `R` makes it emit `(X, T, y)`.
- `integrated_autocorrelation_time` sets `n_skip`.

`l1_ball_offsets(4,2) = 40` against the glueball's 24 → the transport stage
(already 62.8% of a GELT step) grows 1.67× per site.
**`d_qkv ≥ 2D = 8`** is a hard constraint in D=4 (caveat 2) — assert it.

## 7. Build order

| WP | what | gate |
|---|---|---|
| **0** | `topological_charge_density(..., definition="clover")` + the parity test | parity-odd to 1e-14 in complex128 |
| **1** | `gelt/flow.py`: `wilson_flow`, closed-form SU(2) exp + `[·]_TA`, batched; Lüscher RK3 with Euler as cross-check; `flow_trajectory` emits the 5-rung ladder in one pass | flow is gauge covariant, stays on the group, RK3 ≡ Euler(ε→0) |
| **2** | `scripts/measure_topology.py`: ensembles, **Q(t) plateau check**, `τ_int(Q)`, classical arms (identity + best linear filter) | **Gate 0**: `t/a²=2` on a plateau at every β, Q near integer. If not at β=2.3, move the rung and record it |
| **3** | `scripts/train_topology.py`: per-site MSE, one `_build_model()`, `TOPO_ARCH=lcnn` switches architecture and nothing else | matched-budget test (real DOFs within 15%) passes in D=4 |
| **4** | trainings: 3 per-β + 1 mixed-β, × 2 architectures, + untrained controls (3 seeds, **median** fixed in advance) | — |
| **5** | `scripts/fit_topology.py`: M1–M6 offline, correlated `--vs` difference inside every jackknife sample | — |

## 8. Design decisions that are load-bearing

- **β ∈ {2.3, 2.4, 2.5} at L ∈ {8, 12, 16}** so the physical volume is ~constant
  (≈1.3 fm)⁴ while `a` halves — both architectures are translation-equivariant, so
  training at mixed sizes is legitimate (the L-CNN paper trains on 4×8³ and
  evaluates on 8×24³). **Re-derive the scale in-repo** from
  `rectangular_wilson_loop`; do not import `a√σ` from the literature into a
  physics claim.
- **Fixed `t/a²`, not fixed physical `t`.** This holds `r_sm/a` — and therefore the
  receptive-field requirement — constant across β, so a mixed-β model's
  degradation is attributable to *scale adaptivity* and not to a capacity wall
  that bites at one β only. Fixing `t` physically confounds the two.
- **Flow-time ladder `t/a² ∈ {0.5,1,2,4,8}`** (`r_sm/a = 2.0…8.0`), one trajectory,
  five targets. The top rung sits at the receptive-field boundary on purpose:
  it is the built-in capacity control.
- **The best-linear-filter arm is the most important control.** The
  least-squares-optimal Manhattan-8 isotropic filter on `q_clover[U]`, fitted on
  train. The gap between it and the best network *is* the nonlinear content of the
  flow. If it is not clearly beaten, **stop** — the task is a convolution.
- Splits contiguous and chain-ordered, 800/200/200, test at the far end of the
  chain (the glueball protocol; §9.1 of the shootout note is why it matters).

## 9. Readings, fixed in advance

- **R1 (primary, architecture).** Δ(1−R²), mixed-β GELT vs mixed-β L-CNN,
  correlated jackknife. GELT wins at >2σ; parity within 1σ.
- **R2 (control, adaptivity).** Same at fixed β. *Prediction: parity, or a
  clearly smaller margin than R1.* If single-β already separates, the effect is
  not scale adaptivity — re-stratify by measured ρ and say the control failed.
- **R3 (control, capacity).** Accuracy across the flow-time ladder. Both must
  collapse as `r_sm/a → 8`. A later collapse for GELT is receptive-field
  efficiency, not adaptivity, and must be reported as such.
- **R4 (physics, architecture-independent).** Does *either* net reproduce
  χ_top and the Q-histogram from τ=0 within the flowed reference's error? **This
  is the deliverable that does not depend on who wins.**
- **R5 (nulls).** Untrained arms must fail; identity and best-linear-filter must
  be clearly beaten; top-1 configuration share under the §9.2 gate.
- **M4 (stratified) is the reading to look at first** — the dislocation-rich tail
  (top percentile of |q_clover[U]|, top decile of configuration roughness) is
  where the mechanism predicts the separation.

## 10. Odds, honestly

**P(clear GELT win on R1) ≈ 50–55%**, revised down from 65% by this audit:

1. the surviving mechanism is one axis (§2), not four;
2. the raw→flowed map is *mostly* a smoothing, i.e. mostly a convolution — the
   adaptive part is a correction whose size WP2's linear-filter arm measures;
3. the base rate in this repo for "attention will help here" is currently 0 for 1.

**But the risk profile is better than that number.** Two separable deliverables:
*(a)* does a flow-free learned topological density work at all, and how much of
flowed topology is local — novel whoever wins, ~90%; *(b)* does input-dependent
offset weighting help — the coin flip, with a clean pre-registered A/B that makes
a null publishable. A second null after the shootout is genuinely informative:
it would say the mechanism does not pay on *local regression* either, which
sharply narrows where it could ever pay.

**So: frame it as a study of the flow-free topological density with a controlled
architecture A/B inside it, not as "GELT beats L-CNN at topology".**

## 11. Cost

~4–5 V100 nights + ~1 build week: ensembles ~1.5 nights, flow targets 1–2 h,
GELT trainings ~2 nights, L-CNN ~0.5 night, the rest offline.

Two known risks priced in `reports/topology/flow_free_topology.tex` §8:
**topological freezing at β = 2.5** (measure `τ_int(Q)`, not `τ_int(plaquette)`;
if it bites, the per-site readings M1/M4 survive and the aggregate M2/M3 are
quoted at β = 2.3, 2.4 only), and **GELT's transport cost in 4D** — for which
`notes/performance_audit.md` §5.1 (the adjoint SO(3) representation, predicted
1.8× end to end) is a better bet here than it was on the glueball task.
