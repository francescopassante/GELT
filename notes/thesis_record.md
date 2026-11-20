# GELT — the research record

**Consolidated 2026-09-23.** This note replaces the fourteen design records,
audits, status pages and the supervisor update that `notes/` held until today
(list and concordance in §12). It is organised by **aspect**, not by date: each
section takes every attempt that touched one question, states what they add up
to, and says how sure we are. The run-by-run history is kept only where the
history *is* the argument (why anisotropy, why smeared inputs, why a comparator
was replaced). Everything deleted is recoverable with
`git show 570c208:notes/<file>`.

Raw evidence that is not prose stays next to this file, in `notes/raw/`: the
verbatim readouts whose dumps live only on the V100
(`m1_probe_readout_2026-09-16.txt`, `z2_vortex_readout_2026-09-20.txt`,
`beta_transfer_readout_2026-09-20.txt`, and since 2026-09-29
`cnn_vs_gelt_readout_2026-09-29.txt`) and the six figures made for the
18 September update (`notes/raw/figures/`, untracked — `*.png` is gitignored).
`notes/papers_review.md` and `notes/resources.md` are literature, not records,
and are kept as they were.

**Status words used throughout.**

| word | meaning |
|---|---|
| **ESTABLISHED** | measured with its controls, replicated or pinned by a test, survives every audit so far |
| **QUALIFIED** | the measurement stands, but the sentence it is quoted in (main.tex, an old note) says more than it supports |
| **OPEN** | measured once, directional only, or confounded; a named experiment in §11 would settle it |
| **RETRACTED** | was claimed; a later measurement refuted it |
| **DISCARDED** | a study that closed without anything the thesis needs; only its lesson is kept (§10) |

---

## 1. The ledger — every claim, one line each

| # | claim | status | where |
|---|---|---|---|
| L1 | GELT and both L-CNNs are gauge equivariant to machine precision (8.9e−16, complex128) | ESTABLISHED | §2.1 |
| L2 | The samplers are exact (2D SU(2) `I₂/I₁`, anisotropic ξ=1 at 0.1σ, batched multichain pulls ≤ 1.3σ) | ESTABLISHED | §2.1 |
| L3 | GELT (~1.5k par.) regresses a per-site 1×2 Wilson loop that a ~500k-par. CNN cannot | ESTABLISHED | §2.2 |
| L3′ | "GELT also comfortably regresses 2×2, 1×3, 2×3, 3×3 loops" (main.tex) | **no artifact or record anywhere in the repo** for the Z₂ setting it is written in; an SU(2) replacement exists (L3″) | §9 |
| L3″ | In 1+1D SU(2) GELT regresses per-site W²ˣ², W³ˣ³, W⁴ˣ⁴ (R²_β ≥ 0.99998) where a non-equivariant CNN at 40k and at 523k parameters gets R²_β = 0.077 → 0.003 → 0.000 | ESTABLISHED (3 seeds × 2 CNN sizes; the CNN lands on the Letter's own baseline-CNN MSEs) | §2.5 |
| L4 | Our L-CNN reproduces the L-CNN half of PRL 128 032003 Fig. 3 and beats its four MSEs | ESTABLISHED *under the lattice-averaged MSE reading*; flips for W⁴ˣ⁴ under the per-site one | §2.3 |
| L5 | At matched 39.6k parameters GELT solves W⁴ˣ⁴ (per-site R² = 0.99998) but at 16× the L-CNN's MSE | ESTABLISHED (n = 3 vs 3, disjoint ranges); cause OPEN; the same ordering, wider, on W²ˣ² (§2.5) | §2.4 |
| L6 | The 0⁺⁺ needs an anisotropic lattice, smeared multi-level inputs and a per-timeslice network | ESTABLISHED (design facts) | §3.1 |
| L7 | The learned operator has the classical mass and more ground-state weight than the 4-level GEVP on the same inputs: Δm_eff(1) = −0.028 ± 0.007 / −0.038 ± 0.008, ΔA₀ = +0.078 ± 0.022 (3.6σ) | ESTABLISHED, **replicated on a third ensemble**: three-ensemble ΔA₀ = +0.086 ± 0.017 (stat) ± 0.018 (window) = 3.4σ with the window systematic; Δm_eff(1) = −0.037 ± 0.004; ensembles consistent (p = 0.71); a fully correlated fit gives +0.049 ± 0.017 | §3.2 |
| L7′ | The learned operator is a 0⁺⁺ (A₁⁺⁺) operator, not only gauge invariant at zero momentum | ESTABLISHED to 98%: 1.2–2.1% of C(0) outside A₁⁺⁺ over the five trained nets (mostly Eg + T2g), untrained nets 21–83% by init seed (Eg); L8 on the projected 7-level nets: +0.081 ± 0.020 vs `deep`, +0.027 ± 0.023 vs `full`; projecting changes A₀ by +0.005 ± 0.005 (4-level, three ensembles), projected headline ΔA₀ = +0.088 ± 0.016 ± 0.016 (3.9σ); between the pre-registered readings R1 and R2, adoption of the projected headline OPEN | prof_notes §S4 |
| L8 | …and against stronger classical bases | **QUALIFIED**: +0.095 ± 0.021 vs the input-matched 7-level arm, only +0.038 ± 0.027 (1.4σ) once loop shapes are added; A₀ is saturated there. At (t₀, t_d) = (0, 1), where `shapes` / `shapes_sm` pass the gate: +0.085 ± 0.023 vs `deep`, +0.034 ± 0.014 vs `full`, and the **4-level** net is behind the shape bases (−0.034 ± 0.010 vs `full`) | §3.3 |
| L8′ | On its **own training loss** the 4-level net is behind the shape bases on the same four levels (ΔL = +0.021 ± 0.003 vs `shapes_sm`, +0.028 ± 0.003 vs `full`, three ensembles, classical arm cross-fitted) and adds nothing to them (−0.0003 ± 0.0004); the 7-level net ties `full` (−0.002 ± 0.004) and adds −0.004 ± 0.001 | ESTABLISHED (2026-11-12, offline) | §3.8 |
| L8″ | The true ground-state overlap of the 4-level net is ≈ 0.88 (fit-free, 40k configurations), not the 0.93 a one-state fit on [2,7] reads: "A₀ saturates above ≈ 0.9" is the estimator, not the operators | ESTABLISHED for one net | §3.8 |
| L8‴ | Distilling the 4-level net onto its own Euclidean future (§11 P11, ens2, lag 2) leaves it unchanged: ΔL = +0.0020 ± 0.0013, Δm(1) −1.4σ, union with `full` −0.0004 ± 0.0016 — reading R-T0. **Low power**: the objective's only visible gain was the rescale c ≈ C(2)/C(0) and its selection ran on a constant offset, so this does not show the class lacks the evolved operator | OPEN (2026-11-19, one run; the connected-loss rerun settles it) | §11 P11 |
| L9 | The advantage is learned, not architectural | ESTABLISHED (untrained nets below the GEVP at every input depth; decomposition sign flips) | §3.4 |
| L10 | What was learned: ~80% of the out-of-span content is rectangular loops, ~2.5% of the norm² is outside every classical operator | ESTABLISHED (estimator-dependent between 55% and 80%) | §3.5 |
| L10′ | …and that ~2.5% carries ΔA₀ = +0.021 ± 0.009 | **RETRACTED** (2026-10-21): a fitted-mass artifact (m_G − m_P = +0.010 ± 0.004); at a common reference mass removing it gives −0.011 ± 0.005 (35-op `full`), −0.008 ± 0.005 (`full21`) | §3.5 |
| L11 | "The classical smearing ladder is a converged geometric series" | RETRACTED (tail measured at ≈ 4%, non-monotone) | §3.5 |
| L12 | GELT's glueball result is initialisation-robust | ESTABLISHED at n = 3 (ens0: +0.066, +0.067, +0.076) | §3.6 |
| L13 | The attention map is a gauge-invariant lattice operator with a mass tracking ξ | ESTABLISHED, but it is close to a theorem — the random net does it too | §4.1 |
| L14 | Training raises the attention field's ground-state overlap (Z₂ +0.11…+0.27, SU(2) +0.286 ± 0.056) | ESTABLISHED as a trained-minus-random difference; **what it means is OPEN** (no inheritance control) | §4.2 |
| L15 | "The untrained attention field is already better than the classical GEVP" (main.tex, Z₂) | **QUALIFIED, probably false**: the Z₂ GEVP column is one gauge-variant operator; false in SU(2) (0.59 vs 0.88) | §4.3 |
| L16 | "The trained net reads a larger ξ than the GEVP, which means better overlap" (main.tex) | **QUALIFIED**: consistent with the A₀ ordering, no independent reference left in the repo | §4.3 |
| L17 | The attention field reads the ensemble, not the training β (cross-β matrix) | ESTABLISHED (Z₂) | §4.1 |
| L18 | GELT ties our matched L-CNN on the glueball: ΔA₀ = +0.012 ± 0.009 | ESTABLISHED for **our reimplementation** | §5.2 |
| L19 | GELT beats the authors' L-CNN on the glueball: ahead in 4 of 4 pairings, median +0.101 on ens0, GELT's worst run above their best | ESTABLISHED as an accuracy gap at n = 3 on one ensemble; **its cause (representation vs optimisation) is OPEN** | §5.2 |
| L20 | "Attention buys robustness" | **QUALIFIED** to a failure-rate asymmetry on SU(2) tasks that reverses on Z₂ in distribution and is absent in 1+1D | §5.4 |
| L21 | Input-dependent offset weighting (M1) pays | ESTABLISHED on a constructed target (6/6, p = 0.031); **never on a physics task** (0 of 3) | §5.3 |
| L22 | The softmax is worth ≈ 0.48 of R² at matched parameters | ESTABLISHED, as *trainability*; its split into sign / normalisation / bound RETRACTED | §5.3 |
| L23 | Shortest-path averaging of the transport buys accuracy | **OPEN, and never observed** on any task; untested on the glueball | §5.5 |
| L24 | Boundedness (M2) is why L-CNNs fail | OPEN — the counts fit it, nothing separates it from the transport | §5.4 |
| L25 | On 3D Z₂ vortex-cluster size both architectures beat the matched-depth classical algorithm, and tie each other | ESTABLISHED at one β, 70 training configurations | §5.1 |
| L26 | Offset reweighting helps under coupling transfer | RETRACTED (null, p ≥ 0.24) | §5.1 |
| L27 | Our L-CNN is a strict low-rank restriction of the authors' L-CB; the vendored L-CB is one slot wider than the Letter's | ESTABLISHED (constructive, 1.9e−15; Table V reproduced exactly) | §6 |
| L28 | GELT's step costs 3.9× our L-CNN's; the transport is 62.8% of it | ESTABLISHED against ours; unmeasured against theirs | §8 |
| L29 | 4D APE at α = 0.5, n steps ≡ Wilson flow at t = n·α/6 | ESTABLISHED (standing fact) | §7 |
| — | Z₂ APE smearing at α = 0.5 is not gauge covariant and freezes after one step | ESTABLISHED **defect, unfixed** | §9 |

---

## 2. Validation and supervised regression

### 2.1 Equivariance and samplers — ESTABLISHED

- `pytest tests` covers gauge equivariance of `GEMHSA`/`GELT` (SU(2) complex128,
  both gates; Z₂ float64), the optimised attention path against a naive oracle
  (outputs and input gradients to 1e−12, SU(2)/SU(3)/Z₂), the frozen-α arm, and
  the same for `lcnn`, `lcnn_reference`, `lcnn_exact`.
  `scripts/check_gelt_invariance.py` prints the end-to-end drift, 8.9e−16.
- The *architecture* is exactly invariant; the **Z₂ input pipeline is not** at
  α = 0.5 (§9). Say "the network is invariant", not "the pipeline is".
- Heat-bath reproduces `I₂(β)/I₁(β)`; overrelaxation conserves the action to
  round-off; the anisotropic refactor reproduces ξ = 1 at 0.1σ; the batched
  Letter-style Metropolis hits `I₂/I₁` at three couplings at once (pulls −0.5,
  −1.3, −0.4σ). The Creutz `w₀` cap had to go 100 → 1000: at a = β·ξ·6 ≈ 43 one
  failure per ensemble is *expected* (it killed the ens2 phase).

### 2.2 The CNN contrast — ESTABLISHED, one sentence to fix

`train_cnn.py` / `train_gelt.py` are one problem by construction (D = 3, L = 8,
Z₂, Haar links, 1×2 loop). With plaquettes as input the *action* is a linear sum
and both reach R² ≈ 1; the 1×2 loop is a product of transported plaquettes, and
the non-equivariant CNN cannot find it at ~500k parameters while GELT does at
~1.5k (main.tex says ~1k; README says ~1.5k — pick one, from the script's own
count). main.tex's further claim about 2×2, 1×3, 2×3, 3×3 loops has **no
artifact, log or note** anywhere; either run it (§11 P9) or cut it.

### 2.3 Reproducing PRL 128 032003, Fig. 3 (L-CNN half) — ESTABLISHED, conditional

1+1D SU(2), 8×8, the four traced loops W¹ˣ¹, W¹ˣ², W²ˣ², W⁴ˣ⁴, one network per
loop at SM Table V's largest architecture and SM §VI.A's hyper-parameters, the
authors' own layers through `gelt/lcnn_exact.py`.

| loop | ours (`mse_avg`) | Letter |
|---|---|---|
| W¹ˣ¹ | 8.6e−16 | 2.2e−11 |
| W¹ˣ² | 5.3e−16 | 2.1e−9 |
| W²ˣ² | 1.9e−12 | 1.1e−8 |
| W⁴ˣ⁴ | 9.4e−9 | 1.4e−7 |

- The two small loops sit at the **float32 rounding floor** (rms error 0.5 ε);
  W¹ˣ¹ is *constructively* exact in the 12-parameter net (a test).
- The pre-registered "within a factor of 10" fails for three of four — on the
  good side. The criterion should have been one-sided.
- **Everything rests on one reading**: Fig. 3's MSE is lattice-averaged (their
  `mse(global_average=True)` default; the CNN arm has a GAP and cannot be
  per-site). Read per site, W⁴ˣ⁴ goes from 15× better to 3.9× worse. Quote the
  convention with the number.
- Departures, all deliberate: one chain per configuration (independent by
  construction); the β ladder has 11 couplings (SM Table I vs §II disagree);
  links stored in float64. Their baseline-CNN half is not reproduced.

### 2.4 GELT on W⁴ˣ⁴ at matched budget — ESTABLISHED loss, cause OPEN

Same ensemble, splits, label, loss, optimiser, 100 epochs, lr 1e−3 (bracketed
for GELT *and* the Letter's rate); GELT at R = 3, 4 blocks, 39 569 real DOFs
against 39 905.

| | `mse_avg` over 3 seeds | per-site R² | rms error / sd(y) |
|---|---|---|---|
| L-CNN | 9.4e−9 … 3.1e−8 | 0.9999978 | 0.15% |
| GELT | 1.5e−7 … 4.7e−7 | 0.9999843 | 0.40% |

Δlog₁₀ = 1.21 ± 0.22 (16×, ranges disjoint, permutation p = 0.10 = the floor at
3 vs 3). Dispersion equal (3.3× vs 3.1× worst/best). **Read R² first**: both
solve the task, at the loop size where the Letter's baseline CNN predicts the
training mean; GELT lands on the Letter's own published number, and the gap
opened because our L-CNN beat its paper by 15×. Half of GELT's error is coherent
across the lattice (site/avg ratio 26 vs 57 for the L-CNN, 64 if independent).

- **Retracted hypothesis**: "path averaging dilutes a specific-path target".
  Axis-aligned offsets have a unique shortest path, so GELT's `T` there is an
  exact group element (1e−15); a rectangle is built only from those.
- **Leading hypothesis**: α is one convex scalar per (head, offset, site),
  shared by all channels of a head, where the L-CB's ω is a free weight per
  (out, in, in, offset) that can be zero or signed. Untested — §11 P6.
- **Side finding**: with a zero-initialised head the lr dependence is
  non-monotone — 3e−3 never leaves `var(y)`, 1e−2 diverges, 1e−3 works — which
  contradicts `train_gelt.py`'s "raise the rate to beat the stall". Probably
  Adam's `lr/(√v+ε)` with `v ≈ 0`. One geometry; do not generalise either way.

### 2.5 GELT against a non-equivariant CNN on W²ˣ², W³ˣ³, W⁴ˣ⁴ — ESTABLISHED

The SU(2) version of main.tex's validation figure (§2.2), on §2.3's 1+1D
problem: same datasets, splits, per-site MSE loss, AdamW at 1e−3, 100 epochs /
patience 25. `WR_ARCH=cnn` is `LatticeCNN` on the raw re/im components of links
*and* plaquettes (24 channels, 3×3 circular convs, receptive field ±5 ⊇ the
4×4 loop), at `matched` 40 055 parameters and `large` 523 377 (main.tex's
~500k). GELT is §2.4's W⁴ˣ⁴ network on W³ˣ³ and W⁴ˣ⁴ (39 569) and a 2-block
one on W²ˣ² (20 873), so the CNN is never the smaller arm. W³ˣ³ is not in the
Letter; older datasets get it from the float64 links (identical to the
generator's label, checked). Verbatim table:
`notes/raw/cnn_vs_gelt_readout_2026-09-29.txt`; figure
`results/wilson_regression/fig3_cnn_vs_gelt.png` (V100).

**The reading is R²_β = 1 − MSE_site / Var(y | β)**, not the plain R²: the
labels span an 11-coupling ladder, and any network that can read the local mean
plaquette gets the β dependence of ⟨W⟩ for free. R²_β = 0 is "knows the coupling,
nothing else".

| loop | GELT R²_β (best of 3) | CNN matched R²_β | CNN large R²_β | GELT `mse_avg` best / worst | CNN `mse_avg` | Letter's CNN |
|---|---|---|---|---|---|---|
| W²ˣ² | 0.9999979 | 0.077 | 0.074 | 1.3e−8 / 2.7e−8 | 4.0e−3 | 4.0e−3 |
| W³ˣ³ | 0.9999911 | 0.0028 | 0.0024 | 3.7e−8 / 2.9e−7 | 5.0e−3 | — |
| W⁴ˣ⁴ | 0.9999843 | −0.0003 | −0.0002 | 1.5e−7 / 4.7e−7 | 4.3e−3 | 4.2e−3 |

- **GELT solves all three**; 1 − R²_β grows with the loop, 2.1e−6 → 8.9e−6 →
  1.6e−5. The W⁴ˣ⁴ rows are §2.4's dumps.
- **The CNN never forms the loop.** What it has beyond β is 8% of the
  conditional variance at W²ˣ², 0.3% at W³ˣ³, nothing at W⁴ˣ⁴ (slightly below
  the β-only predictor). Its plain per-site R² of 0.13 at W²ˣ² is mostly the
  coupling. **13× the parameters buys nothing**: `large` is within noise of
  `matched` at every loop.
- **Not a budget or a seed**: every CNN run early-stopped at epoch 27–31 on
  patience 25, i.e. its best epoch was 2–6 — it reaches its plateau at once and
  stays there; the three seeds agree to < 1% (median ≈ worst). The learning rate
  was not bracketed for the CNN (1e−3, the Letter's); that is the one knob left,
  and the next point makes it an unlikely one.
- **The CNN lands on the Letter's own baseline-CNN MSEs** — 3.9–4.0e−3 against
  4.0e−3 at W²ˣ², 4.26e−3 against 4.2e−3 at W⁴ˣ⁴ — with a different
  architecture, a per-site head instead of a global pool, and one run per cell
  against their 2 680-model sweep. This reproduces the *number* of the half of
  Fig. 3 §2.3 set aside, not their networks. Consistent with both sitting at the
  floor β leaves, which R²_β ≈ 0 shows for ours at W⁴ˣ⁴; the floor of the
  lattice-averaged MSE itself is not computed.
- **GELT against the L-CNN on W²ˣ²** (a by-product, §5.1 A7): the L-CNN (13 521,
  one seed, §2.3's run) is 6.8e3× lower on `mse_avg` and 1.9e4× per site, at
  fewer parameters than GELT. Same ordering as W⁴ˣ⁴ (§2.4), wider; both R² ≈ 1.
  GELT lands on the Letter's *published* L-CNN number at both loops (1.2× at
  W²ˣ², 1.1× at W⁴ˣ⁴) — noted, not interpreted.
- GELT's seed dispersion (worst/best `mse_avg`) is 2.0× / 7.7× / 3.1× across the
  three loops; W³ˣ³ is the loose one.

**What this licenses in main.tex**: "GELT regresses per-site 2×2, 3×3 and 4×4
SU(2) Wilson loops (R² ≥ 0.99998 with the coupling dependence removed), where a
non-equivariant CNN with 13× more parameters recovers < 8% of the variance at
2×2 and none at 4×4." It does **not** license the sentence as written (§9 item
4): that one is about Z₂ in 3D and 1×3 / 2×3 loops, which were not run.

---

## 3. SU(2) 0⁺⁺ spectroscopy — the physics result

### 3.1 How the pipeline was found (the history that is the argument)

| run | setup | outcome | lesson |
|---|---|---|---|
| 0–1 | isotropic L = 12, β = 2.4; single smeared op, then 4-level GEVP | signal dead by Δ ≈ 3, no plateau for any basis | the lattice, not the basis, is the bottleneck |
| 2–3 | anisotropic 24 × 12³, ξ = 3 | GEVP plateau m·a_t ≈ 0.33 | **anisotropy is required** |
| audit | — | a 4D network voids the transfer-matrix bound (loss gameable toward m → 0); zero-init head ⇒ exactly zero Rayleigh gradient | **per-timeslice 3D network**, `mlp_zero_init=False`, hard train/val/test split |
| 4 | GELT on thin plaquettes | right mass at Δ = 3 but m_eff(1) = 0.53 vs GEVP 0.40; learned ≈ APE×2 (corr 0.87); adding it to the GEVP adds nothing | depth cannot rebuild iterated smearing |
| 5 | inputs = APE levels (0, 2, 4, 6) | beats the GEVP; needed the `(log C(0))²` scale pin (C(0) ran to 1e73 without it) | **smeared multi-level inputs are required** |
| rep. | fresh seed-1 ensemble, from scratch | reproduces, Δm_eff(1) = −0.038 ± 0.008 | the loss floor is ensemble-specific; operator quality is what replicates |

Setup of record: `GELT(D=3, R=2, 4 layers, 2 heads, d_qkv 6, d_model 16)`,
~15.7k real DOFs, loss −mean[C(1)/C(0), C(2)/C(0)] + scale pin, AdamW 3e−3,
batch 6 configs (= 144 slices; **batch is a physics knob** — it sets the VEV
estimate), 70/10/20 contiguous split, 400 test configurations, blocked jackknife
block 10, cosh fit on Δ ∈ [2, 7], A₀ = A(1 + e^{−mN_t})/C(0), GEVP (t₀, t_d) =
(1, 2). The mass (m·a_t ≈ 0.33–0.37) is **not** continuum physics: β_s = 0.8 is
strong coupling and ξ_R ≈ 3.32 ≠ 3.

### 3.2 The headline, as main.tex states it — ESTABLISHED as worded

| | ens0 (Run 5) | ens1 | combined |
|---|---|---|---|
| Δm_eff(Δ=1), GELT − GEVP | −0.028 ± 0.007 | −0.038 ± 0.008 | — |
| A₀ GELT / GEVP | 0.903(47) / 0.837(56) | 1.013(62) / 0.925(71) | |
| ΔA₀ | +0.066 ± 0.031 | +0.089 ± 0.030 | **+0.078 ± 0.022 (3.6σ)** |
| Δm (fit) | −0.008 ± 0.014 | −0.025 ± 0.013 | consistent with 0 |

**Fit-window systematic (supervisor's S1 task 1, 2026-09-27,
`scripts/fit_window_scan.py`).** Over [2,7], [3,7], [2,6], [3,8], [2,8] the
combined ΔA₀ stays ≥ 3.0σ (range +0.074 … +0.111), but its spread, 0.0375,
exceeds the [2,7] error, so by the reading fixed in `notes/prof_notes.md` S1
the headline becomes **+0.077 ± 0.022 (stat) ± 0.019 (window)** and the
abstract's "3.6σ" is rewritten. The Δmin = 3 windows move it *up*, mostly on
ens1. The +0.078 above combines the rounded per-ensemble values; from the
unrounded ones it is +0.077. Δm (fit) stays within 2σ of zero in every window.

**Third ensemble (ens2, seed 2, 2026-09-28; readings fixed beforehand in
`notes/prof_notes.md`).** Same protocol, trained from scratch: ΔA₀ = +0.100 ±
0.028 (3.6σ), Δm_eff(1) = −0.041 ± 0.005. The three ensembles agree (χ² =
0.69 on 2 dof, p = 0.71). **Three-ensemble headline: ΔA₀ = +0.086 ± 0.017
(stat) ± 0.018 (window) = 3.4σ including the window systematic** (5.0σ
statistical; ≥ 4.1σ in every window); Δm_eff(1) = −0.037 ± 0.004 (9.5σ);
plateau at Δ = 1 compatible (+1.3σ); correlated fit +0.049 ± 0.017. The
two-ensemble numbers below are kept as the history.

**Correlated fits (S1 task 2).** The quoted diagonal fits pass the correlated
test (χ²/dof 0.46 / 0.43 at [2,7], p ≈ 0.9, both ensembles summed). The
*correlated fit* gives ΔA₀ = **+0.031 ± 0.021 (1.5σ)**; covariance shrinkage
Σ_λ = (1−λ)Σ + λ·diag Σ puts the whole drop at λ < 0.1 (λ = 0.1 → +0.054,
3.0σ; 0.25 → +0.064; 1 → +0.077). A synthetic test settles it
(`scripts/fit_estimator_mc.py`: pure-cosh truth, the measured joint
covariance, 1000 × 40-block experiments per ensemble, the scan's own fits):
**neither fit is biased on ΔA₀** (|bias| ≤ 0.002), the correlated fit is not
more precise (RMSE 0.021 vs 0.021), and the observed gap −0.047 sits in a
null of −0.001 ± 0.024, **p = 0.057** — a ~1.9σ fluctuation between two
estimators on the same data. So the diagonal fit stays and +0.031 is quoted
as a noisier alternative; an earlier reading of it as a correlated-fit
artifact is retracted. Not pre-registered: λ = 0.1 is the most precise
estimator in the test (RMSE 0.016) and gives +0.054 ± 0.018 (3.0σ) —
reported, not adopted. Each A₀ alone is biased up by +0.004 … +0.010 under
the diagonal fit; it cancels in ΔA₀.

**Autocorrelations and block size (S3 tasks 1–2, 2026-09-29,
`scripts/block_size_scan.py`).** τ_int (Madras–Sokal, c = 6) of Ō averaged
over t, on the 400 chain-ordered test configurations: ≤ 1.12 ± 0.31 saved
configurations for thin, APE×6, GEVP-projected and GELT on all three
ensembles (GELT = GEVP within errors); the per-configuration C(1) term 0.43–0.65;
the b = 1 jackknife pseudo-values of every quantity 0.33–0.65, i.e.
consistent with ½. ρ(t) (kept to t = 20 in the .pt, figure `…_rho.png`):
GEVP and GELT significant only at t = 1–2 (ρ(1) = 0.12–0.34) and nearly
identical; beyond the window, ~0.1 oscillations (ens0 t ≈ 10–11, ens2
t ≈ 7–12, ens2 thin plaquette to t = 8) at 1–2σ per lag. Summed to t = 12 or
20 instead of W, τ_int ≤ 1.56 — still 2τ ≈ 3 < b. σ(b) for b ∈ {1, 2, 5, 10, 20, 40}, fit weights fixed at
b = 10 (gated on the per-ensemble ΔA₀ and Δm_eff(1) above): by the reading
fixed beforehand (flag only if the three-ensemble mean of σ(b)/σ(10) exceeds
1 + noise at *both* b = 20 and 40) **b = 10 is on the plateau for every
quantity** — ratios at 20/40 are 0.84–1.09 — and nothing in chapter 5 is
re-quoted. Combined ΔA₀ error 0.016–0.018 at every b ≤ 20. One caveat: the
combined Δm_eff(1) error at b = 10 (0.0038) sits 17–25% *below* the b ≤ 5
values (ens2 up to 39%); falling, not rising, with pseudo-value τ_int
0.33–0.45, so a 40-block noise fluctuation rather than missed
autocorrelation. At the most conservative b (= 1) it is −0.036 ± 0.005,
6.9σ instead of 9.5σ.

**Boundary-leakage check (S3 task 3, 2026-09-29, `scripts/gap_scan.py`).**
ΔA₀ and Δm_eff(1) after dropping the first g ∈ {0, 5, 10, 20} test
configurations (the only ones adjacent to the validation block), everything
else as in the scan: [2,7], b = 10, fit weights and v₀ recomputed on the
reduced sample; g = 0 gated on the per-ensemble values above. ΔA₀ (ens0 / ens1 /
ens2 / combined): g = 0 +0.066(31) / +0.089(30) / +0.100(28) / +0.086(17);
g = 5 +0.067(32) / +0.093(25) / +0.102(29) / +0.089(16); g = 10 +0.066(31) /
+0.094(31) / +0.099(29) / +0.087(17); g = 20 +0.061(32) / +0.081(29) /
+0.102(29) / +0.083(17). Largest shift from g = 0, in units of the g = 0
error: 0.24σ (ens1, g = 20); the prof_notes reading ("beyond 1σ → escalate")
does not fire. Δm_eff(1) combined −0.037 / −0.035 / −0.037 / −0.037 (±0.004–0.005).
The g = 5 ens1 error (0.025) is smaller than at g = 0 (0.030) — a
40-block fluctuation, not information. Thesis: ch. 4 `tab:gap_leakage`.

**Plateau at Δ = 1 (S1 task 4).** m_eff(1) − m_fit([2,7]) for GELT, same
jackknife sample: +0.037 ± 0.021 (run5), −0.022 ± 0.026 (ens1), **+0.013 ±
0.016 combined (0.8σ)** → "compatible with a plateau from Δ = 1 within
errors" — on the combined value only; run5 alone, whose 0.370(16) main.tex
tabulates, is 1.8σ. Combined Δm_eff(1)(GELT − GEVP) = **−0.032 ± 0.006
(5.9σ)**, the statement to lead with.

Two points that do *not* add independent evidence and should not be sold as
such: "the enlarged GEVP + GELT collapses onto GELT" is what any GEVP does when
one member is the best operator; and "the converged loss decodes to the mass"
assumes a single-state correlator and reproduces the ensemble's own m_eff(0→1),
so it is a consistency check, not a measurement.

### 3.3 Is the comparator a straw man? — the claim QUALIFIED

The 4-level GEVP is four smearing levels of **one** loop shape. The fair fight
built stronger arms and, after two false starts (below), gives one consistent
picture. `deep` = 7 levels (0…16) × plaquette; `full` = 7 levels × {1×1, 1×2,
2×2}, 21 operators. The network was retrained on the 7-level ladder so that
inputs match `deep`.

**Naming, 2026-10-02.** In `su2_fair_fight.py` this 21-operator arm is now
`full21`, and `full` is 7 levels × the five shapes of `shapes_sm` (35
operators, a superset of every arm). **Every `full` elsewhere in this record
is `full21`.**

**The 35-operator `full` at (t₀, t_d) = (0, 1)** (2026-10-02, V100 pass for the
six missing operators, then `SFF_BASES=1 SFF_TRUNCATE=1 SFF_T0=0`; outputs
`su2_fair_fight_full35_{4lv,7lv}_trunc_gevp0-1.pt`). No fallback on any
ensemble; 14 of 35 directions kept (cond C(t₀) ≈ 1.8e10); the superset
ordering holds on every ensemble.

| arm | classical A₀ (run5 / ens1 / ens2) | 4-level GELT, three ensembles | 7-level GELT, run5 + ens1 |
|---|---|---|---|
| `shapes_sm` (20) | 0.947 / 1.025 / 0.965 | −0.037 ± 0.012 (3.2σ) | +0.028 ± 0.011 (2.6σ) |
| `full21` (21) | 0.940 / 1.022 / 0.966 | −0.034 ± 0.010 (3.5σ) | +0.034 ± 0.014 (2.4σ) |
| **`full` (35)** | 0.965 / 1.034 / 0.999 | **−0.060 ± 0.015 (4.0σ)** | **+0.015 ± 0.010 (1.5σ)** |

Per ensemble, 4-level: −0.062 ± 0.029, −0.020 ± 0.032, −0.075 ± 0.021;
7-level: −0.010 ± 0.016, +0.032 ± 0.013. Three things. (i) Levels 8–16 on the
two large shapes are not nothing: `full` is above `shapes_sm` by 0.018 / 0.009
/ 0.034, which qualifies (ii) of the paragraph below ("smearing levels 8–16
add nothing" was `full21` against `shapes_sm`, i.e. without those six
operators). (ii) The 7-level net's lead over the strongest classical basis
drops from 2.4σ to 1.5σ and changes sign on run5: against a basis with its own
inputs plus loop shapes it is **level, not ahead**. (iii) The 4-level net is
behind on all three ensembles.

**Not in-sample optimism.** Half-split (v₀ from one half of the test split,
scored on the other, swapped; one-off check, same convention as the one that
cleared `shapes_sm` and `full21`, whose −0.035 ± 0.013 and −0.033 ± 0.011 it
reproduces): 4-level GELT − `full` = **−0.054 ± 0.016 (3.5σ)** (−0.067 ±
0.030, −0.020 ± 0.034, −0.061 ± 0.022), 7-level **+0.012 ± 0.015 (0.8σ)**
(−0.015 ± 0.023, +0.032 ± 0.020); classical A₀ 0.970 / 1.033 / 0.985. So the
in-sample numbers above carry ≈ 0.005 of optimism, not the effect. **At (1,2)
the 35-operator arm is not a measurement**: the same half-split reads A₀ =
0.72 ± 0.74, 0.18 ± 0.12, 0.54 ± 1.05, as `shapes_sm` does there. **The caveat
that stays:** the classical A₀ is 0.97–1.03, inside the saturated region where
this record says not to rank — read the 4-level result as "behind a basis with
loop shapes at every smearing level", the 7-level one as "level with it".

**The decomposition does not move** (`operator_decomposition.py`, 7-level net,
`--basis=…:deep --shape-span=…:full`, now the 35-operator span; artifacts
`operator_decomposition_deep_7lv_full35`, `…_rnd_fitok_full35`): the fraction
of r's norm² inside the shape span is 0.766 ± 0.004 (run5) and 0.826 ± 0.004
(ens1), against 0.758 and 0.819 on the 21-operator span — §3.5's "80%" stands,
and the six added operators hold under one point of what the network found.
Untrained nets: 0.004–0.009, as before. The canonical `…_deep_7lv` artifacts
are the 21-operator ones and are reproduced exactly by `--shape-span=…:full21`.

**P1 against the 35-operator `full`** (2026-10-03, `p1_meff_strong_basis.py`,
whose arms are now `published`, `deep`, `full21`, `full`; the `full21` rows
reproduce the old `full` rows digit for digit). 7-level GELT − `full`,
Δm_eff(1) [Δm_eff(2)], run5 + ens1: (0,1) half-split **+0.001 ± 0.003 (+0.4σ)**
[+0.003 ± 0.005], (0,1) in-sample +0.003 ± 0.003 (+1.1σ) [+0.004 ± 0.004];
(1,2) half-split −0.15 ± 0.18 (not a measurement), (1,2) in-sample
+0.006 ± 0.005 (run5 alone: ens1 is ±0.06). |Δ| < 2σ everywhere → by the P1
reading, "matches" — level with the 35-operator basis on the unsaturated metric
too, sign now positive. 4-level GELT − `full`, three ensembles, (0,1)
half-split: **+0.027 ± 0.004 (6.8σ)**, behind. The row was added after the
readings were fixed for the 21-operator arm. Thesis: ch. 5 `tab:p1_meff`, row
`full35` (the thesis keeps `full` for the 21-operator arm and calls this one
`full35`).

| comparison (combined, two ensembles) | ΔA₀ |
|---|---|
| 7-level GELT − `published` (4 levels) | +0.131 ± 0.029 |
| **7-level GELT − `deep` (same inputs)** | **+0.095 ± 0.021 (4.5σ)** |
| 7-level GELT − `full` (inputs + loop shapes) | **+0.038 ± 0.027 (1.4σ)** |
| 4-level GELT − `full` | −0.013 ± 0.030 |

The **input↔architecture curve** (same width d_model 24 at every point, a
d_model 16 → 24 width control that moves nothing):

| inputs | classical A₀ | trained A₀ | untrained A₀ (3 seeds) | ΔA₀ trained − classical |
|---|---|---|---|---|
| thin | 0.407 ± 0.054 | 0.680 ± 0.053 | 0.257 ± 0.054 | +0.252 ± 0.056 |
| 4 levels | 0.870 ± 0.044 | 0.949 ± 0.039 | 0.512 ± 0.131 | +0.075 ± 0.019 |
| 7 levels | 0.921 ± 0.045 | 1.008 ± 0.041 | 0.547 ± 0.136 | +0.095 ± 0.021 (saturated) |

What the curve licenses, by its pre-registered tests: the advantage over the
same-input GEVP is positive at every resolvable depth and **shrinks as the
inputs get richer** (not additive, χ²/dof = 9.1 for a constant); a 4-level GELT
is worth the 7-level classical basis (+0.038 ± 0.024), a thin GELT is not worth
the 4-level one (−0.201 ± 0.048); the traces never meet before saturation.

**The consolidated statement for the thesis**: *given the same inputs, one
learned operator carries more ground-state weight than the optimal linear
combination of them (+0.075 at four levels, +0.095 at seven, ≥ 4σ); a classical
basis that adds rectangular loops reduces that to a 1–2σ edge; above ≈ 0.9 the
A₀ observable is saturated and cannot rank methods further.* The 7-level net was
trained **after** `deep` was seen to win on one ensemble — say so.

Estimator lessons that cost the most time and are why the numbers above can be
trusted:

- **A superset arm cannot interpolate worse than its subset.** The first fair
  fight reported an 8.8σ win because `full` read A₀ = 0.07 — a near-null
  direction of an ill-conditioned C(t₀). The script now refuses a verdict when
  the superset ordering fails.
- **Truncate, don't floor.** Flooring C(t₀)'s eigenvalues at ε·s_max and
  whitening gives floored directions weight 1/√ε. `truncate=True` drops them;
  it made `full` readable (9 of 21 directions) and moved its A₀ by 0.004 — the
  floor was not steering it, which nobody could know before trying.
- **The variational gate is a theorem only at t₀ = 0.** At (t₀, t_d) = (1, 2)
  the GEVP maximises C(2)/C(1), the A₀ gate tests C(2)/C(0); `shapes` and
  `shapes_sm` fall back on ens1 for that reason (at cond C(t₀) = 87), and no
  whitening can rescue them. They are off the (1,2) curve. **At t₀ = 0 they
  pass on every ensemble** (2026-10-02, `SFF_T0=0`, next paragraph).

**The fair fight at (t₀, t_d) = (0, 1)** (2026-10-02; `SFF_BASES=1
SFF_TRUNCATE=1 SFF_T0=0 python scripts/su2_fair_fight.py <dumps>`, offline;
without `SFF_T0` the same command reproduces every (1,2) number above). No arm
falls back on any ensemble, 26 curve dumps included. ΔA₀ = GELT − arm:

| arm | classical A₀ (run5 / ens1 / ens2) | 4-level GELT, three ensembles | 7-level GELT, run5 + ens1 |
|---|---|---|---|
| `published` | 0.864 / 0.942 / 0.847 | +0.067 ± 0.015 (4.5σ) | +0.110 ± 0.032 (3.4σ) |
| `deep` | 0.902 / 0.958 / 0.888 | +0.031 ± 0.014 (2.3σ) | +0.085 ± 0.023 (3.6σ) |
| `shapes` | 0.508 / 0.577 / 0.491 | +0.416 ± 0.037 | +0.458 ± 0.054 |
| `shapes_sm` | 0.947 / 1.025 / 0.965 | **−0.037 ± 0.012 (3.2σ)** | +0.028 ± 0.011 (2.6σ) |
| `full` | 0.940 / 1.022 / 0.966 | **−0.034 ± 0.010 (3.5σ)** | +0.034 ± 0.014 (2.4σ) |

`published` and `deep` are S2's (0,1) row (`gevp_time_scan.py`: +0.067 ± 0.015,
+0.031 ± 0.014) by another route. Three things this changes. (i) `shapes` and
`shapes_sm` are measurements, at a third to a half of the fallback's error.
(ii) `shapes_sm` = `full` to 0.007 on every ensemble: with the five shapes at
four levels, smearing levels 8–16 add nothing. (iii) **The 4-level GELT is no
longer level with the shape bases** (−0.001 ± 0.021 vs `full`, +0.003 ± 0.032
vs `shapes_sm` at (1,2)): at (0,1) it is behind both by ≈ 0.035, negative on
all three ensembles. That is at classical A₀ ≈ 0.95–1.0, i.e. in the saturated
region, so read it as "not ahead" and do not rank on it; the 7-level net stays
ahead at ≈ 2.5σ. Not in-sample optimism of a 20-operator v₀: a half-split
(v₀ from one half of the test split, scored on the other, swapped) gives
−0.035 ± 0.013 vs `shapes_sm` and −0.033 ± 0.011 vs `full` (one-off check in
`p1_meff_strong_basis.py`'s convention, no script in the repo), where the same
half-split at (1,2) reads A₀ = 0.21 ± 0.24 for `shapes_sm`. The curve at (0,1)
(`CURVE_EST=_trunc_gevp0-1 python scripts/input_architecture_curve.py`, written
beside the canonical one): classical 0.894 at four levels and 0.926 at seven,
ΔA₀ +0.054 ± 0.019 and +0.085 ± 0.023, rungs +0.024 ± 0.018 (4lv→7lv) and
−0.223 ± 0.045 (thin→4lv), χ²/dof = 11.4 for a constant — R1/R2/R3 all read as
at (1,2).

- An earlier reading "parity at matched inputs, the claim does not survive"
  (against `full` on one ensemble, floor estimator) was **retracted** once both
  ensembles and the truncated estimator were in; so was "full's ΔA₀ swings 8×
  between ensembles" (a 1.4σ difference between small central values).

### 3.4 Learned, not architectural — ESTABLISHED

At every input depth the untrained net is **below** the classical GEVP (last
table), so the advantage needs training. The decomposition (§3.5) sharpens it:
for the trained net, deleting its out-of-span part costs the whole advantage;
for an untrained net, deleting it *helps* (ΔA₀(net − P) = +0.097 ± 0.021
trained, −0.091 ± 0.042 untrained, four converged seeds). The sign flip holds
at a common reference mass too (2026-10-21, §3.5): +0.123 ± 0.011 trained,
−0.106 ± 0.018 untrained (mean ± s.e.m. over the four nets, each between
−0.06 and −0.14); these are the values of the thesis figure
(`fig:decomposition_rnd`, regenerated 2026-10-22).

### 3.5 What the network found — ESTABLISHED, two sentences retracted

Operators are vectors with inner product C_ab(0), so O_GELT = P + r with P the
projection onto a classical span is exact, not fitted. P is the strongest
possible classical opponent for *this* operator, so ΔA₀(GELT − P) is a lower
bound on the advantage over the span.

ΔA₀(GELT − P) is read two ways: from two free-mass single-state fits (each
operator its own m, the original column), and at the common m_ref = m_G that
defines Z_r/Z_G (`a0ref_*` in `operator_decomposition.py`, added 2026-10-21).
Only the second keeps Z_G = Z_P + Z_r additive, and it is the one the thesis
quotes.

| span (7-level net, combined) | norm² outside | Z_r/Z_G | ΔA₀(GELT − P), free-mass fits | ΔA₀(GELT − P), common m_ref |
|---|---|---|---|---|
| 4 levels | 0.211 | 0.229 ± 0.009 | +0.157 ± 0.027 | +0.185 ± 0.015 |
| 7 levels (`deep`) | 0.128 | 0.146 ± 0.007 | +0.097 ± 0.021 | +0.123 ± 0.011 |
| `full21` (21 ops) | 0.025 | 0.013 ± 0.004 | +0.021 ± 0.009 (2.3σ) | −0.008 ± 0.005 (1.6σ) |
| `full` (35 ops, the thesis row) | 0.024 | 0.011 ± 0.004 | +0.022 ± 0.009 (2.5σ) | −0.011 ± 0.005 (2.3σ) |

- **Mechanism: constructive interference.** `r` alone is a poor operator
  (A₀ ≈ 0.4); it wins by adding coherently to P, not by being good.
- **Content**: 80% of `r` against `deep` lies in span{`full`} — rectangular
  loops the network rediscovered (55–60% if `full` is truncated to the
  directions the estimator trusts). The rest, ~2.5% of the norm², is outside
  every classical operator in the repo: content the network built, with no
  resolved effect on A₀ (second RETRACTED bullet below).
- **Distance from the span is architectural, direction is learned**: untrained
  nets are 16–74% outside, the trained one 12.8%; 0.3% of an untrained residual
  is loop-shaped against 80% of the trained one. Quote Z_r/Z_G and the sign
  flip, never the norm fraction, as evidence of learning.
- Not a contact term on the 4-level span (the out-of-span fraction *grows*
  under the C(1), C(2) metrics); the test is undefined on longer spans, whose
  C(τ > 0) Grams are indefinite.
- **RETRACTED**: "the classical ladder is a converged geometric series whose
  tail is worth ≈ 2%". Rungs 8, 12, 16 measured 1.2%, 2.1%, 1.0%; the 4 → 7
  level span absorbs 40% of the learned out-of-span norm.
- **RETRACTED (2026-10-21)**: "the ~2.5% outside every classical operator
  carries ΔA₀ = +0.021 ± 0.009" (+0.022 ± 0.009 against the 35-operator
  `full`). That is a difference of two free-mass fits, and P's fit has the
  lower mass: m_G − m_P = +0.009 ± 0.004 (`full21`) / +0.010 ± 0.004 (`full`),
  almost all of it on ens1 (+0.013 / +0.015 ± 0.005), so P extrapolates to a
  lower amplitude at Δ = 0. At the common m_ref, removing r leaves A₀
  unchanged or slightly better: −0.008 ± 0.005 (`full21`), −0.011 ± 0.005
  (`full`; run5 −0.013 ± 0.007, ens1 −0.010 ± 0.007), and for `full` it moves
  by < 0.001 over m_ref ∈ m_G ± 0.02 and at m_ref = m_P. This is what the norm
  and the amplitude predict: with additive amplitudes
  A₀(G) − A₀(P) ≃ [(2z − f)·A₀(G) − f·A₀(r)]/(1 − f), with z = Z_r/Z_G from
  C_Gr (exact up to the C_Gr − C_rG noise, which per ensemble is ≈ ±0.009);
  the combined `full` values f = 0.0243, z = 0.0107, A₀(G) = 1.014,
  A₀(r) = 0.337 give −0.0114 against −0.0114 measured. The 4-level and `deep`
  rows survive the same check with larger margins, and so does the untrained
  sign flip (§3.4). Raised by the supervisor's final review; the thesis now
  quotes only the common-mass values (ch. 5 `tab:decomposition`,
  `fig:decomposition_rnd`, conclusions), keeps one sentence on the free-mass
  +0.022, and no longer claims that the out-of-span piece improves the
  operator. **Rule**: compare G with P at a common mass; a free-mass
  ΔA₀(G − P) folds in the fitted-mass difference whenever that is resolved —
  on all four spans, free − common ≈ (2.8–3.4)·(m_G − m_P), the lever arm of
  the extrapolation from [2, 7] to Δ = 0. Artifacts:
  `results/glueball/operator_decomposition_{published,deep,full21,full}_7lv_commonmass_2026-10-21.pt`
  and `…_deep_7lv_rnd_fitok_commonmass_2026-10-21.pt`.

### 3.6 Initialisation robustness of the GELT operator — ESTABLISHED (n = 3)

Three GELT initialisations on ens0 (4 levels): ΔA₀ vs GEVP = +0.066, +0.067,
+0.076 (range 0.010), masses consistent with the GEVP, top-1 share of C(0)
0.8–1.0%. The old argument "a loss at the transfer-matrix floor is init-blind"
is now a measurement.

### 3.7 Limitations to state once

One (β, ξ, L), strong-coupling spatial lattice, no continuum limit; A₀ > 1 on
ens1 is a fit artifact of a nearly pure operator (report 1 − A₀ with its sign);
bit-reproducibility of pre-2026-09-08 dumps is gone at the 4e−7 level (the
smearing rewrite) — five orders below any error bar.

### 3.8 Where the headroom is — ESTABLISHED (2026-11-12, offline)

`python scripts/variational_headroom.py` (20 s; `results/glueball/variational_headroom.txt`).
Four measurements with no fit window, no fit model and no A₀ estimator.

**(1) The true overlap is ≈ 0.88, and the plateau starts at Δ ≈ 5.** On the 40k
chain the 4-level ens2 net has cosh m_eff(Δ = 0…7) = 0.3758(12), 0.3472(15),
0.3385(20), 0.3308(29), 0.3260(38), 0.3187(52), 0.3194(65), 0.3128(83). The
fit-free purity η(a) = C(a)²/(C(0)C(2a)) → A₀ (a → ∞), corrected for the
backward-propagating term at m_ref = 0.313, is 0.971(1), 0.948(4), 0.918(9),
0.891(17), 0.877(27) at a = 1…5. The one-state fit on [2,7] reads 0.9335(53) for
the same net on the same chain. So §3.3's "above ≈ 0.9 the observable is
saturated" is a property of that estimator: every operator in this record is
≈ 10–15% from pure, and the contaminant decays slowly (gap ≈ 0.3 a_t⁻¹, i.e. a
state near 2m).

**(2) The net is behind on its own loss.** L = −mean[C(1)/C(0), C(2)/C(0)] on the
test split, the classical arm projected at (0,1) and **cross-fitted** (v from
the even jackknife blocks, scored on the odd ones, swapped). ΔL = L(net) −
L(classical), positive = the net is behind:

| | `published` | `deep` | `shapes_sm` | `full` |
|---|---|---|---|---|
| 4-level net, three ensembles | −0.045 ± 0.003 | −0.023 ± 0.003 | **+0.021 ± 0.003 (7.9σ)** | **+0.028 ± 0.003 (8.5σ)** |
| 7-level net, run5 + ens1 | −0.067 ± 0.005 | −0.052 ± 0.004 | −0.007 ± 0.003 | −0.002 ± 0.004 |

`shapes_sm` is 20 operators built from the *same four smearing levels* the
4-level net is fed. This is the quantity the net was trained to minimise, so
losing here is not saturation and not the A₀ estimator.

**(3) The net adds nothing to the span.** L(arm ∪ net) − L(arm), cross-fitted:
4-level net added to `shapes_sm` −0.0003 ± 0.0004, to `full` −0.0006 ± 0.0007;
7-level net added to `full` −0.0037 ± 0.0014 (2.7σ). For this objective the
4-level operator is a sub-optimal point *inside* the span of smeared
rectangles — the loss-level version of §3.5's "it rediscovered rectangular
loops", and stronger than it.

**(4) Euclidean completeness.** For a span S and probe G,
η_S(a) = c(a)ᵀC_S(0)⁻¹c(a)/C_GG(2a) with c_i(a) = ⟨O_i(t) G(t ± a)⟩ equals
⟨G|T^a P_S T^a|G⟩/⟨G|T^{2a}|G⟩: 1 − η is the share of the Euclidean-evolved
state outside the span, and η_S(a → ∞) is the largest true overlap any operator
of S can have (`tests/test_headroom.py`, on a known spectrum). At high
statistics the 4-level `published` basis recovers none of what the net misses
(span net: 0.971(1), 0.945(4), 0.903(9); span `published` ∪ net: 0.972(1),
0.946(4), 0.905(9), raw). On the 400-configuration splits `full` ∪ net reads
0.983(7), 0.958(19) at a = 1, 2 — not distinguishable from the net alone, and
too noisy at a ≥ 3 to give the 35-operator basis's ceiling. **The missing
measurement is `full` on a high-statistics chain.**

**Two explanations ruled out the same day.**
- *"The transport is thin while the plaquettes are smeared, so the loop-doubling
  step does not close."* `python scripts/hybrid_transport_probe.py` (CPU, 9 s,
  12 × 6³): the hybrid P_n·T₀P_nT₀† correlates 0.995–0.999 with the clean
  smeared 1×2 rectangle at zero momentum and loses 0.003–0.010 of C(1)/C(0).
  Not the bottleneck.
- *"Train at a longer lag."* In a linear span, moving the GEVP from (0,1) to
  (0,2), (0,4), (1,2), (1,3) changes the signal-to-noise at Δ = 5 by +0.5…+2.7%
  at most (table 1 of the script, half-split on the 40k chain); (2,3) is worse.
  The (0,1) vector is already within 3% of the best the span can do: the span,
  not the lag, limits a linear method.

**What this leaves.** Either the function class, as gradient descent on the
Rayleigh loss explores it, stops at rectangles and short of their optimum, or
the loss does. §11 P11 separates the two.

---

## 4. The attention map as a lattice operator

### 4.1 Structural claim — ESTABLISHED, and near-tautological

The score Re Tr[Q†K̃] is gauge invariant, so any per-site reduction of α is a
local scalar lattice operator, and for a per-timeslice network its
zero-momentum correlator has a transfer-matrix decomposition. Measured on 3D Z₂
at β ∈ {0.7450, 0.7520, 0.7560, 0.7585} (24² × 48, R = 6, 1200 unseen configs
per β) and on SU(2) at β = 2.4:

| β | ξ GEVP | ξ_A trained | ξ_A random | A₀ tr./rnd./GEVP | ΔA₀ trained − random |
|---|---|---|---|---|---|
| 0.7450 | 2.04(14) | 2.24(12) | 2.08(13) | 0.85/0.70/0.58 | +0.144(17) |
| 0.7520 | 2.34(16) | 2.76(13) | 2.46(16) | 0.78/0.67/0.58 | +0.110(20) |
| 0.7560 | 4.08(27) | 4.48(30) | 4.08(26) | 0.74/0.53/0.45 | +0.204(12) |
| 0.7585 | 5.59(34) | 6.64(32) | 5.47(31) | 0.75/0.49/0.41 | +0.267(11) |
| SU(2) 2.4 | 2.57(13) | 2.68(12) | 1.86(15) | 0.88/0.59/0.88 | +0.286(56) |

Pearson(ξ_A, ξ_GEVP) = 0.9965 over ×2.96. **Weight it correctly**: any local
gauge-invariant operator decays with the gap, the untrained net tracks ξ too
(Pearson 0.94), and four couplings monotone in β correlate for trivial reasons.
It shows the construction is right; it is not a discovery.

The **cross-β matrix** (five trained nets × five ensembles, Z₂) is the stronger
structural fact: ξ_A varies 12× more across evaluation ensembles than across
training β, the diagonal has no systematic advantage (+0.029, sign-inconsistent)
against a trained − random gap of +0.167, and one net (trained at 0.7585) is the
best operator on three ensembles it never saw. The attention reads the
configuration in front of it.

β = 0.7600 was dropped: its true ξ ≈ 10 against L = 24 (from the since-deleted
dual-Ising reference), so every arm was measuring the box.

### 4.2 Learning claim — ESTABLISHED difference, OPEN meaning

Training raises the attention field's A₀ over the untrained net at every
coupling in both groups, as a correlated difference on shared configurations.
The descriptive δA/A (×13–×46 site-level fluctuation) is confounded by softmax
sharpness and is not evidence. "The network discovers ξ" was never supportable.

**The control that does not exist**: the net was trained to make its *output*
a good ground-state operator, and the attention field is computed from the same
trained features. Any intermediate per-site field of a trained net may inherit
ground-state overlap from the output. Until a non-attention intermediate field
(and the attention field with the output regressed out) is measured the same
way, "training teaches the attention physics" and "trained features correlate
with the trained output" are the same observation. §11 P4.

### 4.3 Two main.tex sentences that do not survive as written

- **"The untrained attention field is already a moderately better operator than
  the classical GEVP" (Z₂).** The Z₂ GEVP column is a single operator: at the
  production α = 0.5 the projected Z₂ APE ladder freezes after one step and is
  gauge-variant (§9); the saved dump has `gevp_fell_back=True, n_ops=1` at every
  β. With a working classical basis this is expected to fail; in SU(2), where
  the basis works, it already fails (0.59 vs 0.88). Rewrite or remove; §11 P2
  measures it.
- **"The trained network reads a larger ξ than GEVP, which means better
  overlap."** The ordering ξ_trained > ξ_random ≈ ξ_GEVP is what the A₀ column
  predicts (contamination biases ξ low), and the since-deleted dual-Ising truth
  put the trained field at +3.8% against −8.9% classical — but a strengthened
  classical arm matched the trained field against that same truth, and nothing
  in the repo now can distinguish "less contaminated" from "fit systematic".
  Say "consistent with", not "means".
- In SU(2) the trained attention field **ties** the GEVP (0.88 vs 0.88); only
  trained − random is significant.

---

## 5. The architecture question — GELT against the L-CNN

### 5.1 Every comparison in one table

| # | task | arms | accuracy | dispersion | status |
|---|---|---|---|---|---|
| A1 | SU(2) 0⁺⁺, Rayleigh loss, 4 levels | GELT vs **our** L-CNN, 2 ens. | ΔA₀ = +0.012 ± 0.009, same mass, same δC/C at every Δ | our L-CNN: 3 of 9 operators put 36–98% of C(0) on one configuration; GELT 0 of 14 | tie |
| A2 | same | GELT vs **authors'** L-CNN, 3+1 seeds | paired ΔA₀ +0.101, +0.283, +0.044 (ens0), +0.003 (ens1); GELT's worst A₀ (0.903) above their best (0.869) | A₀ range 0.010 vs 0.247; one L-CNN run 4.9σ *below* the GEVP; 2 of 4 training curves degrade | **GELT wins**; cause open |
| A3 | M1 probe: constructed per-site T2 on SU(2) slices | gelt, frozen, frozen_matched, lcnn, lcnn_norm; 6 cells | gelt − lcnn +0.348, 4/6, p = 0.69 | sd 0.049 vs 0.415; all 6 GELT runs > 0.81, 3 of 6 L-CNN < 0.25 | no winner |
| A4 | Z₂ 3D vortex-cluster size V1, β = 0.7520 | gelt vs lcnn, 6 seeds each, own bracketed rates | ΔR² = −0.073 ± 0.051 | sd 0.123 vs 0.020 (6.2×, F = 38) **against GELT** | tie |
| A5 | A4's checkpoints at three other β, no retraining | same | contrast +0.007, +0.310, +0.013, p ≥ 0.24 | 1.5× / 1.3× (reversed) / 1.0×; collapses 1 of 6 seeds vs 3 of 6 (Fisher p = 0.545) | null |
| A6 | 1+1D SU(2) W⁴ˣ⁴ regression | GELT vs authors' L-CNN (Letter's L-CB), 3 seeds | GELT 16× worse MSE, both R² > 0.99998 | 3.1× vs 3.3×, equal | L-CNN wins |
| A7 | 1+1D SU(2) W²ˣ² regression (by-product of §2.5) | GELT (20.9k, 3 seeds) vs authors' L-CNN (13.5k, 1 seed) | GELT 6.8e3× worse `mse_avg`, both R² ≈ 1 | n = 1 on the L-CNN side | L-CNN wins |

**The consolidated statement**: on accuracy GELT ties our L-CNN on the physics
task (A1) and on the physics-observable candidate built to favour it (A4),
**beats the authors' L-CNN on the physics task (A2)** — every run, worst GELT
above best L-CNN — and loses on an exact supervised target (A6). The A2 win is
the one accuracy win GELT has against an L-CNN; what is open is whether it is
the authors' architecture that cannot represent a better operator or its
training that does not find one under this loss (§5.2). GELT also **trains
reliably where L-CNN stacks sometimes do not** on SU(2) (A1, A2, A3); that
asymmetry is absent in 1+1D (A6) and reversed on Z₂ in distribution (A4).

### 5.2 The glueball comparison, both baselines

- **Against our reimplementation (A1)**: parity, pre-registered, including in
  signal-to-noise. The ens0 60-epoch L-CNN run was unusable (one configuration,
  unremarkable to every classical operator, carried 71.8% of C(0)); a clean
  20-epoch rerun is the one in the table. `fit_glueball_overlap.py` now prints
  the top-1 share for every arm and refuses an A₀ above 10%. Its LR × init sweep
  was re-run by accident: all four points within 0.6%, `init_scale` a null, the
  rate optimum unbracketed upward.
- **Against the authors' layers (A2)**: the transplant needed three corrections
  before it trained — **no activation layer** (the Letter uses none; `LActPoly`
  made the field scale as `init_w^198`), their `kernel_size` is our K + 1, and a
  one-forward C(0) calibration. Then GELT's worst run (A₀ 0.903) beats their
  best (0.869) on ens0.
- **What A2 establishes, and what it does not.** It is an **accuracy gap, not
  only a dispersion**: on ens0 GELT's worst run (A₀ 0.903) is above the
  authors' best (0.869), and GELT also reaches a lower value of the loss both
  arms optimise (best val −0.6165…−0.6169 against −0.4271…−0.5697). What is
  open is the *cause*. Three facts point to optimisation rather than
  representation, none proves it: (i) our own L-CNN, at the same budget and
  inputs, reached A₀ 0.896 — an L-CNN-type operator of GELT's quality exists;
  (ii) no authors' run reaches GELT's loss, and 2 of 4 degrade after epoch 5–6,
  one back to its untrained loss — the loss is not being minimised, which is an
  optimisation symptom; (iii) their SM §V warns that L-CNNs with a lattice
  average "often did not converge at all". Either way it is a statement about
  **their architecture in this setting**, and the practical comparison — one
  training run each — goes to GELT.
- **Limits.** Three seeds on one ensemble; ens1 has one run, a tie (+0.003) at
  a classical A₀ of 0.925. Excluding ens1 as "saturated" is post hoc — the
  curve's `SATURATED = 0.90` rule is defined on the *combined* classical A₀, and
  the same ens1 is inside the published +0.078 — so ens1 counts as a tie, not as
  missing. §11 P7 (more seeds on both ensembles) and P10 (is the gap
  optimisation?) are what would settle it. The sentence the data support today:
  *at matched budget GELT reached a higher ground-state overlap than the
  authors' L-CNN in every run on ens0 and tied on ens1; our own L-CNN matched
  it.*

### 5.3 The M1 ablation and its descendants

GELT and a matched L-CNN differ in **two** things — input-dependent offset
weights and transport geometry — so no GELT-vs-L-CNN number measures either
alone. The clean test is the ablation `alpha_mode="frozen"`: α = softmax of a
learned table, same transport, same value path, still convex (so boundedness is
held fixed).

- **M1 pays on a constructed target** (A3's T2, the θ = ½ enclosing radius of
  f⁴ over a radius-4 ball): gelt − frozen = +0.144, **6 of 6** cells,
  p = 0.031 (the floor at n = 6); the capacity-matched frozen arm is no closer
  (+0.162, 5/6). The calibration target T0 (a pure convolution) separates
  nothing (3/6), so it is not optimisation. **On physics tasks M1 is 0 for 3**:
  the 0⁺⁺ tie, the vortex +0.066 (one seed, directional), the β-transfer null.
- **Removing the softmax altogether** (`signed_l1`, identical DOFs, α a strict
  superset) costs ≈ 0.48 of R² in 3 of 3 seeds — and *more* on T0 (−0.66), where
  the mechanism is inert. So it is trainability, not expressiveness. The ladder
  that split it into non-negativity / normalisation / bound (0.44 / 0.06 / 0.09)
  is **retracted** by that control.
- **T1** (max over the ball) is **withdrawn**: softmax is a soft argmax, so the
  target is GELT's own primitive (0.996 vs ≤ 0.06).
- **The M1 probe's gate is the template**: the best linear filter over the same
  ball (the M1-free family) reached 0.978 on the first T2 and forced
  `SCALE_POWER = 4` before any training code ran; the radial filter equals the
  full-ball one to four decimals, so the M1-free ceiling is a five-parameter
  object.

### 5.4 Robustness, all readings together — QUALIFIED

| where | measure | GELT | L-CNN |
|---|---|---|---|
| A1 SU(2) glueball | operators with > 10% of C(0) on one config | 0 of 14 | 3 of 9 (ours) |
| A2 SU(2) glueball | runs with ΔA₀ vs the GEVP < 0 | 0 of 3 | 2 of 3 (theirs; one at −4.9σ) |
| A3 SU(2) constructed | runs below R² 0.25 | 0 of 6 | 3 of 6 (ours) |
| A3 learning-rate sweep | diverged runs, 5 rates | 0 of 15 | 4 of 10 |
| A4 Z₂ vortex, in distribution | sd over 6 seeds | 0.123 | 0.020 |
| A5 Z₂ vortex, off coupling | seeds collapsing | 1 of 6 | 3 of 6 |
| A6 1+1D SU(2) regression | worst/best | 3.1× | 3.3× |

What survives: **on SU(2) tasks trained by us, L-CNN runs fail to train
(blow-up, collapse onto a configuration, or stall) at a rate GELT has not shown
in any SU(2) row of this table.** It is a count, never a test; it is not a variance advantage (δC/C
is identical when both train); it reverses on Z₂ in distribution and vanishes in
1+1D. Two confounds keep the *mechanism* open:

- **M2 (boundedness)** predicts it, but the one M2 control (`lcnn_norm`, offset
  weights normalised) is a null on accuracy and only 2.15× on dispersion, and
  did not prevent divergence (6.7e17 vs 3.2e20). The dominant amplifier is the
  L-Bilin degree growth, which nothing here varies. `lcnn_norm` also cannot be
  transported to the authors' merged kernel.
- **The transport does it too**: GELT fed a single-path `T`, same α, same DOFs,
  falls below R² 0.70 in 2 of 12 cells (projected `T`: 3 of 12), `gelt` in 0.

The Z₂ reversal has a candidate explanation nobody has tested: in Z₂ GELT's
path-averaged transport is a **hard vortex mask**
(`T_Δ² = (1 + P_enclosed)/2 ∈ {0, 1}`, exact) — it switches a neighbour off
exactly where the vortex line to be traced crosses — while the L-CNN's axis
transport is the identity. And the vortex study trained on **70
configurations** and scored R² on **20**, so one configuration moves a seed's
R² by tens (it happened once per arm). §11 P5 separates the two.

The Z₂ L-CNN also had **no usable forward pass at initialisation**: at nc = 1,
L-Act multiplies by its own argument, and four layers reach 10²²…∞ at the
production volume while GELT's field stays at 1. `conv_init_scale = 0.2` was
measured at the production volume (0.5 passed on an 8³ box and blew up to 5.8e5
at 48 × 24 × 24 on one seed of two). A deterministic initialisation fact, not a
robustness result.

### 5.5 Path-averaged transport — the untested half of the architecture — OPEN

The architecture has two departures from the L-CNN. The first (attention) has
been measured five ways. The second — **averaging the transport over all
shortest paths** — has been measured three times and never paid on accuracy:
M1 probe R-I = −0.037 (2/6) on the mechanism target, the W⁴ˣ⁴ loss does not
involve it, the vortex M3 = +0.018 (one seed, and in Z₂ it is a mask). It costs
**62.8% of the GELT training step**. Its only support is the consistency count
above (0 of 12 vs 2–3 of 12). It has **never been ablated on the 0⁺⁺ task**,
the one physics result the thesis rests on. §11 P3 is the single most
informative run left for the architecture chapter.

### 5.6 The selection rule the search produced — ESTABLISHED as method

A GELT-vs-L-CNN task is worth running only if (1) the target is per site, not
zero-momentum projected; (2) it involves sparse structures at
configuration-dependent positions and scales with an amplitude-invariant
identity; (3) there is headroom over the best classical local method **at the
architecture's own reach — spatial and iterative** (a BFS run to convergence is
not in a 4-layer network's function class); (4) the target is not the output of
a classical local algorithm; (5) it is not GELT's own primitive restated (T1).
And the gate: **measure the best classical method at matched reach before
writing training code.** It stopped the topology study in ~4 GPU-hours instead
of 60–100, and moved two designs before they ran. This is the durable output of
five attempts and is worth a section of the thesis on its own.

---

## 6. The baseline itself

- **Our `gelt/lcnn.py` is a low-rank member of the authors' L-CB family**:
  given any kernel of theirs, our L-Conv + L-Bilin reproduces it to 1.9e−15 once
  the L-Conv width reaches 2·c_in·(1 + 2DK); production widths are 8% (probe)
  and 1.6% (glueball) of that. The inclusion is proved in one direction only
  (their weights are real, ours complex).
- **Our kernel was one-sided until 2026-09-13** ("the W† channels subsume the
  backward hops" is false — daggering commutes with the adjoint transport); A1
  was run after the fix, with the symmetric kernel.
- **The vendored `LConvBilin` is not the network the Letter counts**: it seeds
  the transported list with the field itself, one slot wider; SM Table V is
  reproduced exactly (ten architectures) by `gelt/lcnn_exact.py`. So every
  "the authors' L-CNN at N parameters" in A2 quotes the wider variant; the
  comparisons are unaffected, the phrase "the paper's network" is not quite true.
- Three traps, each now a test: their `kernel_size` = our K + 1 (passing K would
  silently halve the receptive field); their layout packs links into the field;
  Z₂ must be promoted to complex.

---

## 7. Method — what proved durable

- **Seeds, not jackknives, are the error that matters for learned arms.** A
  blocked jackknife over 400k test sites converges to 1e−4 and never sees the
  across-initialisation spread; the vortex single-seed "3.7σ" became t = 1.4 at
  six seeds. Readings over paired (ensemble, seed) cells with a sign test; the
  error on a median is max(jackknife, half the seed range).
- **Horizon and rate are one choice.** A rate picked on a 6-epoch cosine does
  not transfer to a 40-epoch one; early stopping on a cosine deletes the epochs
  that converge; 120 → 240 epochs *flipped the sign of M1* on the vortex task;
  batch 4 → 1 moved `frozen` by 0.17. Every arm needs an interior optimum at the
  production horizon (`probe_curves.py` asks whether the budget was binding).
- **Silent variation is the recurring defect**: an unrecognised flag ignored
  (now `validate_argv()`), three rates under one run tag, batch size and init
  scale absent from dumps, the obars cache keyed by ensemble only. Every one
  changed a number before it was caught.
- **Read the field, never the output**: a zero-initialised head outputs 0
  whatever the stack is doing.
- **The time-shuffle is not a null**: it keeps each configuration's mean, so it
  measures the finite-N_t zero mode (2ξ − 1)/N_t; the config-scramble is the
  null.
- **4D APE at α = 0.5 is Wilson flow**: n steps ≡ t = n·α/6 (peaks at n = 24 for
  t = 2 and n = 48 for t = 4, Z → 1 at the peak) at ~8% of RK3's cost. A target
  defined by a smoother is reproduced by the smoother (R² = 0.978, zero
  parameters) — the topology study's cause of death.
- **R² has an unbounded tail**: one held-out configuration took a vortex seed to
  R² = −66 (GELT) and −259 (L-CNN); report leave-one-out with the rule applied
  to both arms.

---

## 8. Performance — ESTABLISHED

- 7.77 → **5.04 s/step** on the V100 (1.54×) from four exactly-equivalent
  changes: closed-form SU(2) polar projection (the APE ladder issued ~4.5 M
  2×2 SVDs per step), `ape_smear` vectorised over configurations (7.4×),
  introspection stashes opt-in, and the GEMHSA hot path (shared K/V gather,
  RoPE folded into Q, a gather-based backward for the neighbour gather that
  autograd emitted as an atomic scatter at 64× its forward; 2.16× fwd + bwd).
- Per layer: transport 223 ms fwd / 327 ms bwd, i.e. **62.8% of the step**;
  the adjoint SO(3) representation would take it to ~2.7 s/step (1.8× end to
  end) and is the only large lever left. The "rope_score is the largest stage"
  profile was a micro-benchmark artifact and is retracted.
- Our L-CNN runs 1.29 s/step at the matched glueball shape, so GELT costs 3.9×;
  against the authors' layers this has not been re-measured.
- Rejected with reasons: caching W/T (95 GB), fp16 (C(0) once reached 1e73),
  larger batch (it changes the estimator), real-valued projections (a model
  change).

---

## 9. Live defects and text corrections

1. **Z₂ APE smearing at α = 0.5** (production value in `train_z2_glueball.py`,
   `z2_beta_scan.py`, `z2_attention_correlator.py`). With two staples,
   V = (1 − α)U + (α/2)(s₁ + s₂): the identity for α < ½, a majority-vote
   automaton for α > ½, and at α = ½ exactly `V = 0` whenever the staples
   disagree, which `Z2.project` maps to +1 — not covariant. On production
   configurations the ladder freezes after one step, 99.9% of changed links are
   gauge-dependent tie-breaks, Ō moves 1.82σ under a gauge transformation. So
   the Z₂ classical basis is one operator and the Z₂ nets saw four input
   channels of which three are identical. SU(2) is clean (1.4e−15). The test
   suite never used α = 0.5. Fix: unprojected fat links (linear, hence exactly
   covariant, with a real radius). §11 P2.
2. **main.tex, APE formula**: it writes `U' = P[αU + (1−α)Σ staples]`; the code
   is `P[(1−α)U + (α/n)Σ staples]`. Numerically the same at α = 0.5, but the
   formula is wrong for the code (and the 1/n is missing).
3. **main.tex, the loss**: described as −ln(C(1)/C(0)) "plus terms like
   −ln(C(2)/C(0))"; the code minimises −mean[C(1)/C(0), C(2)/C(0)] plus a
   (log C(0))² scale pin. Say what is run.
4. **main.tex**: §2.2's "~1k" vs README's "~1.5k" parameters; the
   2×2/1×3/2×3/3×3 claim with no artifact (L3′) — replace it by §2.5's SU(2)
   sentence, which has one; §4.3's two sentences.
5. **β = 0.7520 reads low on every arm** in the Z₂ scan (the fair fight noticed
   it on the dual-truth comparison, −0.9% at best, −20.9% at worst); never
   explained. Check its fit window before quoting that row's ξ.
6. **Counts quoted inconsistently**: the robustness census appears as
   "0 of 14 vs 3 of 9" and "0 of 26 vs 3 of 13" in different places. The first
   is the one with a table behind it (§5.4).
7. **main.tex, the abstract's "3.6σ"** (l. 59–61): replace by the
   three-ensemble ΔA₀ = +0.086 ± 0.017 (stat) ± 0.018 (window), "≥ 3σ
   including the fit-window systematic" (3.4σ), and state the correlated fit
   (+0.049 ± 0.017) beside it (§3.2). The "plateau at
   Δ = 1" sentences (l. 59, 406, 427, 436, 505) survive as "compatible with a
   plateau within errors" on both ensembles combined, not on the run5 table
   alone; lead with Δm_eff(1) = −0.032 ± 0.006 combined.
8. **Open after the (0,1) fair fight (2026-10-02, §3.3).** (a) *Which pair
   heads the fair-fight table is not decided*: the (1,2) table stands as
   written and the (0,1) one sits beside it. S2's rule (the pair with the
   highest classical A₀) points to (0,1), and "a 4-level GELT is worth the
   21-operator basis" (−0.001 ± 0.021) then becomes "is behind it in the
   saturated region" (−0.034 ± 0.010). (b)
   `reports/curve/learned_operator_curve.tex` still says `shapes` and
   `shapes_sm` fall back and that the selection rule "has no answer as written"
   (table rows l. 482–483, paragraph l. 525–529); true at (1,2), to be amended
   with the (0,1) rows. (c) The half-split check quoted in §3.3 has no script in
   the repo. (d) `input_architecture_curve.py` knows run5 and ens1 only
   (`ENSEMBLES`, and any other seed is labelled run5), so ens2 is in the
   fair-fight table and not in the curve; no ens2 output exists under
   `_trunc_gevp0-1` for that reason, and none must be written before the label
   is fixed. (e) The canonical curve files (`input_architecture_curve.*`) are
   the (1,2) ones, untouched; the (0,1) curve is
   `input_architecture_curve_trunc_gevp0-1.*`.

---

## 10. Discarded — studied, closed, nothing for the thesis

Each line: what it was, why it closed, the one thing kept.

| study | closed because | kept |
|---|---|---|
| ℓ_att, the attention-range statistic (SU(2) and Z₂) | bounded by R and centred by the ball geometry; landed on its uniform value twice | "read the fluctuation field, not the mean kernel" → §4 |
| dual 3D Ising ground truth, ν fits, Z₂ fair fight (deleted 2026-09-09, `cfa0a7e`) | not in main.tex | ξ(0.7600) ≈ 10 (why four β); the accuracy ordering, since withdrawn |
| rotation-irrep projection, `blocks_bias` | not in main.tex | nothing |
| flow-free topological density (stopped 2026-09-15) | a Manhattan-8 model reaches 15% where 48 APE steps reach 98% at the one topologically meaningful rung | the pre-flight rule, criteria 3–4, APE ≡ flow, the clover density (parity-odd to 1.4e−17, kept in `lattice.py`) |
| M1 target T1 | circular (GELT's primitive) | criterion 5 |
| signed-α ladder (`signed`, `signed_bounded`, `signed_l1`) | its decomposition failed its own T0 control | "removing the softmax costs 0.48, as trainability" |
| `lcnn_norm` / R-D, the M2 control | null on accuracy, did not prevent divergence, not transportable to the authors' kernel | the L-Bilin degree growth is the amplifier |
| β-transfer (attempt 5) | pre-registered claim failed (p ≥ 0.24) | in-distribution score does not predict transfer (GELT's collapsing seed was its 2nd best at β₀) |
| L-CNN init-scale sweeps | init_scale a null (sign flips at 0.001) | spend sweep points on the rate |
| the M1 probe's follow-ups (6 seeds, `signed_bounded` at 1e−1, DP-mode cost ratio) | would firm up a constructed-target result the thesis does not need beyond one paragraph | — |
| vortex replication at β = 0.7450, W-A's comparison half, 6 seeds for `frozen`/`frozen_single` | would decompose a tie | P5 below is the only vortex run still worth doing |
| `update_2026-09-18.md` | a summary of the notes above, superseded by this one | — |

---

## 11. Proposed runs — each confirms or kills something specific

Ordered by value to the thesis per GPU-hour. Readings are written before the
numbers exist; do not edit them afterwards.

**P1 — the unsaturated comparison against the strongest basis. Offline,
seconds, small script.** A₀ is saturated at `deep`/`full`; m_eff(Δ = 1) is not,
and it is where GELT's advantage lives (4σ against the 4-level GEVP). It has
never been computed against `deep` or `full`. Also missing: the classical arm
that solves the network's *own* objective — the GEVP at (t₀, t_d) = (0, 1) is
the optimal linear operator for C(1)/C(0) in the span. From
`dumps/su2_fair_fight_obars_{run5,ens1}.pt` and the two 7-level dumps: project
`deep` and `full` with `gevp_ground_vector(truncate=True)` at (1, 2) and at
(0, 1), fitting v₀ on half the test configurations and evaluating on the other
half (then swapping) so the classical arm is not scored in-sample; correlated
blocked-jackknife Δm_eff(1), Δm_eff(2), combined over ensembles.
*Reading*: Δm_eff(1) vs `full` negative at ≥ 3σ → "better operator than the
strongest classical basis" survives on a metric that is not saturated;
|Δ| < 2σ → the thesis sentence is "matches a 21-operator basis from 7 inputs".

**P2 — the Z₂ attention table on covariant inputs. V100, forward passes, then
optionally 4 retrains.** Add `ape_smear(..., project=False)` (unprojected fat
links) and a test that α = 0.5 projected is *not* covariant; switch the three
Z₂ call sites; evaluate the existing R6 checkpoints through
`z2_attention_correlator.py` first (their `in_channels` is unchanged).
*Reading*: the GEVP column becomes a real 4-operator basis. If A₀_random <
A₀_GEVP at ≥ 3 of 4 β, main.tex's "untrained is already better" is removed; the
trained − random ΔA₀ is expected to survive (same inputs on both arms) — if it
does not, the Z₂ learning claim goes with it.

**P3 — the glueball with single-path transport. V100, 3 trainings × ~8 h.**
Add `GLUEBALL_TRANSPORT_MODE` to `train_glueball.config_inputs`
(`build_transport_average(mode=...)` already exists), tag the stem, run 4
levels, ens0, init seeds 0–2, everything else identical; compare with
`fit_glueball_overlap.py <avg dump> --vs=<single dump>` per seed.
*Reading*: paired ΔA₀(average − single) positive in 3 of 3 and combined > 2σ →
the path average pays on the physics result and the 62.8% cost is justified.
Consistent with 0 → the thesis's second architectural departure has no measured
benefit anywhere, and the chapter must say "a design choice, 1.8–2× the cost".
Either outcome is a sentence the thesis currently cannot write.

**P4 — does the attention field inherit its overlap from the output?
Forward-only, Z₂ ≈ 4 min per β, SU(2) ≈ 40 min.** In the two attention
correlators add two arms for trained and untrained nets: (a) a non-attention
per-site scalar from the same layer, Re Tr of the residual stream's first
channel after that layer, zero-momentum projected; (b) the attention field
after regressing out, per configuration and timeslice, the network's own
per-site output. Same estimator (it is shared by import).
*Reading*: if A₀(a) − A₀(a, random) is comparable to the attention's
+0.11…+0.29, and (b) loses most of the gain, the learning claim is
"trained features correlate with the trained operator", and main.tex must say
that. If (a) gains little and (b) keeps the gain, the attention field carries
ground-state information of its own — a strictly stronger claim than today's.

**P5 — why GELT is the dispersed arm on Z₂. V100, ~10 short runs.** Same
configuration as the §5.1 A4 campaign (`PROBE_GROUP=z2 PROBE_TARGET=V1
PROBE_EPOCHS=240 PROBE_BATCH=1`, its run-tag convention): (i) `PROBE_ARM=gelt_single`
at `PROBE_LR=1e-3`, `PROBE_INIT_SEED=0..5` — the vortex mask removed, α
unchanged; (ii) `gelt` and `lcnn` at `PROBE_N_CONFIGS=400` (the cache holds
2000; no sampling), 3 seeds each.
*Reading*: (i) sd(`gelt_single`) ≲ 0.04 while sd(`gelt`) ≈ 0.12 → the Z₂
reversal is the mask (M3), which reconciles it with the SU(2) readings and
makes "robustness" a property of the averaged transport in non-abelian groups.
(ii) GELT's sd falling toward the L-CNN's → the 6.2× was a 70-configuration
small-sample effect and the "reversal" row of §5.4 is withdrawn.

**P6 — what costs GELT the 16× on W⁴ˣ⁴. V100, 18 min per run, 3 seeds each.**
(i) `WR_TRANSPORT_MODE=single` (exists) — the control; predicted null.
(ii) a `GELT_ARCHS` entry with `nhead` 2 → 8 at fixed budget (`d_qkv` down to
keep 39.6k DOFs; `d_qkv ≥ 2D` must still hold). (iii) a flag for
`alpha_mode="signed"`.
*Reading*: (ii) closing most of the gap → one offset weighting per head is the
bottleneck (a sentence about the architecture's expressiveness per layer);
(iii) closing it → the convexity of α is; neither → the difference is the
L-CB's per-(out, in, in, offset) weights, and "GELT's value path transfers the
L-CNN's loop-doubling argument" needs the qualifier "at lower precision".

**P7 — make robustness one measurement instead of four anecdotes. V100.**
On the glueball, fill in the missing seeds so all three architectures have
three initialisations on both ensembles: our `lcnn` (`GLUEBALL_ARCH=lcnn
GLUEBALL_LR=3e-3 GLUEBALL_INIT_SEED=1,2`, ~2 h each), `lcnn_ref` on ens1
(two more), GELT on ens1 (two more, ~8 h each). Fix the failure criterion now:
*a run fails if its ΔA₀ against the GEVP is negative, or its top-1 share of
C(0) exceeds 10%, or its best validation loss is within 5% of its untrained
loss.* Report the counts per architecture with a Fisher test. This is the only
way the thesis can quote "GELT trains reliably where the L-CNN does not" as a
result on the physics task.

**P10 — is the gap to the authors' L-CNN representation or optimisation?
V100, ~2 h per run.** Three `lcnn_ref` variants on ens0, 3 seeds each, the rest
unchanged: (i) the rate bracketed at the production horizon (`GLUEBALL_LR` at
3e−4, 1e−3, 3e−3); (ii) `GLUEBALL_LCNN_REF_HEAD=linear` (the Letter's head);
(iii) the same arm warm-started from a run that reached a good val loss, trained
twice as long.
*Reading*: if any variant's best run reaches GELT's val loss (≤ −0.61) and
A₀ ≥ 0.90, the gap is optimisation — "the authors' L-CNN is hard to train under
a lattice-summed Rayleigh loss", a statement about trainability. If none gets
past −0.57 and 0.87 across all nine runs, the gap is attributable to what the
architecture reaches at this budget, and A2 becomes an accuracy result without
qualification.

**P8 — optional, only if the robustness sentence is to be a chapter.**
the designed M2 experiment (old `where_attention_can_win.md` §8): low statistics
(`N_train ∈ {100, 400}`, 3 seeds, both architectures, no new sampling), with
P7's failure criterion pooled across stressors. P5(ii) and A5 already exercise
the other two stressors on Z₂.

**P9 — no compute, text only.** §9 items 2–4 in main.tex. For the loop
sentence (item 4) the SU(2) replacement now exists (§2.5, run 2026-09-28): swap
the Z₂ claim for it, or run `train_gelt.py` on the Z₂ 2×2, 1×3, 2×3, 3×3 loops
if the Z₂ wording is to stay.

**P11 — transfer-matrix distillation: is it the class or the loss? V100, one
training (~8 h), then forward-only.** Written 2026-11-12; nothing below has run.
The identity: with U_t the spatial links of slice t (the Markov state of the
transfer matrix) and y any observable on other slices,
g*(U) = E[y(t + a) | U_t = U] = (T^a ŷψ₀)(U)/ψ₀(U), i.e. the multiplication
operator ĝ* acts on the vacuum as T^a ŷ|0⟩. The regression function of the
future on the present *is* the Euclidean-evolved operator, as a legitimate
single-timeslice operator with C_g*(Δ) = C_y(Δ + 2a) — the filter a multilevel
update buys with nested Monte Carlo, amortised into a function fitted once
(checked on a known spectrum, `tests/test_headroom.py`). In a linear span it is
one power-iteration step of C(0)⁻¹C(a), whose ceiling is §3.8's η.
`scripts/train_distill.py`: teacher = the trained 4-level net, frozen; student
= the same architecture **started from the teacher's weights** (so the class is
held fixed and only the objective changes), regressing the teacher's own
per-site field at t ± 2 onto slice t, block-averaged over 3³ plus the
zero-momentum error; selection on the validation loss of that objective, the
start being a candidate. `bash scripts/tmd_batch.sh` (ens2 first: it has the
high-statistics teacher), readout `scripts/tmd_readout.py`, paired on the test
split, then `TMD_HS=1` for both nets on one fresh 40k chain.
*Readings* (also in the readout's docstring, which applies them):
**R-T1** ΔL(student − teacher) < 0 at ≥ 3σ and Δ cosh m_eff(1), m_eff(2) < 0 at
≥ 2σ each → the Rayleigh loss, not the class, was the limit; quote ΔL, the
realised share of the ideal shift m_S(Δ) = m_T(Δ + 4), and η. **R-T2** in
addition L(`full` ∪ student) − L(`full`) < 0 at ≥ 3σ (the teacher's is −0.0006 ±
0.0007) → ground-state content outside all 35 classical operators.
**R-T0** |ΔL| < 2σ → the student is its teacher: the class, as explored from
that start, does not contain the evolved operator at this lag; the 0⁺⁺ on this
lattice is exhausted for this architecture and the thesis sentence is "a point
inside the span of smeared rectangles". One follow-up only (`TMD_LAGS=1`), then
stop. **R-T−** ΔL > 0 at ≥ 3σ → a validation/test mismatch to understand first.
Honest prior: R-T0 is as likely as R-T1. Independent of the outcome, the
forward-only run this record is missing is **`full` on a high-statistics
chain** (§3.8 (4)): it gives the 35-operator basis's true ceiling η(a → ∞) and
needs a streaming version of `su2_fair_fight.py`'s basis builder, not written.

*Run 1 (2026-11-19, ens2, lag 2, the loss as written above — now `TMD_LOSS=raw`;
verbatim in `notes/raw/tmd_ens2_readout_2026-11-19.txt`).* The readout says
**R-T0**: ΔL = +0.0020 ± 0.0013 (+1.5σ), Δ cosh m_eff(0..3) = +2.5σ, −1.4σ,
−0.3σ, −0.7σ, shift realised −0.10 ± 0.09, η equal within errors, both nets
+0.024…+0.035 behind `shapes_sm` / `full` and adding nothing to them. Stopped at
epoch 14, best epoch 8. **The run had no power to separate class from loss**,
for two reasons the training log shows and the teacher's own test dump
(`dumps/…_ens2_test_obars.pt`, the same 400 configurations) quantifies:
(i) *the whole loss decrease is a rescale.* In the teacher's units the
zero-momentum term is 10.64 for the teacher as it is, 6.43 for the teacher
rescaled by c = C(2)/C(0) = 0.49, ≈ 5.94 for the exact regression function
(C_T(Δ + 4)); the student scored **6.51** — the rescale, and none of the 0.50
purification margin, which is r(4) − r(2)² = 0.030 of C(0), ≈ 8% of the term.
(ii) *selection ran on a constant offset.* The validation loss swung 6 → 37 → 6
between epochs; the excesses of the block and zero-momentum terms are in the
ratio V/POOL³ = 64 (epoch 3: +0.52 → +31.5, predicted +33; epoch 7: +0.25 →
+16.5, predicted +16), the signature of a constant of ≈ 0.12σ per site. The
cause is μ/σ ≈ −55 for the teacher's per-site field: rescaling the raw output
by c moves its mean by 55·(1 − c)σ unless the last bias compensates exactly.
So the sentence R-T0 was written to license ("the class does not contain the
evolved operator") is **not** licensed by this run.
*What changed (2026-11-20).* `TMD_LOSS=connected`, now the default: the error is
centred over the minibatch before both terms, and selection is on the *shape
loss* min_c Q(c·s − t), in closed form on the validation split — blind to offset
and scale, as the readout is (`tests/test_distill.py`). The student is tagged
`_tmd2_c`, so the raw run's checkpoint is never overwritten, and the readout
combines dumps only within one loss version. **The pre-registered lag-1
follow-up is dropped**: its margin r(2) − r(1)² = 0.020 of C(0) is *smaller*
than lag 2's 0.030 (7.3% vs 7.8% of the zero-momentum term), so "a stronger
signal" was wrong. This is a deviation from the plan, motivated by the
instrument, not by the sign of the result. *Run 2*: `bash scripts/tmd_batch.sh`
(ens2, lag 2, connected), same readings; **P11 closes on what it reads.** An R-T0
there is the sentence above, with the caveat that the margin is ~8% of the loss
and gradient descent starts at the Rayleigh optimum.

**P12 — the Yang–Mills vacuum wavefunctional, learned by force matching. V100,
one night (`bash scripts/vacuum_batch.sh`, ~10–12 h).** Written 2026-11-18;
nothing below has run at production size. A new observable, not a 0⁺⁺ rerun:
the spatial links of one slice are distributed as ψ₀² (to e^{−m·Nt}), and
∇ log ψ₀² = −∇S_s + E[F_temp | slice] exactly (Lie derivatives, Haar measure;
`gelt/vacuum.py`). A scalar gauge-invariant net trained so that its Lie
gradient matches the temporal force on the slice's links converges to the
exact vacuum — first derivatives only, the existing ensembles only. Three
things make it a measurement rather than a fit: (i) a **conditional chain**
with the slice frozen (`frozen=` in the SU(2) sweeps) gives replicas of the
force, hence an *absolute* error ε = ‖g − E[F|U]‖²/‖E[F|U]‖² with the noise
removed; (ii) **exact relative weights** ψ₀²(U′)/ψ₀²(U) along configuration
ladders, by reweighting with Bennett's acceptance ratio — ground truth no
model saw; (iii) the classical comparators are fitted by the *same* loss in
closed form (least squares): `dr` (ψ₀² ∝ e^{−β′S_s}, the dimensional-reduction
form), thin and smeared loop shapes, a Greensite–Olejník-type B·K·B with a
free radial kernel, and their union `full` (28 operators). The nets are
residual over `full` with a zero-initialised readout, so a run starts at
ε(full) and any gain is structure the classical ansatz lacks.
Pre-flight (local, 8 × 4³, plumbing only): Stein's identity on the slice
marginal holds at z = −0.2 (z = −174 with the temporal force's sign flipped,
`tests/test_vacuum.py`); 13% of the temporal force's variance is predictable
from the slice; the conditional autocorrelation plateaus.
*Readings* (also in `scripts/vacuum_readout.py`, which applies them):
**V0** gate — plateau (first − last late point within 2σ) and Stein |z| < 4 on
the production slices, else nothing is read. **V1** the predictable fraction,
reported. **V2** ε(dr) … ε(full), reported; 1 − ε(dr) is the share of the
vacuum score a rescaled 3D Wilson action explains. **V3** per net,
gain = ε(full) − ε(net + full): **R-V1** > 0 at ≥ 3σ → the network finds vacuum
structure outside every classical ansatz tested ("substantial" if ≥ 25% of
ε(full)); **R-V0** |gain| < 2σ → at this resolution the classical ansatz is
the vacuum as far as this network can tell; **R-V−** < 0 at ≥ 3σ → selection
did not carry over, understand first. **V4** softmax vs frozen α, paired: the
Greensite–Olejník kernel depends on the field through the covariant
Laplacian, so this is the first physics target on which theory *predicts* M1
pays; ≥ 3σ → it does, else a tie. **V5** per ladder passing G-RW (forward and
backward estimates of each rung agree; at most max(1, 5%) rungs off at 3σ): a
model is *validated* on a family if χ²/dof < 2 and max|Δ|/σ < 4; the net
*beats* the classical ansatz if its χ² is under half of `full`'s on both
`noise` and `plane`. `smear` (toward the APE-smoothed slice) is off the
training distribution: reported only.
Publishable if V3 reads R-V1 and V5 validates the net where `full` fails; a
quantitative map of the 3+1D SU(2) vacuum (V2 + V5) is a result even under
R-V0. Honest prior: R-V0 is likelier than R-V1 — on the smoke lattice `dr`
alone already leaves ε ≈ 0.14 and `full` ε ≈ 0.05.

**P13 — the learned K-state spectrum (Ky Fan loss). V100, one training
(~10 h) + its untrained twin (minutes), then offline.** Written 2026-11-19 by
transcribing the readings already coded in `scripts/spectrum_readout.py`; the
`notes/learned_spectrum.md` that the code cites (§5) was never written, and
`spectrum_hs_eval.py` (the high-statistics evaluator its `--hs=` path reads)
does not exist, so RH1/RH3 cannot be read yet. `GLUEBALL_N_STATES=K` gives the
readout K outputs and replaces the Rayleigh quotient by the Ky Fan trace
Tr[C(0)⁻¹C(Δ)] = Σ_{n<K} λ_n (K = 1 is the Rayleigh loss bit for bit;
`tests/test_kyfan.py` recovers ground and first excited state of a known
spectrum end to end). `bash scripts/kyfan_batch.sh` (K = 4, ens0, from
scratch): K = 4 because the net is not O_h-invariant and the 2⁺⁺ E doublet can
enter the span below the 0⁺⁺*. Observable: Ê_n(Δ) = −log λ_n(Δ)/Δ, fit-free,
unsaturated, Ê_n(1) the reading; levels labelled by f_A1 (≥ 0.5 A1, ≤ 0.1
non-A1) against the `full` span; classical arms cross-fitted (even/odd blocks).
*Readings*: **RS1** Ê₁(1) untrained − trained ≥ 5σ → the excited level is
learned, not architectural. **RS3** Ê(1) of the second A1-labelled learned
level − `full` level 1, cross-fitted: ≤ −3σ → the learned basis is better on
the excited 0⁺⁺; ≥ 3σ → classical better; < 2σ → matches; fewer than two A1
levels → not read. **RS4** Ê₀(1) of the K-span against the K = 1 net and
against `full` (k = 1, cross-fitted): ≤ −3σ better, ≥ 3σ behind, < 2σ level —
does asking for K states cost or buy the ground state.

**Worth running only for completeness** (cheap, offline): the excited-state
gate on the `deep` basis (`gevp_eigenvalues(..., truncate=True)`, second
column, blocked jackknife): if m₁(Δ=1) and m₁(Δ=2) agree within 1σ with < 25%
error on one ensemble, a learned two-operator basis becomes the unsaturated
observable A₀ cannot provide; if not, that direction is closed for free.

**Not worth running**: everything in the "Discarded" table; the SU(2) β-scan of
the attention field (it would add points to a correlation that is near-tautological,
§4.1); the adjoint-SO(3) transport (§8) unless P3 says the averaged transport
stays.

---

## 12. Concordance — where each deleted note went

Code comments and docstrings still cite the old files by section; this table
resolves them. Each old note is recoverable with `git show 570c208:notes/<file>`.

| old note | now |
|---|---|
| `glueball_spectroscopy.md` | §3.1–3.2, §3.7 |
| `audit_2026-09-06.md` (§2 Z₂ APE; §6 fair fights) | §9 item 1; §3.3 |
| `fable5.1_10-09_audit.md` (§8.1 estimator, §8.2 curve, §8.3 decomposition, WP7–WP9) | §3.3, §3.4, §3.5; P1, P2, "completeness" |
| `operator_decomposition.md` | §3.5 |
| `attention_as_operator.md` (§6.1 Z₂ table, §6.1.1 cross-β, §6.1.2 ξ overshoot, §6.2 estimator defects, §8 β = 0.760, §9 SU(2)) | §4, §7 |
| `lcnn_shootout.md` (§2 matching, §8 kernel fix, §9 parity, §9.2 top-1 share, §9.3 SNR, §10 sweep) | §5.1–5.2, §6 |
| `lcnn_reference_switch.md` (§1 containment, §7–§9 init / activation, §10 dispersion, §10.1 paired) | §5.2, §6 |
| `where_attention_can_win.md` (§1 M1/M2, §2 attempt 1, §4 topology, §5–§6 rule, §7–§8 M2, §9 vortex, §9.9 result) | §5, §7, §10; P5, P8 |
| `m1_probe.md`, `m1_probe_status.md` (§0 readout, §3 targets and gate, §7.5 grid, §7.6 signed, §7.7 transport, §8 build log) | §5.3–5.5, §7 |
| `beta_transfer.md` (§9) | §5.1 A5, §10 |
| `wilson_regression_1p1d.md` (§4 lcnn_exact, §7 readings, §10 GELT arm) | §2.3–2.4, §6; P6 |
| `performance_audit.md` | §8 |
| `update_2026-09-18.md` | superseded; its figures are in `notes/raw/figures/` |
| `*_readout_*.txt` | `notes/raw/` (unchanged, verbatim) |
