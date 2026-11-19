# Supervisor's review of main.tex — work packages

**Received 2026-09-23.** Four points from the supervisor's reading of
`../report/main.tex`, restated as work packages an agent can pick up, each set
against what `notes/thesis_record.md` already establishes. Read the record's §1
(ledger) and §3 (spectroscopy) before starting any of them; numbers below are
quoted from there and from the code, not re-measured.

Ground rules that apply to all four:

- **Compute**: everything that is not seconds-long runs on the V100 — write the
  script, make it refuse to run without CUDA where it needs a GPU, and hand it
  over. The offline items below read the `dumps/*_test_obars.pt` arrays and the
  `dumps/su2_fair_fight_obars_{run5,ens1}.pt` caches and need no GPU.
- **Readings are fixed before the numbers exist** (the record's §11 rule). Each
  package states what outcome changes which sentence of main.tex; do not edit
  the reading after the run.
- **No new simulations.** The supervisor asks whether everything can be done on
  the existing ensembles: yes, for all four packages.
- The estimator layer is shared (`fit_glueball_overlap.py`,
  `operator_decomposition.py`, `su2_fair_fight.py` all use `FIT_WINDOW = (2, 7)`
  and `gelt.glueball.fit_cosh_correlator`). A change of convention must land in
  all of them together or the tables stop being comparable.

| # | point | kind | cost | overlaps with |
|---|---|---|---|---|
| S1 | fit-window stability, correlated χ², m_eff(1) vs m_fit | offline analysis | seconds–minutes | record §3.2, §11 P1 |
| S2 | is the GEVP a complete enough benchmark? | mostly **done**; (t₀, t_d) scan offline; text | minutes | record §3.3, §3.5, §11 P1 |
| S3 | train/test split, autocorrelations, block size | mostly **thesis writing**; two offline checks | minutes | record §3.1, §7 |
| S4 | is the learned operator in A₁⁺⁺? | new: forward passes on the V100, then offline | hours | none — never studied |

Suggested order: S1 → S2's scan → S3's checks (all offline, one afternoon),
then S4 (the only one that can change the physics statement), then the text.

---

## S1 — Robustness of the mass and overlap extraction

**What the supervisor asks.** The 3.6σ of ΔA₀ = +0.078 ± 0.022 is a
statistical error *given* the analysis choices; it does not contain the
systematic from the fit window. (a) Repeat the single-state fit on at least
Δ ∈ [2,7], [3,7], [2,6]; for each, mass and A₀ for GELT and GEVP with errors
and correlated χ²/dof, in one table or plot. (b) main.tex says GELT plateaus
already at Δ = 1: test m_eff(1) against the fitted mass quantitatively, with
correlations. (c) Optional: a two-state fit with the ground mass shared
between operators.

**Where we are.**

- `scripts/fit_glueball_overlap.py:94` hard-codes `FIT_WINDOW = (2, 7)`; the
  docstring says "window stability should be spot-checked by editing it" —
  it never was, systematically.
- The fits are **uncorrelated**: fixed diagonal σ_Δ from the full-sample
  blocked jackknife (`fit_cosh_correlator(..., sigma=...)`). The χ²/dof it
  prints is the diagonal one. The *errors* are nonetheless correct (the whole
  fit, v₀ included, is redone in every jackknife sample, and ΔA₀ is jackknifed
  as a difference); only the χ² is not the correlated one the supervisor asks
  for.
- The Z₂ correlator (`z2_attention_correlator.py:393`, `_window`) chooses its
  window per arm from the data within bounds (2, 8) — a different convention
  from SU(2), and one that also needs its stability shown if the Z₂ table stays
  in the thesis.
- main.tex Table (l. 443): GELT m_eff(1) = 0.370(16); the fit gives ≈ 0.332.
  The naive difference is 0.038 against an error of ~0.016–0.020, i.e. ≈ 2σ
  before correlations. Expect the correlated test to *not* support "plateau
  at Δ = 1"; be ready for the weaker sentence.

**Tasks.**

1. **Window scan, offline.** Make `FIT_WINDOW` overridable (env/argv, with the
   repo's `validate_argv()` discipline) or write a small
   `scripts/fit_window_scan.py` that imports the estimator from
   `fit_glueball_overlap.py` rather than copying it. Windows: [2,7] (reference),
   [3,7], [2,6], plus [3,8] and [2,8] if the signal allows. For each window and
   each ensemble (`best_glueball_gelt_sm0-2-4-6_test_obars.pt`,
   `…_ens1_test_obars.pt`; also the d24/7-level dumps if S2 quotes them):
   m, A₀ for GELT and the projected GEVP, jackknifed ΔA₀ and Δm, combined over
   the two ensembles the way the record combines them (inverse-variance).
2. **Correlated χ²/dof.** Add a correlated χ² evaluation at the fitted
   parameters using the blocked-jackknife covariance of C(Δ) on the window
   (40 blocks for a 6-point window is estimable but noisy — report the
   covariance's condition number, and if needed a shrinkage/SVD cut, stated).
   Keep the fit itself as it is unless the correlated fit is stable; quote
   both χ² if they differ. Do not change the central estimator silently.
3. **Table + figure** for the thesis: rows = windows, columns = m and A₀ (GELT,
   GEVP), ΔA₀ ± err (σ), χ²/dof (correlated), per ensemble and combined; a
   companion plot of ΔA₀ vs window with the [2,7] band.
4. **m_eff(1) vs m_fit.** In each jackknife sample compute
   d = m_eff^GELT(Δ=1) − m_fit^GELT([2,7]) (same sample, so correlations are
   automatic), jackknife d, report d/σ_d per ensemble and combined; same for
   the GEVP at Δ = 1 and Δ = 2 as the contrast. Also report the already
   established Δm_eff(1)(GELT − GEVP) = −0.028(7) / −0.038(8) next to it —
   that one is the defensible statement.
5. **Optional, only if 1–4 are clean**: a two-state fit
   C_i(Δ) = A_i cosh-form(m₀) + B_i cosh-form(m₁) with m₀ (and m₁) shared by
   GELT and the projected GEVP, simultaneous on both correlators, jackknifed.
   A₀ then comes from A_i/C_i(0) with a shared ground state, which removes the
   "each operator picks its own mass" freedom. Do it on [1,8] or so; if m₁ is
   unconstrained (likely at 400 configs), say so and stop.

**Readings (fixed now).**

- ΔA₀ positive at ≥ 2σ in every window and its central value moving by less
  than its own error → "stable under the fit window", keep 3.6σ as the
  headline and quote the spread as a systematic.
- ΔA₀ moves by more than 1σ between windows, or drops below 2σ in one →
  quote ΔA₀ as the [2,7] value ± stat ± (half the spread across windows) and
  rewrite the abstract's "3.6σ".
- |d|/σ_d < 1.5 → "compatible with a plateau from Δ = 1 within errors".
  Otherwise → main.tex l. 59, 406, 427, 436, 505 say "reduced excited-state
  contamination already at Δ = 1" instead of "plateaus at Δ = 1".

**Task 1 — done 2026-09-27** (`python scripts/fit_window_scan.py`, offline,
1.5 s; imports `fit_one` / `project_ground` / `blocked_jackknife` from
`fit_glueball_overlap.py`, gated on [2,7] reproducing record §3.2 per
ensemble). Δmax = 8 admitted by the rule fixed in the script (C(8) ≥ 2σ for
both operators on both ensembles; S/N 3.0–4.1). Combined ΔA₀:

| window | ΔA₀ (combined) | Δm (combined) |
|---|---|---|
| [2,7] | +0.0772 ± 0.0215 (3.6σ) | −0.0168 ± 0.0095 |
| [3,7] | +0.1093 ± 0.0366 (3.0σ) | −0.0064 ± 0.0125 |
| [2,6] | +0.0739 ± 0.0213 (3.5σ) | −0.0186 ± 0.0094 |
| [3,8] | +0.1114 ± 0.0365 (3.1σ) | −0.0039 ± 0.0125 |
| [2,8] | +0.0810 ± 0.0220 (3.7σ) | −0.0142 ± 0.0097 |

Positive at ≥ 3.0σ in every window, but max − min = 0.0375 > 0.0215, so the
**second reading applies**: ΔA₀ = +0.077 ± 0.022 (stat) ± 0.019 (window);
the abstract's "3.6σ" is to be rewritten. The spread comes from the Δmin = 3
windows and on both ensembles moves ΔA₀ *up* (the GEVP loses more A₀ than
GELT when Δ = 2 leaves the fit). Most of it is ens1, where the GEVP's A₀
drops 0.925 → 0.884/0.846 while GELT's stays at 1.01–1.03 (ΔA₀ +0.089 →
+0.145/+0.168). Two side facts: the combined [2,7] value from unrounded
per-ensemble numbers is **+0.077**, not the +0.078 quoted (that one combines
the rounded 0.066/0.089); the diagonal χ²/dof is 0.01–0.20 everywhere, which
is what an uncorrelated χ² on correlated data looks like — task 2's
correlated one is the readable number.

**Tasks 2 and 4 — done 2026-09-27** (same script; `fit_cosh_correlator`
gained `cov=` for the correlated fit, tested; the diagonal path is
bit-identical and a second gate reproduces Δm_eff(1) = −0.028(7) / −0.038(8)).

*Task 2.* The quoted (diagonal) fits are good fits under the correlated
metric: at [2,7], summed over ensembles, correlated χ²/dof (× Hartlap
32/39) = **0.46 (p = 0.88) GELT, 0.43 (p = 0.90) GEVP**; 0.19–0.57 in every
window. cond of the window correlation matrix 1.4e2–8.9e2. **But the
correlated fit itself does not agree**: ΔA₀ = +0.031 ± 0.021 (1.5σ) at
[2,7], +0.027 … +0.081 across windows. Diagnosis: the window correlation
matrix is ~85% one common mode (top eigenvector flat in Δ), and the
correlated fit sits off the data along it — on run5 every GEVP residual is
positive (+0.1 … +0.7σ), on ens1 GELT's are negative. Fitting C/C(0)
instead does not remove it (+0.036 ± 0.020), so it is the weighting, not the
A₀ ratio. A shrinkage scan Σ_λ = (1−λ)Σ + λ·diag Σ (added after seeing
this, labelled so in the script) places the whole drop in λ < 0.1:

| λ | 0 (correlated) | 0.1 | 0.25 | 0.5 | 1 (the estimator) |
|---|---|---|---|---|---|
| ΔA₀ at [2,7] | +0.031 (1.5σ) | +0.054 (3.0σ) | +0.064 (3.4σ) | +0.071 (3.5σ) | +0.077 (3.6σ) |

One-sign residuals along a common mode are *not* by themselves a failure (a
fit that knows the points move together should not follow a common shift),
so this was settled by a synthetic test rather than argued.

*Synthetic test* (`scripts/fit_estimator_mc.py`, 15 s; readings fixed before
the run). Truth: C(Δ ≥ 1) a pure cosh with the real fits' (m, A), C(0)
observed, so A₀ is known and the model exact; T_diag takes (m, A) from the
diagonal fits (ΔA₀_true = +0.077), T_corr from the correlated ones (+0.030).
Noise: the measured joint jackknife covariance of [C_GELT, C_GEVP]; each of
1000 experiments per ensemble draws 40 Gaussian blocks and redoes the scan's
fits, Σ̂ from those 40 blocks included. Gate passed: synthetic sd of the
diagonal ΔA₀ / real jackknife error = 0.86 (run5), 1.00 (ens1). Combined, at
[2,7] (bias ± MC error):

| λ | bias, T_diag | RMSE, T_diag | bias, T_corr | RMSE, T_corr |
|---|---|---|---|---|
| 0 (correlated) | +0.0001 ± 0.0007 | 0.0213 | −0.0011 ± 0.0007 | 0.0206 |
| 0.1 | +0.0004 ± 0.0005 | **0.0156** | −0.0009 ± 0.0005 | **0.0153** |
| 0.25 | +0.0007 ± 0.0005 | 0.0163 | −0.0008 ± 0.0005 | 0.0166 |
| 0.5 | +0.0008 ± 0.0006 | 0.0182 | −0.0010 ± 0.0006 | 0.0188 |
| 1 (the estimator) | +0.0006 ± 0.0007 | 0.0207 | −0.0017 ± 0.0007 | 0.0218 |

- **Neither fit is biased on ΔA₀** (|bias| ≤ 0.002 at every λ, both truths,
  both ensembles). Each A₀ separately is biased *up*, more by the diagonal fit
  (+0.004 … +0.010) than by the correlated one (+0.001 … +0.004), and it
  cancels in the difference; it is a small part (~0.005) of why ens1's GELT
  A₀ = 1.013 exceeds 1.
- **The disagreement is noise-sized, barely**: under the null
  D = ΔA₀(corr) − ΔA₀(diag) = −0.001 ± 0.024 combined, observed −0.047,
  **two-sided p = 0.057** (T_corr 0.060); per ensemble p = 0.10 (run5,
  null sd 0.025) and 0.20 (ens1, null sd 0.041).
- **The correlated fit is not more precise** (RMSE ratio corr/diag 1.03 under
  T_diag, 0.95 under T_corr).
- **Reading R2 on both truths**: the diagonal fit stays, and +0.031 ± 0.021 is
  quoted as a noisier alternative estimate on the same data, whose distance
  from the headline is a ~1.9σ fluctuation. My earlier diagnosis ("the known
  failure of a correlated fit at 40 blocks") is **retracted**: at this noise
  the correlated fit is unbiased.
- Not pre-registered, for the supervisor: the most precise estimator in the
  test is λ = 0.1 (RMSE 25% below the diagonal fit's, unbiased), and on the
  data it gives **+0.054 ± 0.018 (3.0σ)**. Adopting it now would be choosing
  an estimator after seeing its value; it is reported, not used.
- Model misspecification inside the window is no longer needed to explain
  the gap, but at p = 0.057 it is not excluded either; task 5's two-state
  fit is what would test it.

*Task 3 — done 2026-09-27.* The same run writes
`results/glueball/fit_window_scan.tex` (booktabs tabular: blocks ens0, ens1,
combined; rows = windows; columns m and A₀ for GELT and GEVP, ΔA₀ [σ],
correlated χ²/dof G / P, ΔA₀ of the correlated fit) and
`fit_window_scan.png` (ΔA₀ against the window, diagonal fit only:
per-ensemble in grey, combined in blue; windows ordered
by Δmin so the Δmin = 2 → 3 step is visible; the correlated fit is in the
table only). Regenerated on every run, so they cannot
drift from the printout. No new numbers: they are the tables above.

*Task 4.* d = m_eff(1) − m_fit([2,7]) for GELT: run5 **+0.037 ± 0.021
(1.8σ)**, ens1 −0.022 ± 0.026 (−0.9σ), **combined +0.013 ± 0.016 (0.8σ)**.
Contrast: GELT at Δ = 2 −0.003 ± 0.014; GEVP at Δ = 1 +0.032 ± 0.020
(1.6σ), at Δ = 2 +0.016 ± 0.017. The combined reading (fixed in the script
before the run) gives **"compatible with a plateau from Δ = 1 within
errors"**. Caveat for the text: main.tex's 0.370(16) is the run5 value, and
on run5 alone the same test gives 1.8σ, above the 1.5 threshold — so the
sentence must cite both ensembles, not the run5 table. The defensible
statement next to it: **Δm_eff(1)(GELT − GEVP) = −0.032 ± 0.006 combined
(5.9σ)**; per ensemble −0.028(7), −0.038(8).

---

## S2 — Is the GEVP a complete enough benchmark?

**What the supervisor asks.** The same-input plaquette GEVP is the right test
of "nonlinear combination beats linear combination of the same inputs", but not
of "better than conventional spectroscopy". For the **thesis**: a precise
description of the GEVP choices and, if cheap, stability under the reference
times. For the **paper**: add rectangular spatial Wilson loops built from the
same smeared links and show how GELT's advantage changes going from a
plaquette-only basis to a slightly richer conventional one. Not a large
optimisation campaign; the goal is to show quantitatively that the network
builds useful, possibly complementary, operators.

**Where we are — the paper part is essentially done.** The fair fight
(`scripts/su2_fair_fight.py`, record §3.3) already has exactly this:

| comparison (combined, two ensembles) | ΔA₀ |
|---|---|
| 7-level GELT − `published` (4 levels × plaquette) | +0.131 ± 0.029 |
| 7-level GELT − `deep` (7 levels × plaquette, same inputs) | +0.095 ± 0.021 |
| 7-level GELT − `full` (7 levels × {1×1, 1×2, 2×2}, cubic-symmetrised, 21 ops) | +0.038 ± 0.027 |

and the operator decomposition (record §3.5) answers "complementary":
~80% of what the net adds over `deep` is rectangular-loop content it
rediscovered; ~2.5% of its norm² is outside every classical operator and
carries ΔA₀ = +0.021 ± 0.009. That is the supervisor's requested
"plaquette-only → slightly richer basis" story, already measured, with the
caveat that **A₀ saturates above ≈ 0.9** so it cannot rank `full` against GELT
further — which is why record §11 **P1** (Δm_eff(1) against `deep` and `full`,
offline) is the natural completion and should be run as part of this package.

**Tasks.**

1. **Write the GEVP description** (thesis text, no compute): operator basis
   (`glueball_operator`: sum of the three spatial-plane loops at each APE level,
   spatial-only APE, α = 0.5, levels 0/2/4/6 — and 0…16 for `deep`, shapes for
   `full`), vacuum subtraction, (t₀, t_d) = (1, 2), eigh-whitening vs the
   `truncate=True` variant and why (record §3.3's estimator lessons: floor vs
   truncate, superset ordering gate, the variational property being a theorem
   only at t₀ = 0), the projected-operator comparator (`gevp_ground_vector`,
   v₀ fixed and applied to the basis), v₀ refitted inside every jackknife
   sample. Also fix main.tex's APE formula (record §9 item 2).
2. **(t₀, t_d) stability scan, offline.** `fit_glueball_overlap.py` already
   has `GEVP_TD`; add t₀. Scan (t₀, t_d) ∈ {(0,1), (1,2), (1,3), (2,3)} on the
   `published` and `deep` bases, both ensembles: ΔA₀, Δm_eff(1), and the
   GEVP's own m_eff plateau. Use `truncate=True` for anything with more than
   four operators. (0,1) is also the arm that solves the network's own
   objective — see P1.
3. **Run P1** (record §11) with the half-split v₀ it prescribes so the
   classical arm is not scored in-sample. This is the unsaturated version of
   the table above.
4. **Framing for the thesis/paper**: the claim is not "GELT beats any GEVP"; it
   is the record §3.3 consolidated statement (same inputs: +0.075/+0.095 at
   ≥ 4σ; with rectangular loops: a 1–2σ edge; the residual is ~80% loop shapes
   and a small genuinely new part). Say that the 7-level net was trained after
   `deep` was seen to win on one ensemble.

**Readings.**

- (t₀, t_d) scan: ΔA₀ vs `published` changes by < 1σ across the scan → one
  sentence "stable under the reference times". Larger → the table reports the
  scan and the headline uses the (t₀, t_d) that maximises the classical A₀
  (the conservative choice).
- P1's readings as written in the record (Δm_eff(1) vs `full` ≤ −3σ → "better
  than the strongest classical basis on an unsaturated metric"; |Δ| < 2σ →
  "matches a 21-operator basis from 7 inputs").

**Task 5 (two-state fit) — done 2026-10-01, stopped by its own rule**
(`python scripts/two_state_fit.py`, offline, 2 s). C_i = A_i φ(m₀) + B_i φ(m₁),
m₀ < m₁ shared by GELT and the (1,2)-projected `published` GEVP, amplitudes
profiled out in closed form, window [1,8] (dof 10), whole fit and v₀ redone in
every jackknife sample. **m₁ is not constrained**: jackknife m₁ = 0.59 ± 0.47
(run5), 0.77 ± 0.54 (ens1), and on ens2 it sits at the grid edge (1.5) in 40/40
samples; the correlated A₀'s are unmeasurable (GELT 0.57 ± 0.80 on run5,
1.03 ± 0.35 on ens1), m₀ itself is 0.27 ± 0.17 / 0.37 ± 0.08 / 0.310 ± 0.034
against ±0.01 in the one-state fit, and χ²/dof = 0.02–0.09 (more parameters than
the 400-configuration data can fix). So, as the task anticipated, **m₁ is
unconstrained at this statistics; stop** — the per-ensemble ΔA₀ (+0.09 ± 0.09,
+0.22 ± 0.07, +0.125 ± 0.017) are all positive but the ens2 one is a one-state
fit with a pinned dummy state and the combined number (+0.13 ± 0.02) must not be
quoted. The one-state ΔA₀ stays the estimator; the two-state attempt goes in the
text as "a shared-mass two-state fit does not constrain the first excited mass
on 400 configurations (m₁ = 0.6–0.8 ± 0.5)". It does not test the p = 0.057
misspecification question either way.

**Task 5, high statistics — 2026-10-03.** The 400-configuration limit is the
test split, not the method: the network is frozen (checkpoint chosen on its own
validation split), so `scripts/high_stat_eval.py` samples a fresh SU(2) chain
(same L, Lt, β, ξ, seed 1002, n_skip 5), runs the forward pass only and builds
the classical APE basis on the *same* configurations, streaming (40k configs
would be ~210 GB). **40 000 configurations, ens2's net.** Ō on the new chain
matches the original test split (GELT mean −554.60 vs −554.56, std 0.981 vs
0.972; thin 2539.8 vs 2540.0). τ_int of Ō ≈ 0.5–0.8 saved configs (0.5 =
uncorrelated), so the configurations are effectively independent; jackknife
block 100 (400 blocks). Dump: `dumps/best_glueball_gelt_sm0-2-4-6_ens2_hs_test_obars.pt`.

- Two-state shared-mass fit, `python scripts/two_state_fit.py
  --dump=<dump> --block=100`, window [1,8], dof 10: **m₀ = 0.3145 ± 0.0036**,
  **m₁ = 0.726 ± 0.087** (at the grid edge in 0/400 samples; was 40/40 on
  400 configs), A₀ GELT 0.846 ± 0.023, A₀ GEVP 0.724 ± 0.025, **ΔA₀ = +0.1222
  ± 0.0048** (statistical only). χ²/dof = 0.03: diagonal weights on a
  ~85%-one-common-mode correlation matrix, as before; not an error signal.
- One-state fit, [2,7], same dump (`fit_glueball_overlap.py`): GELT m = 0.3294
  ± 0.0026, A₀ = 0.9335 ± 0.0053; projected GEVP m = 0.3376 ± 0.0029, A₀ =
  0.8458 ± 0.0057. **ΔA₀ = +0.0878 ± 0.0030** (29σ; the original ens2 was
  +0.100 ± 0.028), but **Δm = −0.0082 ± 0.0011 (7.7σ)**: at this statistics the
  two one-state masses are *not* equal, so "same mass as the GEVP" is
  qualified. Both sit above the two-state m₀ = 0.3145, i.e. both are
  contaminated by the first excited state in [2,7], the GEVP more (lower A₀).
  Window scan on the same dump (`fit_window_scan.py <dump> --block=100
  --windows=2-7,3-7,2-6,3-8,4-9,5-10`, V100 not needed afterwards; output
  `fit_window_scan_custom.*`):

  | window | Δm | ΔA₀ (diag) | corr. χ²/dof GELT (p) |
  |---|---|---|---|
  | [2,7] | −0.0082 ± 0.0011 | +0.0878 ± 0.0030 | 12.1 (0.00) |
  | [3,7] | −0.0065 ± 0.0014 | +0.0925 ± 0.0045 | 4.1 (0.01) |
  | [2,6] | −0.0084 ± 0.0010 | +0.0877 ± 0.0029 | 12.7 (0.00) |
  | [3,8] | −0.0061 ± 0.0014 | +0.0934 ± 0.0046 | 4.3 (0.00) |
  | [4,9] | −0.0046 ± 0.0018 | +0.0978 ± 0.0068 | 2.2 (0.07) |
  | [5,10] | −0.0045 ± 0.0023 | +0.0964 ± 0.0102 | 0.56 (0.69) |

  Readings. (i) **The one-state [2,7] fit is misspecified**: correlated χ²/dof
  12 (p ≈ 0) for both operators; the p = 0.057 of the 400-configuration run
  was the same effect at low power. The model fits (p = 0.69) only from
  [5,10]. This answers task 2's open question: excited-state contamination in
  the published window is real. (ii) **Δm shrinks with the window start**, from
  −0.0082 (7.7σ) at [2,7] to −0.0045 ± 0.0023 (2.0σ) at [5,10]; both masses
  fall (GELT 0.3294 → 0.3171, GEVP 0.3376 → 0.3216) towards the two-state
  m₀ = 0.3145 ± 0.0036 and are still above it. So "same mass" holds only at
  the ~2σ level on the latest window, where the errors are 4–5× larger; a
  residual cannot be excluded or shown. (iii) **ΔA₀ is positive at every
  window (≥ 9.5σ) but drifts up with the start**, +0.088 → +0.096…+0.098; spread
  0.0100. By the pre-registered task-1 rule (spread < the [2,7] error 0.0030)
  it is **not stable**, so: **ΔA₀ = +0.088 ± 0.003 (stat) ± 0.005 (window)**,
  15σ with the window. (The rule's threshold is the statistical error, so it
  tightens as statistics grow; applied as written.) (iv) **Task 4: not a
  plateau at Δ = 1**: GELT d = m_eff(1) − m_fit = +0.0174 ± 0.0024 (7.2σ), so
  the main.tex sentences (l. 59, 406, 427, 436, 505) take the "reduced
  excited-state contamination already at Δ = 1" branch — for this net at
  this statistics; at 400 configs the same test did not exclude a plateau.
  (v) **Open**: Δm_eff(1)(GELT − GEVP) = −0.0218 ± 0.0009 here against
  −0.041 ± 0.005 published for ens2 (3.8σ apart). Not understood — a
  400-configuration fluctuation or a definition difference; the scan's
  published-value gate was not run on a custom dump. Check before quoting
  the −0.041.
- What this is and is not. It is the precise value for **one net**; the
  headline ΔA₀ = +0.086 ± 0.017 combines three nets and three ensembles, so
  it contains the net-to-net spread (+0.066…+0.076 over three init seeds) and
  the high-statistics number does not replace it. No headline is chosen by
  which number is larger. The reading that was fixed: the high-statistics run
  *confirms* the original ens2 value if |ΔA₀(hs) − ΔA₀(400)| < 2σ of the
  latter (it is: 0.012 vs 0.028 for the one-state fit) and tests the
  one-state fit's excited-state bias through m₁ and Δm. Still stat only: the
  window systematic (±0.018) now dominates and is the next number.
- Scripts: `two_state_fit.py` gained `--dump=` and `--block=`;
  `fit_window_scan.py` gained `--block=`. Running `fit_glueball_overlap.py`
  on any dump overwrites `results/glueball/glueball_overlap.png`.

**Task 2 — done 2026-10-01** (`python scripts/gevp_time_scan.py`, offline, 5 s;
printout in `results/glueball/gevp_time_scan.txt`). 4-level GELT against the
v₀-projected `published` and `deep`, window [2,7], three ensembles combined by
inverse variance; gate: (1,2) reproduces ΔA₀ = +0.066 / +0.089 (run5 / ens1)
and the three-ensemble +0.086.

| (t₀, t_d) | GEVP A₀ | ΔA₀ vs `published` | Δm_eff(1) | GEVP A₀ (`deep`) | ΔA₀ vs `deep` |
|---|---|---|---|---|---|
| (0,1) | 0.873 | +0.067 ± 0.015 | −0.043 ± 0.005 | 0.907 | +0.031 ± 0.014 |
| (1,2) | 0.852 | +0.086 ± 0.017 | −0.037 ± 0.004 | 0.878 | +0.055 ± 0.020 |
| (1,3) | 0.828 | +0.104 ± 0.020 | −0.038 ± 0.005 | 0.870 | +0.045 ± 0.025 |
| (2,3) | 0.601 | +0.32 ± 0.17 | −0.10 ± 0.08 | 0.637 | +0.30 ± 0.19 |

*Reading.* (2,3) is not a measurement: C(2), C(3) are too noisy for a 4- or
7-operator GEVP, the projected operator's A₀ collapses to 0.6 and the errors
are 10× larger — the three ensembles agree on that, so it is the pair and not
one chain. It is reported and excluded from the reading. On the three usable
pairs ΔA₀ vs `published` moves by −1.09σ / +1.09σ of the (1,2) error, i.e.
just past the "< 1σ" threshold, so **the larger-shift branch applies**: the
table reports the scan, and the conservative headline is the (t₀, t_d) with the
highest classical A₀, **(0,1): ΔA₀ = +0.067 ± 0.015 (4.5σ) vs `published`**,
and **+0.031 ± 0.014 (2.2σ) vs `deep`** (was +0.055 ± 0.020). Both stay
positive on every usable pair and Δm_eff(1) (−0.037 … −0.043, ≥ 7σ vs
`published`) hardly moves; the GEVP's own plateau is stable at m_fit =
0.34–0.35 and m_eff(1) = 0.39 across them. The point for the text: (0,1)
is the pair that solves the network's own objective, and it is where the
classical arm is strongest, so the `deep` margin shrinks to 2.2σ there.

**Task 3 (= P1) — done 2026-10-01** (`python scripts/p1_meff_strong_basis.py`,
offline, 10 s; full printout in `results/glueball/p1_meff_strong_basis.txt`).
Δm_eff = m_eff^GELT − m_eff^classical, classical arm projected with
`gevp_ground_vector(truncate=True, eps=1e-4)`, v₀ refitted inside every
delete-block jackknife sample. Gates passed: 4-level GELT − `published`,
floor whitening, in-sample, reproduces −0.0279(72) / −0.0381(84). Combined
run5 + ens1, **7-level GELT**, Δm_eff(1) [Δm_eff(2)]:

| classical arm | (1,2) half-split | (1,2) in-sample | (0,1) half-split | (0,1) in-sample |
|---|---|---|---|---|
| `published` | −0.056(8) [−0.040(12)] | −0.053(7) [−0.042(12)] | −0.059(8) [−0.048(11)] | −0.058(8) [−0.047(11)] |
| `deep` | −0.042(9) [−0.027(10)] | −0.031(6) [−0.028(10)] | −0.039(7) [−0.034(9)] | −0.038(6) [−0.033(8)] |
| `full` | −0.022(25) [0.000(23)] | +0.004(9) [+0.004(12)] | −0.006(4) [−0.005(5)] | −0.004(4) [−0.004(5)] |

*Reading (record P1).* Against `full`: |Δm_eff(1)| < 2σ in all four variants
(−0.9σ, +0.4σ, −1.5σ, −1.1σ) → **"matches a 21-operator basis from 7
inputs"**; the ≤ −3σ branch is not reached. Against `deep` the 7-level net
leads at 2.6–6.0σ on Δm_eff(1) and Δm_eff(2) in every variant, so the same-input
advantage survives on an unsaturated metric, and it is the shapes that remove it.
Caveats: (i) half-split (1,2) vs `full` has errors 3–10× the others (run5 ±0.080):
200 configurations cannot fix v₀ for 21 operators at t₀ = 1, so only the (0,1)
rows constrain `full`; (ii) the 4-level net, same estimator, ahead of `published`
(−0.041(6)) is level with `deep` (−0.004(8)) and **behind `full`** in-sample
(+0.038(7) at (1,2), +0.017(3) at (0,1) — 5.9σ and 5.6σ, three ensembles), i.e.
on m_eff(1) the 4-level net does not match the 21-operator basis, which A₀
(saturated) could not show; (iii) ens2 has no 7-level dump. Not a selection:
all twelve cells are printed.

---

## S3 — Train/test separation and Monte Carlo autocorrelations

**What the supervisor asks.** Document ensemble generation, simulation
parameters, train/val/test sizes, spacing between saved configurations, and
treatment of autocorrelations; check that strongly autocorrelated
configurations were not split across train and test; clarify whether the
configurations used in the final fit are independent of those used for model
and hyper-parameter selection; motivate the jackknife block size by τ_int or a
block-size stability study — especially for Z₂ near criticality. No new
simulations.

**This is mainly a thesis-writing task** — the design already does the right
thing; it has not been *written down* or *measured* in the form requested. The
facts, from the code:

| | SU(2) glueball (`train_glueball.py`) | Z₂ attention (`train_z2_glueball.py`, `z2_attention_correlator.py`) |
|---|---|---|
| lattice | 24 × 12³, time = axis 0, β = 2.4, ξ = 3 (β_t = β ξ, β_s = β/ξ) | 3D, 48 × 24², β ∈ {0.7450, 0.7520, 0.7560, 0.7585} |
| update | heat-bath + 4 overrelaxation per sweep | exact Z₂ heat-bath (`z2_heatbath_sweep`) |
| thermalisation / spacing | `N_THERM = 300`, `N_SKIP = 5` combined sweeps | `N_THERM = 500`, `N_SKIP = 200` sweeps |
| configurations | 2000 per ensemble; ens0 (seed 0) and ens1 (seed 1) are independent chains | 2000 per β |
| split | contiguous, chain-ordered 70/10/20 → 1400 train / 200 val / 400 test | first `N_USE` (800 at R = 6) for training, split 70/10/20 inside; attention table on the remaining **1200 unseen** configs |
| model selection | best val loss (val split); test untouched until the report | same, on the training slice's val |
| jackknife | blocked, block 10 → 40 blocks on test | blocked, block 20 → 60 blocks |
| τ_int evidence | `scripts/check_glueball_autocorrelation.py` → `results/glueball/glueball_autocorrelation.png`, n_skip ≳ 2 τ_int of the smeared operator | none recorded per β |

**Tasks.**

1. **Collect τ_int** (no sampling): for SU(2), re-read or re-run
   `check_glueball_autocorrelation.py` and put its τ_int (plaquette, smeared
   operator) in the thesis. Then measure τ_int *on the production chain itself*
   with `gelt.sampler.integrated_autocorrelation_time` on the test-split time
   series of Ō(t) summed over t — for the thin, APE×6, GEVP-projected and GELT
   operators — from the dumps (offline; the test split is chain-ordered). For
   Z₂, the same on the cached ensembles per β (the classical operator on all
   2000 configs; CPU, minutes) — the slow β = 0.7585 matters most.
2. **Block-size stability**, offline: the errors on m_fit, A₀, ΔA₀ and
   Δm_eff(1) for block ∈ {1, 2, 5, 10, 20, 40} (SU(2)) and
   {1, 5, 10, 20, 40, 60} (Z₂, 1200 configs). Plot σ(block); the production
   block should sit on the plateau and be ≳ 2 τ_int. If σ is still rising at
   the production block, the quoted errors are too small — report which.
3. **Boundary leakage check**, offline: only the configs adjacent to the
   train→val→test cuts can be correlated across them. Drop the first
   g ∈ {0, 5, 10, 20} test configurations (g in units of saved configs,
   ≫ τ_int) and show ΔA₀ does not move. For SU(2) this also answers "is the
   fit sample independent of model selection": val sits between train and
   test, so train↔test are ≥ 200 saved configurations apart.
   **Done 2026-09-29** (`scripts/gap_scan.py`, offline, seconds, gated on the
   per-ensemble ΔA₀ at g = 0): the largest shift of ΔA₀ from g = 0 is 0.24σ
   (ens1, g = 20); combined +0.086 / +0.089 / +0.087 / +0.083 for
   g = 0 / 5 / 10 / 20. The reading below does not fire. Numbers in
   `thesis_record.md` §3.2; thesis ch. 4 `tab:gap_leakage`.
4. **Selection honesty** (text): checkpoint selection is on val only, but the
   *hyper-parameters and protocol* (input levels, width, the 7-level retraining)
   were chosen with test-split results in view during development — say so, and
   point to ens1 (a fresh chain, trained from scratch, record §3.1 "rep.") as
   the genuinely out-of-sample replication. For Z₂, note that the attention
   table's 1200 configurations were never touched by training or selection.
5. **Write the section**: the table above, τ_int, the σ(block) plot, the
   leakage check, the selection paragraph.

**Readings.** τ_int ≪ block and σ(block) flat at the production block → one
paragraph and a figure, nothing changes. σ(block) still rising for Z₂ at
β = 0.7585 → re-quote that row's errors at the plateau block and say so. Any
movement of ΔA₀ under the boundary gap beyond 1σ → escalate (not expected at
n_skip = 5 with 200 val configs in between).

Note for the Z₂ part: its classical GEVP column is broken independently of
this (record §9 item 1, APE α = 0.5); S3 documents statistics, P2 fixes the
basis.

---

## S4 — Is the learned operator in the A₁⁺⁺ channel?

**What the supervisor asks.** Gauge invariance plus zero-momentum projection do
not put an operator in 0⁺⁺: on the cubic lattice it must also be invariant
under the cubic group (and parity, charge conjugation). Because the network
uses oriented plaquettes, relative offsets and positional encodings, spatial
rotation invariance is not automatic. Either verify the final operator is a
scalar under the relevant lattice symmetries, or project it explicitly into
A₁⁺⁺ (e.g. average over the cubic group), then check that A₀ and the other
results are essentially unchanged. No architectural change required.

**Where we are — never studied, and the concern is real.** The classical
operators are A₁⁺⁺ by construction (`glueball_operator` sums the three spatial
planes; `su2_fair_fight.py`'s shapes are cubic-symmetrised). The network is
not:

- the input plaquette channels are ordered by plane (μ, ν) and mixed by a
  learned `ChannelLift` with free weights per channel — nothing ties the three
  planes together;
- RoPE rotates by axis (`pair_axis`) with learned per-axis frequencies;
- the L1-ball offset set *is* symmetric under signed axis permutations, and
  the transport is covariant, but the weights acting on them need not be.

So Ō_GELT(t) is a zero-momentum, gauge-invariant operator whose A₁⁺⁺ content is
unmeasured. Its non-A₁ components (E, T₁, T₂, and parity-odd parts) cannot
couple to the 0⁺⁺ ground state; they add to C(0) and to the excited-state
tail. Charge conjugation is not an issue for SU(2) (pseudo-real: Re Tr is
C-even identically), which the thesis should state in one line.

**Design.** The spatial cubic group with parity, O_h, is exactly the 48
**signed permutations** of the three spatial axes (time axis 0 untouched, so
the anisotropy is respected). For each g ∈ O_h, transform the test
configurations U → gU, run the trained network on them, and record Ō_g(t).
Since the ensemble measure is O_h-invariant, Ō_g is the operator g⁻¹·O_GELT
evaluated on the original configuration, and

  Ō_A₁⁺⁺(t) = (1/48) Σ_g Ō_g(t)

is the projection onto A₁⁺⁺ (A₁ of O, parity-even). Implementation notes:

- An axis permutation permutes both the link direction index and the lattice
  axes. A reflection x_k → −x_k mod L must use the index order already worked
  out for the parity test (CLAUDE.md caveat 7 / `tests/test_lattice.py`):
  U'_k(y) = U_k(P(y + k̂))†, i.e. shift **down** before reflecting. Gate the
  transform on (i) the Wilson action and (ii) the classical A₁ operator's Ō(t)
  being invariant to round-off — the same "check the action first" discipline.
- Work on links, then rebuild the whole input pipeline (APE levels, plaquettes,
  transport) from the transformed links via `train_glueball.config_inputs`, so
  the smearing is also checked to commute with g.
- Cost: 48 × 400 test configs × 2 ensembles of forward passes. Start with the
  generators (one axis swap, one reflection) as a smoke test; if those are
  already invariant to ~1e−6, the full average is a formality. V100, eval-only,
  write a new `scripts/cubic_projection.py` that dumps
  `{Ō_g}` per g so the analysis is offline.

**Tasks.**

1. Transform + gate (above), with a unit test that `glueball_operator`'s Ō(t)
   is invariant under all 48 elements.
2. Measure the **non-invariance**: ‖Ō_g − Ō_e‖/‖Ō_e‖ per g, and the fraction of
   C(0) outside A₁⁺⁺, i.e. 1 − C_A₁(0)/C(0) computed from the projected and
   unprojected operators.
3. Re-run the full analysis on Ō_A₁⁺⁺: fit (all S1 windows), A₀, ΔA₀ vs GEVP,
   Δm_eff(1), and the operator decomposition — for the 4-level net on both
   ensembles, then the 7-level nets.
4. Optional, same machinery: the untrained (`rnd`) nets, and the L-CNN arms,
   so the A₁ fraction can be compared across architectures.

**Readings.**

- Non-A₁ fraction of C(0) ≲ 1% and ΔA₀ unchanged within errors → one paragraph:
  "the learned operator is an A₁⁺⁺ scalar to x%; projecting it changes nothing".
- Non-A₁ fraction sizeable, and projected A₀ **higher** (expected: the other
  irreps only add C(0) and heavier states) → quote the projected operator as
  the result everywhere, re-derive ΔA₀ and the decomposition on it, and say
  the network learned an approximately but not exactly scalar operator.
- Projected ΔA₀ **lower** than unprojected by > 1σ → the unprojected advantage
  was partly non-0⁺⁺ content (e.g. 2⁺⁺ components whose tails the window
  mistakes for ground state); the headline moves to the projected number and
  S1's windows are re-checked on it.
- Whatever the outcome, the thesis states the symmetry of the final operator
  explicitly. Longer term (not now): impose it by averaging the network over
  O_h at inference, or symmetrise ChannelLift/RoPE — an architectural change the
  supervisor explicitly did not ask for.

**Implementation — 2026-09-28, not yet run on the V100.**

- `gelt/cubic.py`: the 48 elements as `CubicElement(perm, flips)` (reflect,
  then permute; `y = P·F·x`), `apply_cubic` on batched links (index operations
  and daggers only, exact), the ten O_h characters, `irrep_projections`
  (`P_Γ Ō = (d_Γ/48) Σ_g χ_Γ(g) Ō_g`) and `irrep_fractions`.
- `tests/test_cubic.py` (8 tests, 1.5 s): task 1's test — the anisotropic
  action and the classical Ō(t) at APE levels 0/1/3 invariant under all 48 to
  1e-10 — plus the composition law bit for bit (`g₂(g₁U) = (M₂M₁)U`, which ties
  the link maps to the matrices the characters are read from), character
  orthonormality, one plane's plaquette sum = A1g ⊕ Eg with the plane average as
  its A1g part, Σ Im Tr U₁ (SU(3)) = pure T1u. Mutation-checked: reversing the
  reflection's shift fails the action test, swapping χ_T1 ↔ χ_T2 fails T1u.
- `scripts/cubic_projection.py` + `scripts/cubic_batch.sh`: task 2 and the
  inputs of task 3 (details in the script's docstring). Gates before any hours
  are spent: the identity must reproduce the source dump (net and classical Ō,
  ≤ 1e-3 sd), every g preserves the action (1e-10, complex128) and leaves the
  classical Ō invariant at every input level. Writes, per net, the 48 Ō_g and
  an `…_a1_test_obars.pt` in the standard format, so task 3 is the existing
  offline scripts pointed at it.
- **Definition fixed before the numbers exist**: "fraction of C(0) outside
  A₁⁺⁺" is `1 − f_A1g` with `f_Γ = ‖Π_Γ δv‖²/‖δv‖²` on the O_h-augmented sample
  (`irrep_fractions`; Π_Γ the isotypic projectors of the regular
  representation, v the orbit vector Ō(g_k U)). The naive C_Γ(0)/C(0) of the
  projected series do not add up on a finite sample (a 2-configuration test
  gave Σ = 7.3); these add up to 1 exactly and are what the first reading's
  "≲ 1%" is read against. The script also prints C(0)[Ō_A1g]/C(0)[Ō_e].
- Checked locally only as plumbing (CPU, the Run-5 checkpoint, random links):
  the net's Ō already moves under one axis swap and one reflection, i.e. the
  architecture is not invariant by construction — expected, and says nothing
  about the trained operator on equilibrium configurations.

Next: `CUBIC_DRY_RUN=1 bash scripts/cubic_batch.sh`, then the batch on the V100
(phase 1 = the gates on Run 5 in minutes, then ~13 h estimated for the five
trained nets); copy `results/glueball/cubic/` to `dumps/cubic/`; task 3 offline.

**Correction (2026-09-29, same day):** the first version of this outcome and
commit `1fee29a`'s message said untrained nets are "62%" non-scalar; that was
seed 2 only (the log tails show one net per group). The nine range from 21% to
83%, set by the init seed.

**Outcome (2026-09-29).** Batch run with `CUBIC_RND=1 CUBIC_LCNN=1`: every
gate passed on every net (anchor at g = e; action 8e-16; classical Ō
invariant to 0.9–1.9e-4 sd), 131 s per element for a 4-level net, 189 s for a
7-level one, 75 s for the L-CNN. The L-CNN on ens0 did not run (the p2
checkpoint is not in the V100's `results/glueball/`; the script now also looks
beside the dump — copy `dumps/p2/best_glueball_lcnn_sm0-2-4-6_p2.pth` there
and re-run the batch, ~1 h). Dumps in `dumps/cubic/`, the verbatim log tails in
`notes/raw/cubic_readout_2026-09-29.txt`.

*The L-CNN on ens0 (p2), run 2026-09-30.* Under 5 of the 48 elements, one
(configuration, timeslice) of 9600 gives |Ō| between 1.6e4 and 4.2e10 against
a typical 55 ± 0.9. The same three configurations (27, 157, 349) account for all
of it, and they are ordinary in their own orientation. This is an exploding
output on equally probable configurations, not non-scalar content, and it is the
robustness asymmetry of record §5 seen through the symmetry. No GELT net departs
from its own Ō_e by more than 4 sd on any of its 48 × 400 × 24 evaluations
(L-CNN ens1: 0.95 sd). Without the three configurations, a diagnostic chosen
after seeing them: outside A1g 2.14% (T2g 0.6%, T1u 0.5%, parity-odd 1.0%),
δA₀ = +0.014 ± 0.009, ΔA₀ projected +0.072 ± 0.027. Combined with ens1:
δA₀ = +0.008 ± 0.006.

*Task 2 — how scalar is it.* Share of C(0) outside A1g (augmented-sample
definition above), and ν(g) = the per-element change of δŌ:

| net | outside A1g | largest non-A1 parts | ν(g) median (max) |
|---|---|---|---|
| 4-level, run5 / ens1 / ens2 | 1.81% / 1.91% / 1.73% | T2g 0.9%, Eg 0.3% / Eg 1.3%, T2g 0.3% / T2g 0.6%, Eg 0.5% | 0.20 / 0.22 / 0.19 (≤ 0.25) |
| 7-level, run5 / ens1 | 2.11% / 1.21% | T2g 1.3%, Eg 0.6% / Eg 0.4%, T2g 0.4% | 0.22 / 0.16 (≤ 0.24) |
| our L-CNN, ens1 | 1.68% | T2g 0.6%, T1u 0.4% | 0.18 (0.20) |
| untrained 4-level, 9 nets | 21–83% by init seed (rnd0 83%, rnd1 21%, rnd2 62%, the same on every ensemble) | **Eg 20–82%** | 0.78–1.56 |

The non-A₁ part of the trained operators is mostly Eg + T2g, the two cubic
irreps the continuum 2⁺⁺ subduces to (60–88% of the non-scalar part for GELT).
Parity-odd content sums to ≤ 0.5% (0.7% for the L-CNN). Untrained nets are far
from scalar, by an amount set by the init seed (Eg 20–82%: an operator that
weights the three planes differently is A1g ⊕ Eg). **Training removes that almost
entirely** (Eg → 0.3–1.3%), although nothing in the loss asks for it explicitly:
the 0⁺⁺ ground state is the only state the Rayleigh bound rewards. This is a
second "learned, not architectural" fact next to L9.

*Task 3 — does projecting change the result.* A₀(Ō_A1g) − A₀(Ō), same
configurations, correlated jackknife (`fit_glueball_overlap.py <a1> --vs=<orig>`):

| net | run5 | ens1 | ens2 | combined |
|---|---|---|---|---|
| 4-level | +0.003 ± 0.008 | −0.007 ± 0.011 | +0.014 ± 0.008 | **+0.005 ± 0.005** |
| 7-level | −0.002 ± 0.012 | −0.019 ± 0.008 | — | −0.013 ± 0.007 |
| L-CNN | — | +0.004 ± 0.007 | — | — |

Three-ensemble headline on the projected 4-level operators
(`fit_window_scan.py dumps/cubic/…_a1_…`; written to the `_custom` outputs;
every per-net number here is printed by `python scripts/cubic_readout.py`):
**ΔA₀ = +0.088 ± 0.016 (stat) ± 0.016 (window) = 3.9σ with the window
systematic** (per ensemble +0.068(29), +0.082(28), +0.113(28); p = 0.52),
against +0.086 ± 0.017 ± 0.018 (3.4σ) unprojected. ≥ 4.3σ statistical in every
window. Δm_eff(1) = −0.039 ± 0.003 (was −0.037 ± 0.004). The plateau test
still passes (d = +0.016 ± 0.013, 1.2σ). The correlated fit gives +0.053 ± 0.016
(was +0.049 ± 0.017).

*Against the readings fixed above:*
- **R1** (non-A₁ ≲ 1% and ΔA₀ unchanged): ΔA₀ is unchanged (+0.005 ± 0.005,
  i.e. 0.3σ of the headline's error), but the non-A₁ share is 1.2–2.1%. That is
  not "≲ 1%", so R1 is **not met as written**.
- **R2** (non-A₁ sizeable and projected A₀ higher): the projected A₀ is not
  significantly higher (1.0σ paired), and 2% is hard to call sizeable.
- **R3** (projected ΔA₀ lower by > 1σ): not triggered for the 4-level nets.
  For the 7-level ens1 net the drop is −0.019 ± 0.008. That is 2.3σ of the
  paired error but 0.5σ of that net's ΔA₀ error (±0.038), and "1σ" in R3 means
  the headline's σ. So R3 is not triggered. L8 re-derived on the projected
  7-level operators (`SFF_BASES=1 SFF_TRUNCATE=1 su2_fair_fight.py` on the two
  `_a1` dumps, reproducing +0.131 / +0.095 / +0.038 on the originals first):
  vs `published` +0.119 ± 0.028, vs `deep` **+0.081 ± 0.020 (4.0σ)**, vs `full`
  **+0.027 ± 0.023 (1.2σ)**, vs `shapes_sm` +0.042 ± 0.029. The reading of L8
  is unchanged.
- The outcome therefore sits **between R1 and R2**: a ~2% non-scalar
  admixture that changes nothing measurable. Not adopted yet, a decision for the
  author: take R2's action, i.e. quote the projected operator as the result
  (+0.088 ± 0.016 ± 0.016). This is the conservative reading of the
  supervisor's point, not a selection on the number, because it moves the
  headline by +0.002. The thesis sentence either way: "the learned operator is
  an A₁⁺⁺ scalar to 98% of its C(0); the remainder is mostly E/T₂ (2⁺⁺-like);
  projecting it onto A₁⁺⁺ changes A₀ by +0.005 ± 0.005".

---

## Third ensemble (ens2) — readings fixed 2026-09-27, before it exists

**Why.** After S1 the headline is ΔA₀ = +0.077 ± 0.022 (stat) ± 0.019
(window), i.e. 2.7σ with the window systematic, on two ensembles. A third
independent chain (seed 2) trained from scratch shrinks the statistical half
and gives a real test of ensemble-to-ensemble consistency (two ensembles
give one dof).

**Run.** `scripts/ens2_batch.sh` on the V100: (1) sample the seed-2 ensemble
and train the 4-level GELT, every setting at its default; (2) the fair-fight Ō
cache on ens2's test split; (3) three untrained 4-level nets; (4, opt-in
`ENS2_7LV=1`) the 7-level net. Protocol frozen: nothing is tuned on ens2, the
offline analysis is the existing scripts unchanged (their gates stay on
run5/ens1), and ens2 enters every combination once its dump is in `dumps/`.

**Readings** (all at [2,7], diagonal fit, inverse-variance over three
ensembles; `fit_window_scan.py` and `fit_estimator_mc.py` print every number):

- **E1 replication.** ens2's ΔA₀ > 0 and Δm_eff(1)(GELT − GEVP) < 0. No σ
  demand (per-ensemble error ≈ 0.03); a negative ΔA₀ is a failed replication
  and is reported as one.
- **E2 consistency.** χ² of the three per-ensemble ΔA₀ about their mean, 2 dof:
  p ≥ 0.05 → combine. p < 0.05 → the jackknife errors understate the scatter
  between ensembles; S3's block-size study is read before any combined number
  is quoted. (Two ensembles today: χ² = 0.28 on 1 dof, p = 0.59.)
- **E3 headline.** Three-ensemble ΔA₀ ± stat ± window, the window systematic
  re-derived by S1's rule. Total significance x/√(stat² + window²) ≥ 3 → the
  abstract may say "≥ 3σ including the fit-window systematic"; below 3 → it
  quotes the numbers with no σ claim. Expectation, not a reading: if ens2 lands
  near +0.077 the stat error falls to ≈ 0.018 and, with the window systematic
  unchanged, the total is ≈ 3.0σ — the third ensemble buys a clear ≥ 3σ only
  if the Δmin = 3 excursion (driven by ens1) also shrinks.
- **E4 correlated-fit gap.** `fit_estimator_mc.py` on three ensembles, its
  readings unchanged: R2 → closed as noise; R3 → S1 task 5 is required.
- **E5 plateau.** S1 task 4's reading on the three-ensemble d.
- **E6 learned, not architectural** (phase 3). The three untrained nets' A₀
  below ens2's GEVP, as on the other two ensembles.
- Phase 2 (and 4) add no reading of their own: the fair fight and the
  decomposition gain a third ensemble and their §3.3 / §3.5 readings apply.

**Outcome (2026-09-28).** All phases OK (training stopped early at epoch 29,
~10 h). ens2 alone at [2,7]: A₀ GELT 0.924(47), GEVP 0.824(53), **ΔA₀ =
+0.100 ± 0.028 (3.6σ)**, Δm (fit) −0.007 ± 0.010, Δm_eff(1) = −0.041 ± 0.005
(7.6σ).

| | [2,6] | [2,7] | [2,8] | [3,7] | [3,8] |
|---|---|---|---|---|---|
| ΔA₀, three ensembles | +0.081(17) | **+0.086(17)** | +0.090(17) | +0.112(27) | +0.117(26) |

- **E1 replicated**: ΔA₀ > 0 and Δm_eff(1) < 0 on ens2.
- **E2 consistent**: +0.066(31), +0.089(30), +0.100(28); χ² = 0.69 on 2 dof,
  p = 0.71.
- **E3 ≥ 3σ**: ΔA₀ = **+0.086 ± 0.017 (stat) ± 0.018 (window)** = **3.4σ
  including the window systematic** (5.0σ statistical). S1's window test
  still fails (spread 0.036 > 0.017), so the systematic stays quoted; the
  abstract may say "≥ 3σ including the fit-window systematic". ≥ 4.1σ
  statistical in every window.
- **E4 not read — the gate failed on ens2**: synthetic sd of ΔA₀ 0.019 against
  the real jackknife error 0.028 (ratio 0.70). Cause, measured: the real
  jackknife refits v₀ in every sample, the synthetic does not. With v₀ fixed
  the real errors are 0.026 / 0.030 / 0.018, matching the synthetic 0.026 /
  0.030 / 0.019; refitting v₀ adds +18% on run5, +2% on ens1, +51% on ens2. For
  the record only: the correlated fit gives +0.049 ± 0.017 (2.9σ) on three
  ensembles (ens2 alone +0.087 ± 0.030), λ = 0.1 gives +0.067 ± 0.015, and the
  gap to the diagonal fit sits at p = 0.053 against a null that lacks the v₀
  noise. Reading E4 needs the synthetic to draw the basis and refit v₀.
- **E5 plateau**: d = +0.017 ± 0.013 (1.3σ) combined → "compatible with a
  plateau from Δ = 1 within errors" (per ensemble +0.037(21), −0.022(26),
  +0.023(22)). Combined **Δm_eff(1) = −0.037 ± 0.004 (9.5σ)**.
- **E6, split by method.** Free cosh fit on [2,7]: ens2's untrained A₀ =
  0.447(146), 0.442(50), 0.855(426) against GEVP 0.824(53) — two below, the
  third's fit locks onto m = 0.84 ± 0.23, not the ground state (0.32), and its
  central value is 0.03 above. With the mass fixed at the GEVP's (as
  `operator_decomposition.py --m-ref` does for untrained nets): 0.18(2),
  0.37(2), 0.22(2) on ens2 and 0.15–0.42 on all nine untrained nets, against
  GEVP A₀ of 0.82–0.92 → below, by far.
- **Fair fight on three ensembles** (4-level GELT, truncated whitening):
  vs `deep` +0.055 ± 0.020 (2.7σ), vs `full` −0.001 ± 0.021, vs `shapes_sm`
  +0.003 ± 0.032 — a 4-level GELT is worth the 21-operator classical basis
  and no more, as on two ensembles.
- **…and at (t₀, t_d) = (0, 1)** (2026-10-02, `SFF_T0=0`; `shapes` /
  `shapes_sm` fall back at (1,2) and pass here on every ensemble): vs `deep`
  +0.031 ± 0.014 (2.3σ), vs `full` **−0.034 ± 0.010 (3.5σ)**, vs `shapes_sm`
  **−0.037 ± 0.012 (3.2σ)** — at S2's conservative pair the 4-level GELT is
  behind the shape bases, in the saturated region (classical A₀ 0.95–1.0).
  7-level GELT (run5 + ens1): vs `deep` +0.085 ± 0.023, vs `full`
  +0.034 ± 0.014, vs `shapes_sm` +0.028 ± 0.011. Table in
  `notes/thesis_record.md` §3.3.

## main.tex sentences these packages touch

To be edited only after the corresponding package has run:

- abstract l. 59–61 — "plateau already at Δ = 1" (S1), "3.6σ" (S1, S4);
- l. 406, 427, 436, 505 — the plateau sentences and captions (S1);
- the GEVP section — reference times, whitening, basis description (S2);
- a new data/statistics subsection — ensembles, splits, τ_int, block size (S3);
- a symmetry paragraph — A₁⁺⁺, parity, C for SU(2) (S4);
- plus the record's §9 items 2–4, which are independent of these.
