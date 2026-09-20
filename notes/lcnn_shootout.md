# The matched-parameter L-CNN shootout — design record

**Status (2026-09-13): built, not yet run.** The code path, the matched
geometry, the driver and the tests are in; no GPU hour has been spent. Nothing
below is a result. §8 records what reading the authors' own implementation
changed — including one defect in ours that would have made the baseline unfair.

---

## 1. The question

Every published claim in the glueball program compares GELT to a *classical*
operator: APE-smeared Wilson loops, the multi-level GEVP, the strengthened arms
of the fair fight. Both audits pushed on the classical side (is the comparator
strong enough? is the learned operator outside its span?). Neither asked the
question an examiner asks first:

> Is the advantage **attention**, or would any gauge-equivariant network
> trained on the same inputs with the same loss do the same?

The thesis names the baseline that answers it — the L-CNN of Favoni et al.
(2012.12901), whose primitives GELT is built on — and has never run it. The
architecture chapter's central substitution (L-Conv + L-Bilin → an attention
block with a matrix-bilinear value path) is currently argued, not measured.

`reports/curve` §7 pre-registers the expectation: **parity**. The smeared inputs
hand both networks the staple content, and at matched depth the loop degree is
the same. Parity is the honest form of "attention costs nothing, and the
attention field is a measurement a convolution cannot produce".

## 2. What "matched" means here

Matched-parameter alone would be a weak match — two networks can have the same
budget and see different neighbourhoods. The defaults in `train_glueball.py`
(`LCNN_K = 2`, `LCNN_C_HIDDEN = 6`, `LCNN_LAYERS = 4`) match GELT on four axes:

| axis | GELT | L-CNN |
|---|---|---|
| parameters (real DOFs) | 15 693 (d_model 16, 4 levels) | 14 305 at `c_hidden = 5` — 0.91× |
| parameters (the 7-level net) | 23 741 (d_model 24, 7 levels) | 26 033 at `c_hidden = 6` — 1.10× |
| receptive field | R = 2 L1-ball × 4 layers → Manhattan 8 | K = 2 **symmetric** × 4 layers → Manhattan 8 |
| loop degree | bilinear value path doubles per layer → ≤ 16 | L-CB doubles per layer → ≤ 16 |
| inputs | 3 spatial-plaquette channels per APE level | identical — `in_channels = 3·len(INPUT_SMEAR_LEVELS)` |

The budget is matched to *the net it is compared against*, not once and for all:
`c_hidden = 5` is the default because the batch trains the 4-level net; a
7-level shootout runs `--lcnn-c-hidden=6`.
`tests/test_lcnn.py::test_lcnn_default_geometry_matches_the_glueball_gelt_budget`
pins the budget ratio, so a later change to either architecture breaks the test
rather than the comparison.

The **remaining** asymmetry is deliberate and is the thing under test: L-Conv's
steps are axis-aligned, so the L-CNN reaches off-axis sites only by stacking
layers, where GELT's shortest-path-averaged transport reaches the whole L1-ball
within one layer. Correcting for that would be correcting away the architecture.
What is *not* deliberate is reaching in one direction only — see §8.1.

The inputs matter as much as the budget. Feeding the L-CNN thin plaquettes while
GELT eats the 4-level ladder would repeat the `published` straw-man mistake with
the sign flipped — `notes/audit_2026-09-06.md` §6.4 is the precedent.

## 3. Why it is a switch inside `train_glueball.py`

`GLUEBALL_ARCH=lcnn` (or `--arch=lcnn`) changes the model and its transport.
Everything else — ensemble and cache key, the contiguous chain-ordered split,
the input channels, the multi-Δ Rayleigh loss and its scale pin, the val-loss
checkpoint selection, the blocked jackknife, the dump — is the *same code*, for
the same reason `train_gelt.py` and `train_cnn.py` are one problem: a sibling
script drifts, and the comparison silently stops being one.

Two consequences worth stating:

* The per-timeslice constraint is unchanged. The axis transports are built from
  the same least-smeared 3D slice (`config_inputs`), so the L-CNN is a
  single-timeslice operator too and the variational bound holds for both.
* The dump keeps the `gelt_obar` key whatever the architecture is. Every offline
  consumer — `fit_glueball_overlap.py`, `su2_fair_fight.py`,
  `operator_decomposition.py`, `input_architecture_curve.py` — compares *one
  learned operator* against the classical span, which is exactly what an L-CNN
  dump is. `meta["arch"]` is how a reader tells them apart. So the whole
  analysis layer applies with no new code.

## 4. The hyperparameters are the real cost

`LR = 3e-3`, `INIT_SCALE = 10`, `WEIGHT_DECAY = 1e-3` were tuned for GELT. An
L-CNN that loses at GELT's settings is uninterpretable — it may only be
undertrained. So the batch runs four short (10-epoch) runs first, over
`lr ∈ {3e-3, 1e-3}` × `init_scale ∈ {1, 1e-4}`, on ens0 only, under disposable
`_sweep_*` tags that cannot collide with a real run.

The init-scale arm exists because the reference L-CNN init is scale-agnostic
(the paper's losses are supervised) while the Rayleigh loss is scale-*invariant*
except through the `(log C(0))²` pin: a probe batch starts at C(0) ~ 1e10, i.e.
a pin term of ~5 against ratio terms bounded by 1, so the first steps travel in
λ rather than in shape. GELT's own `INIT_SCALE = 10` exists for the same reason
and starts at C(0) ~ 0.3.

## 5. Memory

The whole step, profiled on the V100 at `BATCH_CONFIGS = 6` (`--arch=lcnn`):

| | s/step | s/epoch | peak |
|---|---|---|---|
| before §8 (`c_hidden = 6`, one-sided kernel) | 13.31 | 3115 | 9.42 GiB |
| after §8, gradient checkpointing on | 1.56 | 366 | 4.70 GiB |
| after §8, **checkpointing off** | **1.29** | **302** | 7.86 GiB |

**10.3× on the step**, against a kernel now doing 1.9× more shift terms — part
the rewrites, part the narrower matched width. Checkpointing is off for this arm
in `lcnn_shootout.sh`: at 2.6 GiB a block it buys memory nobody needs and costs
an extra forward per layer. That is an L-CNN-only setting — GELT OOMs at batch 4
without it. `BATCH_CONFIGS` still may not move whatever the profile says: it is
the VEV-estimate knob of the ratio estimator, so a different batch stops the run
being comparable to the GELT runs it is measured against. If a step ever does not
fit, `c_hidden` is the knob.

At 1.29 s/step the batch is one night: ~50 min per sweep arm (10 epochs), ~5.0 h
per full 60-epoch training, nearer 2 h if it early-stops where GELT did.

**The input stage is now the largest single term** — `config_inputs` is 44.9% of
the step (579 ms), of which the APE ladder is 489 ms, against 380 ms of forward
and 329 ms of backward. See `notes/performance_audit.md` §6.1: caching it was
rejected partly because "the input stage is not what dominates", which is no
longer true for this arm. (It may not anyway: it is also the VEV-estimate knob of the ratio
estimator, so changing it stops the run being comparable to the GELT runs it is
measured against. If a step ever does not fit, `c_hidden` is the knob.)

That figure was taken *before* §8.2, which removed the channel-pair outer
product — the `(B, c_left, c_right, *Λ, nc, nc)` tensor, 1.3 GiB per block at
`c_hidden = 6` — from the L-Bilin path entirely. `LCNN` keeps its
`grad_checkpoint` flag (exact: `tests/test_lcnn.py` pins identical gradients),
and part 0 of the driver profiles one step before anything long starts.

## 6. How to run it

```bash
LCNN_DRY_RUN=1 bash scripts/lcnn_shootout.sh        # what would run, no GPU
nohup bash scripts/lcnn_shootout.sh > logs/lcnn_shootout.log 2>&1 &   # parts 0,1
# read the four sweep logs, then:
LCNN_PARTS=2,3 LCNN_LR=<lr> LCNN_INIT=<scale> bash scripts/lcnn_shootout.sh
```

Then, offline and already written: one
`SFF_TRUNCATE=1 SFF_BASES=1 python scripts/su2_fair_fight.py <lcnn dump>` per
dump for ΔA₀ against the classical arms, `operator_decomposition.py` for what
the L-CNN found outside the span, and `input_architecture_curve.py` for the
fourth trace of the figure.

## 7. Reading the outcome (fixed in advance)

* **Parity** (|ΔA₀(GELT − L-CNN)| within ~1σ, both beating the GEVP) — the
  pre-registered expectation. It *strengthens* the general claim: a learned
  equivariant operator beats the classical GEVP, and the result is not an
  attention artefact. The attention-as-operator chapter is untouched either way,
  since the L-CNN has no attention map to correlate. The architecture chapter
  then says attention buys the L1-ball receptive field and an interpretable
  field, not raw overlap.
* **L-CNN worse** — the architectural claim gets the baseline it has been
  missing. Report the per-axis match (§2) alongside it, or the reading is
  "GELT had more capacity".
* **L-CNN better** — worth knowing before an examiner finds it, and it points at
  the transport average / receptive field rather than at the loss.

None of the three is a wasted run, which is the argument for spending the time.

## 8. What the authors' own implementation changed (2026-09-13)

`lge-cnn-master/` is the code for Favoni et al. (2012.12901). Reading it against
`gelt/lcnn.py` found one defect and three optimisations. The first is a
correctness issue for the *comparison*; the rest are cost.

### 8.1 Our L-Conv kernel reached in one direction only

`lge_cnn/nn/layers.py` loops over both orientations in both of its convolutions
(`LConv`: `for orientation in [+1, -1]`; `LConvBilin`: `for i, o in zip([0, 1],
[-1, +1])`, with `use_symmetric=True` making `kernel_range = [-(ks-1), ks-1]`).
Ours gathered `W(x + k·μ̂)` only, on the docstring's claim that "negative shifts
are subsumed by the W† channels of the augmentation".

**That claim is false.** Daggering commutes with the adjoint transport —
`T·W†·T† = (T·W·T†)†` — so a W† channel is the dagger of a *forward*-transported
matrix, never a backward-transported one. Perturbing a single site confirmed it:
the output at `x` moved only for perturbations at `x` and `x + μ̂`, so a stack of
four layers saw a one-sided cone, not a ball. Against GELT's signed L1-ball
transport that is not a matched receptive field, it is a handicapped baseline —
and the result would have been unreadable either way (a loss could have been the
handicap; a win would have been remarkable and suspect).

Fixed: `LConv(..., symmetric=True)` is the default and gathers `±k·μ̂` for
`k = 1 … K`, with `T_{−k,μ}(x) = U^(k)†_μ(x − k·μ̂)` read off the same link
products. `tests/test_lcnn.py::test_the_kernel_reaches_both_directions` pins the
support of a layer in both settings. The shift count per layer goes from
`D·(K+1)` to `1 + 2·D·K` (the local term is now one slot, as in the reference,
rather than duplicated per axis), which is why the matched width moved from
`c_hidden = 6` to `5`.

### 8.2 The bilinear never needs the channel-pair product

The reference's `LConvBilin` offers five orderings and marks
`bilin_implementation = 2` "the good one":

```python
tmp = einsum('uvw, bxwjk -> bxuvjk', weight, t_w)   # mix the transported axis first
w_c = einsum('bxvij, bxuvjk -> bxuik', w, tmp)      # then one bilinear contraction
```

The large transported-channel axis is contracted *before* any per-site matrix
product, so the `(c_left × c_right)` matrices per site never exist. Ours built
them (`L_e @ R_e` → 169 channel pairs per site at `c_hidden = 6`) and then
contracted. Adopted: `LBilin` now mixes the right channel axis into
`(c_out, c_left)` with one GEMM and closes with a single batched matmul, the
left channel axis folded into the contraction dimension.

### 8.3 Augment *after* transporting, not before

Ours augmented to `2C + 1` channels and then transported all of them. The same
identity as §8.1 says the W† half is free — and the identity channel transports
to itself — so only the `C` raw channels need the two products. That is a 2.2×
saving in the transport stage, and the W† half of the weight contraction comes
back through `Σ_j ω_j S_j† = (Σ_j conj(ω_j) S_j)†`, one GEMM and one dagger of
the *output*.

### 8.4 Fold the channel axis into the matrix dimension

At `nc = 2` every product here is a 2×2 matmul, and the reference is no better
off — which is why the repo also ships `layers_cuda.py`, ~2000 lines of
hand-written numba-CUDA kernels with explicit backward passes for exactly these
operations. We are not writing kernels for a baseline. The PyTorch-level version
of the same idea is the one GEMHSA already uses
(`notes/performance_audit.md` §3.4): fold the channel axis into the matrix
dimension so `T · [W_1 | … | W_C] · T†` is two batched `(nc × C·nc)` products
per site instead of `C` separate `(nc × nc)` ones.

**Measured (CPU, B = 8 slices of 8³, C = 5, K = 2, fwd+bwd):** 36.4 ms before →
18.8 ms for the same one-sided kernel (**1.9×**), and 27.3 ms for the corrected
symmetric kernel — i.e. 1.3× faster than the old block while doing 1.9× more
shift terms. Every rewrite is pinned against the naive definition it implements
(`test_lconv_matches_the_naive_definition`,
`test_lbilin_matches_the_naive_definition`, 1e-12 in complex128).

**Measured on the V100** (`scripts/bench_lcnn_reference.py`, B = 144 slices of
12³, C = 5, K = 2 — the production shape), one block:

| block | forward | fwd + bwd | peak |
|---|---|---|---|
| ours, `gelt.lcnn.LCB` | 85.3 ms | **180.5 ms** | 2.60 GiB |
| reference, `LConvBilin` (v2) | 85.9 ms | 377.0 ms | 4.34 GiB |

Ours is **2.1× faster end to end and 1.7× lighter**, with the forward a dead
heat — the whole gap is the backward. Two caveats before that is read as a
verdict on their code: their merged layer carries a full 3-index kernel,
7205 real weights against our factored pair's 2640 (2.7×), so it computes a
richer function per block; and their production path is the numba-CUDA kernels
(§8.4), not this one. What the number does establish is that our block is not
the slow one by construction, which is what the shootout needed.

### 8.5 What stays different, on purpose

* **Parametrisation.** Their `LConvBilin` is one 3-index kernel
  `weight[n_out, n_in1, n_in2]` (real-valued, with explicit conjugate channels);
  ours factors that through L-Conv's `c_out` and uses complex weights — a
  low-rank version of the same family, and the one the paper's Eq. 5/Eq. 6
  separation describes.
* **Transport construction.** They hop one link at a time, reusing the previous
  hop; we build the axis products `U^(k)` once per configuration
  (`build_axis_transports`, 7.1 ms of a 13 s step — not worth changing).
* **`scripts/bench_lcnn_reference.py`** times our block against theirs at the
  production shape, so "is ours slow?" has an answer that is not a guess. Their
  CUDA path is deliberately out of scope.

### 8.6 Nothing in their code speeds up GELT

Worth stating, because it is the natural next question. Both of the tricks §8.2
and §8.4 lift from the reference are **already in GELT's hot path** — the
channel fold in `GEMHSA.transport` (`notes/performance_audit.md` §3.4) and the
"weight the sum before the matrix product" identity in the value path
(`blocks.py:450`, `Σ_n α_n (Q_v† Ṽ_n) = Q_v† (Σ_n α_n Ṽ_n)`). That is the
explanation for the 13.3 s vs 7.77 s gap: not attention being expensive, but one
block having had two rounds of optimisation and the other none. The details, and
the three negative readings that came with them, are in
`notes/performance_audit.md` §6.5.

---

## 9. Result (2026-09-14) — parity, as pre-registered

Ran, offline, on the test-split dumps. `fit_glueball_overlap.py` gained
`--vs=<dump>`: a second learned operator from the same ensemble and split,
differenced against the first **inside every jackknife sample**. That is the
number this whole exercise was for, and it cannot be got by subtracting the two
ΔA₀-against-GEVP values by hand — they are correlated through the classical arm,
and the correlated error is ~4× smaller than the naive one.

Fit window Δ ∈ [2, 7], GEVP (t0, td) = (1, 2), blocked jackknife, block 10,
400 held-out configurations per ensemble — the published protocol, unchanged.

| | ens0 | ens1 | combined |
|---|---|---|---|
| A₀(GELT) | 0.903 ± 0.047 | 1.013 ± 0.062 | |
| A₀(L-CNN) | 0.896 ± 0.047 | 0.997 ± 0.062 | |
| A₀(GEVP-projected) | 0.837 ± 0.056 | 0.925 ± 0.071 | |
| ΔA₀(GELT − GEVP) | +0.066 ± 0.031 | +0.089 ± 0.030 | **+0.077 ± 0.022** (3.6σ) |
| ΔA₀(L-CNN − GEVP) | +0.058 ± 0.031 | +0.072 ± 0.026 | **+0.067 ± 0.020** (3.4σ) |
| **ΔA₀(GELT − L-CNN)** | +0.007 ± 0.013 (0.6σ) | +0.016 ± 0.013 (1.3σ) | **+0.012 ± 0.009** (1.3σ) |
| Δm(GELT − L-CNN) | +0.0019 ± 0.0052 | +0.0016 ± 0.0060 | consistent with 0 |

(ens0 is the clean 20-epoch rerun of §9.1, `_p2`; masses: GELT 0.3324 ± 0.0268 /
0.3745 ± 0.0308, L-CNN 0.3305 ± 0.0274 / 0.3730 ± 0.0314.)

(The GELT column reproduces the published +0.078 ± 0.022 from the same dumps, so
the combination convention here is the published one.)

**The pre-registered expectation holds.** A matched-parameter L-CNN, on the same
inputs, loss, splits and estimator, is the same operator to within **1.3σ**: the
same mass (Δm consistent with zero on both ensembles), an A₀ lower by
0.012 ± 0.009, and it beats the classical GEVP by the same margin. GELT is ahead
on both ensembles and by at most ~0.03 in A₀ at 2σ — a direction worth stating,
not a difference worth claiming. So §7's first reading applies:

* The general claim gets **stronger**: "a learned gauge-equivariant operator
  beats the classical multi-level GEVP" is not an artefact of attention.
* The attention-specific chapters are untouched — the L-CNN has no attention map,
  so §6.1/§6.3 are not claims parity can dent.
* The architecture chapter must now say what attention *does* buy: the L1-ball
  receptive field within a layer, an attention field that is itself a measurable
  lattice operator, and — from `notes/performance_audit.md` §5.0 — a 3.9× more
  expensive step. Not raw overlap. That is the honest sentence, and it was
  written down before the run.

### 9.1 The ens0 60-epoch run was unusable — **resolved 2026-09-14**

Rerun at 20 epochs (`_p2`, same lr and init scale, ~1.7 h): val −0.6146 — better
than the 60-epoch run's −0.6143 — test ratios 0.675 / 0.467 / 0.331, and the
top-1 configuration back at 0.9%, 4× the median. The §9 table above uses it; the
paragraphs below are the record of what the bad run was, and §9.2 is why.


`best_glueball_lcnn_sm0-2-4-6_test_obars.pt` (ens0, 60 epochs) reports the best
validation loss of every run — −0.6143 — and a **test** correlator that is
garbage: C(1)/C(0) = 0.171 against 0.674 for the four 10-epoch sweep arms on the
*same ensemble and the same splits*, i.e. a test Rayleigh loss of −0.143 where
val says −0.614. Its A₀ fit does not converge (0.24 ± 0.60).

Every other run is internally consistent (val ≈ −0.61, test ≈ −0.57). Two
candidate causes were considered — chain-local overfitting (val sits adjacent to
train in chain order, test at the far end) and a checkpoint collision — and
**both were wrong**: see §9.2. The rerun is kept separate rather than
overwriting, and the bad dump is kept as the evidence.

### 9.2 What actually happened to that run: one configuration in 400

Not chain-overfitting and not a checkpoint collision. **A single test
configuration carries 71.8% of that operator's C(0)** — config 371, 1187× the
median configuration's contribution. Drop it and the correlator is ordinary:
C(1)/C(0) goes 0.171 → 0.634, and dropping a second gets 0.673, which *is* the
sweep arms' value. The Δ ≥ 1 shape was never wrong — the run's own log shows
m_eff(Δ=1) = 0.396 ± 0.041 against the GEVP's 0.398 ± 0.016 — only the A₀
denominator was, which is why the mass looked fine and the overlap did not.

Config 371 is unremarkable to every classical operator: rank 145/400 by variance
at thin links, 1.07–1.44× the median across the smearing ladder. So it is not a
bad configuration. It is the network.

Measured across every dump we have (share of C(0) on its single largest
configuration):

| operator | top-1 share | × median |
|---|---|---|
| GELT, 12 untrained + 2 trained | 0.5 – 1.0% | 2.2 – 4.2 |
| L-CNN, trained ens1 + all 4 sweeps | 0.5 – 0.9% | 2.1 – 4.2 |
| **L-CNN trained ens0 (60 epochs)** | **71.8%** | 1187 |
| **L-CNN untrained, seed 0, ens0** | **35.9%** | 1.2 × 10⁶ |
| **L-CNN untrained, seed 0, ens1** | **97.9%** | 3.2 × 10⁶ |

0 of 14 GELT operators, 3 of 9 L-CNN ones.

> **"It is the architecture" has been qualified twice since, and both
> qualifications are measurements** (added 2026-09-20). First,
> `notes/where_attention_can_win.md` §9.9: on the 3D Z₂ vortex task the
> ordering **reverses** — GELT is 6.2× the more dispersed arm. Second,
> `notes/beta_transfer.md` §9.1: that reversal itself holds only at the
> coupling both arms were trained on, and one β away the ratio is 1.5 / 1.3 /
> 1.0. What survives all three measurements is narrower and still worth having:
> **a failure-rate asymmetry that appears in the predicted direction whenever
> the input distribution is stressed** — here, in `m1_probe.md` §0 (all six
> GELT runs above R² 0.81, three of six L-CNN below 0.25), and in
> `beta_transfer.md` §9.2 (1 of 6 seeds against 3 of 6 under coupling
> transfer, Fisher p = 0.545). Three directional observations, none
> significant on its own, is what `where_attention_can_win.md` §8 exists to
> turn into one measurement. Read the sentence below as the *mechanism* being
> proposed, not as a result it has earned.

The mechanism: an L-CB stack is a raw matrix polynomial in the plaquettes —
degree ≤ 16 after four layers — aggregated by L-Conv with unbounded weights, so a
configuration in the tail of the input distribution is amplified without limit.
GEMHSA aggregates over its neighbourhood with a **softmax**: a convex combination
with weights in (0, 1), on a score already normalised by √(d_qkv·nc). It cannot
put 72% of its output variance on one configuration, and across 14 measured
operators it never does.

**This is a real result of the shootout, and parity in A₀ does not contain it.**
The honest pair of sentences: at matched parameters the two architectures reach
the same ground-state overlap (§9), and the attention block is markedly more
robust across configurations than the convolutional one. It also costs 3.9× more
per step (`notes/performance_audit.md` §5.0) — all three belong in the write-up.

Consequences, in order:

1. `fit_glueball_overlap.py` now prints the top-1 share for **every** arm and
   refuses to let an A₀ be quoted above 10% — uniform, so it is a gate and not a
   thing applied to an arm one dislikes. It fires on exactly the rows above.
2. The ens0 row of §9 is the clean `_p2` rerun. Deleting config 371 by hand was
   never an option: the published protocol has no outlier rejection, and adding
   one for a single arm would be exactly the move this repo's audits exist to
   catch. Note what the rerun does *not* do — it does not make the blow-up go
   away, it re-rolls it. Three of nine L-CNN operators hit it; the next one
   might too.
3. **The untrained L-CNN trace is unusable at seed 0** on both ensembles. Any
   mean over seeds would be meaningless; the rule has to be fixed in advance
   (median over seeds, or the §9.2 gate as an exclusion criterion applied to
   every architecture) before those numbers go anywhere near the curve.

### 9.3 Parity extends to the error bars, and what the one real difference is (2026-09-14)

Two things needed pinning down before parity could be read correctly, and both
came out of auditing the follow-on question (*given parity, where could attention
ever win?* — `notes/where_attention_can_win.md`).

**Parity also holds in signal-to-noise.** The natural inference from §9.2 is that
GELT's bounded aggregation should buy smaller error bars. It does not. Blocked
jackknife of `δC(Δ)/C(Δ)` for the two learned operators on the *same* 400 held-out
configurations, ens1 (the clean trained pair):

| arm | Δ=1 | Δ=2 | Δ=4 | Δ=6 |
|---|---|---|---|---|
| GELT | 0.0322 | 0.0396 | 0.0720 | 0.1695 |
| L-CNN | 0.0322 | 0.0399 | 0.0728 | 0.1676 |

Indistinguishable. The pathological ens0 60-epoch arm is 1.7–2.3× worse at every
Δ, as its 71.8% top-1 share predicts — but that is a *failure event*, not a
systematic variance edge. So §9.2 is a statement about **failure rate across
training runs** (3 of 9 L-CNN arms unusable, 0 of 14 GELT), and the honest pair
of sentences stays as written there. **Do not quote it as an error-bar
advantage**, and do not build a follow-on proposal on one.

**The architectural difference is narrower than "polynomial vs rational".** The
loose framing — an L-CB stack is a raw matrix polynomial, so its weights are
fixed — is wrong in a way that matters for designing the next experiment.
`LAct` is `g(Re Tr W_c / nc) · W_c`, i.e. *input-dependent* gating, and L-Conv
mixes channels **before** it, so the gate's argument is a learned linear
combination of transported channels: an L-CNN can realise
`ReLU(Re Tr[A − B])`, a data-dependent difference test between two scales held in
different channels. GELT carries the *same* L-Act on its residual branch. So:

> **The only architectural difference between the two arms is whether the weights
> over *offsets* are input-dependent.** Gating, bilinearity, receptive field,
> loop degree, inputs, loss, splits and estimator are matched.

That is a better sentence for the architecture chapter than the one §9 currently
offers, and it makes this shootout a controlled A/B on a single mechanism. The
mechanism is **normalisation**: softmax depends only on score *differences* and is
invariant under a common additive shift, so α is a scale-free ratio across the
neighbourhood; `ReLU(Re Tr[A − B])` thresholds an *absolute* magnitude and must be
calibrated to an amplitude the fixed weights cannot adapt.

**Why the tie was structural.** Read against that statement, the 0⁺⁺ task offers
the mechanism nothing: the Rayleigh loss is built from the lattice-summed `Ō(t)`,
so site-to-site variation is integrated away before the loss sees it
(`notes/attention_as_operator.md` §1 made this argument about ℓ_att; it applies
verbatim to the operator); the optimum is a linear functional of loop shapes
(2.5% outside the `full` span, ~80% of that rectangular loops —
`notes/fable5.1_10-09_audit.md` §8.3); and there is one β, one scale, one
maximally symmetric channel. Under those three, softmax attention degenerates into
a convolution with learned per-offset weights, which is L-Conv. §7's first reading
was right for a stronger reason than it knew.

The selection rule that follows is in `notes/where_attention_can_win.md` §3.
The one task that appeared to satisfy it — flow-free topological charge density —
was built, run and **stopped on a measurement** (§4 there), which added two
criteria the rule was missing and turned the search toward §9.2's mechanism
rather than this section's.
