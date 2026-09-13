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

Measured on the V100 at the production batch: **9.4 GiB peak**, against GELT's
~25 GiB — batch 6 is safe with room to spare, and `BATCH_CONFIGS` never has to
move. (It may not anyway: it is also the VEV-estimate knob of the ratio
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
shift terms. GPU numbers should be better, since what was removed is mostly
launch overhead. Every rewrite is pinned against the naive definition it
implements (`test_lconv_matches_the_naive_definition`,
`test_lbilin_matches_the_naive_definition`, 1e-12 in complex128).

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
