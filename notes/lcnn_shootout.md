# The matched-parameter L-CNN shootout — design record

**Status (2026-09-13): built, not yet run.** The code path, the matched
geometry, the driver and the tests are in; no GPU hour has been spent. Nothing
below is a result.

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
| parameters (real DOFs) | 15 693 (d_model 16, 4 levels) / 23 741 (d_model 24, 7 levels) | 17 345 / 22 097 — 1.11× and 0.93× |
| receptive field | R = 2 L1-ball × 4 layers → Manhattan 8 | K = 2 × 4 layers → Manhattan 8 |
| loop degree | bilinear value path doubles per layer → ≤ 16 | L-CB doubles per layer → ≤ 16 |
| inputs | 3 spatial-plaquette channels per APE level | identical — `in_channels = 3·len(INPUT_SMEAR_LEVELS)` |

One L-CNN geometry brackets *both* trained GELT nets within ~10%, which is why
the shootout needs one geometry and not two.
`tests/test_lcnn.py::test_lcnn_default_geometry_matches_the_glueball_gelt_budget`
pins the budget ratio, so a later change to either architecture breaks the test
rather than the comparison.

The **remaining** asymmetry is deliberate and is the thing under test: L-Conv's
steps are axis-aligned, so the L-CNN reaches off-axis sites only by stacking
layers, where GELT's shortest-path-averaged transport reaches the whole L1-ball
within one layer. Correcting for that would be correcting away the architecture.

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

L-Bilin forms the channel-pair outer product `(B, c_left, c_right, *Λ, nc, nc)`
— 13×13 at `c_hidden = 6`, 25×13 in the first block with a 4-level input. At the
production `B = 6 configs × 24 timeslices` that is the memory wall, so `LCNN`
gained a `grad_checkpoint` flag (exact: `tests/test_lcnn.py` pins identical
gradients). **`BATCH_CONFIGS` may not be lowered to dodge an OOM** — it is also
the VEV-estimate knob of the ratio estimator, and changing it stops the run
being comparable to the GELT runs it is measured against. Part 0 of the driver
profiles one step before anything long starts.

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
