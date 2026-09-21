# Switching the baseline to the authors' own L-CNN

**Status (2026-09-20): built, tested, nothing measured.** Everything below
labelled *measured* was measured; the widths and the run order are fixed here
before the first production run, on purpose.

Every GELT-vs-L-CNN number in this repo was produced against **our**
reimplementation, `gelt/lcnn.py`. `scripts/bench_lcnn_reference.py` has said in
its own docstring from the start that ours and theirs are *not the same
function*. The baseline a thesis is read against should be the authors', so
`gelt/lcnn_reference.py` wraps the vendored `lge-cnn-master/` layers and
`lcnn_ref` is an arm wherever `lcnn` is.

The methodological point does not depend on the numbers changing. It is the
first question an examiner asks about a baseline, and "we reimplemented it" is
not the answer that closes it.

---

## 1. What actually differs

| | ours (`gelt/lcnn.py`) | theirs (`LConvBilin`) |
|---|---|---|
| bilinear kernel | L-Conv `ω[c_out, 2c_in+1, n_shifts]` then L-Bilin `β[c_out, c_left, c_right]` | one merged `weight[n_out, 2n_in+1, 2n_in·n_terms+1]` |
| family | the merged kernel **factored through `c_out`** — a low-rank member | unrestricted in the (local, transported) index pair |
| activation | `LAct`: `g(Re Tr W/nc)·W`, relu or softplus | `LActPoly`: `Σ_s a_s · (Re Tr W)^{s}·W`, plus an optional relu term |
| weights | complex | real, cast to complex at use |
| transport | precomputed axis products `(B, D, K, *Λ, nc, nc)` | one step at a time, inside the layer, from the raw links |
| layout | `(B, C, *Λ, nc, nc)` complex | `(B, N^D, C, nc, nc, 2)`, links packed into the field tensor |

The wrapper takes their layer **and** their activation. "The authors' L-CNN"
stops meaning anything if half the stack is still ours.

### The expressiveness question — *answered 2026-09-20, constructively*

Ours composes to `M_i[v,w] = Σ_m β_i[v,m]·ω[m,w]` over the (local-augmented,
transported-augmented) index pair. So the reachable kernel is confined to the
row space of ω — at most `2·c_out + 1` of the `t_w` available directions, the
factor 2 from the daggered right-hand block — while theirs is unconstrained.

**Theirs is contained in ours once the L-Conv width is large enough, and the
construction is explicit** (`test_our_lcb_reproduces_their_kernel_exactly_at_sufficient_width`):
take a random full `weight[n_out, 2c_in+1, t_w]` of theirs, set ω to select the
transported basis one direction per channel and β to their kernel verbatim, and
the two layers agree to **1.9e-15**. The correspondence between their column
order (−1…−K then +1…+K per axis) and ours (+1, −1, +2, −2) is *discovered* by
matching the two bases numerically, so the test cannot pass by re-implementing
their ordering and then checking its own re-implementation.

The width that suffices is `c_out = t_w − 1 = 2·c_in·(1+2DK)`. **Our arms run at
a small fraction of it:**

| task | our `c_hidden` | width for full expressiveness | fraction |
|---|---|---|---|
| M1 probe (3 channels in) | 6 | 78 | 8% |
| glueball (12 channels in) | 5 | 312 | 1.6% |

So the reimplementation is a **strict and large restriction at production
widths**, not a cosmetic difference — which is a reason to expect the re-runs to
move, and the honest answer to "is it expressiveness or parametrisation": it is
expressiveness, and the gap is two orders of magnitude in reachable kernel
directions.

One asymmetry is *not* quantified: their weights are real and ours are complex,
so at equal widths the inclusion runs only in the direction proved above. Whether
complex coefficients buy anything their basis cannot already reach was not
measured (the feature maps are not linearly independent at the shape the test
runs, so a rank argument there would not have been sound).

## 2. Three traps, each now a test

`tests/test_lcnn_reference.py`, 18 cases, all passing.

1. **The kernel-size convention.** Theirs is `kernel_range = [-(k-1), k-1]`;
   ours is `K` hops per axis in both orientations. **Their `kernel_size` is our
   `K + 1`.** At `D = 3, K = 2` both then carry 13 transported terms. Passing
   their `kernel_size=2` for our `K=2` would have given the baseline **half the
   receptive field per layer** — Manhattan 4 against GELT's 8 over four layers —
   with every other check still passing. `test_our_K_maps_to_their_kernel_size_plus_one`
   and the delta-probe support test pin it as integer set arithmetic.
2. **The layout.** Flattened lattice, split complex axis, links inside the field
   tensor. Pinned bit-exact both ways, *and* against their own `shift`: rolling
   in their layout must equal rolling in ours on every axis.
3. **Z₂ is real, their code is not.** Promoted to complex on the way in. A Z₂
   arm therefore costs 2× the memory here; that is their parametrisation, not a
   defect.

Plus gauge invariance of the per-site readout (SU(2) complex128, Z₂ float64, on
stacked multi-level inputs), non-cubic lattices, checkpoint exactness, and a
refusal if an axis-transport tensor is passed where raw links belong — it has
the same `D` on axis 1 and would otherwise flatten into the channel axis with
plausible-looking numbers.

## 3. The matched width is *not* ours, and it is forced

Their kernel is quadratic in the input width, `(2c+1)·(2c(1+2DK)+1)` per output
channel, so the **first** layer carries most of the budget and a single uniform
integer steps over the tolerance band in one hop. Their weights are also real,
so GELT's own count halves at `nc = 1` while theirs does not — which is why Z₂
has no uniform integer width inside the band at all.

Per-layer channel lists are **their** idiom (`conv_ch` in their models), and are
the fine adjustment:

**The selection rule**, applied to all three tasks: among per-layer shapes
inside `DOF_TOLERANCE`, take the one whose *narrowest* layer is widest, ties
broken by closeness to 1.0. The glueball's `[1, 4, 4, 4]` is nearer on ratio
(1.024) and is rejected by it — squeezing 12 input channels through a width-1
layer is a bottleneck GELT's ChannelLift does not have, and handicapping the
baseline is the failure mode this whole arm exists to remove.

| group | GELT | `lcnn` (ours) | `lcnn_ref` channels | `lcnn_ref` DOFs | ratio |
|---|---|---|---|---|---|
| SU(2) probe | 15405 | 14801 (0.961) | `[5, 4, 4, 4]` | 16461 | **1.069** |
| Z₂ vortex | 8253 | 7625 (0.924) | `[4, 3, 3, 3]` | 8661 | **1.049** |
| glueball, 4 levels | 15693 | 14305 (0.91) | `[2, 2, 2, 2]` | 17457 | **1.112** |

*Measured 2026-09-20* by `scripts/z2_dof_table.py`, which now prints the arm
under either group. `gelt.lcnn_reference.reference_dof_count` reproduces the
built model to the parameter (a test), so a width can be chosen before a GPU is
involved. A 7-level glueball net wants `[1, 1, 1, 1]` (1.009×), and both
`GLUEBALL_LCNN_REF_CHANNELS` and the probe's `LCNN_REF_CHANNELS` are overridable.

## 4. What does not transport

- **R-D dies.** `lcnn_norm` bounds the per-offset profile `Σ_s|ω̂| = 1`; in their
  kernel the offset axis is fused into `t_w_size` together with the channels and
  that profile does not exist. R-D stays with **our** implementation and must be
  labelled as a control on a reimplementation, not as a baseline reading.
- **The 3.9× step-cost ratio.** "5.04 s vs 1.29 s" is measured against ours and
  is quoted in three places (`update_2026-09-18.md` §1 and §5.3 point 4,
  `CLAUDE.md` Performance). Their merged einsum is a different cost and it must
  be re-measured — `bench_lcnn_reference.py` for the per-block half,
  `profile_glueball_step.py` for the step.
- **`init_w` is not `conv_init_scale`.** Both are init-scale knobs on the same
  layer family, written against different fan-ins. The 0.2 measured for ours
  (`z2_init_gate.py`, after 0.5 was falsified at the production volume) says
  **nothing** about theirs. `Z2GATE_ARM=lcnn_ref` gates it, over the grid
  `0.2, 0.5, 1.0, 1.5`, and the script refuses to run if the configured value is
  outside the grid — otherwise the verdict passes vacuously. The configured
  value is their own default, 1.0, **pending that measurement**.

## 5. What has to be re-run, and what does not

Re-run: `update_2026-09-18.md` §1 (parity + the robustness census) and §5.3's
L-CNN half (R-A on T0, R-C, the dispersion ordering), §6's cost ratio,
`lcnn_shootout.md` §9, `where_attention_can_win.md` §9.9, and the `lcnn`
checkpoints that feed `beta_transfer.md`.

**Untouched**, because no L-CNN enters them: R-B / R-B′ (M1 is a GELT-internal
ablation), R-I / R-I′, §7.6's signed ladder, the spectroscopy against the
classical GEVP, the two audits of `update_2026-09-18.md` §2 and §4, the
input↔architecture curve, and everything in `attention_as_operator.md`.

## 6. Order of execution

Local, done: the wrapper, 18 tests, the DOF table under both groups, and the
wiring — `lcnn_ref` is an arm in `probe_common.ARMS`, `GLUEBALL_ARCH=lcnn_ref`
in `train_glueball.py` (so `profile_glueball_step.py` picks it up through the
shared `_build_model`), `Z2GATE_ARM=lcnn_ref` in `z2_init_gate.py`, and the
labels in `fit_glueball_overlap.py` and `train_probe.py`. Full suite: 222 passed.

On the V100, in this order — each stage's outcome gates the next:

1. `Z2GATE_ARM=lcnn_ref python scripts/z2_init_gate.py` — forward-only, minutes.
   Also run it under SU(2) geometry before the probe: their init is not ours.
2. The learning-rate sweep **at the full 40-epoch horizon and on 2–3 seeds**.
   Not at 6 epochs and not on one seed: §8 of `notes/m1_probe.md` records that a
   rate chosen on a 6-epoch cosine is not transferable to a 40-epoch one, our
   `lcnn` arm's rate was never re-gated after that was found, and one seed of an
   arm whose range is 0.96 picks a rate for one initialisation.
3. `bench_lcnn_reference.py` and `profile_glueball_step.py` → the new cost ratio.
4. Probe M1, `lcnn_ref`: 3 targets × 3 seeds × 2 ensembles = 18 runs.
5. The glueball shootout with the new arm — `GLUEBALL_ARCH=lcnn_ref`, wired and
   smoke-tested at the production shape — then §9.9's Z₂ campaign.
6. Regenerate `update_2026-09-18.md` §1, §5.3, §6 and the two figures.

## 7. Readings, fixed before the runs

- **X-1 — does the baseline's accuracy move?** median ΔR²(`lcnn_ref` − `lcnn`)
  on T2 over the six paired cells. A null says our factorisation was not the
  limitation and every re-run number lands where the old one did.
- **X-2 — does the *dispersion* move?** sd over six T2 runs, `lcnn_ref` against
  `lcnn`'s 0.415 and `gelt`'s 0.049, with the distribution-free count (runs
  below R² = 0.25) alongside. This is the reading that matters: the dispersion
  is what `update_2026-09-18.md` §1 and §5.3 rest on, and the mechanism proposed
  for it — bilinear degree growth with unbounded weights — is **identical in
  both implementations**, so the prediction on record is that it does not move.
  If it does, the M2 story is wrong and the robustness claim goes with it.
- **X-3 — the cost ratio**, replacing 3.9× wherever it is quoted.
- **X-4 — the glueball parity**, ΔA₀(GELT − `lcnn_ref`) against the +0.012 ±
  0.009 on record, and the top-1 share census against 3-of-13.

X-2 is the one to read first: if it is a null, the programme's conclusions are
unchanged and the re-run has bought the baseline's provenance, which was the
point.

---

## 7. The initialisation, measured — *2026-09-21*. The knob is not a knob.

`glueball_init_gate.py`, 3 seeds, production shape, field entering the head:

| `init_w` | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 0.3 | — | 1.2e−31 | 1.6e−21 |
| 0.5 | 1.6e−16 | 2.0e−23 | 8.4e−16 |
| **1.0** | **5.2e−06** | **9.6e−08** | **4.5e−05** |
| 1.5 | 3.5e+29 | 1.5e+21 | 4.3e+31 |

**The field at four layers goes as `init_w^198`.** Fitting the two decades
between 1.0 and 1.5 on seed 0 gives that exponent, and with it: a **1% change
in `init_w` moves the field by 7.2×**, the value that would land on O(1) is
≈ 1.064, and at fixed `init_w = 1.0` **three seeds already span 471×**. The
usable window is narrower than the seed-to-seed spread, so `init_w` is not a
tunable parameter here — it is a knife edge. That is the doubly-exponential
degree growth of an L-CB stack with nothing normalising it, which is the same
mechanism as the Z₂ cliff, with the sign reversed by their fan-in at `nc = 2`.

**But a small field is not, by itself, fatal** — the head is linear, so a
uniformly small operator is a *scale*, and the Rayleigh loss is invariant under
`Ō → λŌ`. What killed the first run is that λ was left where the init put it:
C(0) = 1.2e−10 against the loss's floor `EPS = 1e−8`, so every batch came back
non-finite. The scale-free quantity is `C(0)/⟨Ō⟩²`, and that is what the gate
now reports alongside C(0).

`--calibrate-c0` exists for that — one forward at initialisation that rescales
the readout so C(0) starts at 1, fixing a gauge freedom the way GELT's
hand-set `INIT_SCALE = 10` does. **But it is the wrong cure here, and the
scale-free reading is what showed it.** Measured 2026-09-21 with 8
configurations, `init_w = 1.0`:

| seed | field @ layer 4 | ⟨Ō⟩ | C(0) | C(0)/⟨Ō⟩² |
|---|---|---|---|---|
| 0 | 4.8e−06 | 404 | 1.16e−10 | **7.1e−16** |
| 1 | 1.4e−07 | −315 | **0** | 0 |
| 2 | 3.0e−05 | −269 | 1.57e−10 | 2.2e−15 |

⟨Ō⟩ is **large**. The operator is not small, it is **constant** — constant to
float32 round-off, whose relative ε of 1.2e−7 floors `C(0)/⟨Ō⟩²` at ~1e−14, and
one seed is exactly 0. The mechanism is the head: with the field at 5e−6,
`relu(W·trace + b) ≈ relu(b)`, so the readout emits its own bias identically at
every site and every configuration and the signal is lost in the rounding.
Rescaling would multiply the round-off by 9e4 and buy a run that trains on
noise for hours, so `--calibrate-c0` now **refuses** below
`C(0)/⟨Ō⟩² = 1e−10`, with the diagnosis.

The stack's scale therefore does have to move, and the window is narrow but
may not be empty. Extrapolating the measured `init_w^198` from each seed's own
field: the value that lands on O(1) is 1.054, 1.064 and 1.083 for the three
seeds, and a single **`init_w ≈ 1.064` puts them at 0.03, 1.04 and 6.5** — a
220× spread, but all three inside a range a linear head can work with. That is
an extrapolation over a narrow interval and it is the next thing to *measure*,
not to assume. Upward is settled and closed: 2.0, 4.0 and 8.0 all overflow to
`nan` by layer 3 or 4, in every seed.

Two things this does **not** settle, and neither should be quoted as settled:

* whether the instability is a property of their architecture or of this
  transplant. Their own models train a supervised fixed-point action with their
  own depths, widths and input normalisation; four layers on 12 smeared channels
  under a Rayleigh loss is our setting, not theirs. The honest statement is
  about the transplant.
* whether `degree_range = 2` with the relu term is the right `LActPoly`. Their
  models take a per-layer list and allow 0 (no activation at all), so a milder
  nonlinearity is inside *their* design space and is the first thing to vary if
  calibration alone does not make the arm trainable.

One unrelated defect fixed on the way: their `layers.py` decorates `unpack_x`
and friends with `@torch.compile`, and inductor raised `BackendCompilerFailed`
on the V100, taking `profile_glueball_step.py` with it while the training path
survived. `reference_layers()` now unwraps those decorators in our copy of the
module — surgical, reversible, vendored source untouched, and narrower than a
global `TORCHDYNAMO_DISABLE` that would also silence `PROFILE_COMPILE`.

---

## 8. The paper, read — *2026-09-21*. Two of our choices were not theirs.

`PhysRevLett.128.032003.pdf` and `sm.pdf`, Tables V–VI of the Supplemental
Material, list every L-CNN architecture they report. In 1+1D and in 3+1D, every
one of them is:

    L-CB(K, n_in, n_out) × n  →  Trace  →  Linear(2·n_out, 1)

**There is no activation layer anywhere.** The Letter introduces L-Act under
"Additional layers" (Eq. 7, suggesting `g = ReLU(ReTr[W])`) and never uses it —
"L-Bilins are already nonlinear". `LActPoly` does not appear in the Letter at
all; it belongs to their later fixed-point-action code, which is the part of
the repository we are *not* reproducing. We had it after every layer.

That is almost certainly §7's catastrophe: a random polynomial multiplier per
layer compounding the degree growth L-Bilin already has. `use_act` now defaults
to **False**.

**And their head is a single `Linear` per site** — no hidden layer, no ReLU
(Table V: `Trace, Linear(4, 1)` for two channels). Ours is `Linear(2c, 32) →
ReLU → Linear(32, 1)`, inherited from `gelt/lcnn.py` so that the head matches
GELT's in the shootout. That is a defensible choice for the comparison and a
poor one for §7's failure: a saturating ReLU on a hidden layer is exactly what
turns a small field into an *exactly constant* output. `head_hidden=0` is the
paper's, `--lcnn-ref-head=linear`.

Three smaller divergences, all deliberate, all recorded here so nobody has to
rediscover them:

1. **Positive shifts only.** SM §III: "we restrict our convolutions to only
   consider positive shifts along the lattice axes, i.e. we use 0 ≤ k ≤ K
   unless explicitly specified otherwise", and SM Table VIII counts the loops
   that choice *misses*. We pass `use_symmetric=True`, as our own `gelt/lcnn.py`
   does, because the receptive field has to match GELT's — `notes/lcnn_shootout.md`
   §8.1 is where that was decided. Their code supports both.
2. **Kernel size grows with depth** in their four-layer models —
   `L-CB(2,·), L-CB(2,·), L-CB(3,·), L-CB(3,·)` for `W^(4×4)`, and (3,3,4,4) for
   the medium one. Ours is uniform `K = 2`. Matching depth and reach to GELT is
   why; it is a difference, not a defect.
3. **Their 3+1D four-layer models are 2109 and 14377 parameters**, so our
   matched `[2,2,2,2]` at 17433 sits just above their largest reported 3+1D
   architecture. The budget is not the problem.

Two things the paper *confirms* rather than contradicts:

* **Label rescaling is their own device.** SM Eq. (13): `q̃ = C·q` with `C = 100`
  for the topological-charge regression, "similar to simple whitening
  transformations… which normalize the domains of the output and input data".
  `--calibrate-c0` is the same move made on the operator's scale instead of the
  label's, and it is now precedented in their own work.
* **Four L-CB layers is inside their range** — the Letter says "up to four
  layers of L-Conv + L-Bilin and ≈40000 trainable parameters".

And one warning that lands directly on the glueball task: SM §V, "we have found
that leaving out the lattice average for L-CNNs led to much easier training in
terms of convergence. **L-CNNs with global average pooling often did not
converge at all.**" Our Ō is a zero-momentum projection, i.e. a lattice sum —
exactly the setting they warn about. GELT and our own `lcnn` arm train through
it, so it is not fatal; it is a reason to read a non-converging `lcnn_ref` run
as possibly *theirs* rather than *ours*, and to say so rather than quietly
tuning until it goes away.
