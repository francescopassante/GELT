# Reproducing Fig. 3 of PRL 128, 032003 — 1+1D Wilson-loop regression

**Status 2026-09-22: wired, smoke-tested, not yet run at production size.**
Everything below is fixed in advance; §7 is the pre-registration.

The target: on an 8 × 8 SU(2) lattice, regress the four traced Wilson loops
`W^(1×1)`, `W^(1×2)`, `W^(2×2)`, `W^(4×4)` from the gauge field with the
authors' own L-CNN, at the architectures and hyper-parameters the Supplemental
Material prints, and land on the four MSEs of Fig. 3. **One network per loop
shape, four runs.**

    bash scripts/wilson_regression.sh          # data + four trainings + the figure
    WR_DRY_RUN=1 bash scripts/wilson_regression.sh   # what would run

**Scope: the L-CNN half of Fig. 3.** Their panels also carry a baseline CNN,
whose point is that it degrades with loop size until at 4 × 4 it predicts the
training mean and nothing else. That half is **not** reproduced. It is not the
claim this repo's L-CNN arm needs to stand on, and their baseline study is
2 680 models over 264 architectures and four activation functions — a
reproduction of it would be the larger half of the work and would say nothing
about whether our L-CNN is theirs.

---

## 1. What is being reproduced

Fig. 3's four panels, and the four MSEs printed in their upper-left corners:

| loop | L-CNN (the target) | CNN (not reproduced, for scale) |
|------|--------------------|---------------------------------|
| `W^(1×1)` | **2.2 × 10⁻¹¹** | 1.0 × 10⁻⁹ |
| `W^(1×2)` | **2.1 × 10⁻⁹**  | 2.0 × 10⁻³ |
| `W^(2×2)` | **1.1 × 10⁻⁸**  | 4.0 × 10⁻³ |
| `W^(4×4)` | **1.4 × 10⁻⁷**  | 4.2 × 10⁻³ |

`PAPER_MSE_LCNN` in `scripts/wilson_regression_common.py` carries the left
column, and every run prints its own number next to it. The right column is
here so the size of the gap the Letter reports is on the page; nothing computes
it.

The observable is Eq. (12), `W^(m×n)_{x,01} = Re Tr[U^(m×n)_{x,01}]/N_c`, per
site, in the single plane 1+1D has. That is exactly
`gelt.lattice.rectangular_wilson_loop(U, SU(2), R=m, T=n, mu=0, nu=1)`.

## 2. The datasets

`scripts/wilson_regression_data.py`. The Monte Carlo is theirs (SM §I):
Metropolis with `U' = V U`, `V = exp(i Σ_a T^a X^a)`, `X^a = A η^a` with
`η^a` standard normal and amplitude `A = 0.5`, **ten hits per link per sweep**,
`N_warmup = 2 × 10³` sweeps from a random start. The proposal is
`gelt.sampler._su2_proposal_paper`, in closed form — `T^a = σ^a/2` makes the
exponent `(X·σ)/2`, so `V` is the unit quaternion `(cos|X|/2, sin|X|/2 · X̂)`
and there is no matrix exponential and no projection. It is **not**
`_su2_proposal`, whose vector part is uniform on a cube and cannot reach `−𝟙`.

Three deliberate departures, all recorded here so none of them is a surprise:

1. **One chain per configuration, not one per coupling.** They run a single
   chain per β and save every `N_obs = 10²` sweeps. Here every configuration is
   its own chain, all thermalised together in one batched sweep
   (`gelt.sampler.metropolis_sweep_multichain`, with β a per-chain vector). The
   samples are then independent by construction rather than by a skip interval,
   and the whole ladder costs `N_warmup` sweeps instead of
   `N_warmup + N_obs × N_cfg`. `WR_CHAINS` caps the batch when memory binds, and
   the shortfall is made up their way — further snapshots `WR_SKIP` sweeps
   apart down the same chains.

2. **The β ladder has 11 entries, not 10.** SM Table I says
   `β ∈ {0.1, …, 6.0}`; SM §II says the step is `(β_max − β_min)/N_β` with
   `N_β = 10`. Ten steps between those endpoints is *eleven* couplings. This
   takes the table's endpoints literally: β = 0.1, 0.69, …, 6.0, and per-coupling
   counts of 910 / 91 / 91 put the totals at 10 010 / 1 001 / 1 001 against their
   10⁴ / 10³ / 10³. `WR_N_BETA=10` drops the top coupling and makes the totals
   exact. Nothing in the published material decides which they ran; the knob
   exists because of that, not despite it.

3. **Links and labels are stored in float64.** The Letter's best MSE is
   2.2 × 10⁻¹¹, i.e. errors of ~5 × 10⁻⁶ on a quantity of order one, and a
   16-link `W^(4×4)` accumulated in complex64 is good to ~10⁻⁶. The label must
   not be the noise floor of the thing being measured. Training casts down to
   float32 at load, which is what their code runs in.

**The correctness gate is exact, not conventional.** 2D SU(2) is solvable and
`⟨Re Tr P⟩/N_c = I₂(β)/I₁(β)`;
`tests/test_sampler.py::test_multichain_metropolis_mean_plaquette_matches_exact_2d`
pins the batched sampler against it at three couplings at once, so a β-broadcast
bug shows up as the *ladder* being wrong rather than the sampler. Measured at
512 chains and 600 sweeps: pulls −0.5σ, −1.3σ, −0.4σ at β = 0.5, 2.0, 4.0.

## 3. The architectures

`LCNN_ARCHS` is SM Table V verbatim, in the table's own `L-CB(k, n_in, n_out)`
notation, not paraphrased into a different parametrisation, so it can be checked
against the PDF line by line — and it is checked against the table's own
`N_param` column **at build time**, because a silently wider network is the one
way this reproduction could "succeed" without reproducing anything.

The L-CNN stack is a bare `L-CB × n` → `Trace` → one per-site `Linear(2·n_out, 1)`.
No L-Act (there is none anywhere in the Letter), no global average pool (SM §V:
"leaving out the lattice average for L-CNNs led to much easier training"),
positive shifts only (SM §III), and `k` grows with depth in the `W^(4×4)`
networks — which is why `LCNNRef` now takes a **per-layer** `K`. Their
`kernel_size` is our `K + 1`.

## 4. The divergence that forced `gelt/lcnn_exact.py`

The vendored `LConvBilin` is the authors' code, but it is **not** the network
whose parameter counts Table V prints. Their `forward` seeds the transported
list with the field itself (`t_w = [w]`) and then appends the `D(k−1)` shifted
copies, so the bilinear runs over `1 + D(k−1)` slots. Table V is reproduced
exactly — all ten architectures, four kernel sizes, five widths — by
`max(1, D(k−1))` slots instead: the transported copies **alone**, with the local
field kept only when there are none (`k = 1`, the `W^(1×1)` network, where the
list would otherwise be empty).

| architecture | Table V | vendored | `lcnn_exact` |
|---|---|---|---|
| `W^(1×1)` small | 12 | 12 | 12 |
| `W^(1×2)` small | 35 | 47 | 35 |
| `W^(1×2)` medium | 117 | 141 | 117 |
| `W^(1×2)` large | 329 | 377 | 329 |
| `W^(2×2)` small | 125 | 177 | 125 |
| `W^(2×2)` medium | 1 305 | 1 617 | 1 305 |
| `W^(2×2)` large | 13 521 | 15 745 | 13 521 |
| `W^(4×4)` small | 465 | 597 | 465 |
| `W^(4×4)` medium | 4 833 | 5 721 | 4 833 |
| `W^(4×4)` large | 39 905 | 46 481 | 39 905 |

This is not cosmetic. The extra slot is the local `W` on the *right* of the
bilinear, so the vendored layer can form `W_i · W_j` at one site while the
Letter's can only form `W_i · (transported W_j)` — the unit element already
supplies `W · 𝟙`. A strictly larger function class at strictly more parameters.

Which is right is not decidable from the published source: the repository has
moved on since the Letter, and there is no tagged version. So neither is treated
as the correction of the other. `WR_CONV_IMPL=ref` is **the code as published**,
`WR_CONV_IMPL=exact` is **the architecture as reported**, and the batch defaults
to `exact` because the claim being reproduced is "this many parameters reach
this MSE". `tests/test_lcnn_exact.py` pins the table, pins the excess, and —
the load-bearing one — shows the two are the *same function* once the slots the
vendored class adds are zeroed, which tests the ordering of that axis and not
merely its length.

Note the consequence for the rest of the repo: every "the authors' own L-CNN at
N parameters" in `notes/lcnn_reference_switch.md`, `notes/lcnn_shootout.md` and
`notes/m1_probe.md` is quoting the *wider* variant. Those comparisons are
matched to **our** L-CNN by parameter count and are unaffected as comparisons;
what changes is only the sentence "this is the paper's network", which it is
not, quite.

## 5. Training protocol

SM §VI.A, transcribed into `TRAIN_HP` and asserted by a test:

* AdamW, **zero weight decay**, for everything.
* `W^(1×1)`, `W^(1×2)`: lr 3 × 10⁻³, ≤ 20 epochs, early stopping patience 5.
* `W^(2×2)`, `W^(4×4)`: lr 1 × 10⁻³, ≤ 100 epochs, patience 25.
* Batch size 50 throughout.

Selection is on validation loss, which is also what early stopping watches, and
the figure's "best model" is the best **validation** seed — never the best test
MSE, since the test MSE is what the plot is evidence about.

Nothing else is tuned. If a number misses, it misses.

## 6. The two MSEs, and which one is theirs

Fig. 3 plots **one point per test configuration**, not one per site: their
`LCNN.mse(global_average=True)` averages the per-site predictions over the
lattice before comparing, because the baseline CNNs end in a global average pool
and can only produce one number per configuration. So the number to quote is the
**lattice-averaged** MSE, `mse_avg`.

`mse_site` is reported next to it every time, because it is the loss actually
minimised and because an operator can be right on average and wrong site by
site — the averaged number alone cannot tell those apart, and on an 8 × 8
lattice it is 64 times more forgiving. Both are in every dump, and the dumps
keep the **per-site** predictions, so either can be recomputed offline.

## 7. Pre-registered readings

* **R1 (the headline).** `mse_avg` for the best L-CNN at each loop, against
  2.2e−11 / 2.1e−9 / 1.1e−8 / 1.4e−7. Reproduced = **within a factor of 10**,
  which is the resolution at which the Letter itself prints them (one
  significant figure, best of ten seeds, a different ensemble).
* **R2 (the shape of the curve).** The four MSEs should climb with loop size —
  theirs go 2.2e−11 → 2.1e−9 → 1.1e−8 → 1.4e−7, four orders over the four
  panels — because the number of L-CB layers grows only as
  `⌈log₂(N²)⌉` while the number of Wilson loops a layer must combine grows
  exponentially with path length (SM Tables VII–IX). A flat curve would mean the
  ensemble, not the architecture, is setting the floor.
* **R3 (the scatter).** Each panel should sit on the 45° line over the *whole*
  range the labels span, not only near their mean. `W^(4×4)` at β = 0.1 has
  labels near zero and at β = 6.0 near the maximum; a model that fits the bulk
  and misses the tails has a small MSE and a visibly bent panel.
* **R4 (precision floor).** `W^(1×1)` is exactly representable by the
  12-parameter network (constructed, not fitted, in
  `test_w11_is_exactly_representable_by_the_12_parameter_net`), so its MSE is a
  float32 floor rather than an approximation error. A smoke run on 120
  configurations reached **4.8 × 10⁻¹⁵**, below the paper's 2.2 × 10⁻¹¹ — this
  reading is a *consistency* check, not a difficulty.
* **R5 (`ref` vs `exact`).** Same loop, same seed, both L-CB parametrisations
  (`WR_CONV_IMPL=ref` reruns the four). The extra slot is extra capacity, so
  `ref ≤ exact` in MSE is expected and a large gap would mean the same-site
  square `W_i · W_j` matters for these loops — worth saying, since the Letter's
  loop-counting analysis (SM Tables VIII–IX) assumes the published slot set.
  This is a *side* reading: the reproduction is the `exact` column.
* **R6 (volume transfer).** The 8 × 8-trained L-CNN evaluated at 16/32/64
  without retraining (`WR_TEST_SIZES`). Exactness of the mechanism is already a
  test (a 2 × 2 tiling reproduces the tiled output to 1e−10); the reading here is
  whether the *MSE* survives, which is a statement about the ensemble, not the
  architecture.

## 8. What is run and what is not

**Wired and smoke-tested** (tiny ensemble, CPU): data generation, all four
loops, both L-CB parametrisations, the figure, the transfer path. `W^(1×1)`
reaches 4.8e−15 on 120 configurations in 600 epochs.

**Not run**: the production ensemble (10 010 configurations at 11 couplings,
2 000 warmup sweeps — the one phase that wants the V100), and therefore no
reading in §7 has a value yet.

**How many trainings: four.** `scripts/wilson_regression.sh` trains **one
network per loop shape**, at the size the paper's reported MSE comes from —
`WR_SIZES=best`, which is Table V's largest architecture for that loop
(`W^(1×1)` has only one, `L-CB(1,1,1)` at 12 parameters) — and one seed.

The three sizes are **a sweep, not a result**. The Letter lists small / medium /
large per loop because it explores models "of various sizes" and reports the
best; the MSE printed in a Fig. 3 panel is the *minimum* over them, not three
numbers. Likewise the ten seeds: SM §VI.A trains each architecture ten times
with random initialisation and reports the best, which is why the Letter's own
1+1D count is `(1 + 3 + 3 + 3) × 10 = 100` L-CNN models.

    WR_SEEDS=10 WR_SIZES=small,medium,large bash scripts/wilson_regression.sh

reaches for that (100 runs) and its only effect on the headline is to make a bad
initialisation less likely to be the number quoted. **What the four-run default
buys, and what it costs**: `W^(1×1)` is unaffected — the target is exactly
representable in the 12-parameter network (§7 R4), so the result is a float32
floor with no seed dependence. `W^(4×4)` is where a single draw can miss, since
four L-CB layers at `init_w = 1` are a degree-16 matrix polynomial and a bad
start is a real failure mode. If a panel misses badly, the first thing to try is
more seeds at that loop only — `WR_SEEDS=5 WR_TARGETS=W44` — not a
hyper-parameter change, which would stop being a reproduction.

The figure selects on **validation** loss, never on the test MSE, and prints the
median and spread over whatever seeds exist next to the best. At one seed those
three columns coincide, which is the honest way for the table to say that no
best-of-N was taken.

Cost: tens to tens of thousands of parameters on a 64-site lattice. Only
`W^(4×4)` large (39 905 parameters, four L-CB layers, up to 100 epochs) is worth
a GPU; the Letter's own figure for it is ~36 s/epoch.

**Not attempted**, and deliberately: **the baseline CNN half of Fig. 3** (their
2 680 models over 264 architectures and four activation functions — §1), the
3+1D half of the Letter (Table VI, Fig. 5 — topological charge on 4 × 8³ under
Wilson flow), and the adversarial-attack study of Fig. 4.
