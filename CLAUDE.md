# GELT — Gauge-Equivariant Neural Networks for Lattice Gauge Theory

Master's thesis codebase. A **gauge-equivariant lattice transformer** for
SU(N_c) lattice gauge theory, using 2D/3D Z₂ as a debug-friendly testbed and
4D SU(2) for the physics results.

The architecture follows the L-CNN framework (Favoni et al. 2012.12901) for
primitives and gauge-equivariance proofs, with two departures: (i) the
L-Conv + L-Bilin stack is replaced by an attention block whose **value path is
matrix-bilinear** (`α · Q† · Ṽ`), so L-CNN's loop-doubling universality argument
transfers directly; (ii) parallel transport between sites is **averaged over all
shortest lattice paths** in the L1-ball of Manhattan radius R (a DP recursion,
not enumeration), so each block already reaches the full L1-ball receptive field
with non-axis-aligned loop content.

## Scope — read this first

**2026-09-09: the repo was pruned to what reproduces `../report/main.tex`**, plus
two audits of that content (the SU(2) fair fight, the operator decomposition).
`README.md` is the reproduction guide — which script produces which table and
figure — and the entry point for orientation.

Deleted, not archived (history at commit `cfa0a7e`): the dual-Ising ground truth
and the ν-exponent fits (`gelt/ising.py`, `scripts/dual_ground_truth.py`,
`notes/dual_ground_truth.md`), the topological-localization study
(`gelt/topology.py`, `scripts/topology_attention.py`, `scripts/check_cooling.py`),
the ℓ_att attention-range readouts (`scripts/z2_attention_readout.py`,
`scripts/beta_scan.py`, `scripts/visualize_glueball_attention.py`), the rotation-
irrep study, the Z₂ fair fight, `scripts/compare_transport_modes.py`, and the
`blocks_bias` block variant. When a comment in the surviving code says "retired"
or "removed in the 2026-09-09 cleanup", that is where it went.

## Documents

- `README.md` — **the reproduction guide**: main.tex section → script → artifact.
- `PLANS.md` — future directions. Proposals only; nothing there is implemented.
- `notes/glueball_spectroscopy.md` — the run-by-run record of the SU(2) 0⁺⁺
  program: the anisotropy pivot, the multi-level smeared inputs, Runs 4–5, the
  §6.2 audit, the presentation layer, the replication.
- `notes/attention_as_operator.md` — the design record for "the attention map is
  a lattice operator": why ℓ_att failed and the correlator of the attention field
  does not, the three arms, the Z₂ result (§6.1) and its transport to SU(2) (§9).
  §8 records why the scan has four β and not five.
- `notes/operator_decomposition.md` — `O_GELT = P + r` against the classical
  span: what the network found, and the two controls the study does not have.
- `notes/audit_2026-09-06.md` — the repo audit: the **Z₂ APE smearing defect**
  (§2, live caveat), the fair-fight results and the corrections they forced
  (§6.4/§6.5), the ranked plan (§4, re-ranked after the cleanup).
- `notes/performance_audit.md` — the source of truth for anything touching the
  GELT hot path: the four measured optimisations that landed, what was rejected,
  and the ranked backlog.
- `notes/papers_review.md` — literature review (L-CNN, Nagai–Tomiya, CASK).
  Sections 0 (lattice primer) and 1 (L-CNN) are the architecture prerequisites.
- `notes/resources.md` — textbooks and lecture notes, with a reading order.
- `reports/` — the LaTeX write-ups and their PDFs: `paper/` (the full draft),
  `glueball/` (spectroscopy), `attention/` (attention as operator). They pull
  figures from `results/` via `\graphicspath`.

## Status

**Wilson-loop regression** (main.tex §Validation). GELT recovers per-site Wilson
loops a CNN with orders of magnitude more parameters cannot. `train_gelt.py` and
`train_cnn.py` are deliberately the same problem (D=3, L=8, Z₂, Haar links, 1×2
loop, same splits); change one and change the other.

**0⁺⁺ glueball spectroscopy in 4D SU(2)** (main.tex §SU(2) Glueball
spectroscopy). GELT trained as a variational operator on the Rayleigh loss
`−C(1)/C(0)`, per-timeslice 3D, on an anisotropic lattice (L=12, Lt=24, β=2.4,
ξ=3.0). Against the classical multi-level GEVP on the same smeared basis and the
same held-out configurations: same mass, more ground-state weight —
**ΔA₀ = +0.078 ± 0.022 (3.6σ)** over two independent ensembles, m_eff flat from
Δ=1, and the enlarged GEVP+GELT basis collapses onto pure GELT. Two enablers,
both hard-won: **anisotropy** (the isotropic lattice cannot resolve the 0⁺⁺ at
all) and **multi-level smeared input channels** (with thin plaquettes the network
only relearns APE×2's staple content).

**Attention as a lattice operator** (main.tex §Attention as a physical field).
The attention score is gauge invariant, so any reduction of the attention map is
a local scalar lattice operator and its connected correlator has a mass. Measured
on 3D Z₂ at four β approaching β_c ≈ 0.7614 (ξ = 2 … 6) and on anisotropic SU(2)
at β = 2.4: ξ_A tracks the classical correlation length, and **training raises
the attention field's ground-state overlap** far above both the random-init
network and the classical GEVP (Z₂ ΔA₀ = +0.110 … +0.267; SU(2) +0.286 ± 0.056).
The random-init arm tracks ξ too — so the *structural* claim (equivariant
attention maps are lattice operators with a mass) is established by ξ_A, while
the *learning* claim rests on ΔA₀. "The network discovers ξ" is **not**
supportable.

**The two audits.** (i) *Is the classical comparator a straw man?* Against the
input-matched strengthened arm (`deep`), the spectroscopy claim survives:
ΔA₀ = +0.097 ± 0.021 (4.6σ) combined. Two other arms (`full`, `shapes_sm`) are
numerically unusable — singular `C(t0)` on both ensembles — so "does real
loop-shape variety close the gap?" is **open**, not closed
(`notes/audit_2026-09-06.md` §6.5). (ii) *What did the network find?* 12.9%/11.7%
of `O_GELT`'s norm² lies outside the span of the whole classical basis on two
ensembles, and removing it costs the entire advantage — ΔA₀ = +0.076 ± 0.019
(4.0σ) at unchanged mass. Not a contact term. `r` alone is a poor operator
(A₀ = 0.43) that wins by constructive interference.

## Layout

Library in `gelt/`, entry points in `scripts/`, pytest in `tests/`. Installed
editable via `pyproject.toml`. Device order: cuda → mps → cpu.

### `gelt/`

- **`lattice.py`** — `GaugeGroup` ABC with `Z2` and `SU(N)`; pure tensor
  functions. `SU.project` (nearest group element) takes a closed-form route at
  `nc == 2` (`_polar_factor_2x2`), the same map with no `linalg` kernels (~23×).
  - `random_links(L, D, group, dtype, Lt=None)` → `(D, *Λ, nc, nc)`. `Lt` gives a
    non-cubic `Λ = (Lt,) + (L,)*(D-1)` lattice (time = axis 0) for anisotropy.
  - `plaquette_tensor(U, group)` → `(D(D-1)/2, *Λ, nc, nc)`.
  - `action(U, group, beta, plaquettes=None, xi=1.0, time_axis=0)` → Wilson
    action `β Σ_p (1 − Re Tr P / nc)`. **Anisotropic** when `xi ≠ 1`: temporal
    plaquettes weighted `β_t = β·ξ`, spatial `β_s = β/ξ` (tree-level); `xi = 1` is
    the bit-exact isotropic path.
  - `topological_charge_density` / `topological_charge` — naive (plaquette)
    `q_x`, clover-free, `F_{μν} = (P − P†)/2i`; and its lattice sum.
  - `rectangular_wilson_loop(U, group, R, T, mu, nu)` → `Re Tr W/nc` per site.
  - `link_gauge_transformation` / `local_gauge_transformation` — apply site-local
    Ω to links (`U_μ(x) → Ω(x) U_μ(x) Ω†(x+μ̂)`) or to an adjoint field
    (`W(x) → Ω(x) W(x) Ω†(x)`).
  - `l1_ball_offsets(D, R)` → signed Δx with `1 ≤ |Δx|₁ ≤ R`, ordered by `|Δx|₁`.
  - `build_transport_average(U, R, group, mode="average")` — DP routine
    materialising `T_Δx(x)` over the full signed L1-ball. Expects batched links
    `(N, D, *Λ, nc, nc)`. `"average"` is shortest-path-averaged (the default and
    the architecture's choice); `"single"` is one canonical path.
- **`sampler.py`** — `staple_sum`, `metropolis_sweep` (checkerboard-vectorised;
  proposal routed by `_PROPOSAL_FN[type(group)]`), `mcmc_ensemble` (thermalise +
  decorrelate + collect, sweep routed by `_SWEEP_FN`; **Metropolis is the
  registry default for both groups**), `haar_ensemble`.
  - `staple_sum(..., batched=True)` accepts a leading configuration axis; it is
    bit-identical to looping and is what lets `ape_smear` vectorise. The MCMC
    sweeps are one sequential chain and keep the unbatched form.
  - **Anisotropy:** `staple_sum` and all sweeps take `xi=1.0, time_axis=0`; the
    per-plane coupling ratio ξ^±1 is folded into the staple, so `xi = 1` is
    bit-exact backward-compatible. Opt in by binding ξ into the sweep, e.g.
    `functools.partial(heatbath_overrelaxation_sweep, n_or=4, xi=ξ)`.
  - **SU(2) heat-bath + overrelaxation** (`heatbath_sweep`,
    `overrelaxation_sweep`, `heatbath_overrelaxation_sweep`) — exact,
    tuning-free, beats Metropolis critical slowing; the prerequisite for
    resolvable spectroscopy. **Opt-in, not the registry default**: pass it as
    `sweep_fn=` to `mcmc_ensemble`. Heat-bath factors the staple `A = k·V` and
    draws the scalar part via Creutz (`_sample_su2_w0`, `max_iter=1000` — the 100
    of the original cost the ens2 replication run); overrelaxation reflects
    `U' = V†·U†·V†` and re-projects with the closed-form `_project_su2`.
    SU(2) only — SU(N≥3) needs Cabibbo–Marinari.
  - **`z2_heatbath_sweep`** — exact Z₂ heat-bath, `P(U=+1) = σ(2βs)`. Near β_c
    the Metropolis flip proposal's acceptance collapses to 0.02; this does not.
  - **`integrated_autocorrelation_time(series, c=6, max_lag=None)`** → `(rho,
    tau_int, window)`, Madras–Sokal windowing. Samples `n_skip ≳ 2·τ_int` apart
    are effectively independent.
- **`data.py`** — `build_plaquette_datasets(N, D, L, group, target, ...)`.
  `target` is a callable `target(configs, group) -> Tensor` (bind extras with
  `functools.partial`). `structured=True` keeps the `(N, n_pairs, *Λ, nc, nc)`
  matrix layout for GELT; `structured=False` calls `flatten_color` for the CNN
  baseline. With `R` set, the transport is precomputed per config and the splits
  yield `(X, T, y)` triples. `save=True` writes to `datasets/`.
- **`blocks.py`** — the one GELT block. `GEMHSA`: augment (append daggers +
  identity) → fused Q/Q_v/K/V projections → adjoint transport of K, V via
  `T`/`T_dag` → gauge-invariant score `Re Tr[Q† K̃]` + RoPE → softmax →
  multiplicative value `Σ α · Q_v† · Ṽ` → channel mix → residual + L-Act gate.
  Also `ChannelLift`, `Trace`, `MLP`, `GELT` (ChannelLift → stacked GEMHSA →
  Trace → per-site MLP → spatial reduction). `GELT.forward(W, T)` computes
  `T_dag` once and prepends the Δx = 0 transport slot once for the whole stack.
  `_last_score` / `_last_alpha` are stashed per layer under `no_grad` for the
  interpretability studies — **opt-in**, via
  `GELT.set_introspection(store_attention=True)`, along with the `_last_*_norm`
  scalars under `diagnostics=True`.
- **`glueball.py`** — classical 0⁺⁺ spectroscopy. Time is lattice axis 0.
  - `ape_smear(U, group, alpha=0.5, n_steps=1, directions=None)` — spatial-only
    APE smearing (time links untouched, so the transfer-matrix interpretation
    holds). Vectorised over the configuration batch, sliced by a `chunk_bytes`
    budget; bit-exact under any chunking.
  - `glueball_operator` → `(B, *Λ)` sum of spatial-plane R×T Wilson loops;
    `zero_momentum` → `(B, Nt)`; `connected_correlator` → vacuum-subtracted
    `C(Δ)` (0⁺⁺ has a nonzero VEV — the subtraction is essential);
    `effective_mass`; `jackknife_effective_mass`.
  - **Multi-level GEVP**: `smearing_operator_basis`, `connected_correlator_matrix`,
    `gevp_eigenvalues` (eigh-whitening with an eigenvalue floor, not Cholesky —
    low statistics can make `C(t0)` indefinite), `gevp_effective_mass`,
    `jackknife_gevp_effective_mass`.
  - **Fit / overlap layer**: `gevp_ground_vector(C, t0, td)` → the ground-state
    generalized eigenvector, defining the *projected operator* — the optimal
    single operator in a basis's span, i.e. the apples-to-apples comparator for a
    single learned operator; `fit_cosh_correlator(C, dmin, dmax, sigma)` →
    `(m, A, χ²)` for `A·[e^(−mΔ) + e^(−m(Nt−Δ))]` (profiled-A grid scan +
    parabolic refine — dependency-free, safe inside every jackknife sample). The
    ground-state overlap fraction is `A₀ = A·(1+e^(−m·Nt))/C(0)`.
- **`lcnn.py`** — Favoni et al. L-CNN: `build_axis_transports` (axis-aligned link
  products — distinct from GELT's L1-ball `T`), `LConv`, `LBilin`, `LCB`,
  `LAct`, `Trace`, `LCNN`. Mirrors `GELT`'s I/O so the two are
  matched-parameter comparable. The matched shootout has **not** been run.
- **`cnn_baseline.py`** — `LatticeCNN`: non-equivariant baseline; `Conv2d`/
  `Conv3d` for D=2/3 and a roll-based `_RollConvND` for D≥4.

### `scripts/`

Flat and self-contained: each defines its own `evaluate` / `train_model` inline
(there is no shared `gelt/train.py`). Cross-imports go through
`sys.path.insert(0, dirname(__file__))`, which is why they are not in
subdirectories. `README.md` has the one-line table; the details that matter:

- **`train_glueball.py`** — the variational operator. `GELT(reduction="none")` on
  the Rayleigh loss, **per-timeslice 3D** (any temporal receptive field voids the
  transfer-matrix bound and makes the loss gameable toward m → 0),
  `mlp_zero_init=False` (zero init ⇒ exactly zero Rayleigh gradient — training
  never starts), `INPUT_SMEAR_LEVELS = (0, 2, 4, 6)`, `SCALE_REG` pinning the
  `Ō → λŌ` flat direction. Env overrides: `GLUEBALL_ENSEMBLE_SEED`,
  `GLUEBALL_INIT_SEED`, `GLUEBALL_RESUME`, `GLUEBALL_EVAL_ONLY`; non-default
  seeds get a `RUN_TAG` suffix so Run-5 artifacts are never overwritten. Dumps
  the test-split Ō arrays to `datasets/…_test_obars.pt` (also kept in `dumps/`).
- **`z2_attention_correlator.py`** — the Z₂ table. GEVP t0=1/td=2, window
  Δ∈[2,8], cosh + A₀, blocked jackknife block 20, config-scramble null,
  time-shuffle zero-mode check, correlated ΔA₀. `ZAC_REPLOT=<dump.pt>` regenerates
  every figure and table offline (no GPU, no ensembles, no checkpoints).
- **`su2_attention_correlator.py`** — the same measurement on anisotropic SU(2);
  **imports the estimator layer from `z2_attention_correlator.py`**, so identical
  conventions is a fact about the call graph. Defaults to one row at β = 2.4 (the
  only trained coupling ⇒ a true diagonal cell); `SAC_BETAS` runs a scan. Refuses
  to print aggregates (Pearson, slope, dynamic range) below three couplings
  rather than emitting NaN. **Does not run on MPS** (SU(2) projection needs
  complex `linalg.det`/`svd`) — use `SAC_DEVICE=cpu` for a smoke test.
- **`su2_fair_fight.py`** — the strengthened classical arms (`published` /
  `deep` / `shapes` / `full`), gated on reproducing the published numbers first.
  `SFF_NOCACHE=1` runs the dump-only path offline.
- **`operator_decomposition.py`** — `O_GELT = P + r` in the exact Hilbert-space
  metric `C_ab(0)`, offline from the `dumps/` Ō arrays. Prints the published
  comparison first as a gate.
- **`profile_glueball_step.py`** — where one optimizer step goes, per stage,
  forward **and backward** separately. It goes through
  `train_glueball.config_inputs`, i.e. the pipeline the training loop actually
  runs; an earlier version profiled a different one and produced the false
  "transport is 1.9%" figure. Env: `PROFILE_DIAGNOSTICS`, `PROFILE_LEGACY_SMEAR`,
  `PROFILE_MICRO`, `PROFILE_COMPILE`.

### `tests/`

- **`test_lattice.py`** — gauge invariance of `plaquette_tensor` / `action`
  (bit-exact in Z₂ float64), anisotropic-action invariance, the `xi = 1` match,
  non-cubic shapes, and `SU(2).project`'s closed-form route against the SVD polar
  it replaced (far-from-group and near-group input; SU(3) still takes the general
  route).
- **`test_transport.py`** — `l1_ball_offsets` counts, brute-force per-octant
  pattern, octant-relation consistency, gauge covariance for Z₂ and `nc = 2`.
- **`test_blocks.py`** — gauge equivariance of `GEMHSA`/`GELT` (SU(2) complex128,
  both gates; Z₂ float64), the optimised attention path against a naive oracle
  (outputs, `_last_score`, `_last_alpha` and input gradients to 1e-12 for
  SU(2)/SU(3)/Z₂), `_OffsetGather`'s hand-written backward against autograd's,
  the Δx = 0 prepend, introspection off by default, gradient checkpointing, plus
  the four cases ported from the retired `blocks_bias` suite.
- **`test_sampler.py`** — overrelaxation conserves the Wilson action to machine
  precision and stays on the group; heat-bath reproduces the exact 2D SU(2) mean
  plaquette `I₂(β)/I₁(β)`; anisotropic versions of both.
- **`test_glueball.py`** — APE smearing is gauge covariant and stays on the group;
  the operator is gauge invariant; correlator / `m_eff` / jackknife recover a
  known mass from a synthetic correlator; the GEVP recovers both masses of a
  synthetic two-state matrix; `gevp_ground_vector` kills the excited state
  exactly; `fit_cosh_correlator` recovers `(m, A)`.
- **`test_data_model.py`** — split validation and CNN-baseline shape guards.

## Conventions

- **Tensor layouts** (the only spec, no OO wrappers): links
  `(D, L, …, L, nc, nc)` — direction first, spatial axes, color last;
  plaquettes `(n_pairs, L, …, L, nc, nc)` with `n_pairs = D(D-1)/2`, ordered by
  `(μ, ν)`, μ < ν lexicographically.
- **Color axes are always present**, even for Z₂ where `nc = 1`. Every product is
  `A @ B` and every inverse `group.dagger(A)`, so the code ports verbatim to
  U(1)/SU(N). For Z₂ both are no-ops; any laxity is a bug Z₂ cannot catch.
- **Plaquette convention:** `P_{μν}(x) = U_μ(x) U_ν(x+μ̂) U_μ†(x+ν̂) U_ν†(x)`.
- **Periodic BCs:** `torch.roll`. Never manual modulo arithmetic on indices.
- **Wilson action:** `S = β Σ_p (1 − Re Tr P / nc)`.
- **Parallel transport:** sum over **all** shortest lattice paths in the L1-ball
  — never a single axis-aligned path (unless `mode="single"`). The octant
  identity `T_{−Δx}(x) = dagger(T_Δx(x − Δx))` holds as a math property and is a
  test-suite check; it is **not** relied on at build time (one auditable DP
  surface beats a 2× memory saving, and mixed-sign offsets cannot be derived from
  positive-octant data anyway).
- **Float32** for training; `dtype=torch.float64` (→ `complex128`) through the
  dataset builders for high-precision gauge-invariance tests.
- **Time is lattice axis 0** throughout the spectroscopy code.

## Performance

Full record and the algebra behind each number: `notes/performance_audit.md`.
The ens1 replication logged **7.77 s/step, 1942 s/epoch** on a 32 GB V100 for a
~5k-parameter model. Four exactly-equivalent changes landed, each measured and
each covered by a test:

- **`SU(2).project` in closed form** (`gelt.lattice._polar_factor_2x2`). The APE
  ladder was issuing ~4.5 M batched 2×2 complex SVDs *per step*. `H = M†M` is 2×2
  Hermitian PSD, so `√H = (H + √det H·𝟙)/√(tr H + 2√det H)`: two matmuls plus real
  scalar arithmetic, agreeing with the SVD route to 3.9e-7 on 248 832 real
  production matrices with a *better* unitarity residual. `H` is formed in
  float64; MPS (no float64) stays in complex64 — where it now runs at all.
- **`ape_smear` vectorised over the configuration batch** (`staple_sum` gained
  `batched=True`): **7.4×** on real configs (5.85 s → 0.79 s per step), bit-exact
  under any chunking.
- **Introspection stashes are opt-in.** The two K̃/Ṽ norms alone re-read ~12% of
  the block's memory traffic, and the ~10 `.item()` calls per block are ~10 GPU
  syncs — all of it running again inside every gradient-checkpoint recompute.
- **The GEMHSA hot path**: K and V share one gather; the Δx = 0 transport slot is
  prepended once in `GELT.attn`; the RoPE rotation is folded into the *query*
  (`GEMHSA.rope_score` — Q carries no offset axis, so n_offsets times less data);
  and the neighbour gather's backward is written as a gather (`_OffsetGather`)
  instead of the atomic scatter-add autograd emits, which measured **64× its own
  forward**. Per-layer materialised bytes 43.7 → 29.8 GiB, non-view kernels
  76 → 44, `.item()` syncs ~10 → 0. **2.16×** on forward + backward, outputs
  identical to 1.2e-7.

**No total speedup is claimed** — every timing behind these numbers is CPU. Run
`profile_glueball_step.py` on the V100 before spending effort on the next tier;
the ranked candidates (adjoint SO(3) transport, Q/K/V layout, unmaterialised
contractions, offset chunking, the schedule, `torch.compile`) are §5 of the note.

## Known caveats

1. **Z₂ APE smearing is broken at the production `SMEAR_ALPHA = 0.5`**
   (`notes/audit_2026-09-06.md` §2). With two spatial staples the update is
   `V = (1−α)U + (α/2)(s₁+s₂)`: the exact identity for α < ½, a majority-vote
   automaton for α > ½, and at α = ½ exactly `V = 0` whenever the staples disagree
   with the link, where `Z2.project` sends `0 → +1` — a value that does not
   transform. Measured on production configs: the ladder **freezes after one
   step**, 99.9% of changed links are gauge-dependent tie-breaks, Ō(t) moves 1.82σ
   under a gauge transformation, 0.67% of the network's smeared input plaquettes
   flip sign. So the Z₂ classical basis is one operator, not four (the dumps
   already say `gevp_fell_back=True, n_ops=1`), and the Z₂ nets were trained on
   four channels of which three are byte-identical. **SU(2) is verified clean**
   (1.4e-15). `tests/test_glueball.py` misses it because its covariance cases use
   α = 0.6/0.7, never 0.5. Fix: α = 0.7, or unprojected fat links.
2. **RoPE axis coverage:** `pair_axis = [p % D for p in range(n_pairs)]` rotates
   only `n_pairs` axes, so `d_qkv < 2D` leaves whole axes on the identity
   rotation. Enforce `d_qkv ≥ 2D` (8 in D=4, 6 in D=3).
3. **Datasets do not store β.** Multi-β training (and any conditioning on the
   coupling) needs β to become part of the dataset.
4. **Sampler dispatch is by group type.** `metropolis_sweep` looks up
   `_PROPOSAL_FN[type(group)]`, `mcmc_ensemble` looks up `_SWEEP_FN`; adding
   U(1)/SU(N) is a registry entry plus the proposal/sweep function.
5. **No worst-case-Ω stress test.** `check_gelt_invariance.py` is the quick
   version; the systematic drift-vs-Ω study does not exist.
6. **Offset-chunked attention does not exist** — it is the memory gate on running
   the attention studies at physically large R.

## Running

```bash
python scripts/validate_sampler_z2.py        # Metropolis four-panel check (Z₂)
python scripts/validate_sampler_su2.py       # …and SU(2)
python scripts/validate_anisotropy.py        # anisotropic-lattice checks
python scripts/check_gelt_invariance.py      # 60-second gauge-invariance check
python scripts/train_cnn.py                  # CNN baseline, 1×2 Wilson loop
python scripts/train_gelt.py                 # GELT, the same problem
python scripts/train_lcnn.py                 # L-CNN baseline
python scripts/check_glueball_autocorrelation.py   # τ_int → production n_skip
python scripts/measure_glueball.py           # classical 0⁺⁺ baseline + ensemble
python scripts/train_glueball.py             # GELT as a variational operator
python scripts/fit_glueball_overlap.py [dump]      # cosh fits + A₀ (offline)
bash   scripts/overnight_replication.sh      # fresh ensemble + retraining (~24 h)
python scripts/operator_decomposition.py     # O = P + r (offline, seconds)
SFF_NOCACHE=1 python scripts/su2_fair_fight.py     # reproduce ΔA₀ offline
python scripts/z2_beta_scan.py               # Z₂ classical mass vs β
Z2G_R=6 Z2G_N_USE=800 python scripts/train_z2_glueball.py 0.756
python scripts/z2_attention_correlator.py    # the Z₂ attention table
python scripts/su2_attention_correlator.py   # the SU(2) row (~40 min)
PROFILE_DIAGNOSTICS=1 python scripts/profile_glueball_step.py
pytest tests
```

`.venv/` is local (uv-style, not gitignored). `datasets/`, `results/`, `*.pth`,
`*.png` are gitignored.

## Suggested next steps

Ranked in `notes/audit_2026-09-06.md` §4, and unchanged by the cleanup:

1. **The random-init control for the operator decomposition** — one GPU eval
   pass, no training; it is what separates *learned* from *architectural*.
2. **Fix the Z₂ smearing** (caveat 1) and add the α = 0.5 covariance case to
   `tests/test_glueball.py`. Changing the Z₂ inputs needs a retrain for a clean
   end-to-end statement; evaluating existing checkpoints on covariant inputs is
   the cheap robustness check first.
3. **Prune near-degenerate operators before the GEVP** so the `full`/`shapes_sm`
   arms become readable and "does loop-shape variety close the gap?" gets an
   answer. Not a bigger `GEVP_EPS` floor.
4. **Dump per-config Ō from `z2_attention_correlator.py`** so the decomposition
   transports to the attention field without a GPU re-run.
5. **The matched-parameter L-CNN shootout** on the per-timeslice glueball task —
   the one baseline the thesis names and has never run.

## Things to keep in mind

- **Do not silently broadcast across color axes.** A missed dagger or a wrong
  shortest-path step passes every Z₂ test and fails at the first non-abelian Ω.
- **Do not remove comments unless the code they describe is being deleted.**
- The two attention scripts share one estimator by import, and the two
  Wilson-loop scripts share one problem by construction. Both pairs must be
  changed together or the comparison silently stops being one.
