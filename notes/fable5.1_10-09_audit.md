# Plan — the input↔architecture curve (audit of 2026-09-10, Fable 5.1)

Written 2026-09-10 after reading every note in `notes/`, `../report/main.tex`,
`README.md`, `CLAUDE.md`, `PLANS.md`, the scripts and the local `results/` and
`logs/`. It is the **execution plan for the next block of thesis work**, written
so that an agent who has `CLAUDE.md` but not the conversation that produced this
can carry it out. §9 keeps the first-pass ranking of options that this plan
supersedes, so the reasoning trail is not lost.

Compute rule, from the user's standing instruction: **everything with a progress
bar runs on the V100.** Locally only seconds-long offline work on downloaded
`.pt` files (fits, plots, table generation, `pytest`). Push a script, let the
user run it; `scripts/overnight_replication.sh` is the batch-driver pattern.

---

## 0. Read this first — the pre-registration discipline

§1 and §2 of this note are a **pre-registration**. They fix the observable, the
estimator, the fit conventions, the grid of points, and the three readings with
their operational tests, *before* the new numbers exist. The value of the curve
over a single ΔA₀ is precisely that it removes the question "did you choose the
opponent after seeing the numbers?", and that value is lost if this note is
edited after the fact. Therefore:

1. Fill §0.1 (the run currently in flight) **first**, from the V100, without
   reading that run's results.
2. Commit this note (`git add notes/fable5.1_10-09_audit.md && git commit`)
   **before** reading any number produced by WP1–WP6.
3. Results go into a new §8 written *after* the fact, dated, and §1–§2 are not
   modified — if a convention has to change mid-way, say so in §8 with the
   reason, and keep the original text.

### 0.1 The run in flight — FILLED 2026-09-10 (the run had already finished)

A training was running on the V100 when this plan was written; which one is not
recorded anywhere. Identify it and write it here before anything else:

```bash
nvidia-smi
ps -ef | grep -E "train_|correlator|fair_fight|measure_glueball" | grep -v grep
ls -lt logs/ | head
head -30 logs/<newest>.log        # the config banner: seeds, levels, d_model
```

    script         : scripts/train_glueball.py, then scripts/su2_fair_fight.py on both 7-level dumps
    argv / env     : GLUEBALL_ENSEMBLE_SEED=1  --input-smear-levels=0,2,4,6,8,12,16  --d-model=24
    started        : not recorded locally (finished before 2026-09-09 17:15, commit cfa0a7e)
    log            : V100 only; no copy in logs/ on the laptop
    grid point (§2): 7lv, ensemble ens1 (seed 1) — the 7lv trained point that §2.1 lists as existing
    width d_model  : 24

**It was not in flight.** Identified by the user on 2026-09-10: the run was the
ens1 retraining on the deep ladder, followed by the SU(2) fair fight of both
7-level nets against every classical arm. It had finished, and its results were
already recorded in `notes/audit_2026-09-06.md` §6.5 (2026-09-09) — the same
numbers this plan quotes as WP1's gate. So §0's "do not read it before the
commit" could not apply: they were known when §1–§2 were written, and this
point carries the reactive-provenance flag of §1.5 on both ensembles.

Its final numbers, from §6.5 (floor estimator, `GEVP_EPS = 1e-4`, window
Δ ∈ [2, 7], jackknife block 10, 400 test configs per ensemble):

    arm          n_ops   cond C(t0) (run5 / ens1)   gate (run5 / ens1)      ΔA₀ = GELT(7lv) − arm
    published        4   2.4e3  / 2.4e3             ok / ok                 combined +0.131 ± 0.029 (4.5σ)
    deep             7   1.9e6  / 7.0e6             ok / ok                 +0.052 ± 0.033 / +0.127 ± 0.027, combined +0.097 ± 0.021 (4.6σ)
    shapes           5   85     / 87                ok / FELL BACK (m. 1)   —
    shapes_sm       20   5.6e10 / 2.8e9             ok / FELL BACK (m. 19)  unusable (different estimator on each ensemble)
    full            21   inf    / inf               ok / ok                 +0.013 ± 0.029 / +0.105 ± 0.056 — unusable (singular C(t0))

§6.5 recorded only the differences. The per-arm A₀, printed from the V100
`results/fair_fight/su2_fair_fight.pt` on 2026-09-10 (same run, same estimator):

    arm          A₀ (run5)      A₀ (ens1)
    gelt (7lv)   0.955(57)      1.066(59)
    published    0.837(56)      0.925(71)
    deep         0.903(62)      0.939(66)
    shapes       0.500(110)     0.612(110)
    shapes_sm    0.934(67)      1.003(56)   ← ens1 is the single-member fallback
    full         0.942(54)      0.960(62)

Best validation Rayleigh loss (from the dump `meta`): −0.635 (run5), −0.580
(ens1). The two dumps are now tracked in `dumps/` (WP0 step 2).

If it is one of the grid points of §2, it becomes that point; if it is something
else, it is not part of the curve and is reported separately.

---

## 1. The claim, as a curve

### 1.1 Definition

The spectroscopy chapter currently rests on one number, ΔA₀ = +0.078 ± 0.022
against the 4-level GEVP, restated by the fair fight as +0.097 ± 0.021 against
the input-matched `deep` arm (`notes/audit_2026-09-06.md` §6.5). The plan
replaces the single number by a **function**:

    A₀(x)  for  x ∈ { thin, 4lv, 7lv, 4lv+5sh, 7lv+3sh }

where x is the *input content* handed to both methods, ordered by inclusion
along the main chain thin ⊂ 4lv ⊂ 7lv ⊂ 7lv+3sh (with 4lv+5sh an off-chain
point, since it is neither a subset nor a superset of 7lv), and three traces:

| trace | what it is | where the number comes from |
|---|---|---|
| **classical** GEVP(x) | the v₀-projected GEVP over the classical basis with content x | `su2_fair_fight.py` arm of that name, pruned/truncated estimator (§1.4) |
| **trained** GELT(x) | a GELT trained on input channels with content x | `train_glueball.py` dump for that ladder, fitted by the same fair-fight run |
| **random** GELT(x) | the same architecture, untrained, same inputs — the architecture-only baseline | eval-only dumps, three init seeds, same fair-fight run |

Every point is measured on the **same 400 held-out test configurations** of an
ensemble, and every difference between two traces at the same x, or between
GELT(x_k) and GEVP(x_{k+1}), is a **correlated blocked-jackknife difference** on
shared configurations. Two ensembles (run5 = seed 0, ens1 = seed 1), combined
inverse-variance as §6.2 already does.

### 1.2 The three readings, with their operational tests (pre-registered)

Let ΔA₀(x) = A₀_GELT(x) − A₀_GEVP(x), correlated, combined over ensembles, and
let a point be **resolvable** if A₀_GEVP(x) < 0.90 on the combined value
(saturation clause, §1.3). "Within errors" means |difference| < 2σ.

- **R1 — the architecture buys input depth.** ΔA₀(x) > 2σ at every resolvable
  x, **and** rung-equivalence holds: A₀_GELT(x_k) ≥ A₀_GEVP(x_{k+1}) within
  errors for k = thin→4lv and 4lv→7lv. Statement licensed: "one GELT at k input
  levels is worth at least the classical basis at the next rung; the advantage
  is the architecture, and it persists until the observable saturates."
- **R2 — the architecture is an efficient shortcut, not an extension.**
  ΔA₀(x) consistent with zero (< 2σ) at some **resolvable** x where the GEVP
  still has 1 − A₀ more than 3σ from zero, i.e. the traces meet before
  saturation. Statement licensed: "the learned operator reproduces the optimal
  classical combination from poorer inputs, and adds nothing once the classical
  basis is rich enough."
- **R3 — additive advantage.** ΔA₀(x) consistent with a constant c > 0 over the
  resolvable points (χ²/dof of a constant fit < 2 and c > 3σ). Statement
  licensed: "the learned operator carries a fixed amount of ground-state weight
  the classical span does not, independent of how much input it is given."

R1 and R3 can both hold (constant gap *and* rung-equivalence); that is expected
from the two points that already exist, and is the honest statement if it comes
out that way. R2 excludes both.

**The random trace's own reading.** Expected: A₀_random(x) ≤ A₀_GEVP(x) at every
x. If A₀_random(x) ≥ A₀_GEVP(x) anywhere, the *architecture alone* beats the
optimal linear combination of its own inputs; report it as such and the
"learned" attribution of that x rests on trained − random only.

### 1.3 Saturation clause (pre-registered limitation)

A₀ is bounded by one and every strong arm on this ensemble already reads
0.93–0.96, one fit above one (a fit artifact of a nearly pure operator,
`notes/operator_decomposition.md` §7). At such points the observable cannot
distinguish R1 from R2. Points with combined A₀_GEVP ≥ 0.90 are therefore
**excluded from the R1/R2/R3 tests** and shown on the figure as saturated. The
second panel of the figure is ΔA₀(x) with its error at every x, saturated or
not, so the reader sees the compression directly. Only a harder observable
(an excited state, §3 WP9; a different channel; a coarser ensemble) could
extend the curve past this ceiling, and the note says so.

### 1.4 Fixed conventions (pre-registered)

| item | value | source |
|---|---|---|
| fit window | Δ ∈ [2, 7], cosh, profiled-A grid | `fit_glueball_overlap.py` FIT_WINDOW |
| jackknife | delete-block, block 10 configs, whole fit redone per replica | same |
| GEVP | t0 = 1, td = 2, v₀-projected operator | `gevp_ground_vector` |
| estimator | truncated whitening (eps 1e-4 relative, directions **dropped**, not floored) + collinearity pruning only if needed — selection rule in WP1 | this note |
| test split | contiguous last 20% of each 2000-config ensemble | `train_glueball.py` |
| ensembles | run5 (seed 0) and ens1 (seed 1); combined inverse-variance | §6.2 |
| width policy | every **new** trained point at d_model = 24 (the deep net's width); the 4lv points stay at 16 as trained; one width-control run resolves the confound (WP3) | this note |
| training protocol | the Run-5 protocol as it stands in `train_glueball.py` today: scale pin on, LOSS_DELTAS (1, 2), AdamW, cosine, batch 6, patience 10 | this note |
| random trace | three init seeds (0, 1, 2), reported as mean with the larger of the seed spread and the mean jackknife error | this note |

### 1.5 Provenance, stated exactly

- The classical arms (`ARM_SPEC` in `su2_fair_fight.py`) were named before the
  fair fight ran, and the MAX_LEVEL comment predates it. That pedigree is real.
- The **7-level trained GELT was trained after** the `deep` arm was seen to win
  on run5 (2026-09-08, `notes/audit_2026-09-06.md` §6.4). It is a reactive
  point and the thesis must say so.
- The thin retrain, the width control, the random trace, the pruned estimator
  and the readings above are fixed **by this note, on 2026-09-10, before any of
  their numbers exist**.

---

## 2. The grid — what exists and what is missing

### 2.1 Points

| x | classical arm (`ARM_SPEC`) | n_ops | trained GELT | width | random GELT |
|---|---|---|---|---|---|
| thin | `thin` = ([0], 1×1) — **must be added** to ARM_SPEC | 1 | Run-4 ckpt exists (run5 only, **old loss, pre scale-pin**) → **retrain both ensembles** (WP3) | 24 (new) | eval only (WP2) |
| 4lv | `published` = [0,2,4,6] × 1×1 | 4 | exists, both ensembles | 16 | eval only |
| 7lv | `deep` = [0,2,4,6,8,12,16] × 1×1 | 7 | exists, both ensembles | 24 | eval only |
| 4lv+5sh | `shapes_sm` = [0,2,4,6] × {1×1,1×2,2×2,2×3,3×3} | 20 | none (needs loop input channels, WP6, optional) | — | eval only if WP6 |
| 7lv+3sh | `full` = [0,2,4,6,8,12,16] × {1×1,1×2,2×2} | 21 | none (WP6, optional) | — | eval only if WP6 |

The thin classical point is a single operator with poor overlap; its cosh fit
over [2, 7] may not converge (the thin operator's signal dies by Δ ≈ 3, Run 0/4).
If it does not, report the point as unfittable rather than moving the window.

### 2.2 Files

`results/` is gitignored, and on the V100 it is **root-owned** (jobs run in a
container): jobs write there, users cannot `scp` into it. Small artifacts that
must travel go through the tracked `dumps/` directory (see `dumps/README.md`).

| artifact | where | notes |
|---|---|---|
| `datasets/glueball_configs_L12_Lt24_b2.4_xi3.0_N2000.pt`, `…_seed1.pt` | V100 only | the two ensembles; ~2 GB each |
| `results/glueball/best_glueball_gelt.pth` | V100 | Run 4, thin, d16, old loss — **do not reuse, do not overwrite** (see WP3 naming) |
| `results/glueball/best_glueball_gelt_sm0-2-4-6.pth`, `…_ens1.pth` | V100 + local | Run 5 / ens1, d16 |
| `dumps/best_glueball_gelt_sm0-2-4-6_test_obars.pt`, `…_ens1_…` | tracked | their test-split Ō dumps |
| `results/glueball/best_glueball_gelt_sm0-2-4-6-8-12-16.pth`, `…_ens1.pth` | V100 only | the 7-level nets, **d_model 24 although the name does not say so** |
| `results/glueball/best_glueball_gelt_sm0-2-4-6-8-12-16{,_ens1}_test_obars.pt` | V100 only | their dumps — **copy into `dumps/` and track them** (WP0) |
| `results/fair_fight/su2_fair_fight_obars_{run5,ens1}.pt` | V100 (all arms); local copy has `published` only | per-arm Ō series on the test split: `{"labels", "bases": {arm: (n_ops, 400, 24) float32}, "gelt", "t0", "td"}`. Keyed by **ensemble only**, so the `gelt` inside is whichever dump ran last — take `gelt` from the dumps, never from here |
| `results/fair_fight/su2_fair_fight.pt` | V100 (both ensembles, all arms); local has run5/published | the §6.5 numbers |
| `results/attention/beta_scan.pt` | local + V100 | classical SU(2) masses at 5 β, for WP8's cross-check |

---

## 3. Work packages, in execution order

Each package: goal → files → exact change → commands → runtime → gate → outputs.
Nothing in WP1–WP5 changes a published number's code path without an opt-in
flag, so every existing table can still be regenerated bit-for-bit (modulo the
4e-7 smearing change already recorded in `notes/performance_audit.md` §7.3).

### WP0 — pre-register and secure the inputs (laptop + V100, 1 hour)

1. Fill §0.1. Commit this note.
2. On the V100, copy the two 7-level dumps into `dumps/` (they are ~386 KB
   each) and commit them: `cp results/glueball/best_glueball_gelt_sm0-2-4-6-8-12-16*_test_obars.pt dumps/`.
   Extend `dumps/README.md` with the two lines and the width note.
3. Verify the V100 obars caches hold every arm:
   `python -c "import torch;d=torch.load('results/fair_fight/su2_fair_fight_obars_run5.pt',weights_only=False);print(d['labels'].keys(), {k:tuple(v.shape) for k,v in d['bases'].items()})"`
   — expected keys `published, deep, shapes, shapes_sm, full`. If any is
   missing, WP1 rebuilds the table (it has the ensemble).
4. Copy both obars caches to the laptop (`scp` from `results/fair_fight/`,
   readable) — every offline step below runs on them locally in seconds.

### WP1 — the estimator: make the shape arms readable (laptop, 1 day; V100 offline, minutes)

**Goal.** `full` (cond C(t0) = ∞ on both ensembles) and `shapes_sm` (fell back
to one operator on ens1) are not readable (`audit_2026-09-06.md` §6.5). The
fix the audit asks for: drop near-null directions instead of flooring them,
and prune collinear operators before the GEVP. Not a larger floor.

**Library change — `gelt/glueball.py`.** Add `truncate: bool = False` to
`gevp_eigenvalues` and `gevp_ground_vector`, sharing one helper:

```python
def _whiten(Ct0, eps, truncate):
    s, Q = torch.linalg.eigh(Ct0)                     # ascending
    if truncate:
        keep = s > eps * s[-1]                        # drop null / negative directions
        return Q[:, keep] * s[keep].rsqrt()           # (n_ops, k), k ≤ n_ops
    s = s.clamp_min(eps * s[-1].clamp_min(eps))       # the existing floor path, untouched
    return Q * s.rsqrt()
```

`gevp_ground_vector`: `W = _whiten(C[t0], eps, truncate)`; `M = Wᵀ C[td] W`
(k×k); `v0 = W @ V[:, -1]` is still `(n_ops,)`. `gevp_eigenvalues` with
`truncate=True` returns `(Nt, k)`; document it. Default `False` keeps every
existing caller bit-identical.

**Tests — `tests/test_glueball.py`**, next to `test_gevp_ground_vector_kills_excited_state`:

- `test_gevp_truncate_is_identity_when_well_conditioned`: the existing
  two-state synthetic C; `truncate=True` and `False` give the same v₀ and the
  same eigenvalues to 1e-12.
- `test_gevp_ground_vector_survives_duplicated_operator`: three-operator basis
  with Z[2] = Z[0] exactly (C(t0) singular); with `truncate=True, eps=1e-8` the
  projected correlator's `effective_mass` is flat at the ground mass to 1e-8.
  Assert nothing about the floor path on this input.

**Script change — `scripts/su2_fair_fight.py`.**

1. `ARM_SPEC["thin"] = ([0], ((1, 1),), False)` — the leftmost classical point.
2. `SFF_TRUNCATE=1` → `_project` calls `gevp_ground_vector(..., truncate=True)`.
   `SFF_PRUNE=<rho_max>` (unset = off) → before the GEVP, greedy collinearity
   pruning ported from `z2_attention_correlator._prune` with **no cap**
   (`max_ops = n_ops`): rank by C(td)/C(0), add greedily, skip any operator
   whose |corr| with an already-kept one exceeds `rho_max`. The kept index set
   is chosen **once on the full sample** and held fixed across jackknife
   replicas (as `_jack` does in the Z₂ script). Print `n_kept` per arm.
3. Per-dump outputs: `OUT_PT`/`OUT_PNG` default to
   `results/fair_fight/su2_fair_fight_<dump stem>.pt/.png`; `SFF_OUT=<path>`
   overrides. The obars cache is written only if absent (never overwritten).
4. `SFF_BASES=<obars.pt>`: load `bases`/`labels` from the cache instead of
   re-smearing the ensemble; the slice gate then compares the cached
   `published` basis to the dump's `Obar_basis` (same check, no GPU). With this
   flag the script runs **offline on the laptop in seconds**, which is what the
   curve needs (six dumps × two ensembles × several estimator settings).
5. Nothing else changes: window, block, t0/td, the superset gate, the
   variational gate all stay.

**Estimator selection rule (pre-registered).** Run every arm on **both** dumps of
the 7-level nets under, in this order: (a) floor eps 1e-4 (today's §6.5
numbers — the gate), (b) truncate eps 1e-4, (c) truncate + prune 0.999,
(d) truncate + prune 0.99. Quote the **least invasive setting under which every
arm passes the superset gate and the variational gate on both ensembles**, and
under which `published`'s A₀ is unchanged to the third decimal. That setting is
then used for every point of the curve. The choice is made by this rule, not by
the A₀ it yields; report the full (a)–(d) table in §8 regardless.

**Gate.** Under the chosen setting `published` reproduces +0.066/+0.089 and
`deep` reproduces +0.052/+0.127 (§6.5) — otherwise the estimator changed
something it should not have, stop.

**Outputs.** `results/fair_fight/su2_fair_fight_<stem>.pt` for the two 7-level
dumps; the (a)–(d) table; a one-paragraph §8.1 with the now-readable verdict on
`full` and `shapes_sm`, i.e. the answer to "does loop-shape variety close the
gap?" (`CLAUDE.md` Status, audit §6.5). The pre-registered expectation from
the two unstable numbers (+0.013 ± 0.029, +0.105 ± 0.056, both positive):
parity within a few hundredths or a 1–2σ GELT edge.

### WP2 — the random trace (V100, ~2 h of evals; laptop, minutes)

**Goal.** The architecture-only baseline at every x, and control 1 of
`notes/operator_decomposition.md` §5 (is the out-of-span content learned or
architectural?).

**Script change — `scripts/train_glueball.py`.**

1. `RANDOM_INIT = _env_flag("GLUEBALL_RANDOM_INIT", False)`. When set: force
   `RESUME = False` and `EVAL_ONLY = True`; skip the `if not os.path.exists(CHECKPOINT): return`
   guard and the final `load_state_dict`; save the *untrained* `state_dict` to
   `CHECKPOINT` so checkpoint and dump stay paired. The init is already seeded by
   `torch.manual_seed(INIT_SEED)` immediately before `GELT(...)`.
2. Tags. Extend the stem so nothing collides:
   `best_glueball_gelt` + (`_sm<levels>` unless thin) + (`_d<D_MODEL>` if D_MODEL ≠ 16)
   + (`_ens<k>` if ENSEMBLE_SEED ≠ 0) + (`_rnd<k>` if RANDOM_INIT else `_init<k>` if INIT_SEED ≠ 0)
   + `GLUEBALL_RUN_TAG` (free-form, must start with `_`, default empty).
   The `_ens1` piece must stay in the basename — `su2_fair_fight._cache_for` and
   `_tag` key the ensemble off it. The two existing 7-level checkpoints predate
   the `_d24` suffix; list them in §2.2 as they are and do not rename them.
3. Write `d_model` and `random_init` into the dump `meta`.

**Runs** (all on the V100, each ≈ 5 min: smearing the 400 test configs plus one
forward pass; batch them in `scripts/curve_batch.sh`, WP4):

```
for ENS in 0 1; do for SEED in 0 1 2; do
  GLUEBALL_RANDOM_INIT=1 GLUEBALL_ENSEMBLE_SEED=$ENS GLUEBALL_INIT_SEED=$SEED GLUEBALL_INPUT_SMEAR_LEVELS=0                  GLUEBALL_D_MODEL=24 python -u scripts/train_glueball.py
  GLUEBALL_RANDOM_INIT=1 GLUEBALL_ENSEMBLE_SEED=$ENS GLUEBALL_INIT_SEED=$SEED GLUEBALL_INPUT_SMEAR_LEVELS=0,2,4,6            GLUEBALL_D_MODEL=16 python -u scripts/train_glueball.py
  GLUEBALL_RANDOM_INIT=1 GLUEBALL_ENSEMBLE_SEED=$ENS GLUEBALL_INIT_SEED=$SEED GLUEBALL_INPUT_SMEAR_LEVELS=0,2,4,6,8,12,16    GLUEBALL_D_MODEL=24 python -u scripts/train_glueball.py
done; done
```

(`--input-smear-levels=` / `--d-model=` argv forms exist too; env vars do not
survive every container wrapper — check the banner line of each log.) Width
follows the trained point at that x (§1.4). 18 dumps → copy to `dumps/`.

**Then**, laptop: `SFF_BASES=… SFF_TRUNCATE=1 [SFF_PRUNE=…] python scripts/su2_fair_fight.py <dump>`
on each of the 18 dumps with the WP1 setting → 18 per-dump `.pt` files.

**Pre-registered expectation** (`operator_decomposition.md` §5): the random net
is also far outside the classical span (the L1-ball transport reaches content
no smeared plaquette has) but carries little ground-state amplitude; for the
random net the projection onto the span should be a *better* operator than the
net itself, so ΔA₀(net − P) flips sign relative to the trained case (WP5).

### WP3 — the thin point and the width control (V100, 3 trainings)

**Goal.** The leftmost trained point on both ensembles, under today's protocol,
plus the one run that resolves the width confound.

Naming trap: with `INPUT_SMEAR_LEVELS=(0,)` the stem is `best_glueball_gelt`,
which is Run 4's checkpoint; `RESUME` defaults to True and would warm-start from
it, and a from-scratch run would overwrite it. Always pass `GLUEBALL_RESUME=0`
and a tag. With the WP2 naming, `GLUEBALL_D_MODEL=24` already yields
`best_glueball_gelt_d24.pth`; add `GLUEBALL_RUN_TAG=_p5` anyway so the protocol
version is visible in the name.

```
run_phase thin_run5      env GLUEBALL_INPUT_SMEAR_LEVELS=0       GLUEBALL_D_MODEL=24 GLUEBALL_RUN_TAG=_p5 GLUEBALL_RESUME=0 TQDM_MININTERVAL=30 python -u scripts/train_glueball.py
run_phase thin_ens1      env GLUEBALL_ENSEMBLE_SEED=1 GLUEBALL_INPUT_SMEAR_LEVELS=0 GLUEBALL_D_MODEL=24 GLUEBALL_RUN_TAG=_p5 GLUEBALL_RESUME=0 TQDM_MININTERVAL=30 python -u scripts/train_glueball.py
run_phase width_ctrl_run5 env GLUEBALL_INPUT_SMEAR_LEVELS=0,2,4,6 GLUEBALL_D_MODEL=24 GLUEBALL_RUN_TAG=_p5 GLUEBALL_RESUME=0 TQDM_MININTERVAL=30 python -u scripts/train_glueball.py
```

Runtime: the ens1 replication logged 15 h per training **before** the
optimisations of `notes/performance_audit.md`; the post-optimisation V100 number
has never been measured. Run `PROFILE_DIAGNOSTICS=1 python scripts/profile_glueball_step.py`
once first (minutes) and write the s/step into §8; budget 8–15 h per training
until then. The width control is third in the queue: if time runs out it is the
one to drop, and the curve is then reported with the width caveat of §1.4.

If the run in flight (§0.1) is one of these three, it counts and is not repeated.

**Gate per run.** The banner shows the intended levels, width and tag; the val
Rayleigh loss saturates near the ensemble's floor (−0.62 on run5, −0.57 on
ens1, `notes/glueball_spectroscopy.md` "Replication results") — a thin net will
sit well above it (Run 4 reached −0.43) and that is expected; the classical
anchor printed at the end lands at m_eff(Δ=1) ≈ 0.39–0.40.

**Outputs.** Three checkpoints + dumps (copy dumps to `dumps/`), then one
fair-fight run per dump on the laptop as in WP2.

### WP4 — the batch driver (laptop, 1 hour)

`scripts/curve_batch.sh`, modelled line-for-line on `overnight_replication.sh`
(`set -u`, `run_phase`, per-phase logs, failures logged and skipped): profile
step → the three WP3 trainings → the 18 WP2 evals → (optional) WP8's scan last.
**Sequential only**: training uses essentially the whole 32 GB (batch 8 OOMs),
so nothing runs beside it. Print `ls -l` of every new dump at the end, as the
replication driver does.

### WP5 — the decomposition companions and the curve figure (laptop, offline, 1–2 days)

**5a. `scripts/operator_decomposition.py` — a basis from the obars cache.**
Add `--basis=<obars.pt>:<arm>` (replaces the dump's `Obar_basis` with
`bases[arm].double()` and `levels` with `labels[arm]`; `jack_block` stays the
dump's 10) and `--m-ref=<float>` (fixes the reference mass for the amplitude
split; for a random-init dump use the trained net's m on that ensemble, since a
poor operator's own cosh fit is not a usable reference). The ladder-increment
loop assumes nested rungs: run it for `deep` only; for `full` report the
out-of-span fraction, Z_r/Z_G and ΔA₀(GELT − P) alone. Add `--shape-span=<obars.pt>:<arm>`:
after the decomposition against `deep`, project **r** onto span{`arm`} and print
the fraction of r's norm² inside it — "is the out-of-span content rectangular
loops the network rediscovered, or something no planar loop expresses?" This is
the cheapest new interpretability statement available and it costs nothing.

Runs, all offline:

```
python scripts/operator_decomposition.py dumps/best_glueball_gelt_sm0-2-4-6-8-12-16_test_obars.pt      --basis=…obars_run5.pt:deep --shape-span=…obars_run5.pt:full
python scripts/operator_decomposition.py dumps/best_glueball_gelt_sm0-2-4-6-8-12-16_ens1_test_obars.pt --basis=…obars_ens1.pt:deep --shape-span=…obars_ens1.pt:full
python scripts/operator_decomposition.py dumps/<random dumps at 7lv>  --basis=…:deep --m-ref=<trained m on that ensemble>
```

Gate: without `--basis` the script still reproduces the published gate numbers
(it prints them first). Expected against `deep`: a smaller out-of-span fraction
than 12.9%/11.7% and ΔA₀(GELT − P) positive but below +0.097.

**5b. `scripts/input_architecture_curve.py` — the figure and the table.**
Offline, CPU, seconds. Inputs: the per-dump fair-fight `.pt` files
(`rows[0]["arms"][arm]["A0"|"A0_err"]`, `rows[0]["delta"][arm]["dA0"|"dA0_err"]`)
plus a small manifest mapping each file to (ensemble, x, trace, seed). Output:
`results/fair_fight/input_architecture_curve.{png,pt,tex}`.

- Panel 1: A₀ vs x (ordinal axis: thin, 4lv, 7lv, 7lv+3sh; 4lv+5sh as an
  off-chain marker), three traces, two ensembles as two marker styles, random
  trace as mean over seeds with the §1.4 error rule; a shaded band at
  A₀ ≥ 0.90 marks saturation.
- Panel 2: ΔA₀(x) = trained − classical, correlated, per ensemble and combined,
  with the R1/R2/R3 tests printed as text.
- Table (`.tex`, paste-ready like `su2_attention_correlator_table.tex`): one
  row per x: A₀ classical / trained / random per ensemble, ΔA₀ combined, and
  the rung-equivalence entry ΔA₀(GELT(x_k) − GEVP(x_{k+1})), which is simply
  `delta["deep"]` in the 4lv dump's file and `delta["published"]` in the thin
  dump's file — no new estimator.
- The script evaluates the three tests of §1.2 mechanically and prints which
  readings hold; the text in §8 is written from that output.

### WP6 — optional: the shape-input GELT (V100, 2 trainings + 1 day of code)

Only if WP1–WP5 are done and ≥ 2 weeks remain. Feed matrix-valued
cubic-symmetrised R×T loops as extra adjoint channels (`rectangular_wilson_loop`
builds the matrix and traces it; expose the matrix, symmetrise r×t with t×r,
stack per plane and level in `config_inputs`), `in_channels = 3·n_levels·n_shapes`,
d_model 24 or wider as needed, `GLUEBALL_RUN_TAG=_sh3` in the name. Train at
7lv+3sh on both ensembles. Without it the right end of the curve compares GELT
without shapes to a GEVP with shapes, which is a fair reading as long as the
figure says so.

### WP7 — fix the Z₂ input pipeline and redo the Z₂ table (V100 ≈ 12–16 h; laptop 1 day)

Independent of the curve; second in value (§9). The defect:
`notes/audit_2026-09-06.md` §2 — at the production α = 0.5 the projected Z₂
ladder freezes after one step and is not gauge covariant; the Z₂ classical
column is one operator and the nets saw four channels of which three are
byte-identical.

Use **unprojected fat links**, not α = 0.7 (at 0.7 the projected ladder still
reaches a fixed point within ~5 steps, so the deep levels stay near-identical).
The retired implementation is `smear_steps(U, alpha, n_steps, project=False)`
in `git show cfa0a7e:scripts/z2_fair_fight.py` (lines ~254–283): the same APE
update `V = (1−α)U + (α/n_staples)·staples` without the sign projection,
linear in the links hence exactly covariant.

1. `gelt/glueball.py`: `ape_smear(..., project: bool = True)`; `project=False`
   skips `gaugegroup.project`. Document that the links then leave the group and
   that the loops built from them are still gauge invariant.
2. Tests: `test_ape_smear_z2_alpha_half_projected_is_not_covariant` (asserts
   the defect on Haar Z₂ at α = 0.5 with `project=True`, so the suite records
   it) and `test_ape_smear_unprojected_is_gauge_covariant` (Z₂ and SU(2),
   α = 0.5, exact to 1e-12).
3. Switch the three Z₂ call sites to `project=False`: `train_z2_glueball.config_inputs`,
   `z2_beta_scan.py` (regenerates `results/attention/z2_beta_scan.pt`, the
   classical mass the training gate reads), `z2_attention_correlator.py`
   (classical basis at line ~749). Keep `INPUT_SMEAR_LEVELS = (0, 4, 8, 16)`
   and `SMEAR_LEVELS = [0, 4, 8, 16]`: with fat links these are four distinct
   operators. Add `_fat` to `train_z2_glueball.artifact_tag` so the R6
   checkpoints and the correlator dumps are not overwritten.
4. Cheap robustness check first (no training): evaluate the existing R6
   checkpoints on fat inputs (`in_channels` is unchanged, so they load) through
   `z2_attention_correlator.py` and compare ξ_A and A₀ with the published table.
5. Retrain: `Z2G_R=6 Z2G_N_USE=800 python scripts/train_z2_glueball.py <β>` for
   β ∈ {0.745, 0.752, 0.756, 0.7585}. Budget 4 h each (the R=12/N_USE=400 logs
   show 5.7 s/step, 70 steps/epoch, 40 epochs; R = 6 has 85 offsets against
   313); check for an R6 log on the V100 for the real number.
6. `python scripts/z2_attention_correlator.py` (≈ 4 min per ensemble), then
   `ZAC_REPLOT` locally.

Pre-registered expectation: the trained-versus-random ΔA₀ column survives (it
compares two operators on identical inputs); the classical column moves up
(a real 4-operator ladder); the sentence in `main.tex` "the untrained attention
field is already a moderately better operator than the classical GEVP" is
**expected to die** and must be rewritten from the new table.

### WP8 — GPU filler: the SU(2) β-scan of the attention field (V100 ≈ 6 h, zero code)

`SAC_BETAS=2.1,2.3,2.4,2.5,2.7 TQDM_MININTERVAL=30 nohup python -u scripts/su2_attention_correlator.py > logs/sac_scan.log 2>&1 &`
— samples its own seed-11, N = 1600 ensembles at the four new β (~40 min each),
evaluates the β = 2.4 trained net off-diagonal (licensed by
`notes/attention_as_operator.md` §6.1.1/§9.3), prints the Pearson against
`beta_scan.pt` only with ≥ 3 couplings. Turns Table `tab:su2_train_rnd_gevp`
from one row into a scan. Run it **last in the queue** — it must not share the
GPU with a training.

### WP9 — the excited-state gate for a learned two-operator basis (laptop, minutes)

Do **not** write training code before this passes. From the obars cache:

```python
import torch
from gelt.glueball import connected_correlator_matrix, gevp_eigenvalues, gevp_effective_mass
ob = torch.load("results/fair_fight/su2_fair_fight_obars_run5.pt", weights_only=False)
b = ob["bases"]["deep"].double()                              # (7, 400, 24)
lam = gevp_eigenvalues(connected_correlator_matrix(b), t0=1, truncate=True)
print(gevp_effective_mass(lam)[1:4, :2])                      # columns: ground, first excited
```

plus a delete-block jackknife (block 10) over the 400 configs for the second
column. **Pass**: m₁(Δ=1) and m₁(Δ=2) agree within 1σ and the Δ=2 error is
below 25%, on at least one ensemble. If it fails, the learned two-operator
basis has no classical comparator on this ensemble and the option is closed
for free; record the numbers in §8. If it passes, the design is: `mlp_out=2`,
loss −Tr[(C(0)+ε)⁻¹C(1)] over the learned 2×2 block (Ky Fan: maximum at
e^{−m₁}+e^{−m₂}, no orthogonality penalty needed, collapse onto two copies is
penalised because C(0) goes singular), scale pin on log det C(0); GEVP on the
learned pair against `deep` for the first excited state. Two trainings.

### WP10 — the matched L-CNN (after the curve, if ≥ 3 weeks remain)

`gelt/lcnn.py` mirrors GELT's I/O; `train_glueball.py` needs a model switch and
`build_axis_transports(U3_first, K=2, gaugegroup)` in place of
`build_transport_average`. Same loss, splits, jackknife, smeared channels,
~5k parameters, 4lv and 7lv on run5 first, then ens1. It becomes a **fourth
trace** on the WP5 figure — a better use of it than a standalone number.
Pre-registered expectation: parity with GELT on A₀ (the smeared inputs hand
both nets the staple content; at matched depth the loop degree is the same);
parity is the honest "attention costs nothing and the attention field is what
a convolution cannot produce".

---

## 4. Sequence and budget

| order | package | where | GPU | wall |
|---|---|---|---|---|
| 1 | WP0 pre-register, secure dumps | both | 0 | 1 h |
| 2 | WP1 estimator, shape verdict | laptop (+V100 offline if a cache is missing) | 0 | 1 day |
| 3 | WP4 driver, then WP3 + WP2 in the queue | V100 | 3 trainings + 18 evals | 1–2 days GPU |
| 4 | WP5 decomposition + curve figure | laptop | 0 | 1–2 days |
| 5 | WP7 Z₂ fix | both | 12–16 h | 3 days |
| 6 | WP8 scan (filler), WP9 gate | V100 / laptop | 6 h / 0 | — |
| 7 | WP6 or WP10 | V100 | 2 trainings | 1–2 weeks |

The minimal curve (three x, three traces, two ensembles) is steps 1–4, about
one week. Everything after step 4 is separable and can be dropped without
affecting the figure.

---

## 5. Traps (each one cost a day somewhere in `notes/`)

- **`results/` is root-owned on the V100.** Jobs write there; you cannot copy
  into it. Anything that must travel goes through `dumps/` and git.
- **The obars cache is keyed by ensemble only** and its `gelt` is whichever
  dump last ran there. Never read `gelt` from it; WP1 makes it write-once.
- **`_ens1` must be in every dump's basename** or `su2_fair_fight` pairs it
  with the wrong ensemble (`_cache_for`, `_tag`).
- **Thin-input naming collides with Run 4's checkpoint** and `RESUME` defaults
  to True — always tag, always `GLUEBALL_RESUME=0` (WP3).
- **`d_model` is not in the checkpoint name** for the existing 7-level nets
  (width 24). New runs get `_d24`; the old ones are listed in §2.2.
- **Env vars do not survive every container wrapper**: both scripts accept
  `--flag=value` argv forms; read the banner line of each log to confirm what
  actually ran.
- **Bit-reproducibility of old dumps is gone** since the smearing rewrite
  (`performance_audit.md` §7.3): re-evaluating an old checkpoint reproduces Ō
  to ~4e-7, which is five orders below any error bar. Do not chase it.
- **The published slice gate must pass** in every fair-fight run (`verify_slice`
  or its cached form): a failure means wrong configurations, not physics.
- **The superset gate**: a superset arm reading below `published` by > 2σ is
  the estimator failing, not a result; the WP1 selection rule handles it. With
  pruning the superset property is only approximate — that is why pruning is
  last resort and `published` must stay untouched under the chosen setting.
- **A₀ > 1 is a fit artifact** of a nearly pure operator; report 1 − A₀ with
  its sign, do not clip.
- **`report()`'s verdict prose** in `su2_fair_fight.py` is written for the
  single-claim audit. For the curve use the numbers in the `.pt`, not the
  sentences.
- **`SFF_NOCACHE=1`** means dump-only (published arm only) — it is *not* the
  offline mode for the curve; `SFF_BASES` is.
- **Never read the run in flight before §0.1 and the commit.**

---

## 6. Definition of done

- `results/fair_fight/input_architecture_curve.{png,pt,tex}` exist and the
  script that made them runs offline from tracked inputs.
- §8 of this note carries: the §0.1 record; the WP1 (a)–(d) estimator table and
  the chosen setting; the verdict on `full`/`shapes_sm`; the per-x table with
  both ensembles; which of R1/R2/R3 hold by the §1.2 tests; the random-trace
  reading; the decomposition against `deep` and the r-versus-shapes fraction;
  the profiler's s/step; and every deviation from §1–§2 with its reason.
- `pytest tests` green, including the new truncation tests.
- `CLAUDE.md` Status paragraph and `README.md`'s reproduction table gain one
  line each for the curve script; `notes/audit_2026-09-06.md` §6.5's "still
  open" sentence gets a pointer to §8 here.

## 7. What changes in the text once the numbers exist

- **Abstract and §SU(2)** of `../report/main.tex`: the single "+0.078 ± 0.022
  (3.6σ)" sentence becomes the curve statement (whichever of R1/R2/R3 holds),
  with the 4-level number kept as the first result and the matched-input pair
  as the headline; the figure replaces or joins `fig:glueball_overlap`.
- **Table `tab:overlap`** becomes the per-x table (WP5b).
- The fair fight and the decomposition, which `main.tex` does not mention at
  all today, get one paragraph each: the comparator was strengthened before the
  curve was drawn (§1.5), and the advantage is 12.9%/11.7% outside the span,
  learned if WP2 says so.
- §Attention: the "untrained attention field is already better than the
  classical GEVP" sentence is rewritten after WP7.

---

## 8. Results — TO BE WRITTEN AFTER THE FACT (dated, §1–§2 untouched)

### 8.0 Deviations from §2.2's file inventory (2026-09-10)

`ls results/glueball/` on the V100 returns only the two 7-level checkpoints and
their two dumps, the two 4-level checkpoints (`…_sm0-2-4-6.pth`, `…_ens1.pth`),
and `glueball_gelt{,_ens1}.png`. Against §2.2:

- **Run 4's `best_glueball_gelt.pth` does not exist** on the V100, and not on
  the laptop either. The thin point has no old checkpoint left, so WP3's thin
  retrain overwrites nothing. It still gets the `_d24` / `_p5` tags so the name
  says what it is.
- **The 4-level test dumps are not in `results/glueball/`** on the V100. They
  exist only as the tracked copies in `dumps/`, which is where every fair-fight
  run should read them from.
- **The Run-5 4-level checkpoint is there**, so the width control (4lv at
  d_model 24) still collides with it under today's naming. WP2's `_d<D_MODEL>`
  suffix has to land before that run.

### 8.0.1 Profiler, V100, 2026-09-10 (`PROFILE_DIAGNOSTICS=1 profile_glueball_step.py`)

Measured at the Run-5 shape (4 levels, d_model 16, batch 6 = 144 slices,
R = 2, 4 layers, gradient checkpointing on):

    config_inputs (W, T)    646 ms   12.8%   (APE ladder 489 ms of it)
    forward (GELT)         1155 ms   22.9%
    backward               3242 ms   64.3%
    TOTAL                  5046 ms   → 1181 s/epoch, peak 20.0 GiB

That is **5.05 s/step against the ens1 run's logged 7.77 s/step: 1.54× end to
end** on the V100, the first measured total for the optimisations of
`notes/performance_audit.md`. Introspection stashes cost 304 ms/step (6%).
At ~25 epochs to early stop (the ens1 run), a training is ≈ 8 h. This replaces
§3 WP3's "budget 8–15 h" for the 4-level shape. The thin (1 level) and 7-level
d24 runs have a different shape and were not profiled.

The per-stage micro-benchmark's `transport (2 bmm)` case OOMed (a 4.45 GiB
allocation) and has no number. It is a benchmark case, not the training path,
which ran at the 20 GiB peak above.

---

## 9. Superseded: the first-pass ranking (2026-09-10, earlier the same day)

Before the curve was proposed, the audit ranked five options; the curve
absorbed option A and moved the rest down. Kept here so the reasoning trail is
intact; the numbers are unchanged.

| option | what | new code | V100 time | wall | P(positive) |
|---|---|---|---|---|---|
| A | decomposition controls + pruned GEVP (now WP1, WP2, WP5a) | small | ~30 min | 2–3 days | 0.75 "learned"; 0.85 readable shape verdict |
| B | matched L-CNN shootout (now WP10) | 1–2 days | 2 trainings | ~1 week | 0.30 GELT wins; 0.55 parity |
| C | fix Z₂ smearing, retrain 4 nets, redo table (now WP7) | small | ~12 h | 3 days | 0.85 core claim survives |
| D | learned two-operator basis, the 0⁺⁺* state (now WP9 gate) | ~1 week | 2–4 trainings | 2–3 weeks | 0.45 |
| E | SU(2) β-scan of the attention field (now WP8) | none | ~6 h | 1 day | 0.65 |

Where the thesis was exposed, in the order a committee would find it: the
spectroscopy headline quoted against the 4-level arm with the fair fight
unmentioned and the shape arms unreadable; the "why attention" argument
resting on a non-equivariant CNN with the L-CNN never run on the glueball
task; the Z₂ table's one-operator classical column and gauge-variant inputs;
no continuum physics (stated honestly, no cheap fix).

Why the curve ranks first: it is the presentation the fair fight and the
matched-input retrains were missing, every point that already exists lies on
it, three of option A's pieces become its prerequisites or its third trace,
and it removes the "opponent chosen after the fact" objection by construction —
provided §0's discipline is kept.

`PLANS.md`: parts B and C are month-to-year programmes (flows, fermions, neural
quantum states, curved lattices, RG) with success probabilities well under one
half; none fits the final phase of a thesis. Learned smearing is already
implicit in the smeared-input GELT; control variates is a different thesis;
the continuum limit of the learned operator (A.7) is the right follow-up if
this becomes a paper but needs 3–4 retrains and anisotropy tuning; the
expressivity theory (A.10) is an appendix, not a next step.
