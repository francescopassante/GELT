#!/usr/bin/env bash
# The M1 probe batch (V100, unattended; ~1 h sweep + ~7 h grid).
#
# notes/m1_probe.md is the design record and holds the pre-registered readings.
# The question: does *input-dependent offset weighting* pay, with the transport
# confound removed? The GELT-vs-L-CNN comparison can never answer it — those two
# differ in two things at once — so the primary arm here is GELT against GELT
# with the softmax frozen.
#
# Modelled on lcnn_shootout.sh: phases are independent, a failure is logged and
# the batch continues, every phase has its own log, and a phase whose stats dump
# already exists is skipped, so a restart resumes where the batch stopped.
#
#   part 0  the gates, seconds, no GPU: the matched-DOF table and the §5
#           pre-flight on both ensembles. **Read the pre-flight before part 1.**
#           A target whose best linear filter at matched reach already explains
#           ≥ 0.90 of it is dropped — no architecture whose only extra
#           ingredient is input-dependent offset weighting can separate in what
#           is left. (This is not hypothetical: it is how T2 acquired its
#           SCALE_POWER, notes/m1_probe.md §3.1.)
#   part 1  the LR sweep. GELT's 3e-3 was tuned for the Rayleigh loss on the
#           glueball task; a losing arm at another arm's LR is uninterpretable.
#           Three LRs × five arms on T2 / ens0 / seed 0, 6 epochs, disposable
#           `_sweep_*` tags. Read the best val per arm, then set it in part 2.
#   part 2  T0, the calibration target: gelt, frozen, lcnn × 3 seeds × 2
#           ensembles. R-A must come back within 1σ or every other reading is
#           void, so this runs first and is worth reading on its own.
#   part 3  T1 and T2: gelt, frozen, lcnn, lcnn_norm × 3 seeds × 2 ensembles.
#           R-B (primary), R-C, R-D.
#   part 4  R-E, the null: the stack frozen at init, only the head trained.
#           Not "no training" — every arm's head is zero-initialised, so an
#           untrained net scores R² = 0 by construction and measures nothing.
#   part 5  R-B', the matched-parameter frozen arm. It was conditional on R-B
#           favouring GELT at > 2σ. **That condition fired on 2026-09-15** (the
#           gate measured ~+0.21 at twenty times its error bars), so
#           `frozen_matched` is now an arm of parts 2 and 3 like any other and
#           this part is left only for re-running it alone.
#
# Everything downstream is offline: scripts/probe_readings.py turns the dumps
# into the pre-registered table and a LaTeX fragment.
#
# Check which phases are done and which would run (no GPU needed):
#   PROBE_DRY_RUN=1 bash scripts/probe_batch.sh
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/probe_batch.sh > logs/probe_batch.log 2>&1 &
#   tail -f logs/probe_batch.log
#
#   PROBE_PARTS=0,1              the gates and the sweep (the default)
#
#   ── the Z₂ vortex study (notes/where_attention_can_win.md §9) ──
#   PROBE_PARTS=z-gate           the gates: the matched-DOF table under the
#                                switch, and the **initialisation gate at the
#                                production volume**. The L-CNN's field grows
#                                doubly exponentially with depth at nc = 1 and
#                                the scale that tames it was first set from an
#                                8³ box, which the gate falsified — so this runs
#                                before anything trains, every time.
#   PROBE_PARTS=z-sweep          the per-arm LR sweep (and, for the L-CNN arms,
#                                init scale) at β = 0.7520, on **V1 masked and
#                                V2 together**: an arm that is down on the
#                                calibration target is down for reasons the
#                                study does not measure (m1_probe.md §0 point 6).
#   PROBE_PARTS=z-grid           the grid at the primary coupling, six seeds.
#                                **Read W-A (V2) first** — it is the gate — and
#                                run the replication only if it passes:
#                                  PROBE_PARTS=z-grid PROBE_Z2_BETAS=0.745 …
#   PROBE_PARTS=2,3,4            the grid, after the sweep has been read
#   PROBE_PARTS=8,9              the transport arms (§4.4), after the CPU gate
#   PROBE_PARTS=6 then 7         the signed-α side experiment (§8); independent
#                                of 2-4, so it can share the night on a 2nd GPU
#   PROBE_LR_GELT=… PROBE_LR_FROZEN=… PROBE_LR_LCNN=… PROBE_LR_LCNN_NORM=…
#                                per-arm learning rates for parts 2+
#   PROBE_ENSEMBLES="0 1"        which cached ensembles to use
#   PROBE_SEEDS="0 1 2"          init seeds

set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/m1_probe

PARTS="${PROBE_PARTS:-0,1}"
ENSEMBLES="${PROBE_ENSEMBLES:-0 1}"
SEEDS="${PROBE_SEEDS:-0 1 2}"
EPOCHS="${PROBE_EPOCHS:-20}"
SWEEP_EPOCHS="${PROBE_SWEEP_EPOCHS:-6}"
# The sweep grid. **An arm whose best LR sits at an edge of this list is not
# bracketed**, and running the grid on it would be the same defect part 1 exists
# to prevent — so extend and re-run part 1 (finished points are skipped):
#   PROBE_PARTS=1 PROBE_SWEEP_LRS="3e-2 3e-4" bash scripts/probe_batch.sh
SWEEP_LRS="${PROBE_SWEEP_LRS:-1e-2 3e-3 1e-3}"
# …and which arms to sweep, so chasing one unbracketed arm down does not re-run
# the four that are already settled:
#   PROBE_PARTS=1 PROBE_SWEEP_LRS=1e-4 PROBE_SWEEP_ARMS="lcnn lcnn_norm" …
SWEEP_ARMS="${PROBE_SWEEP_ARMS:-gelt frozen lcnn lcnn_norm frozen_matched}"
# Per-arm learning rates for parts 2–5, set from part 1's val curves. The
# defaults are GELT's glueball LR for every arm, which is exactly the situation
# part 1 exists to fix — do not run the grid on them without reading the sweep.
LR_GELT="${PROBE_LR_GELT:-3e-3}"
LR_FROZEN="${PROBE_LR_FROZEN:-3e-3}"
LR_LCNN="${PROBE_LR_LCNN:-3e-3}"
# lcnn_norm gets its own knob: bounding ω changes the optimisation enough that
# its sweep optimum (3e-4) is a factor of three below plain lcnn's (1e-3), and
# giving the control arm the other arm's rate is exactly the handicap part 1
# exists to remove. Defaults to LR_LCNN when unset.
LR_LCNN_NORM="${PROBE_LR_LCNN_NORM:-${LR_LCNN}}"
LR_SIGNED="${PROBE_LR_SIGNED:-1e-2}"
# Which signed arms parts 6/7 cover, and over which rates.
SIGNED_ARMS="${PROBE_SIGNED_ARMS:-signed signed_bounded signed_l1}"
SIGNED_LRS="${PROBE_SIGNED_LRS:-3e-2 1e-2 3e-3 1e-3}"
LR_SIGNED_BOUNDED="${PROBE_LR_SIGNED_BOUNDED:-${LR_SIGNED}}"
LR_SIGNED_L1="${PROBE_LR_SIGNED_L1:-${LR_SIGNED}}"
# Parts 8/9, the transport arms. They are the `gelt` network to the byte fed a
# different T, so `gelt`'s rate is the honest default — but "same parameters"
# does not imply "same optimisation", which is what part 8 checks before part 9
# spends six cells on it.
TRANSPORT_ARMS="${PROBE_TRANSPORT_ARMS:-gelt_single gelt_projected}"
TRANSPORT_LRS="${PROBE_TRANSPORT_LRS:-1e-2 3e-3}"
LR_GELT_SINGLE="${PROBE_LR_GELT_SINGLE:-${LR_GELT}}"
LR_GELT_PROJECTED="${PROBE_LR_GELT_PROJECTED:-${LR_GELT}}"

# ── the Z₂ vortex study ──────────────────────────────────────────────────────
# β is the ensemble's identity here: train_z2_glueball.py keys its cache on the
# coupling, so a second ensemble is a second β rather than a second chain.
# §9.6 fixed 0.7520 as primary and 0.7450 as the replication, in writing, from
# the production pre-flight — do not add couplings without re-reading it.
# Written as Python's float repr, because that is what the cache key and the
# dump stems are built from (probe_common.cache_path, train_probe.STEM): 0.752
# *is* §9.6's 0.7520, and spelling it with the trailing zero would silently
# break the skip-on-existing-dump rule that makes a restart resume.
Z2_BETAS="${PROBE_Z2_BETAS:-0.752}"
Z2_SEEDS="${PROBE_Z2_SEEDS:-0 1 2 3 4 5}"
Z2_TARGETS="${PROBE_Z2_TARGETS:-V2 V1}"
# The seven arms: the 2 × 2 of §9.3 ({softmax,frozen} × {average,single}) plus
# the capacity control, plus the two L-CNN arms.
Z2_ARMS="${PROBE_Z2_ARMS:-gelt frozen frozen_matched gelt_single frozen_single lcnn lcnn_norm}"
# Half-decade steps. An arm whose optimum sits at an edge is not bracketed and
# the grid must not be run on it — extend and re-run z-sweep, finished points
# are skipped.
Z2_SWEEP_LRS="${PROBE_Z2_SWEEP_LRS:-1e-2 3e-3 1e-3 3e-4}"
# …and the L-CNN arms also over init scale, because 0.2 is a gate verdict and
# not an optimum. Only scales the gate passed belong here.
Z2_SWEEP_INITS="${PROBE_Z2_SWEEP_INITS:-0.1 0.2 0.3}"
Z2_SWEEP_EPOCHS="${PROBE_Z2_SWEEP_EPOCHS:-${EPOCHS}}"
# Per-arm rates for z-grid, set from z-sweep's val curves. The defaults are one
# rate for every arm, which is exactly what the sweep exists to fix.
z2_arm_lr() {
  local v
  case "$1" in
    gelt)           v="${PROBE_Z2_LR_GELT:-}";;
    frozen)         v="${PROBE_Z2_LR_FROZEN:-}";;
    frozen_matched) v="${PROBE_Z2_LR_FROZEN_MATCHED:-}";;
    gelt_single)    v="${PROBE_Z2_LR_GELT_SINGLE:-}";;
    frozen_single)  v="${PROBE_Z2_LR_FROZEN_SINGLE:-}";;
    lcnn)           v="${PROBE_Z2_LR_LCNN:-}";;
    lcnn_norm)      v="${PROBE_Z2_LR_LCNN_NORM:-}";;
    *)              v="";;
  esac
  echo "${v:-${PROBE_Z2_LR:-3e-3}}"
}

stamp() { date "+%F %T"; }
wants() { case ",${PARTS}," in *",$1,"*) return 0;; *) return 1;; esac; }

arm_lr() {
  case "$1" in
    gelt) echo "${LR_GELT}";;
    frozen|frozen_matched) echo "${LR_FROZEN}";;
    lcnn) echo "${LR_LCNN}";;
    lcnn_norm) echo "${LR_LCNN_NORM}";;
    signed) echo "${LR_SIGNED}";;
    signed_bounded) echo "${LR_SIGNED_BOUNDED}";;
    signed_l1) echo "${LR_SIGNED_L1}";;
    gelt_single) echo "${LR_GELT_SINGLE}";;
    gelt_projected) echo "${LR_GELT_PROJECTED}";;
    *) echo "${LR_GELT}";;
  esac
}

need_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping the batch rather than"
    echo "   training on the CPU. Check nvidia-smi, then re-run: finished phases"
    echo "   are skipped."
    exit 1
  fi
}

# run_phase <name> <dump that marks it done> <command...>
run_phase() {
  local name="$1" dump="$2"; shift 2
  if [ -n "${dump}" ] && [ -e "${dump}" ]; then
    echo "[$(stamp)] ── phase ${name}: done already (${dump}) — skipped"
    return
  fi
  if [ -n "${PROBE_DRY_RUN:-}" ]; then
    echo "[$(stamp)] ── phase ${name}: WOULD RUN: $*"
    return
  fi
  need_cuda
  echo "[$(stamp)] ── phase ${name}: $*"
  if "$@" > "logs/${name}.log" 2>&1; then
    echo "[$(stamp)]    OK  ${name}"
  else
    echo "[$(stamp)]    FAILED  ${name} (see logs/${name}.log) — continuing"
  fi
}

# train <arm> <target> <ensemble> <seed> [extra flags…]
train() {
  local arm="$1" tgt="$2" ens="$3" seed="$4"; shift 4
  local stem="results/m1_probe/probe_${arm}_${tgt}_ens${ens}_init${seed}"
  local name="probe_${arm}_${tgt}_ens${ens}_s${seed}"
  for a in "$@"; do
    case "$a" in --null*) stem="${stem}_null"; name="${name}_null";; esac
  done
  run_phase "${name}" "${stem}_stats.pt" \
    python -u scripts/train_probe.py --arm="${arm}" --target="${tgt}" \
    --ensemble-seed="${ens}" --init-seed="${seed}" --epochs="${EPOCHS}" \
    --lr="$(arm_lr "${arm}")" "$@"
}

# z2_train <arm> <target> <beta> <seed> [extra flags…]
# The Z₂ study's dumps live under results/z2_vortex/probe and their stems carry
# β rather than an ensemble seed (probe_common.cache_path), so the skip-on-dump
# rule needs its own stem builder rather than a flag on the one above.
z2_train() {
  local arm="$1" tgt="$2" beta="$3" seed="$4"; shift 4
  local tag="" stem name
  for a in "$@"; do
    case "$a" in --run-tag=*) tag="${a#--run-tag=}";; esac
  done
  # train_probe.STEM appends _n<N> whenever PROBE_N_CONFIGS is not the group
  # default, so the stem this builds has to as well or skip-on-existing-dump
  # silently stops firing and a restart re-runs everything.
  local nsuf=""
  if [ -n "${PROBE_N_CONFIGS:-}" ] && [ "${PROBE_N_CONFIGS}" != "100" ]; then
    nsuf="_n${PROBE_N_CONFIGS}"
  fi
  stem="results/z2_vortex/probe/probe_${arm}_${tgt}_b${beta}_init${seed}${nsuf}${tag}"
  name="z2_${arm}_${tgt}_b${beta}_s${seed}${tag}"
  run_phase "${name}" "${stem}_stats.pt" \
    python -u scripts/train_probe.py --group=z2 --arm="${arm}" --target="${tgt}" \
    --z2-beta="${beta}" --init-seed="${seed}" "$@"
}

export TQDM_MININTERVAL=30

# ── part 0: the gates (no GPU) ───────────────────────────────────────────────
if wants 0; then
  echo "[$(stamp)] ══ part 0: matched-DOF table and the §5 pre-flight"
  python -u - <<'PY'
import sys; sys.path.insert(0, "scripts")
import probe_common as pc
print("arm              real DOFs   ratio to gelt")
for k, (d, r) in pc.dof_table().items():
    note = "nested ablation, fewer by design" if k == "frozen" else (
        "OK" if abs(r - 1) <= pc.DOF_TOLERANCE else "** OUT OF TOLERANCE **")
    print(f"{k:16s} {d:9d}   {r:5.3f}   {note}")
PY
  for ENS in ${ENSEMBLES}; do
    run_phase "probe_preflight_ens${ENS}" "results/m1_probe/preflight_ens${ENS}.pt" \
      python -u scripts/probe_preflight.py --ensemble-seed="${ENS}"
  done
  echo "[$(stamp)]    READ logs/probe_preflight_ens*.log before part 1: a target"
  echo "[$(stamp)]    that fails its gate must not be trained."
fi

# ── part 1: the per-arm LR sweep ─────────────────────────────────────────────
if wants 1; then
  echo "[$(stamp)] ══ part 1: LR sweep on T2 / ens0 / seed 0, ${SWEEP_EPOCHS} epochs"
  echo "[$(stamp)]    grid: ${SWEEP_LRS}   arms: ${SWEEP_ARMS}"
  for ARM in ${SWEEP_ARMS}; do
    for SLR in ${SWEEP_LRS}; do
      TAG="_sweep_lr${SLR}"
      run_phase "probe_sweep_${ARM}_lr${SLR}" \
        "results/m1_probe/probe_${ARM}_T2_ens0_init0${TAG}_stats.pt" \
        python -u scripts/train_probe.py --arm="${ARM}" --target=T2 \
        --ensemble-seed=0 --init-seed=0 --epochs="${SWEEP_EPOCHS}" \
        --lr="${SLR}" --run-tag="${TAG}"
    done
  done
  echo "[$(stamp)]    grep -H 'best epoch' logs/probe_sweep_*.log"
  echo "[$(stamp)]    Check each arm's winner is NOT at an edge of the grid"
  echo "[$(stamp)]    before believing it. Then run"
  echo "[$(stamp)]    PROBE_PARTS=2,3,4 PROBE_LR_GELT=… PROBE_LR_FROZEN=… PROBE_LR_LCNN=… bash scripts/probe_batch.sh"
fi

# ── part 2: T0, the calibration target ───────────────────────────────────────
if wants 2; then
  echo "[$(stamp)] ══ part 2: T0 (calibration) — gelt, frozen, frozen_matched, lcnn"
  for ENS in ${ENSEMBLES}; do
    for SEED in ${SEEDS}; do
      for ARM in gelt frozen frozen_matched lcnn; do
        train "${ARM}" T0 "${ENS}" "${SEED}"
      done
    done
  done
  echo "[$(stamp)]    read R-A now: python scripts/probe_readings.py"
fi

# ── part 3: T1 and T2 — the primary grid ─────────────────────────────────────
if wants 3; then
  echo "[$(stamp)] ══ part 3: T2 then T1 — gelt, frozen, frozen_matched, lcnn, lcnn_norm"
  # Target outermost, so every T2 cell is finished before any T1 cell starts:
  # T2 is the primary target and a batch that has to be cut short should leave
  # a complete primary reading, not two half-finished ones.
  for TGT in T2 T1; do
    for ENS in ${ENSEMBLES}; do
      for SEED in ${SEEDS}; do
        for ARM in gelt frozen frozen_matched lcnn lcnn_norm; do
          train "${ARM}" "${TGT}" "${ENS}" "${SEED}"
        done
      done
    done
  done
fi

# ── part 4: R-E, the random-features null ────────────────────────────────────
if wants 4; then
  echo "[$(stamp)] ══ part 4: null arms (stack frozen at init, head trained)"
  for TGT in T1 T2; do
    for ARM in gelt frozen lcnn; do
      train "${ARM}" "${TGT}" 0 0 --null=1
    done
  done
fi

# ── part 5: R-B', conditional on R-B ─────────────────────────────────────────
if wants 5; then
  echo "[$(stamp)] ══ part 5: frozen_matched — ONLY if R-B favours GELT at > 2σ"
  for ENS in ${ENSEMBLES}; do
    for SEED in ${SEEDS}; do
      for TGT in T2 T1; do
        train frozen_matched "${TGT}" "${ENS}" "${SEED}"
      done
    done
  done
fi

# ── part 6: the signed arms' LR sweep, at the grid's own horizon ─────────────
# The side experiment of notes/m1_probe.md §8: why does a matched L-CNN beat
# GELT on a target built to favour attention? The standing hypothesis is that
# softmax does two things — reads the input (worth +0.21, R-B) and forces a
# convex combination (a layer can only average neighbours, never subtract one
# from another, and T2 is a shell *difference*). `signed` and `signed_bounded`
# keep the first and drop the second, at an identical parameter count.
#
# Sweeps at PROBE_EPOCHS, not at 6: a rate chosen on a short anneal does not
# transfer to a long one, which is the lesson of §8's first budget gate.
# Independent of parts 2-4, so it can run on a second GPU alongside them:
#   CUDA_VISIBLE_DEVICES=1 PROBE_PARTS=6 PROBE_EPOCHS=40 bash scripts/probe_batch.sh
if wants 6; then
  echo "[$(stamp)] ══ part 6: LR sweep for the signed arms, ${EPOCHS} epochs"
  for ARM in ${SIGNED_ARMS}; do
    for SLR in ${SIGNED_LRS}; do
      TAG="_sweep40_lr${SLR}"
      run_phase "probe_sweep40_${ARM}_lr${SLR}" \
        "results/m1_probe/probe_${ARM}_T2_ens0_init0${TAG}_stats.pt" \
        python -u scripts/train_probe.py --arm="${ARM}" --target=T2 \
        --ensemble-seed=0 --init-seed=0 --epochs="${EPOCHS}" \
        --lr="${SLR}" --run-tag="${TAG}"
    done
  done
  echo "[$(stamp)]    grep -H 'best epoch\|R² =' logs/probe_sweep40_*.log"
  echo "[$(stamp)]    winners must be interior, then run part 7 with"
  echo "[$(stamp)]    PROBE_LR_SIGNED=… PROBE_LR_SIGNED_BOUNDED=…"
fi

# ── part 7: the signed arms proper ───────────────────────────────────────────
# T2 is the reading and T0 is its calibration (§4.2), 3 seeds on ens0. The
# comparisons are against `gelt`, whose cells parts 2-3 already produce.
if wants 7; then
  echo "[$(stamp)] ══ part 7: signed arms on T2 and T0, 3 seeds, ens0"
  for TGT in T2 T0; do
    for SEED in ${SEEDS}; do
      for ARM in ${SIGNED_ARMS}; do
        train "${ARM}" "${TGT}" 0 "${SEED}"
      done
    done
  done
fi

# ── part 8: the transport arms' rate check ───────────────────────────────────
# `gelt_single` and `gelt_projected` are `gelt` with a different T (§4.4), so
# `gelt`'s 1e−2 is the default — but the transport is what the score path reads,
# and an arm fed an on-group T has different score statistics from one fed a
# sub-unitary T. Two rates, not four: this is a confirmation that 1e−2 is not
# wrong, not a search. **Run scripts/probe_transport_gate.py first** — if the
# two transports agree on production configurations, nothing below is worth a
# GPU-hour, and that is a CPU measurement of a couple of minutes.
if wants 8; then
  echo "[$(stamp)] ══ part 8: rate check for the transport arms, ${EPOCHS} epochs"
  for ARM in ${TRANSPORT_ARMS}; do
    for SLR in ${TRANSPORT_LRS}; do
      TAG="_sweep40_lr${SLR}"
      run_phase "probe_sweep40_${ARM}_lr${SLR}" \
        "results/m1_probe/probe_${ARM}_T2_ens0_init0${TAG}_stats.pt" \
        python -u scripts/train_probe.py --arm="${ARM}" --target=T2 \
        --ensemble-seed=0 --init-seed=0 --epochs="${EPOCHS}" \
        --lr="${SLR}" --run-tag="${TAG}"
    done
  done
  echo "[$(stamp)]    grep -H 'R² =' logs/probe_sweep40_gelt_*.log"
fi

# ── part 9: the transport arms proper ────────────────────────────────────────
# T2 is the reading and T0 the calibration, the full 3 seeds × 2 ensembles so
# R-I is read on the same six paired cells as R-B and R-C. `gelt`'s own cells
# are what they are differenced against, and parts 2-3 already produced those.
if wants 9; then
  echo "[$(stamp)] ══ part 9: transport arms on T2 and T0, 3 seeds, 2 ensembles"
  for ENS in ${ENSEMBLES}; do
    for TGT in T2 T0; do
      for SEED in ${SEEDS}; do
        for ARM in ${TRANSPORT_ARMS}; do
          train "${ARM}" "${TGT}" "${ENS}" "${SEED}"
        done
      done
    done
  done
fi

# ── part z-gate: the Z₂ study's gates (no training) ──────────────────────────
if wants z-gate; then
  echo "[$(stamp)] ══ part z-gate: matched-DOF table under the switch, then the"
  echo "[$(stamp)]    initialisation gate at the production volume"
  python -u scripts/z2_dof_table.py --group=z2
  run_phase "z2_init_gate" "results/z2_vortex/init_gate.pt" \
    python -u scripts/z2_init_gate.py \
      --z2gate-betas="$(echo ${Z2_BETAS} | tr ' ' ',')"
  echo "[$(stamp)]    READ logs/z2_init_gate.log before z-sweep. A FAIL there"
  echo "[$(stamp)]    means the arms' conv_init_scale does not hold at the"
  echo "[$(stamp)]    production volume, and nothing should train until it does."
fi

# ── part z-sweep: per-arm LR (and init scale) at the primary coupling ─────────
# On V1 *and* V2, because m1_probe.md §0 point 6 is the lesson: an arm that is
# down on the calibration target is down for reasons the study does not measure,
# and reading that only after the grid wastes the grid.
if wants z-sweep; then
  echo "[$(stamp)] ══ part z-sweep: β ${Z2_BETAS}, ${Z2_SWEEP_EPOCHS} epochs, seed 0"
  echo "[$(stamp)]    arms: ${Z2_ARMS}"
  echo "[$(stamp)]    rates: ${Z2_SWEEP_LRS}   L-CNN init scales: ${Z2_SWEEP_INITS}"
  for BETA in ${Z2_BETAS}; do
    for TGT in ${Z2_TARGETS}; do
      for ARM in ${Z2_ARMS}; do
        for SLR in ${Z2_SWEEP_LRS}; do
          case "${ARM}" in
            lcnn|lcnn_norm)
              for CI in ${Z2_SWEEP_INITS}; do
                z2_train "${ARM}" "${TGT}" "${BETA}" 0 \
                  --epochs="${Z2_SWEEP_EPOCHS}" --lr="${SLR}" \
                  --z2-lcnn-conv-init="${CI}" \
                  --run-tag="_sweep_lr${SLR}_ci${CI}"
              done
              ;;
            *)
              z2_train "${ARM}" "${TGT}" "${BETA}" 0 \
                --epochs="${Z2_SWEEP_EPOCHS}" --lr="${SLR}" \
                --run-tag="_sweep_lr${SLR}"
              ;;
          esac
        done
      done
    done
  done
  echo "[$(stamp)]    grep -H 'best epoch' logs/z2_*_sweep_*.log"
  echo "[$(stamp)]    Check every arm's winner is NOT at an edge of the grid,"
  echo "[$(stamp)]    on BOTH V1 and V2, before believing it. Then:"
  echo "[$(stamp)]    PROBE_PARTS=z-grid PROBE_Z2_LR_GELT=… PROBE_Z2_LR_LCNN=… \\"
  echo "[$(stamp)]      PROBE_Z2_LCNN_CONV_INIT=… bash scripts/probe_batch.sh"
fi

# ── part z-grid: the grid at the primary coupling ────────────────────────────
# V2 first and outermost: it is W-A, the calibration, and it is the gate. A
# batch cut short should leave the reading that can stop the study complete.
if wants z-grid; then
  echo "[$(stamp)] ══ part z-grid: β ${Z2_BETAS}, seeds ${Z2_SEEDS}, targets ${Z2_TARGETS}"
  for BETA in ${Z2_BETAS}; do
    for TGT in ${Z2_TARGETS}; do
      for SEED in ${Z2_SEEDS}; do
        for ARM in ${Z2_ARMS}; do
          z2_train "${ARM}" "${TGT}" "${BETA}" "${SEED}" \
            --epochs="${EPOCHS}" --lr="$(z2_arm_lr "${ARM}")"
        done
      done
    done
  done
  echo "[$(stamp)]    READ W-A (V2) FIRST. The replication at 0.745 runs only"
  echo "[$(stamp)]    if it passes:  PROBE_PARTS=z-grid PROBE_Z2_BETAS=0.745 …"
fi

echo "[$(stamp)] ══ batch done. Readings:  python scripts/probe_readings.py"
