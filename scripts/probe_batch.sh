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
#   part 5  R-B', the matched-parameter frozen arm. **Conditional**: run it only
#           if R-B favours GELT at > 2σ, when "capacity or mechanism?" becomes a
#           real question. Not in the default parts for that reason.
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
#   PROBE_PARTS=2,3,4            the grid, after the sweep has been read
#   PROBE_LR_GELT=… PROBE_LR_FROZEN=… PROBE_LR_LCNN=…   per-arm LRs for part 2+
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

stamp() { date "+%F %T"; }
wants() { case ",${PARTS}," in *",$1,"*) return 0;; *) return 1;; esac; }

arm_lr() {
  case "$1" in
    gelt) echo "${LR_GELT}";;
    frozen|frozen_matched) echo "${LR_FROZEN}";;
    lcnn|lcnn_norm) echo "${LR_LCNN}";;
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
  echo "[$(stamp)] ══ part 2: T0 (calibration) — gelt, frozen, lcnn"
  for ENS in ${ENSEMBLES}; do
    for SEED in ${SEEDS}; do
      for ARM in gelt frozen lcnn; do
        train "${ARM}" T0 "${ENS}" "${SEED}"
      done
    done
  done
  echo "[$(stamp)]    read R-A now: python scripts/probe_readings.py"
fi

# ── part 3: T1 and T2 — the primary grid ─────────────────────────────────────
if wants 3; then
  echo "[$(stamp)] ══ part 3: T1 and T2 — gelt, frozen, lcnn, lcnn_norm"
  for ENS in ${ENSEMBLES}; do
    for SEED in ${SEEDS}; do
      for TGT in T2 T1; do
        for ARM in gelt frozen lcnn lcnn_norm; do
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

echo "[$(stamp)] ══ batch done. Readings:  python scripts/probe_readings.py"
