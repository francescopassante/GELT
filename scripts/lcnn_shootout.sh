#!/usr/bin/env bash
# The matched-parameter L-CNN shootout (V100, unattended; ~4 h sweep + 2 × ~5 h).
#
# reports/curve §"7. The matched-parameter L-CNN" — the one baseline the thesis
# names and has never run. Modelled on curve_batch.sh: phases are independent, a
# failure is logged and the batch continues, every phase has its own log, and a
# phase whose test dump already exists is skipped, so a restart resumes where
# the batch stopped.
#
#   part 0  profile one L-CNN step. Measured 2026-09-13 after the rewrite of
#           notes/lcnn_shootout.md §8: 1.29 s/step, 302 s/epoch, 7.86 GiB peak
#           with --grad-checkpoint=0 (1.21× faster than recomputing, and the
#           memory is there). BATCH_CONFIGS may NOT be moved whatever the
#           profile says — it is also the VEV-estimate knob, so a different
#           batch stops the run being comparable to the GELT runs it is
#           measured against; c_hidden is the knob if one is ever needed.
#   part 1  the hyperparameter sweep. GELT's LR and INIT_SCALE were tuned for
#           GELT; a losing L-CNN at GELT's settings is uninterpretable. Four
#           short runs (--epochs=10, ~50 min each), tagged so none of them can
#           collide with the real thing. Read the val curves, pick, then set
#           LR/INIT below before running part 2.
#   part 2  the two real trainings, ens0 (run5's ensemble) and ens1, at the
#           chosen settings — the same two ensembles every other claim uses.
#   part 3  the untrained control, free: three init seeds per ensemble,
#           eval-only. It is what makes ΔA₀(L-CNN − P) separable from "any
#           equivariant net at init is outside the classical span".
#
# Everything downstream is offline and already written: the dumps keep the
# `gelt_obar` key, so scripts/fit_glueball_overlap.py, su2_fair_fight.py,
# operator_decomposition.py and input_architecture_curve.py read them unchanged.
# meta["arch"] = "lcnn" is how a reader tells the dumps apart.
#
# Check which phases are done and which would run (no GPU needed):
#   LCNN_DRY_RUN=1 bash scripts/lcnn_shootout.sh
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/lcnn_shootout.sh > logs/lcnn_shootout.log 2>&1 &
#   tail -f logs/lcnn_shootout.log
# Afterwards copy every new results/glueball/*_test_obars.pt into dumps/.
#
# Parts run independently — the sweep must be READ before part 2 is worth
# starting, so the default is to stop after part 1:
#   LCNN_PARTS=0,1     profile + sweep (the default)
#   LCNN_PARTS=2,3     the trainings and the untrained control
#   LCNN_LR=1e-3 LCNN_INIT=1.0   the settings part 2 uses (defaults below)

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

PARTS="${LCNN_PARTS:-0,1}"
LR="${LCNN_LR:-3e-3}"
INIT="${LCNN_INIT:-1.0}"
LEVELS=0,2,4,6           # the 4-level ladder of the d_model=16 GELT net
# Gradient checkpointing off for this arm: one L-CNN block peaks at 2.6 GiB, so
# the whole step fits in 7.86 GiB of the card's 32 and the recompute is pure
# cost (1.56 s/step with it, 1.29 s without). L-CNN-only — GELT OOMs at batch 4
# without it.
CKPT_FLAG=--grad-checkpoint=0
STEM=results/glueball/best_glueball_lcnn_sm0-2-4-6

stamp() { date "+%F %T"; }
wants() { case ",${PARTS}," in *",$1,"*) return 0;; *) return 1;; esac; }

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
  if [ -n "${LCNN_DRY_RUN:-}" ]; then
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

export TQDM_MININTERVAL=30

# ── part 0: does a step fit, and what does it cost? ──────────────────────────
if wants 0; then
  echo "[$(stamp)] ══ part 0: profile one L-CNN step at BATCH_CONFIGS=6"
  run_phase lcnn_profile "" \
    python -u scripts/profile_glueball_step.py --arch=lcnn
  echo "[$(stamp)]    read logs/lcnn_profile.log: peak memory and s/step. If"
  echo "[$(stamp)]    batch 6 OOMs, the fix is c_hidden or the L-Bilin path —"
  echo "[$(stamp)]    NOT BATCH_CONFIGS (it changes the VEV estimate)."
fi

# ── part 1: the hyperparameter sweep (short runs, disposable tags) ───────────
if wants 1; then
  echo "[$(stamp)] ══ part 1: LR / init-scale sweep, 10 epochs each"
  for CFG in "3e-3 1.0" "1e-3 1.0" "3e-3 1e-4" "1e-3 1e-4"; do
    set -- ${CFG}; SLR="$1"; SINIT="$2"
    TAG="_sweep_lr${SLR}_is${SINIT}"
    run_phase "lcnn_sweep_lr${SLR}_is${SINIT}" "${STEM}${TAG}_test_obars.pt" \
      python -u scripts/train_glueball.py --arch=lcnn --resume=0 ${CKPT_FLAG} \
      --ensemble-seed=0 --input-smear-levels=${LEVELS} --epochs=10 \
      --lr=${SLR} --lcnn-init-scale=${SINIT} --run-tag=${TAG}
  done
  echo "[$(stamp)]    compare the four logs' best val Rayleigh loss, then run"
  echo "[$(stamp)]    LCNN_PARTS=2,3 LCNN_LR=<lr> LCNN_INIT=<scale> bash scripts/lcnn_shootout.sh"
fi

# ── part 2: the two real trainings ───────────────────────────────────────────
if wants 2; then
  echo "[$(stamp)] ══ part 2: full trainings at lr=${LR}, init_scale=${INIT}"
  for ENS in 0 1; do
    E=$([ "${ENS}" = 0 ] && echo "" || echo "_ens${ENS}")
    run_phase "lcnn_train_ens${ENS}" "${STEM}${E}_test_obars.pt" \
      python -u scripts/train_glueball.py --arch=lcnn --resume=0 ${CKPT_FLAG} \
      --ensemble-seed=${ENS} --input-smear-levels=${LEVELS} \
      --lr=${LR} --lcnn-init-scale=${INIT}
  done
fi

# ── part 3: the untrained control (eval-only, ~5 min each) ───────────────────
if wants 3; then
  echo "[$(stamp)] ══ part 3: untrained L-CNNs, 3 init seeds × 2 ensembles"
  for ENS in 0 1; do
    E=$([ "${ENS}" = 0 ] && echo "" || echo "_ens${ENS}")
    for SEED in 0 1 2; do
      run_phase "lcnn_rnd_ens${ENS}_s${SEED}" "${STEM}${E}_rnd${SEED}_test_obars.pt" \
        python -u scripts/train_glueball.py --arch=lcnn --random-init=1 ${CKPT_FLAG} \
        --ensemble-seed=${ENS} --init-seed=${SEED} \
        --input-smear-levels=${LEVELS} --lcnn-init-scale=${INIT}
    done
  done
fi

echo "[$(stamp)] batch done. L-CNN dumps:"
ls -l results/glueball/best_glueball_lcnn*_test_obars.pt 2>/dev/null
