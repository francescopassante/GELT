#!/usr/bin/env bash
# The third ensemble (V100, unattended; ~1 h sampling + up to ~20 h training).
#
# Supervisor's S1 (notes/prof_notes.md) left the headline at ΔA₀ = +0.077 ±
# 0.022 (stat) ± 0.019 (fit window) on two ensembles. A third, independently
# sampled ensemble (seed 2) trained from scratch with the frozen protocol is
# the replication that tightens the statistical half. Its readings are fixed in
# notes/prof_notes.md ("Third ensemble") BEFORE this runs — read them first.
#
#   phase 1  train4_ens2   sample the seed-2 ensemble (the cache
#                          datasets/glueball_configs_…_N2000_seed2.pt) and train
#                          the 4-level GELT with every default untouched. The
#                          one phase the readings need.
#   phase 2  fairfight_ens2  the classical arms (deep, shapes, full) on ens2's
#                          test split — writes the Ō cache
#                          results/fair_fight/su2_fair_fight_obars_ens2.pt that
#                          su2_fair_fight.py (SFF_BASES) and
#                          operator_decomposition.py read offline. Needs phase 1.
#   phase 3  rnd4_ens2_s{0,1,2}  three UNTRAINED 4-level nets, eval only
#                          (~5 min each): "learned, not architectural" on a
#                          third ensemble.
#   phase 4  train7_ens2   the 7-level net (d_model 24) — only with ENS2_7LV=1.
#                          Needed only to put S2's +0.095 / +0.038 on three
#                          ensembles; another ~day of GPU.
#
# Why not overnight_replication.sh: its phase 1 retrains ens1 from scratch
# (--resume=0) and would overwrite ens1's checkpoint and dump. Everything here
# carries _ens2, so nothing on run5 or ens1 can be touched.
#
# Same guards as curve_batch.sh: the batch STOPS if CUDA is not visible (a lost
# driver once sent a phase to the CPU at ~15× the step time), and a phase whose
# output already exists is skipped, so a restart resumes where it stopped.
#
# Check first, without running anything (no GPU needed):
#   ENS2_DRY_RUN=1 bash scripts/ens2_batch.sh
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/ens2_batch.sh > logs/ens2_batch.log 2>&1 &
#   tail -f logs/ens2_batch.log logs/train4_ens2.log
# Afterwards copy the new *_ens2*_test_obars.pt and su2_fair_fight_obars_ens2.pt
# into dumps/ and commit them: every offline script then picks ens2 up.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/fair_fight

stamp() { date "+%F %T"; }

need_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping the batch rather than"
    echo "   training on the CPU. Check nvidia-smi, then re-run this script: finished"
    echo "   phases are skipped."
    exit 1
  fi
}

# run_phase <name> <file that marks it done> <command...>
run_phase() {
  local name="$1" done_file="$2"; shift 2
  if [ -e "${done_file}" ]; then
    echo "[$(stamp)] ── phase ${name}: done already (${done_file}) — skipped"
    return
  fi
  if [ -n "${ENS2_DRY_RUN:-}" ]; then
    echo "[$(stamp)] ── phase ${name}: WOULD RUN (no ${done_file})"
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
G=results/glueball/best_glueball_gelt
DUMP4="${G}_sm0-2-4-6_ens2_test_obars.pt"

# ── phase 1: the replication ─────────────────────────────────────────────────
# Every setting as argv (env vars do not survive every container wrapper), and
# every one at its default: the protocol is frozen, nothing is tuned on ens2.
run_phase train4_ens2 "${DUMP4}" python -u scripts/train_glueball.py \
  --resume=0 --ensemble-seed=2 --init-seed=0 --input-smear-levels=0,2,4,6 --d-model=16

# ── phase 2: the classical arms on ens2's test split ─────────────────────────
if [ -e "${DUMP4}" ] || [ -n "${ENS2_DRY_RUN:-}" ]; then
  run_phase fairfight_ens2 results/fair_fight/su2_fair_fight_obars_ens2.pt \
    python -u scripts/su2_fair_fight.py "${DUMP4}"
else
  echo "[$(stamp)] ── phase fairfight_ens2: skipped — phase 1 left no ${DUMP4}"
fi

# ── phase 3: the untrained controls ──────────────────────────────────────────
for SEED in 0 1 2; do
  run_phase "rnd4_ens2_s${SEED}" "${G}_sm0-2-4-6_ens2_rnd${SEED}_test_obars.pt" \
    python -u scripts/train_glueball.py \
    --random-init=1 --ensemble-seed=2 --init-seed=${SEED} \
    --input-smear-levels=0,2,4,6 --d-model=16
done

# ── phase 4 (opt-in): the 7-level net ────────────────────────────────────────
if [ -n "${ENS2_7LV:-}" ]; then
  run_phase train7_ens2 "${G}_sm0-2-4-6-8-12-16_d24_ens2_test_obars.pt" \
    python -u scripts/train_glueball.py \
    --resume=0 --ensemble-seed=2 --init-seed=0 \
    --input-smear-levels=0,2,4,6,8,12,16 --d-model=24
else
  echo "[$(stamp)] ── phase train7_ens2: not requested (ENS2_7LV=1 to add it)"
fi

echo "[$(stamp)] batch done. New artifacts:"
ls -l results/glueball/*_ens2*_test_obars.pt results/fair_fight/*ens2* 2>/dev/null
