#!/usr/bin/env bash
# The input↔architecture curve batch (V100, unattended; ~1.5 h + 3 × ~8 h).
#
# notes/fable5.1_10-09_audit.md WP4. Modelled on overnight_replication.sh:
# phases are independent, a failure is logged and the batch continues, and
# every phase has its own log. Sequential only — a training uses ~20 GiB of the
# 32 GB card, so nothing may run beside it.
#
#   part 1  the random trace (WP2) — 18 eval-only passes of UNTRAINED nets:
#           2 ensembles × init seeds 0,1,2 × {thin d24, 4lv d16, 7lv d24}.
#           ~5 min each. Run first on purpose: they exercise the new artifact
#           naming in minutes, before 24 h of training depend on it.
#   part 2  the trained points (WP3) — thin on run5, thin on ens1, and the
#           width control (4lv at d_model 24) on run5. The width control is
#           last: if the clock runs out, it is the one to drop.
#
# Every setting is passed as argv (env vars do not survive every container
# wrapper) and printed in each log's banner — check it. Nothing here can
# overwrite an existing checkpoint: the random nets are tagged _rnd<k>, the new
# trainings _d24 + _p5, and every training runs with --resume=0.
#
# Two guards, both added after the first batch lost the GPU mid-run (the
# driver became unreachable between two phases, and the next phase — a fresh
# process — silently fell back to the CPU at ~15× the step time):
#   - every phase first checks that CUDA is visible, and the batch STOPS if it
#     is not. Nothing here is worth running on the CPU.
#   - a phase whose test dump already exists is skipped, so a restart resumes
#     where the batch stopped. The dump is written only at the very end of a
#     run, so its presence means the phase completed.
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/curve_batch.sh > logs/curve_batch.log 2>&1 &
#   tail -f logs/curve_batch.log
# Afterwards copy every new results/glueball/*_test_obars.pt into dumps/.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%F %T"; }

need_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping the batch rather than"
    echo "   training on the CPU. Check nvidia-smi, then re-run this script: finished"
    echo "   phases are skipped."
    exit 1
  fi
}

# run_phase <name> <dump that marks it done> <command...>
run_phase() {
  local name="$1" dump="$2"; shift 2
  if [ -e "${dump}" ]; then
    echo "[$(stamp)] ── phase ${name}: done already (${dump}) — skipped"
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

# ── part 1: the random trace ─────────────────────────────────────────────────
G=results/glueball/best_glueball_gelt
for ENS in 0 1; do
  E=$([ "${ENS}" = 0 ] && echo "" || echo "_ens${ENS}")
  for SEED in 0 1 2; do
    run_phase "rnd_thin_ens${ENS}_s${SEED}" "${G}_d24${E}_rnd${SEED}_test_obars.pt" \
      python -u scripts/train_glueball.py \
      --random-init=1 --ensemble-seed=${ENS} --init-seed=${SEED} \
      --input-smear-levels=0 --d-model=24
    run_phase "rnd_4lv_ens${ENS}_s${SEED}" "${G}_sm0-2-4-6${E}_rnd${SEED}_test_obars.pt" \
      python -u scripts/train_glueball.py \
      --random-init=1 --ensemble-seed=${ENS} --init-seed=${SEED} \
      --input-smear-levels=0,2,4,6 --d-model=16
    run_phase "rnd_7lv_ens${ENS}_s${SEED}" "${G}_sm0-2-4-6-8-12-16_d24${E}_rnd${SEED}_test_obars.pt" \
      python -u scripts/train_glueball.py \
      --random-init=1 --ensemble-seed=${ENS} --init-seed=${SEED} \
      --input-smear-levels=0,2,4,6,8,12,16 --d-model=24
  done
done

# ── part 2: the trained points ───────────────────────────────────────────────
run_phase thin_run5 "${G}_d24_p5_test_obars.pt" python -u scripts/train_glueball.py \
  --resume=0 --ensemble-seed=0 --input-smear-levels=0 --d-model=24 --run-tag=_p5
run_phase thin_ens1 "${G}_d24_ens1_p5_test_obars.pt" python -u scripts/train_glueball.py \
  --resume=0 --ensemble-seed=1 --input-smear-levels=0 --d-model=24 --run-tag=_p5
run_phase width_ctrl_run5 "${G}_sm0-2-4-6_d24_p5_test_obars.pt" python -u scripts/train_glueball.py \
  --resume=0 --ensemble-seed=0 --input-smear-levels=0,2,4,6 --d-model=24 --run-tag=_p5

echo "[$(stamp)] batch done. New dumps:"
ls -l results/glueball/*_rnd*_test_obars.pt results/glueball/*_p5_test_obars.pt 2>/dev/null
