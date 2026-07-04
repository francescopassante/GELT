#!/usr/bin/env bash
# Overnight §6.2 replication batch (~24 h unattended, V100).
#
# Closes the remaining honesty caveats of the Run-5 result without touching
# the Run-5 artifacts (non-default seeds get their own cache / checkpoint /
# dump / plot names via RUN_TAG in train_glueball.py):
#
#   phase 1  replication_ens1 — sample a FRESH ensemble (seed 1; the long
#            pole: ~2300 heat-bath+OR sweep bundles at 12^3x24) and train
#            GELT from scratch on it. An independent ensemble replicates both
#            the m_eff and the A0 statements; two independent ~2σ ΔA0
#            measurements combine to ~3σ.
#   phase 2  init_seed1       — retrain from scratch on the ORIGINAL ensemble
#   phase 3  init_seed2         with different init/batch-order seeds: shows
#            Run 5 was not a lucky initialization (the converged Rayleigh
#            loss and test m_eff/A0 should agree across inits).
#
# Phases are independent: a failure is logged and the batch continues.
# Each phase leaves a test-Ō dump in datasets/*_test_obars.pt — copy those
# to the laptop and run scripts/fit_glueball_overlap.py on each.
#
# Run (from the repo root, inside the venv):
#   nohup bash scripts/overnight_replication.sh > logs/overnight.log 2>&1 &

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

stamp() { date "+%F %T"; }

run_phase() {
  local name="$1"; shift
  echo "[$(stamp)] ── phase ${name}: $*"
  if "$@" > "logs/${name}.log" 2>&1; then
    echo "[$(stamp)]    OK  ${name}"
  else
    echo "[$(stamp)]    FAILED  ${name} (see logs/${name}.log) — continuing"
  fi
}

run_phase replication_ens1 env \
  GLUEBALL_ENSEMBLE_SEED=1 GLUEBALL_INIT_SEED=0 GLUEBALL_RESUME=0 \
  python scripts/train_glueball.py

run_phase init_seed1 env \
  GLUEBALL_ENSEMBLE_SEED=0 GLUEBALL_INIT_SEED=1 GLUEBALL_RESUME=0 \
  python scripts/train_glueball.py

run_phase init_seed2 env \
  GLUEBALL_ENSEMBLE_SEED=0 GLUEBALL_INIT_SEED=2 GLUEBALL_RESUME=0 \
  python scripts/train_glueball.py

echo "[$(stamp)] batch done. Artifacts:"
ls -l datasets/*_test_obars.pt best_glueball_gelt*.pth glueball_gelt*.png 2>/dev/null
