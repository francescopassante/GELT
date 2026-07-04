#!/usr/bin/env bash
# Overnight §6.2 replication batch (~24 h unattended, V100).
#
# Closes the "one ensemble" caveat of the Run-5 result without touching the
# Run-5 artifacts (non-default seeds get their own cache / checkpoint / dump /
# plot names via RUN_TAG in train_glueball.py):
#
#   phase 1  replication_ens1 — sample a FRESH ensemble (seed 1; the long
#            pole: ~2300 heat-bath+OR sweep bundles at 12^3x24) and train
#            GELT from scratch on it. An independent ensemble replicates both
#            the m_eff and the A0 statements.
#   phase 2  replication_ens2 — same again with seed 2 (bonus): three fully
#            independent A0 measurements (Run 5 + ens1 + ens2) combine to
#            ~3.5σ on the overlap headline. If the clock runs out mid-phase,
#            phase 1's results are already on disk.
#
# (Init-seed robustness runs were considered and dropped: a converged loss
# that saturates the transfer-matrix floor is already init-blind evidence —
# any init reaching the variational bound has found the same optimum.)
#
# Phases are independent: a failure is logged and the batch continues.
# Each phase leaves a test-Ō dump in datasets/*_test_obars.pt — copy those
# to the laptop and run scripts/fit_glueball_overlap.py on each.
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/overnight_replication.sh > logs/overnight.log 2>&1 &
#
# Watch progress (sampler sweep bar, epoch/batch bars, per-epoch losses):
#   tail -f logs/overnight.log logs/replication_ens1.log
# python -u keeps prints unbuffered; TQDM_MININTERVAL=30 makes every bar
# write a fresh line every ~30 s instead of spamming the log.

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
  TQDM_MININTERVAL=30 \
  python -u scripts/train_glueball.py

run_phase replication_ens2 env \
  GLUEBALL_ENSEMBLE_SEED=2 GLUEBALL_INIT_SEED=0 GLUEBALL_RESUME=0 \
  TQDM_MININTERVAL=30 \
  python -u scripts/train_glueball.py

echo "[$(stamp)] batch done. Artifacts:"
ls -l datasets/*_test_obars.pt best_glueball_gelt*.pth glueball_gelt*.png 2>/dev/null
