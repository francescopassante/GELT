#!/usr/bin/env bash
# Transfer-matrix distillation on the 0⁺⁺ glueball (V100, unattended; one
# student ≈ one train_glueball run, ~25 min per epoch, 20 epochs at most).
#
# The question (notes/thesis_record.md §11 P11): the trained 4-level net is
# beaten on its own loss by 20 classical loops on the same inputs (+0.021 ±
# 0.003) and its true ground-state overlap is ≈ 0.88, not the 0.93–1.0 a
# one-state fit reads (scripts/variational_headroom.py). Is that the function
# class, or the Rayleigh loss that explored it? The student has the teacher's
# architecture and starts from its weights; only the objective changes — it
# regresses the teacher's own field at t ± a onto slice t, whose exact solution
# is the Euclidean-evolved operator T^a·teacher (scripts/train_distill.py).
#
# ONE student per invocation.
#   TMD_ENS=0|1|2     ensemble seed (default 2: the one with a high-statistics
#                     teacher dump to compare against)
#   TMD_LAGS=2        the lag(s) a (default 2; ideal student: C_S(Δ) = C_T(Δ+4))
#   TMD_LOSS=connected  the offset-free loss with selection on the shape loss
#                     (default since 2026-11-20; the student is tagged _tmd<lags>_c).
#                     TMD_LOSS=raw is the first run's loss (_tmd<lags>, the R-T0
#                     of 2026-11-19, notes/raw/tmd_ens2_readout_2026-11-19.txt).
#   TMD_TAG=          extra suffix, e.g. _r2 for a second round
#   TMD_TEACHER=      teacher checkpoint (default: the production net of TMD_ENS;
#                     for a second round, the first round's student)
#   TMD_LR=3e-4 TMD_EPOCHS=20 TMD_POOL=3 TMD_P0_WEIGHT=1   passed through
#   TMD_HS=1          afterwards, evaluate student AND teacher on one fresh
#                     40k-configuration chain (high_stat_eval.py, HS_ALSO; ~the
#                     cost of the original high-statistics run, twice the
#                     forward passes). HS_N overrides the length.
#
# Readings: fixed in scripts/tmd_readout.py (R-T1 / R-T2 / R-T0 / R-T−), which
# this batch runs at the end. In one line: ΔL(student − teacher) < 0 at ≥ 3σ
# with Δm_eff(1), Δm_eff(2) < 0 at ≥ 2σ → the training principle was the limit;
# |ΔL| < 2σ → the class was. Either is a sentence the thesis cannot write today.
#
# Order of the campaign, each step only if the previous one read R-T1:
#   1. TMD_ENS=2                          (this default)
#   2. TMD_ENS=2 TMD_HS=1                 (the same student at high statistics)
#   3. TMD_ENS=0 and TMD_ENS=1            (replication on the other ensembles)
#   4. TMD_ENS=2 TMD_TAG=_r2 TMD_TEACHER=results/glueball/best_glueball_gelt_sm0-2-4-6_ens2_tmd2.pth
#                                         (a second round: power iteration)
# If step 1 reads R-T0, the one follow-up worth its GPU time is the same run
# with TMD_LAGS=1 (a stronger signal, a weaker filter); after that, stop.
#
# What actually happened (2026-11-19/20): step 1 with the raw loss read R-T0 at
# low power — the student learned the rescale c ≈ C(2)/C(0) and nothing else,
# and selection ran on a constant offset (train_distill.py's docstring). The
# lag-1 follow-up was dropped: its purification margin, r(2) − r(1)² = 0.020 of
# C(0), is smaller than lag 2's r(4) − r(2)² = 0.030, not larger. Instead: one
# rerun of step 1 with TMD_LOSS=connected (the default now), then P11 closes
# whatever it reads.
#
# Check first, without running anything (no GPU needed):
#   TMD_DRY_RUN=1 bash scripts/tmd_batch.sh
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/tmd_batch.sh > logs/tmd_batch.log 2>&1 &
# Afterwards copy results/glueball/*_tmd*_test_obars.pt into dumps/ and commit.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

ENS="${TMD_ENS:-2}"
LAGS="${TMD_LAGS:-2}"
TAG="${TMD_TAG:-}"
LOSS="${TMD_LOSS:-connected}"
LR="${TMD_LR:-3e-4}"
EPOCHS="${TMD_EPOCHS:-20}"
POOL="${TMD_POOL:-3}"
P0="${TMD_P0_WEIGHT:-1}"
HS="${TMD_HS:-0}"
DRY="${TMD_DRY_RUN:-0}"
stamp() { date "+%F %T"; }

ENS_TAG=""; [ "${ENS}" != "0" ] && ENS_TAG="_ens${ENS}"
LOSS_TAG=""; [ "${LOSS}" = "connected" ] && LOSS_TAG="_c"
LAG_TAG="_tmd$(echo "${LAGS}" | tr ',' '-')${LOSS_TAG}${TAG}"
STEM="results/glueball/best_glueball_gelt_sm0-2-4-6${ENS_TAG}"
TEACHER="${TMD_TEACHER:-${STEM}.pth}"
STUDENT="${STEM}${LAG_TAG}.pth"
DUMP="${STEM}${LAG_TAG}_test_obars.pt"
HS_DUMP="${STEM}${LAG_TAG}_hs_test_obars.pt"
NAME="tmd_ens${ENS}${LAG_TAG}"

echo "[$(stamp)] transfer-matrix distillation: ensemble ${ENS}, lags ${LAGS}, loss ${LOSS}"
echo "   teacher: ${TEACHER}"
echo "   student: ${STUDENT}"
echo "   dump   : ${DUMP}"

if [ "${DRY}" = "1" ]; then
  echo "DRY RUN — would run, in order:"
  echo "  0. python scripts/train_distill.py --tmd-smoke=1          (plumbing, seconds)"
  [ -e "${DUMP}" ] && echo "  1. (skipped: ${DUMP} exists)" || \
    echo "  1. python scripts/train_distill.py --ensemble-seed=${ENS} --tmd-lags=${LAGS} --tmd-loss=${LOSS} --tmd-lr=${LR} --tmd-epochs=${EPOCHS} --tmd-pool=${POOL} --tmd-p0-weight=${P0} --tmd-teacher=${TEACHER}${TAG:+ --tmd-tag=${TAG}}"
  echo "  2. python scripts/tmd_readout.py ${DUMP}"
  if [ "${HS}" = "1" ]; then
    [ -e "${HS_DUMP}" ] && echo "  3. (skipped: ${HS_DUMP} exists)" || \
      echo "  3. GLUEBALL_ENSEMBLE_SEED=${ENS} GLUEBALL_RUN_TAG=${LAG_TAG} HS_ALSO=${TEACHER} python scripts/high_stat_eval.py"
    echo "  4. python scripts/tmd_readout.py --hs=${HS_DUMP}"
  fi
  [ -e "${TEACHER}" ] || echo "  ** note: no teacher checkpoint at ${TEACHER} on this machine"
  exit 0
fi

if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "[$(stamp)] ** CUDA is not available — stopping rather than training on"
  echo "   the CPU. Check nvidia-smi, then re-run this script."
  exit 1
fi
if [ ! -e "${TEACHER}" ]; then
  echo "[$(stamp)] ** no teacher checkpoint at ${TEACHER} — set TMD_TEACHER."
  exit 1
fi
export TQDM_MININTERVAL=30

# ── 0. plumbing, on this machine's GPU stack ─────────────────────────────────
echo "[$(stamp)] 0. plumbing check"
if ! python scripts/train_distill.py --tmd-smoke=1 > "logs/${NAME}_smoke.log" 2>&1; then
  echo "[$(stamp)] ** the plumbing check failed — see logs/${NAME}_smoke.log"
  exit 1
fi

# ── 1. the student ───────────────────────────────────────────────────────────
if [ -e "${DUMP}" ]; then
  echo "[$(stamp)] 1. ${DUMP} exists — training skipped"
else
  echo "[$(stamp)] 1. training the student → logs/${NAME}.log"
  # Every setting as argv AND env (env vars do not survive every container wrapper).
  GLUEBALL_ENSEMBLE_SEED="${ENS}" python scripts/train_distill.py \
    --ensemble-seed="${ENS}" --tmd-lags="${LAGS}" --tmd-loss="${LOSS}" --tmd-lr="${LR}" \
    --tmd-epochs="${EPOCHS}" --tmd-pool="${POOL}" --tmd-p0-weight="${P0}" \
    --tmd-teacher="${TEACHER}" ${TAG:+--tmd-tag="${TAG}"} > "logs/${NAME}.log" 2>&1
  if [ ! -e "${DUMP}" ]; then
    echo "[$(stamp)] ** no dump was written — see logs/${NAME}.log"
    exit 1
  fi
fi

# ── 2. the paired readout on the 400 test configurations ─────────────────────
echo "[$(stamp)] 2. readout → logs/${NAME}_readout.log"
python scripts/tmd_readout.py "${DUMP}" 2>&1 | tee "logs/${NAME}_readout.log"

# ── 3–4. optional: both nets on one fresh high-statistics chain ──────────────
if [ "${HS}" = "1" ]; then
  if [ -e "${HS_DUMP}" ]; then
    echo "[$(stamp)] 3. ${HS_DUMP} exists — evaluation skipped"
  else
    echo "[$(stamp)] 3. high-statistics evaluation → logs/${NAME}_hs.log"
    GLUEBALL_ENSEMBLE_SEED="${ENS}" GLUEBALL_RUN_TAG="${LAG_TAG}" HS_ALSO="${TEACHER}" \
      python scripts/high_stat_eval.py > "logs/${NAME}_hs.log" 2>&1
  fi
  if [ -e "${HS_DUMP}" ]; then
    echo "[$(stamp)] 4. high-statistics readout → logs/${NAME}_hs_readout.log"
    TMD_LAGS="${LAGS}" python scripts/tmd_readout.py --hs="${HS_DUMP}" 2>&1 \
      | tee "logs/${NAME}_hs_readout.log"
  else
    echo "[$(stamp)] ** no high-statistics dump — see logs/${NAME}_hs.log"
  fi
fi
echo "[$(stamp)] done."
