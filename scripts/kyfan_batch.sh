#!/usr/bin/env bash
# The learned K-state spectrum on the 0⁺⁺ ensemble: the Ky Fan loss (V100,
# unattended; ~10 h for the training, minutes for the untrained twin).
#
# train_glueball.py with GLUEBALL_N_STATES=K: the readout gets K outputs per
# site and the Rayleigh quotient becomes the Ky Fan trace Tr[C(0)⁻¹ C(Δ)] =
# Σ_{n<K} λ_n, maximal only on the span of the K lightest states the network
# can reach (gelt.glueball.kyfan_trace, tests/test_kyfan.py). Default K = 4:
# the net is not O_h-invariant, so the 2⁺⁺ E doublet can enter the span below
# the 0⁺⁺*, and the excited A1 reading (RS3) needs two A1-labelled levels.
#
# ONE ensemble per invocation, trained from scratch (no GLUEBALL_WARM_FROM).
#   KF_ENS=0|1|2   ensemble seed (default 0)
#   KF_K=4         number of states K (default 4)
#   KF_DRY_RUN=1   list what would run, no GPU needed
#
# Phases, each skipped if its dump exists:
#   1. the K-state training                   (~10 h, the production schedule)
#   2. its untrained twin, --random-init=1    (minutes; RS1 needs it)
#   3. scripts/spectrum_readout.py on the trained dump — the readings RS1, RS3,
#      RS4 of notes/thesis_record.md §11 P13, applied by the script.
#
# Run (from the repo root, inside the venv; on a multi-GPU node pick the card):
#   mkdir -p logs
#   CUDA_VISIBLE_DEVICES=1 nohup bash scripts/kyfan_batch.sh > logs/kyfan_batch.log 2>&1 &
# Afterwards copy results/glueball/*_k<K>*_test_obars.pt into dumps/ and commit.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

ENS="${KF_ENS:-0}"
K="${KF_K:-4}"
DRY="${KF_DRY_RUN:-0}"
stamp() { date "+%F %T"; }

ENS_TAG=""; [ "${ENS}" != "0" ] && ENS_TAG="_ens${ENS}"
STEM="results/glueball/best_glueball_gelt_sm0-2-4-6_k${K}${ENS_TAG}"
DUMP="${STEM}_test_obars.pt"
TWIN="${STEM}_rnd0_test_obars.pt"
NAME="kyfan_k${K}_ens${ENS}"
# Every setting as argv (env vars do not survive every container wrapper); all
# but --n-states are the production defaults of the 4-level net.
COMMON="--ensemble-seed=${ENS} --n-states=${K} --input-smear-levels=0,2,4,6 --d-model=16"

echo "[$(stamp)] Ky Fan K-state basis: K = ${K}, ensemble ${ENS}"
echo "   trained dump: ${DUMP}"
echo "   twin dump   : ${TWIN}"

if [ "${DRY}" != "1" ]; then
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping rather than training on"
    echo "   the CPU. Check nvidia-smi, then re-run this script."
    exit 1
  fi
  export TQDM_MININTERVAL=30
fi

# ── 1. the training ──────────────────────────────────────────────────────────
if [ -e "${DUMP}" ]; then
  echo "[$(stamp)] 1. ${DUMP} exists — training skipped"
elif [ "${DRY}" = "1" ]; then
  echo "  1. would run: python -u scripts/train_glueball.py --resume=0 ${COMMON} > logs/${NAME}.log"
else
  echo "[$(stamp)] 1. training → logs/${NAME}.log"
  # shellcheck disable=SC2086
  if ! python -u scripts/train_glueball.py --resume=0 ${COMMON} > "logs/${NAME}.log" 2>&1 \
      || [ ! -e "${DUMP}" ]; then
    echo "[$(stamp)] ** training failed or wrote no dump — see logs/${NAME}.log"
    exit 1
  fi
fi

# ── 2. the untrained twin ────────────────────────────────────────────────────
if [ -e "${TWIN}" ]; then
  echo "[$(stamp)] 2. ${TWIN} exists — skipped"
elif [ "${DRY}" = "1" ]; then
  echo "  2. would run: python -u scripts/train_glueball.py --random-init=1 ${COMMON} > logs/${NAME}_rnd0.log"
else
  echo "[$(stamp)] 2. untrained twin → logs/${NAME}_rnd0.log"
  # shellcheck disable=SC2086
  python -u scripts/train_glueball.py --random-init=1 ${COMMON} > "logs/${NAME}_rnd0.log" 2>&1 \
    || echo "[$(stamp)] ** twin failed — RS1 will not be read (see logs/${NAME}_rnd0.log)"
fi

# ── 3. the readout ───────────────────────────────────────────────────────────
if [ "${DRY}" = "1" ]; then
  echo "  3. would run: python scripts/spectrum_readout.py ${DUMP} --out=results/glueball/spectrum_readout_k${K}${ENS_TAG}"
  for f in "dumps/su2_fair_fight_obars_$( [ "${ENS}" = 0 ] && echo run5 || echo "ens${ENS}").pt" \
           "dumps/best_glueball_gelt_sm0-2-4-6${ENS_TAG}_test_obars.pt"; do
    [ -e "${f}" ] || echo "  ** note: ${f} missing — git pull first"
  done
  exit 0
fi
echo "[$(stamp)] 3. readout → logs/${NAME}_readout.log"
python scripts/spectrum_readout.py "${DUMP}" --out="results/glueball/spectrum_readout_k${K}${ENS_TAG}" \
  2>&1 | tee "logs/${NAME}_readout.log"
echo "[$(stamp)] done."
