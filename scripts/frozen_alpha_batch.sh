#!/usr/bin/env bash
# The M1 ablation on the 0⁺⁺ glueball (V100, unattended; ~10 h per training).
#
# Does input-dependent attention pay on the physics task? The thesis
# (ch. 5, "Comparison with the L-CNN") proposes it as a possible explanation of
# GELT's edge over the authors' L-CNN; the M1 probe showed it on a constructed
# target only (notes/thesis_record.md §5.3), and it was never run on the 0⁺⁺.
# The arm: GELT with alpha_mode="frozen" — the softmax over the input-dependent
# score Re Tr[Q_s† K̃] is replaced by a softmax over a learned (head, offset)
# logit table, the same at every site of every configuration. Transport, value
# path, channel mix, gate, readout, inputs, loss, splits and checkpoint
# selection are those of the production GELT. Frozen drops Q_s, K and RoPE, so
# it runs at d_qkv 10: 14793 real DOFs against GELT's 15693 (0.94×), the M1
# probe's `frozen_matched` rule.
#
# ONE training per invocation, on purpose: the plan is to stop at the first
# tie. Defaults are ensemble 0, init seed 0, paired with the production GELT of
# the same ensemble (dumps/best_glueball_gelt_sm0-2-4-6[_ens<k>]_test_obars.pt).
#
#   FROZEN_ENS=0|1|2   ensemble seed (default 0)
#   FROZEN_INIT=<k>    init seed of the frozen net (default 0); paired with the
#                      GELT run of the same init seed if its dump exists, else
#                      with the init-0 GELT of that ensemble
#
# Readings, fixed before the run (paired ΔA₀ = A₀(GELT) − A₀(frozen), computed
# inside every jackknife sample by fit_glueball_overlap.py --vs=, window [2,7];
# the paired difference does not depend on the GEVP reference times):
#   ΔA₀ > 0 at ≥ 2σ   → attention pays on the 0⁺⁺: train the next ensemble
#                       (FROZEN_ENS=1, then 2) before writing it in the thesis.
#   |ΔA₀| < 2σ        → tie: stop. The thesis keeps the explanation as a
#                       hypothesis and does not report the run.
#   ΔA₀ < 0 at ≥ 2σ   → the frozen arm is better: stop, and the "possible
#                       explanation" paragraph of ch. 5 has to be revised.
# Δm_eff(1) of the pair, the unsaturated cross-check, is computed offline from
# the two dumps.
#
# Same guards as ens2_batch.sh: the batch STOPS if CUDA is not visible, and a
# training whose dump already exists is skipped (re-running only redoes the
# readout).
#
# Check first, without running anything (no GPU needed):
#   FROZEN_DRY_RUN=1 bash scripts/frozen_alpha_batch.sh
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/frozen_alpha_batch.sh > logs/frozen_alpha_batch.log 2>&1 &
#   tail -f logs/frozen_alpha_batch.log logs/frozen_ens0_init0.log
# Afterwards copy results/glueball/*_frozen_q10*_test_obars.pt into dumps/ and
# commit it; the readout is logs/frozen_readout_ens<k>_init<k>.log.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

ENS="${FROZEN_ENS:-0}"
INIT="${FROZEN_INIT:-0}"
stamp() { date "+%F %T"; }

need_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping rather than training on"
    echo "   the CPU. Check nvidia-smi, then re-run this script."
    exit 1
  fi
}

export TQDM_MININTERVAL=30

ENS_TAG=""; [ "${ENS}" != "0" ] && ENS_TAG="_ens${ENS}"
INIT_TAG=""; [ "${INIT}" != "0" ] && INIT_TAG="_init${INIT}"
NAME="frozen_ens${ENS}_init${INIT}"
FROZEN_DUMP="results/glueball/best_glueball_gelt_sm0-2-4-6_frozen_q10${ENS_TAG}${INIT_TAG}_test_obars.pt"

# The production GELT of the same ensemble (and init seed, when it exists).
GELT_DUMP="dumps/best_glueball_gelt_sm0-2-4-6${ENS_TAG}${INIT_TAG}_test_obars.pt"
if [ ! -e "${GELT_DUMP}" ]; then
  GELT_DUMP="dumps/best_glueball_gelt_sm0-2-4-6${ENS_TAG}_test_obars.pt"
fi
if [ ! -e "${GELT_DUMP}" ]; then
  echo "[$(stamp)] ** no GELT dump to pair with (${GELT_DUMP}) — git pull first."
  exit 1
fi

echo "[$(stamp)] frozen-α ablation: ensemble ${ENS}, init ${INIT}"
echo "   frozen dump: ${FROZEN_DUMP}"
echo "   paired with: ${GELT_DUMP}"

# ── the training ─────────────────────────────────────────────────────────────
# Every setting as argv (env vars do not survive every container wrapper); all
# but the two that define the arm are the production defaults.
if [ -e "${FROZEN_DUMP}" ]; then
  echo "[$(stamp)] ── training: done already — skipped"
elif [ -n "${FROZEN_DRY_RUN:-}" ]; then
  echo "[$(stamp)] ── training: WOULD RUN (no ${FROZEN_DUMP})"
else
  need_cuda
  echo "[$(stamp)] ── training ${NAME} (log: logs/${NAME}.log)"
  if python -u scripts/train_glueball.py \
      --resume=0 --ensemble-seed="${ENS}" --init-seed="${INIT}" \
      --input-smear-levels=0,2,4,6 --d-model=16 \
      --alpha-mode=frozen --d-qkv=10 > "logs/${NAME}.log" 2>&1; then
    echo "[$(stamp)]    OK  ${NAME}"
  else
    echo "[$(stamp)]    FAILED  ${NAME} (see logs/${NAME}.log)"
    exit 1
  fi
fi

# ── the readout ──────────────────────────────────────────────────────────────
# GELT first, so the paired line reads GELT − frozen.
if [ -e "${FROZEN_DUMP}" ]; then
  echo "[$(stamp)] ── readout (log: logs/frozen_readout_ens${ENS}_init${INIT}.log)"
  python -u scripts/fit_glueball_overlap.py "${GELT_DUMP}" --vs="${FROZEN_DUMP}" \
    > "logs/frozen_readout_ens${ENS}_init${INIT}.log" 2>&1 \
    || echo "[$(stamp)]    readout FAILED (see the log)"
  grep -A3 " − " "logs/frozen_readout_ens${ENS}_init${INIT}.log" | tail -n 8
fi
echo "[$(stamp)] done."
