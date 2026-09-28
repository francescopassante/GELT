#!/usr/bin/env bash
# S4 — the A₁⁺⁺ projection of the learned operators (V100, unattended, eval only).
#
# Supervisor's S4 (notes/prof_notes.md): is Ō_GELT a scalar under the spatial
# cubic group with parity? scripts/cubic_projection.py evaluates each trained
# net on its test split transformed by all 48 elements of O_h and writes the
# A₁⁺⁺-projected operator as a standard dump (…_a1_test_obars.pt). Readings
# are fixed in notes/prof_notes.md §S4 — read them first.
#
#   phase 1  gen_run5     identity + one axis swap + one reflection on Run 5's
#                         4-level net (minutes). Runs the anchor, action and
#                         classical-invariance gates on the real ensemble; if it
#                         FAILS the batch stops — every later phase would fail
#                         the same gate after loading its ensemble.
#   phase 2  a1_4lv_ens{0,1,2}  all 48 on the three 4-level nets — the reading.
#   phase 3  a1_7lv_ens{0,1}    all 48 on the two 7-level nets (d_model 24).
#   phase 4  rnd4_ens{0,1,2}    the untrained 4-level nets (three per ensemble,
#                         one group each) — only with CUBIC_RND=1.
#   phase 5  lcnn_ens{0,1}      our matched L-CNN (ens0 = the p2 rerun) — only
#                         with CUBIC_LCNN=1.
#
# Cost, estimated rather than measured: one element ≈ one eval pass over the
# 400 test configurations (~3 min for a 4-level net), i.e. ~2.5 h per group
# and ~13 h for phases 2–3. A group of three nets shares the input pipeline but
# not the forward passes, and GELT's forward is dominated by its own transport,
# so phase 4 is likely comparable to phases 2–3 together; the per-element
# seconds in logs/a1_4lv_ens0.log are the measurement to extrapolate from.
# Phases 4–5 run last, so they never delay the trained nets' results. Progress is saved after every element and a restart
# resumes; a phase whose output exists is skipped.
#
# Check first, without running anything (no GPU needed):
#   CUBIC_DRY_RUN=1 bash scripts/cubic_batch.sh
#
# Run (from the repo root, inside the venv):
#   mkdir -p logs
#   nohup bash scripts/cubic_batch.sh > logs/cubic_batch.log 2>&1 &
#   tail -f logs/cubic_batch.log logs/a1_4lv_ens0.log
# Afterwards copy results/glueball/cubic/ into dumps/cubic/ and commit it: the
# _a1 dumps then run through fit_window_scan.py / su2_fair_fight.py /
# operator_decomposition.py offline.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs results/glueball/cubic

stamp() { date "+%F %T"; }

need_cuda() {
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping the batch. Check"
    echo "   nvidia-smi, then re-run this script: finished phases are skipped."
    exit 1
  fi
}

# run_phase <name> <file that marks it done> <command...>; returns the command's status
run_phase() {
  local name="$1" done_file="$2"; shift 2
  if [ -e "${done_file}" ]; then
    echo "[$(stamp)] ── phase ${name}: done already (${done_file}) — skipped"
    return 0
  fi
  if [ -n "${CUBIC_DRY_RUN:-}" ]; then
    echo "[$(stamp)] ── phase ${name}: WOULD RUN (no ${done_file})"
    return 0
  fi
  need_cuda
  echo "[$(stamp)] ── phase ${name}: $*"
  if "$@" > "logs/${name}.log" 2>&1; then
    echo "[$(stamp)]    OK  ${name}"
  else
    echo "[$(stamp)]    FAILED  ${name} (see logs/${name}.log)"
    return 1
  fi
}

export TQDM_MININTERVAL=30
OUT=results/glueball/cubic
D=dumps/best_glueball_gelt
P="python -u scripts/cubic_projection.py"

# ── phase 1: the gates on the real ensemble ──────────────────────────────────
run_phase gen_run5 "${OUT}/cubic_best_glueball_gelt_sm0-2-4-6_gen.pt" \
  ${P} --elements=generators "${D}_sm0-2-4-6_test_obars.pt" \
  || { echo "[$(stamp)] gates failed — stopping. Read logs/gen_run5.log."; exit 1; }

# ── phase 2: the three 4-level nets ──────────────────────────────────────────
for E in 0 1 2; do
  T=$([ "${E}" = 0 ] && echo "" || echo "_ens${E}")
  run_phase "a1_4lv_ens${E}" "${OUT}/best_glueball_gelt_sm0-2-4-6${T}_a1_test_obars.pt" \
    ${P} "${D}_sm0-2-4-6${T}_test_obars.pt"
done

# ── phase 3: the two 7-level nets ────────────────────────────────────────────
for E in 0 1; do
  T=$([ "${E}" = 0 ] && echo "" || echo "_ens${E}")
  run_phase "a1_7lv_ens${E}" "${OUT}/best_glueball_gelt_sm0-2-4-6-8-12-16${T}_a1_test_obars.pt" \
    ${P} "${D}_sm0-2-4-6-8-12-16${T}_test_obars.pt"
done

# ── phase 4 (opt-in): the untrained nets, three per ensemble in one group ────
if [ -n "${CUBIC_RND:-}" ]; then
  for E in 0 1 2; do
    T=$([ "${E}" = 0 ] && echo "" || echo "_ens${E}")
    run_phase "rnd4_ens${E}" "${OUT}/best_glueball_gelt_sm0-2-4-6${T}_rnd2_a1_test_obars.pt" \
      ${P} "${D}_sm0-2-4-6${T}_rnd0_test_obars.pt" "${D}_sm0-2-4-6${T}_rnd1_test_obars.pt" \
      "${D}_sm0-2-4-6${T}_rnd2_test_obars.pt"
  done
else
  echo "[$(stamp)] ── phase rnd4: not requested (CUBIC_RND=1 to add it)"
fi

# ── phase 5 (opt-in): the matched L-CNN ──────────────────────────────────────
# ens0 is the clean 20-epoch rerun in dumps/p2/, the one the record's table
# uses: the 60-epoch run in dumps/lcnn_shootout/ put 71.8% of C(0) on one
# configuration (notes/thesis_record.md §5.2) and must not be projected instead.
if [ -n "${CUBIC_LCNN:-}" ]; then
  run_phase lcnn_ens0 "${OUT}/best_glueball_lcnn_sm0-2-4-6_p2_a1_test_obars.pt" \
    ${P} dumps/p2/best_glueball_lcnn_sm0-2-4-6_p2_test_obars.pt
  run_phase lcnn_ens1 "${OUT}/best_glueball_lcnn_sm0-2-4-6_ens1_a1_test_obars.pt" \
    ${P} dumps/lcnn_shootout/best_glueball_lcnn_sm0-2-4-6_ens1_test_obars.pt
else
  echo "[$(stamp)] ── phase lcnn: not requested (CUBIC_LCNN=1 to add it)"
fi

echo "[$(stamp)] batch done. New artifacts:"
ls -l ${OUT} 2>/dev/null
