#!/usr/bin/env bash
# The Yang–Mills vacuum wavefunctional by force matching (V100, unattended,
# ~10–12 h at the defaults; notes/thesis_record.md §11 P12).
#
# The identity (gelt/vacuum.py): the spatial links of one timeslice are
# distributed as ψ₀², and ∇ log ψ₀² = −∇S_s + E[F_temp | slice]. A scalar
# gauge-invariant network trained to reproduce the temporal force on the slice's
# links converges to the exact vacuum. Ensemble, splits and architecture are
# the glueball study's; nothing new is sampled except conditional chains.
#
# Phases (VAC_PARTS, comma list; default all, in this order):
#   smoke      the whole pipeline on an 8 × 4³ ensemble sampled on the spot
#              (results/vacuum/smoke, wiped first) — plumbing, minutes
#   replicas   conditional chains on 200 test configs: signal, noise floor,
#              the per-config replicas every ε is read on      (~90 min)
#   classical  force-matching least squares of the six classical arms, ε  (~7 h)
#   gelt       GELT, residual over `full`                       (~3–4 h)
#   frozen     the same with frozen offset weights (M1 ablation) (~3–4 h)
#   lcnn       our matched L-CNN, residual over `full`           (~1–2 h)
#   rw         exact relative weights, families noise / plane / smear, every
#              classical arm and every net above scored on them (~30 min)
#   readout    the pre-registered readings V0–V5 (vacuum_readout.py)
#   rw2        (not in the default list) V5 on more than one configuration:
#              `noise` on test configs VAC_RW2_CONFIGS (default 0,100,200,300;
#              config 0 at the default spread exists from run 1) and `plane`
#              at VAC_RW2_PLANE_STD (0.5) on the same configs — run 1's plane
#              failed G-RW at spread 1.0. Every net is scored. (~1 h)
#                VAC_PARTS=rw2,readout bash scripts/vacuum_batch.sh
#   rescore    (not in the default list) ε of the classical arms and of every
#              trained net on the *current* replica dump, nothing re-accumulated
#              or retrained (VAC_EVAL_ONLY=1) — after the replicas are rerun
#              (minutes). Run 2 of P12 (2026-11-21), after moving run 1 away:
#                mv results/vacuum/replicas_ens0.pt results/vacuum/replicas_ens0_run1.pt
#                VAC_PARTS=replicas,rescore,readout bash scripts/vacuum_batch.sh
#   alone      (not in the default list) GELT with no classical residual: does
#              the network alone reach the 24-operator ansatz?    (~3–4 h)
# A phase whose output exists is skipped (replicas resumes instead).
#
#   VAC_ENS=0          ensemble seed (0, 1 or 2 — train_glueball's caches)
#   VAC_DRY_RUN=1      list what would run, no GPU needed
#
# Check first:   VAC_DRY_RUN=1 bash scripts/vacuum_batch.sh
# Run:           mkdir -p logs && nohup bash scripts/vacuum_batch.sh > logs/vacuum_batch.log 2>&1 &
# Afterwards copy results/vacuum/{replicas,classical,net_*,rw_*}_ens<k>*.pt
# (not the .pth) back; the readout runs offline on them.

set -u
cd "$(dirname "$0")/.."
mkdir -p logs

ENS="${VAC_ENS:-0}"
PARTS="${VAC_PARTS:-smoke,replicas,classical,gelt,frozen,lcnn,rw,readout}"
DRY="${VAC_DRY_RUN:-0}"
OUT="results/vacuum"
E="_ens${ENS}"
stamp() { date "+%F %T"; }
want() { case ",${PARTS}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

NET_GELT="${OUT}/net_gelt_sm0_resfull_init0${E}"
NET_FROZEN="${OUT}/net_gelt_frozen_sm0_resfull_init0${E}"
NET_LCNN="${OUT}/net_lcnn_sm0_resfull_init0${E}"
A="--vac-ensemble-seed=${ENS}"

run() {  # run <log name> <command...>; env VAC_ENSEMBLE_SEED as well as argv
  local log="logs/vacuum_$1${E}.log"; shift
  if [ "${DRY}" = "1" ]; then echo "  would run: $* > ${log}"; return 0; fi
  echo "[$(stamp)] $* > ${log}"
  VAC_ENSEMBLE_SEED="${ENS}" "$@" > "${log}" 2>&1
  local rc=$?
  [ ${rc} -ne 0 ] && echo "[$(stamp)] ** exit ${rc} — see ${log}"
  return ${rc}
}

echo "[$(stamp)] vacuum wavefunctional, ensemble ${ENS}, parts ${PARTS}"
if [ "${DRY}" != "1" ]; then
  if ! python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "[$(stamp)] ** CUDA is not available — stopping. Check nvidia-smi."
    exit 1
  fi
  export TQDM_MININTERVAL=30
fi

if want smoke; then
  [ "${DRY}" = "1" ] || rm -rf "${OUT}/smoke"
  for s in "scripts/vacuum_replicas.py" "scripts/vacuum_classical.py" "scripts/train_vacuum.py"; do
    VAC_SMOKE=1 run "smoke_$(basename "${s}" .py)" python "${s}" --vac-smoke=1 || exit 1
  done
  VAC_SMOKE=1 run smoke_rw python scripts/vacuum_relative_weights.py --vac-smoke=1 \
    --vac-family=plane --vac-rw-nets="${OUT}/smoke/net_gelt_sm0_resfull_init0${E}.pth" || exit 1
  VAC_SMOKE=1 run smoke_readout python scripts/vacuum_readout.py --vac-smoke=1 || exit 1
fi

if want replicas; then
  run replicas python scripts/vacuum_replicas.py "${A}" || exit 1
fi

if want classical; then
  if [ -e "${OUT}/classical${E}.pt" ]; then echo "  classical: ${OUT}/classical${E}.pt exists — skipped"
  else run classical python scripts/vacuum_classical.py "${A}" || exit 1; fi
fi

train() {  # train <phase> <stem> <flags...>
  local phase="$1" stem="$2"; shift 2
  if want "${phase}"; then
    if [ -e "${stem}.pt" ]; then echo "  ${phase}: ${stem}.pt exists — skipped"
    else run "train_${phase}" python scripts/train_vacuum.py "${A}" "$@"; fi
  fi
}
train gelt   "${NET_GELT}"
train frozen "${NET_FROZEN}" --vac-alpha-mode=frozen
train lcnn   "${NET_LCNN}"   --vac-arch=lcnn
train alone  "${OUT}/net_gelt_sm0_resnone_init0${E}" --vac-residual=none

if want rescore; then
  if [ -e "${OUT}/classical${E}.pt" ] || [ "${DRY}" = "1" ]; then
    VAC_EVAL_ONLY=1 run rescore_classical python scripts/vacuum_classical.py "${A}" \
      --vac-eval-only=1 || exit 1
  else echo "  rescore: no ${OUT}/classical${E}.pt — nothing to rescore"; fi
  rescore() {  # rescore <name> <stem> <flags...>
    local name="$1" stem="$2"; shift 2
    if [ -e "${stem}.pth" ] || [ "${DRY}" = "1" ]; then
      VAC_EVAL_ONLY=1 run "rescore_${name}" python scripts/train_vacuum.py "${A}" \
        --vac-eval-only=1 "$@"
    else echo "  rescore ${name}: no ${stem}.pth — skipped"; fi
  }
  rescore gelt   "${NET_GELT}"
  rescore frozen "${NET_FROZEN}" --vac-alpha-mode=frozen
  rescore lcnn   "${NET_LCNN}"   --vac-arch=lcnn
  rescore alone  "${OUT}/net_gelt_sm0_resnone_init0${E}" --vac-residual=none
fi

NETS=""
for n in "${NET_GELT}" "${NET_FROZEN}" "${NET_LCNN}"; do
  if [ -e "${n}.pth" ] || [ "${DRY}" = "1" ]; then NETS="${NETS:+${NETS},}${n}.pth"; fi
done

if want rw; then
  for fam in noise plane smear; do
    if [ -e "${OUT}/rw_${fam}_cfg0${E}.pt" ]; then echo "  rw ${fam}: exists — skipped"
    else run "rw_${fam}" python scripts/vacuum_relative_weights.py "${A}" \
      --vac-family="${fam}" ${NETS:+--vac-rw-nets="${NETS}"}; fi
  done
fi

if want rw2; then
  PSTD="${VAC_RW2_PLANE_STD:-0.5}"
  for k in $(echo "${VAC_RW2_CONFIGS:-0,100,200,300}" | tr ',' ' '); do
    if [ -e "${OUT}/rw_noise_cfg${k}${E}.pt" ]; then echo "  rw2 noise cfg${k}: exists — skipped"
    else run "rw_noise_cfg${k}" python scripts/vacuum_relative_weights.py "${A}" \
      --vac-family=noise --vac-rw-config="${k}" ${NETS:+--vac-rw-nets="${NETS}"}; fi
    if [ -e "${OUT}/rw_plane_cfg${k}_std${PSTD}${E}.pt" ]; then
      echo "  rw2 plane cfg${k} std${PSTD}: exists — skipped"
    else run "rw_plane_cfg${k}_std${PSTD}" python scripts/vacuum_relative_weights.py "${A}" \
      --vac-family=plane --vac-rw-config="${k}" --vac-rw-target-std="${PSTD}" \
      ${NETS:+--vac-rw-nets="${NETS}"}; fi
  done
fi

if want readout; then
  if [ "${DRY}" = "1" ]; then echo "  would run: python scripts/vacuum_readout.py ${A}"
  else python scripts/vacuum_readout.py "${A}" 2>&1 | tee "logs/vacuum_readout${E}.log"; fi
fi
echo "[$(stamp)] done."
