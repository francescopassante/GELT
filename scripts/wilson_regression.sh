#!/usr/bin/env bash
# Fig. 3 of PRL 128, 032003, end to end: one L-CNN per Wilson-loop shape.
#
# Phases, each skipped if its artifacts already exist:
#   data    the SU(2) 8x8 datasets (one batched thermalisation, ~2e3 sweeps)
#   train   four networks, one per loop, at SM Table V's own architectures
#   figure  the four scatter panels and the MSE table (offline, seconds)
#
# The size is the largest Table V gives for that loop -- W(1x1) has only one --
# because the MSE printed in a Fig. 3 panel is the best over the sizes the
# Letter tried, not one per size. The three sizes and the ten random
# initialisations of SM SS VI.A are a *sweep*, not the result; reaching for
# them is
#
#   WR_SEEDS=10 WR_SIZES=small,medium,large bash scripts/wilson_regression.sh
#
# and its only effect on the headline is to make a bad initialisation less
# likely to be the number quoted. The figure selects on validation loss, never
# on the test MSE.
#
#   WR_DRY_RUN=1 bash scripts/wilson_regression.sh      # list what would run
#   WR_PARTS=figure bash scripts/wilson_regression.sh   # replot from the dumps
#
# Unlike the glueball and probe batches this does *not* refuse to run on a CPU:
# the smallest model here is twelve parameters on a 64-site lattice. The data
# phase and the W(4x4) network are the only parts that want a GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

# Same convention as lcnn_shootout.sh: the interpreter on PATH, overridable.
# A hardcoded .venv/bin/python is a local-machine assumption and breaks on any
# box where the environment is a module load or a conda env.
#   PY=.venv/bin/python bash scripts/wilson_regression.sh
PY="${PY:-python}"
PARTS=${WR_PARTS:-data,train,figure}
SEEDS=${WR_SEEDS:-1}
TARGETS=${WR_TARGETS:-W11,W12,W22,W44}
SIZES=${WR_SIZES:-best}
IMPL=${WR_CONV_IMPL:-exact}
DRY=${WR_DRY_RUN:-0}
DUMPS=${WR_DUMP_DIR:-dumps}
DATA=${WR_DATA_DIR:-datasets/wilson1p1d}

run() {
  echo "+ $*"
  [ "$DRY" = "1" ] || "$@"
}
has_part() { case ",$PARTS," in *",$1,"*) return 0;; *) return 1;; esac; }

# The sizes to train for one loop. "best" is the largest architecture Table V
# gives for it, which is where the paper's reported MSE comes from; the table
# has a single W(1x1) network (L-CB(1,1,1), 12 parameters) and nothing to
# choose.
sizes_for() {
  if [ "$SIZES" != "best" ]; then echo "${SIZES//,/ }"; return; fi
  if [ "$1" = "W11" ]; then echo small; else echo large; fi
}

if has_part data; then
  if [ -f "$DATA/train_L8.pt" ] && [ "$DRY" != "1" ]; then
    echo "= data: $DATA/train_L8.pt exists, skipping"
  else
    run $PY -u scripts/wilson_regression_data.py
  fi
fi

IFS=',' read -ra TGTS <<< "$TARGETS"

if has_part train; then
  for t in "${TGTS[@]}"; do
    for s in $(sizes_for "$t"); do
      # Table V has no W(1x1) beyond 'small': nothing to sweep there.
      [ "$t" = "W11" ] && [ "$s" != "small" ] && continue
      for seed in $(seq 0 $((SEEDS - 1))); do
        d="$DUMPS/wilson_regression_${t}_${s}_${IMPL}_s${seed}.pt"
        if [ -f "$d" ] && [ "$DRY" != "1" ]; then
          echo "= $d exists, skipping"; continue
        fi
        run env WR_TARGET=$t WR_SIZE=$s WR_CONV_IMPL=$IMPL WR_SEED=$seed \
            $PY -u scripts/wilson_regression.py
      done
    done
  done
fi

has_part figure && run $PY -u scripts/wilson_regression_figure.py
echo "done."
