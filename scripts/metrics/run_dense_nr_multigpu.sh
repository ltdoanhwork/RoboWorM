#!/usr/bin/env bash
# Launch a MULTI-GPU box as several consecutive machine indices of one dense-NR wave.
#
#   MACHINES=8 bash scripts/metrics/run_dense_nr_multigpu.sh <first index> <n gpus>
#   # e.g. box with 4 GPUs taking indices 4,5,6,7 of an 8-machine wave:
#   MACHINES=8 bash scripts/metrics/run_dense_nr_multigpu.sh 4 4
#
# WHY A 4-GPU BOX IS 4 MACHINES, NOT 1
#   The partition is `rows[MACHINE*W + shard :: W*MACHINES]` with W = workers per GPU.
#   A machine index owns exactly W shard slots, i.e. ONE GPU's worth. Running 4 GPUs under
#   a single index would need W=32 on that box and W=8 elsewhere -- but W is a factor of the
#   global `stride`, so a box with a different W computes a different stride and the
#   "partition" stops partitioning: some rows are owned by nobody, others twice, and since
#   every write is idempotent NOTHING ERRORS to tell you.
#
#   So: same W everywhere, and a 4-GPU box claims 4 consecutive indices. With 4 single-GPU
#   boxes + 1 four-GPU box, that is MACHINES=8 (indices 0-3 single, 4-7 on the big box).
#
# MEASURED: 8 workers/GPU is safe on a 32-core box even with 4 GPUs
#   Per worker on this workload: ~0.93 core, ~3.0 GB RSS, and the GPU sits at 99-100%
#   util with 8 workers using 57 GB of 80 GB VRAM. So 4 GPUs x 8 workers = 32 workers needs
#   ~30 of 32 cores and ~96 GB RAM -- fits, and the work is GPU-bound (util 99%+), not
#   CPU-bound, so cores are not the limit. VRAM is per-GPU, so 57 GB/GPU is unaffected by
#   running four of them.
#
#   If the box has FEWER cores than 0.93 * 8 * n_gpus, lower WORKERS -- but then EVERY box
#   in the wave must use the same lower value, and MACHINES must be recomputed.
#
# ENV
#   MACHINES  total machine indices in the wave (single-GPU boxes + one per GPU here)
#   WORKERS   workers per GPU (default 8). MUST be identical on every box in the wave.
#   RUN       shared run dir
#   FIRST_GPU CUDA index of the first GPU to use (default 0)
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
FIRST="${1:?usage: MACHINES=<N> bash scripts/metrics/run_dense_nr_multigpu.sh <first index> <n gpus>}"
NGPU="${2:?usage: MACHINES=<N> bash scripts/metrics/run_dense_nr_multigpu.sh <first index> <n gpus>}"
MACHINES="${MACHINES:?set MACHINES to the TOTAL machine-index count of the wave}"
WORKERS="${WORKERS:-8}"
FIRST_GPU="${FIRST_GPU:-0}"
RUN="${RUN:-$REPO/metrics/WorldScore/results/dense_nr_full}"
TARGET="${TARGET:-300}"

LAST=$((FIRST + NGPU - 1))
if [ "$LAST" -ge "$MACHINES" ]; then
  echo "ERROR: indices $FIRST..$LAST do not fit in [0, $MACHINES)." >&2
  echo "  A 4-GPU box needs 4 indices, so MACHINES must count every GPU in the wave:" >&2
  echo "  4 single-GPU boxes + this 4-GPU box  ->  MACHINES=8" >&2
  exit 1
fi

# Cores are the one resource that is shared across this box's GPUs.
CORES=$(nproc)
NEED=$(awk "BEGIN{printf \"%d\", 0.93 * $WORKERS * $NGPU + 0.5}")
echo "cores $CORES, estimated need $NEED (0.93/worker x $WORKERS workers x $NGPU GPUs)"
if [ "$NEED" -gt "$CORES" ]; then
  echo "WARNING: oversubscribed. Lower WORKERS on EVERY box in the wave (and recompute" >&2
  echo "         MACHINES), or use fewer GPUs here. Set FORCE=1 to proceed." >&2
  [ "${FORCE:-0}" = "1" ] || exit 1
fi

# Row list: must match the single-GPU boxes EXACTLY (see the wave-mode lock in
# _run_dense_nr_full.sh). This used to hardcode ONLY_IDS="" = whole manifest, while
# run_dense_nr_grid.sh defaults to TARGET=300 = a ~51k-row subset, so the documented
# "4 boxes via grid + this box via multigpu" plan had the two halves walking different
# lists and silently failing to partition. Honour the same FULL / TARGET knobs here.
IDS=""
if [ "${FULL:-0}" != "1" ]; then
  IDS="$RUN/manifest_grid_target${TARGET}.json"
  if [ ! -f "$IDS" ]; then
    echo "ERROR: no id subset at $IDS" >&2
    echo "  Either build it first (any box):" >&2
    echo "    models/dreamgen/.venv/bin/python scripts/metrics/nr_table_coverage.py \\" >&2
    echo "      --run $RUN --target $TARGET --ids-out $IDS" >&2
    echo "  or run the whole manifest with FULL=1 (must match the other boxes)." >&2
    exit 1
  fi
  echo "subset: $IDS  sha $(sha256sum "$IDS" | cut -c1-12)  <- must match every box"
else
  echo "row list: WHOLE manifest (FULL=1)  <- every other box must also use FULL=1"
fi

for i in $(seq 0 $((NGPU - 1))); do
  m=$((FIRST + i))
  g=$((FIRST_GPU + i))
  echo "=== machine index $m  ->  GPU $g ==="
  # Each index gets its own worker_m<m>_*.log, so the logs never collide.
  MACHINE="$m" MACHINES="$MACHINES" GPU="$g" WORKERS="$WORKERS" RUN="$RUN" ONLY_IDS="$IDS" \
    "$REPO/scripts/metrics/_run_dense_nr_full.sh"
done
echo
echo "launched $NGPU machine indices ($FIRST..$LAST) of $MACHINES on GPUs $FIRST_GPU..$((FIRST_GPU + NGPU - 1))"
echo "verify: bash scripts/metrics/dense_nr_status.sh   (MACHINES=$MACHINES)"
