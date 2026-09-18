#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TP="$REPO/metrics/WorldScore/worldscore/benchmark/metrics/third_party"
export PYTHONPATH="$REPO:$REPO/scripts:$TP:$TP/SEA-RAFT/core:$TP/SEA-RAFT:$TP/VFIMamba:$TP/groundingdino:$TP/sam2:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# Measured: 10 workers on an 80GB A100 sit at ~8GB each = the whole card, and 20 of 154
# clips died with OOM after all 4 retries (13% loss) on multiview 4-view-grid clips, whose
# SAM2 propagation peaks far above the median clip. The allocator's own suggestion is the
# right fix: expandable segments stop each worker's cache from fragmenting into
# unreturnable blocks, which is what turns a transient peak into a hard OOM here.
# Keep workers <= 8 as well; throughput is GPU-bound past ~5 anyway (2.1 -> 3.0 clips/min
# from 5 -> 10 workers), so the last two workers buy little and cost reliability.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$REPO/metrics/WorldScore"
exec "$REPO/models/dreamgen/.venv/bin/python" -u "$REPO/scripts/run_dense_nr.py" "$@"
