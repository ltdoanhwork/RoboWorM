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
# Some Determined images (e.g. the one behind the VLAq shell) ship without OpenCV runtime
# deps, so every worker there died instantly with
#   ImportError: libGL.so.1: cannot open shared object file
# while the launcher still printed a normal banner + PGID, so it looked like a good launch.
# Those containers have no sudo, but /pfss is shared, so libs copied from a working image
# (tmp/20261005_refactor/vendor/libgl/, override with LIBGL_DIR; 10 files: libGL, libGLX, libGLdispatch, libX11, libXext, libxcb, libXau,
# libXdmcp, libgthread-2.0, libglib-2.0) are enough -- verified import cv2 -> 4.11.0.
# Append rather than prepend so an image keeps its own newer libs in front.
LIBGL_DIR=${LIBGL_DIR:-$REPO/tmp/20261005_refactor/vendor/libgl}
if [ -d "$LIBGL_DIR" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$LIBGL_DIR"
fi
cd "$REPO/metrics/WorldScore"
exec "$REPO/models/dreamgen/.venv/bin/python" -u "$REPO/scripts/run_dense_nr.py" "$@"
