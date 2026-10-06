#!/usr/bin/env bash
# Same venv + vendored PYTHONPATH as run_dense_nr.sh, for the E2 text-shuffle control.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
TP="$REPO/metrics/WorldScore/worldscore/benchmark/metrics/third_party"
export PYTHONPATH="$REPO:$REPO/scripts:$TP:$TP/SEA-RAFT/core:$TP/SEA-RAFT:$TP/VFIMamba:$TP/groundingdino:$TP/sam2:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$REPO/metrics/WorldScore"
exec "$REPO/models/dreamgen/.venv/bin/python" -u "$REPO/scripts/metrics/_exp_text_shuffle.py" "$@"
