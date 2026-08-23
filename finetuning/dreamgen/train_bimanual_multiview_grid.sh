#!/bin/bash
# =============================================================================
# End-to-end: segment grid videos -> T5 embeddings -> launch training
# DreamGen bimanual multiview GRID, sliding-window segmented (10s win / 5s stride).
#
# Long grid episodes are sliced into overlapping sub-clips so the model sees
# every part of each episode (not just a random 93-frame crop). The segmented
# experiment uses override_fps=16 to condition the 30fps clips correctly.
#
# Usage:
#   NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid.sh
#   SKIP_PREP=1 SKIP_T5=1 NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid.sh
#   WIN=10 STRIDE=5 NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid.sh
#
#   # Anti-overfit v2 run (EMA on, save_iter=5k, stronger LR decay, max_iter=30k):
#   SKIP_PREP=1 SKIP_T5=1 NPROC=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
#     EXPERIMENT=predict2_video2world_training_2b_bimanual_multiview_seg_v2 \
#     bash finetuning/dreamgen/train_bimanual_multiview_grid.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_bimanual_multiview_seg}"
SOURCE_DIR="${PROJECT_ROOT}/datasets/bimanual_multiview_grid_dreamgen"
DATASET_DIR="${PROJECT_ROOT}/datasets/bimanual_multiview_grid_seg_dreamgen"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_T5="${SKIP_T5:-0}"
NPROC="${NPROC:-4}"
WIN="${WIN:-10}"
STRIDE="${STRIDE:-5}"

DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"
VENV="${DREAMGEN_ROOT}/.venv"

echo "============================================"
echo "DreamGen Bimanual Multiview GRID (segmented)"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Experiment:   ${EXPERIMENT}"
echo "  Source:       ${SOURCE_DIR}"
echo "  Segmented:    ${DATASET_DIR}"
echo "  Window:       ${WIN}s  Stride: ${STRIDE}s"
echo "  GPUs (NPROC): ${NPROC}"
echo "============================================"

# ---- Step 1: Segment the grid videos ----
N_SEG_EXISTING=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
if [[ "${SKIP_PREP}" != "1" && "${N_SEG_EXISTING}" -eq 0 ]]; then
    echo ""
    echo "=== [Step 1/3] Segmenting grid videos (sliding window) ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/segment_bimanual_multiview_grid_dreamgen.py" \
        --source "${SOURCE_DIR}" \
        --target "${DATASET_DIR}" \
        --win "${WIN}" \
        --stride "${STRIDE}" \
        --min-keep-whole "${WIN}"
else
    echo ""
    echo "=== [Step 1/3] Skipping segmentation (SKIP_PREP=${SKIP_PREP}, existing=${N_SEG_EXISTING}) ==="
fi

# Verify data exists
N_VIDEOS=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
N_METAS=$(find "${DATASET_DIR}/metas" -name "*.txt" 2>/dev/null | wc -l)
echo "  Segments: ${N_VIDEOS} videos, ${N_METAS} metas"
if [[ "${N_VIDEOS}" -eq 0 ]]; then
    echo "ERROR: No segmented videos found in ${DATASET_DIR}/videos/. Run without SKIP_PREP=1." >&2
    exit 1
fi

# ---- Step 2: Compute T5-XXL embeddings ----
if [[ "${SKIP_T5}" != "1" ]]; then
    mkdir -p "${DATASET_DIR}/t5_xxl"
    N_T5=$(find "${DATASET_DIR}/t5_xxl" -name "*.pickle" 2>/dev/null | wc -l)
    if [[ "${N_T5}" -ge "${N_VIDEOS}" ]]; then
        echo ""
        echo "=== [Step 2/3] T5 embeddings already computed (${N_T5} files), skipping ==="
    else
        echo ""
        echo "=== [Step 2/3] Computing T5-XXL embeddings (${N_T5}/${N_VIDEOS} done) ==="
        source "${VENV}/bin/activate"
        export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
        export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints"
        cd "${DREAMGEN_ROOT}"
        python -m scripts.get_t5_embeddings --dataset_path "${DATASET_DIR}"
        cd "${PROJECT_ROOT}"
        echo "  Done: $(find "${DATASET_DIR}/t5_xxl" -name "*.pickle" | wc -l) embeddings"
    fi
else
    echo ""
    echo "=== [Step 2/3] Skipping T5 embeddings (SKIP_T5=1) ==="
fi

# Final check
N_T5=$(find "${DATASET_DIR}/t5_xxl" -name "*.pickle" 2>/dev/null | wc -l)
if [[ "${N_T5}" -eq 0 ]]; then
    echo "ERROR: No T5 embeddings found. Cannot train without them." >&2
    exit 1
fi
echo "  Ready: ${N_VIDEOS} videos, ${N_T5} T5 embeddings"

# ---- Step 3: Launch training ----
echo ""
echo "=== [Step 3/3] Launching training ==="
echo "  Experiment: ${EXPERIMENT}"
echo "  NPROC: ${NPROC}"

NPROC="${NPROC}" bash "${PROJECT_ROOT}/finetuning/dreamgen/launch.sh" "${EXPERIMENT}"
