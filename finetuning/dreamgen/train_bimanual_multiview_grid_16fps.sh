#!/bin/bash
# =============================================================================
# End-to-end (16fps FIX): re-encode segmented grid clips 30fps -> 16fps ->
# reuse T5 embeddings -> launch training FROM SCRATCH (original DiT).
#
# WHY: the original bimanual_multiview_seg run conditioned 30fps clips as 16fps
# (override_fps=16) while the dataloader grabs 93 consecutive native frames, so
# the model learned motion at ~2x the wrong speed and never converged. This run
# uses videos RE-ENCODED to a true 16fps and an experiment WITHOUT override_fps.
#
# Source (must already exist, produced by train_bimanual_multiview_grid.sh):
#   datasets/bimanual_multiview_grid_seg_dreamgen/{videos,metas,t5_xxl}
# Target (created here):
#   datasets/bimanual_multiview_grid_seg16_dreamgen/{videos,metas,t5_xxl}
#
# Usage:
#   NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid_16fps.sh
#   SKIP_REENCODE=1 NPROC=4 bash finetuning/dreamgen/train_bimanual_multiview_grid_16fps.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
#     bash finetuning/dreamgen/train_bimanual_multiview_grid_16fps.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_bimanual_multiview_seg16}"
SOURCE_DIR="${PROJECT_ROOT}/datasets/bimanual_multiview_grid_seg_dreamgen"
DATASET_DIR="${PROJECT_ROOT}/datasets/bimanual_multiview_grid_seg16_dreamgen"
SKIP_REENCODE="${SKIP_REENCODE:-0}"
NPROC="${NPROC:-4}"
WORKERS="${WORKERS:-8}"

DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"
VENV="${DREAMGEN_ROOT}/.venv"

echo "============================================"
echo "DreamGen Bimanual Multiview GRID (16fps FIX)"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Experiment:   ${EXPERIMENT}"
echo "  Source (30fps): ${SOURCE_DIR}"
echo "  Target (16fps): ${DATASET_DIR}"
echo "  GPUs (NPROC): ${NPROC}"
echo "============================================"

# ---- Step 1: Re-encode segments to true 16fps (reusing captions + T5) ----
mkdir -p "${DATASET_DIR}/videos" "${DATASET_DIR}/metas" "${DATASET_DIR}/t5_xxl"
N_SEG16_EXISTING=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
if [[ "${SKIP_REENCODE}" != "1" ]]; then
    if [[ ! -d "${SOURCE_DIR}/videos" ]]; then
        echo "ERROR: source segmented dataset not found at ${SOURCE_DIR}/videos." >&2
        echo "       Run finetuning/dreamgen/train_bimanual_multiview_grid.sh first." >&2
        exit 1
    fi
    echo ""
    echo "=== [Step 1/2] Re-encoding segments 30fps -> 16fps (existing=${N_SEG16_EXISTING}) ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/reencode_bimanual_multiview_seg_16fps.py" \
        --source "${SOURCE_DIR}" \
        --target "${DATASET_DIR}" \
        --workers "${WORKERS}"
else
    echo ""
    echo "=== [Step 1/2] Skipping re-encode (SKIP_REENCODE=1, existing=${N_SEG16_EXISTING}) ==="
fi

# Verify data + T5 exist
N_VIDEOS=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
N_METAS=$(find "${DATASET_DIR}/metas" -name "*.txt" 2>/dev/null | wc -l)
N_T5=$(find "${DATASET_DIR}/t5_xxl" -name "*.pickle" 2>/dev/null | wc -l)
echo "  Ready: ${N_VIDEOS} videos, ${N_METAS} metas, ${N_T5} T5 embeddings"
if [[ "${N_VIDEOS}" -eq 0 ]]; then
    echo "ERROR: No 16fps videos in ${DATASET_DIR}/videos/. Run without SKIP_REENCODE=1." >&2
    exit 1
fi
if [[ "${N_T5}" -eq 0 ]]; then
    echo "ERROR: No T5 embeddings in ${DATASET_DIR}/t5_xxl/. Cannot train without them." >&2
    exit 1
fi

# ---- Step 2: Launch training from scratch (original DiT, no resume) ----
echo ""
echo "=== [Step 2/2] Launching training ==="
echo "  Experiment: ${EXPERIMENT}"
echo "  NPROC: ${NPROC}"

NPROC="${NPROC}" bash "${PROJECT_ROOT}/finetuning/dreamgen/launch.sh" "${EXPERIMENT}"
