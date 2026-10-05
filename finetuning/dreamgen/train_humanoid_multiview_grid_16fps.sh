#!/bin/bash
# =============================================================================
# End-to-end (16fps FIX): prepare 2x2 grid -> segment long clips ->
# compute T5 embeddings -> re-encode segments 10fps -> 16fps ->
# launch training FROM SCRATCH (original DiT).
#
# WHY: the humanoid multiview grid dataset is short (clips 3.2-31.6s) and was
# stitched at 10fps from 30fps source, so at num_frames=93 only 132/248 clips
# survive the dataloader's frame filter, and clips are conditioned at native
# 10fps while the base ckpt is model-480p-16fps. Segmenting long clips +
# re-encoding to a true 16fps recovers ~204/248 clips and fixes the fps
# mismatch. The seg16 experiment drops override_fps (fps on disk is truly 16).
# Mirrors finetuning/dreamgen/train_bimanual_multiview_grid_16fps.sh.
#
# Pipeline:
#   datasets/humanoid/multiview (30fps, 4 cams)
#     -> datasets/humanoid_multiview_grid_dreamgen        (10fps 2x2 grid; reused if present)
#     -> datasets/humanoid_multiview_grid_seg_dreamgen    (10s win / 5s stride)
#        (+ T5 embeddings)
#     -> datasets/humanoid_multiview_grid_seg16_dreamgen  (true 16fps; T5 copied)
#     -> train predict2_video2world_training_2b_humanoid_multiview_seg16
#
# Usage:
#   NPROC=4 bash finetuning/dreamgen/train_humanoid_multiview_grid_16fps.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4 \
#     bash finetuning/dreamgen/train_humanoid_multiview_grid_16fps.sh
#   SKIP_GRID=1 SKIP_SEGMENT=1 SKIP_T5=1 SKIP_REENCODE=1 NPROC=4 \
#     bash finetuning/dreamgen/train_humanoid_multiview_grid_16fps.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_humanoid_multiview_seg16}"
SOURCE_RAW="${PROJECT_ROOT}/datasets/humanoid/multiview"
GRID_DIR="${PROJECT_ROOT}/datasets/humanoid_multiview_grid_dreamgen"
SEG_DIR="${PROJECT_ROOT}/datasets/humanoid_multiview_grid_seg_dreamgen"
DATASET_DIR="${PROJECT_ROOT}/datasets/humanoid_multiview_grid_seg16_dreamgen"

SKIP_GRID="${SKIP_GRID:-0}"
SKIP_SEGMENT="${SKIP_SEGMENT:-0}"
SKIP_T5="${SKIP_T5:-0}"
SKIP_REENCODE="${SKIP_REENCODE:-0}"
NPROC="${NPROC:-4}"
WORKERS="${WORKERS:-8}"
WIN="${WIN:-10}"
STRIDE="${STRIDE:-5}"
MIN_KEEP_WHOLE="${MIN_KEEP_WHOLE:-10}"

DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"
VENV="${DREAMGEN_ROOT}/.venv"

# --- Environment (needed for prep/T5 steps; launch.sh sets its own too) ---
source "${VENV}/bin/activate"
export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints/base"
export IMAGINAIRE_OUTPUT_ROOT="${PROJECT_ROOT}/checkpoints/finetuned/dreamgen"
export OMP_NUM_THREADS=8

echo "============================================"
echo "DreamGen Humanoid Multiview GRID (16fps FIX)"
echo "  Project root:   ${PROJECT_ROOT}"
echo "  Experiment:     ${EXPERIMENT}"
echo "  Raw source:     ${SOURCE_RAW}"
echo "  Grid (10fps):   ${GRID_DIR}"
echo "  Seg (10fps):    ${SEG_DIR}"
echo "  Seg16 (16fps):  ${DATASET_DIR}"
echo "  GPUs (NPROC):   ${NPROC}"
echo "============================================"

# ---- Step 1: Prepare 2x2 grid dataset (reuse if already present) ----
N_GRID_EXISTING=$(find "${GRID_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
if [[ "${SKIP_GRID}" != "1" && "${N_GRID_EXISTING}" -eq 0 ]]; then
    echo ""
    echo "=== [Step 1/5] Preparing 2x2 grid dataset ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/prepare_humanoid_multiview_grid_dreamgen.py" \
        --source "${SOURCE_RAW}" \
        --target "${GRID_DIR}"
else
    echo ""
    echo "=== [Step 1/5] Skipping grid prep (SKIP_GRID=${SKIP_GRID}, existing=${N_GRID_EXISTING}) ==="
fi
N_GRID=$(find "${GRID_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
echo "  Grid videos: ${N_GRID}"
if [[ "${N_GRID}" -eq 0 ]]; then
    echo "ERROR: No grid videos in ${GRID_DIR}/videos/. Check raw source." >&2
    exit 1
fi

# ---- Step 2: Segment long grid clips into overlapping windows ----
mkdir -p "${SEG_DIR}/videos" "${SEG_DIR}/metas"
N_SEG_EXISTING=$(find "${SEG_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
if [[ "${SKIP_SEGMENT}" != "1" ]]; then
    echo ""
    echo "=== [Step 2/5] Segmenting grid clips (win=${WIN}s stride=${STRIDE}s, existing=${N_SEG_EXISTING}) ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/segment_humanoid_multiview_grid_dreamgen.py" \
        --source "${GRID_DIR}" \
        --target "${SEG_DIR}" \
        --win "${WIN}" \
        --stride "${STRIDE}" \
        --min-keep-whole "${MIN_KEEP_WHOLE}"
else
    echo ""
    echo "=== [Step 2/5] Skipping segment (SKIP_SEGMENT=1, existing=${N_SEG_EXISTING}) ==="
fi
N_SEG=$(find "${SEG_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
echo "  Segment videos: ${N_SEG}"
if [[ "${N_SEG}" -eq 0 ]]; then
    echo "ERROR: No segment videos in ${SEG_DIR}/videos/." >&2
    exit 1
fi

# ---- Step 3: Compute T5-XXL embeddings on the segmented dataset ----
# Captions are task+layout level (fps-independent); computing here lets the
# re-encode step copy the pickles verbatim to the seg16 dataset.
if [[ "${SKIP_T5}" != "1" ]]; then
    echo ""
    echo "=== [Step 3/5] Computing T5-XXL embeddings on segments ==="
    ( cd "${DREAMGEN_ROOT}" && python -m scripts.get_t5_embeddings --dataset_path "${SEG_DIR}" )
else
    echo ""
    echo "=== [Step 3/5] Skipping T5 (SKIP_T5=1) ==="
fi
N_T5_SEG=$(find "${SEG_DIR}/t5_xxl" -name "*.pickle" 2>/dev/null | wc -l || true)
echo "  T5 embeddings (seg): ${N_T5_SEG}"
if [[ "${N_T5_SEG}" -eq 0 ]]; then
    echo "ERROR: No T5 embeddings in ${SEG_DIR}/t5_xxl/. Cannot proceed." >&2
    exit 1
fi

# ---- Step 4: Re-encode segments to true 16fps (reusing captions + T5) ----
mkdir -p "${DATASET_DIR}/videos" "${DATASET_DIR}/metas" "${DATASET_DIR}/t5_xxl"
N_SEG16_EXISTING=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
if [[ "${SKIP_REENCODE}" != "1" ]]; then
    echo ""
    echo "=== [Step 4/5] Re-encoding segments 10fps -> 16fps (existing=${N_SEG16_EXISTING}) ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/reencode_humanoid_multiview_seg_16fps.py" \
        --source "${SEG_DIR}" \
        --target "${DATASET_DIR}" \
        --workers "${WORKERS}"
else
    echo ""
    echo "=== [Step 4/5] Skipping re-encode (SKIP_REENCODE=1, existing=${N_SEG16_EXISTING}) ==="
fi

# Verify data + T5 exist
N_VIDEOS=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l || true)
N_METAS=$(find "${DATASET_DIR}/metas" -name "*.txt" 2>/dev/null | wc -l || true)
N_T5=$(find "${DATASET_DIR}/t5_xxl" -name "*.pickle" 2>/dev/null | wc -l || true)
echo "  Ready: ${N_VIDEOS} videos, ${N_METAS} metas, ${N_T5} T5 embeddings"
if [[ "${N_VIDEOS}" -eq 0 ]]; then
    echo "ERROR: No 16fps videos in ${DATASET_DIR}/videos/. Run without SKIP_REENCODE=1." >&2
    exit 1
fi
if [[ "${N_T5}" -eq 0 ]]; then
    echo "ERROR: No T5 embeddings in ${DATASET_DIR}/t5_xxl/. Cannot train without them." >&2
    exit 1
fi

# ---- Step 5: Launch training from scratch (original DiT, no resume) ----
echo ""
echo "=== [Step 5/5] Launching training ==="
echo "  Experiment: ${EXPERIMENT}"
echo "  NPROC: ${NPROC}"

NPROC="${NPROC}" bash "${PROJECT_ROOT}/finetuning/dreamgen/launch.sh" "${EXPERIMENT}"
