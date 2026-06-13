#!/bin/bash
# =============================================================================
# End-to-end: prepare → T5 embeddings → train
# DreamGen humanoid GR1 single ego view @ 16fps (fps-normalized from 20fps).
#
# Data source: datasets/humanoid/singleview/{makovian,non_makovian}
#   single ego camera observation.images.ego_view_freq20 (GR1T2 humanoid).
# Prepared into datasets/humanoid_dreamgen_16fps (transcoded 20fps → 16fps).
#
# This script runs everything in sequence — no external dependencies.
#
# Usage:
#   NPROC=8 bash finetuning/dreamgen/train_humanoid_16fps.sh
#   SKIP_PREP=1 SKIP_T5=1 NPROC=8 bash finetuning/dreamgen/train_humanoid_16fps.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="predict2_video2world_training_2b_humanoid_singleview_16fps"
DATASET_DIR="${PROJECT_ROOT}/datasets/humanoid_dreamgen_16fps"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_T5="${SKIP_T5:-0}"
NPROC="${NPROC:-4}"

DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"
VENV="${DREAMGEN_ROOT}/.venv"

echo "============================================"
echo "DreamGen Humanoid Singleview 16fps"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Experiment:   ${EXPERIMENT}"
echo "  Dataset:      ${DATASET_DIR}"
echo "  GPUs (NPROC): ${NPROC}"
echo "============================================"

# ---- Step 1: Prepare 16fps data ----
if [[ "${SKIP_PREP}" != "1" ]]; then
    echo ""
    echo "=== [Step 1/3] Preparing humanoid ego view @ 16fps (transcode from 20fps) ==="
    python "${PROJECT_ROOT}/finetuning/dreamgen/scripts/prepare_humanoid_dreamgen_16fps.py" \
        --source "${PROJECT_ROOT}/datasets/humanoid/singleview" \
        --target "${DATASET_DIR}"
else
    echo ""
    echo "=== [Step 1/3] Skipping data prep (SKIP_PREP=1) ==="
fi

# Verify data exists
N_VIDEOS=$(find "${DATASET_DIR}/videos" -name "*.mp4" 2>/dev/null | wc -l)
N_METAS=$(find "${DATASET_DIR}/metas" -name "*.txt" 2>/dev/null | wc -l)
echo "  Videos: ${N_VIDEOS}, Metas: ${N_METAS}"
if [[ "${N_VIDEOS}" -eq 0 ]]; then
    echo "ERROR: No videos found in ${DATASET_DIR}/videos/. Run without SKIP_PREP=1." >&2
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
