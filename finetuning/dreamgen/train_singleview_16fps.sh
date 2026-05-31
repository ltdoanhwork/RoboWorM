#!/bin/bash
# =============================================================================
# End-to-end: prepare 16fps data → compute T5 embeddings → launch training
# for DreamGen singleview on single_arm and/or bimanual.
#
# This script fixes the fps mismatch issue:
#   - Single-arm: 15fps → 16fps (minor adjustment)
#   - Bimanual:   30fps → 16fps (critical fix: 93 frames now = 5.8s, was 3.1s)
#
# Available experiments:
#   predict2_video2world_training_2b_single_arm_singleview_16fps
#   predict2_video2world_training_2b_bimanual_singleview_16fps
#   predict2_video2world_training_2b_arm_bimanual_singleview_16fps  (combined)
#
# Usage:
#   # Train on combined single_arm + bimanual (default):
#   NPROC=4 bash finetuning/dreamgen/train_singleview_16fps.sh
#
#   # Train on single_arm only:
#   EXPERIMENT=predict2_video2world_training_2b_single_arm_singleview_16fps \
#     NPROC=4 bash finetuning/dreamgen/train_singleview_16fps.sh
#
#   # Train on bimanual only:
#   EXPERIMENT=predict2_video2world_training_2b_bimanual_singleview_16fps \
#     NPROC=4 bash finetuning/dreamgen/train_singleview_16fps.sh
#
#   # Skip data preparation (already done):
#   SKIP_PREP=1 NPROC=4 bash finetuning/dreamgen/train_singleview_16fps.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="${EXPERIMENT:-predict2_video2world_training_2b_arm_bimanual_singleview_16fps}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_T5="${SKIP_T5:-0}"
NPROC="${NPROC:-4}"

echo "============================================"
echo "DreamGen Singleview 16fps Training Pipeline"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Experiment:   ${EXPERIMENT}"
echo "  GPUs (NPROC): ${NPROC}"
echo "  Skip prep:    ${SKIP_PREP}"
echo "  Skip T5:      ${SKIP_T5}"
echo "============================================"

# ---- Step 1: Prepare 16fps datasets ----
if [[ "${SKIP_PREP}" != "1" ]]; then
    echo ""
    echo "=== [Step 1/3] Preparing 16fps datasets ==="

    if [[ "${EXPERIMENT}" == *"single_arm"* ]] || [[ "${EXPERIMENT}" == *"arm_bimanual"* ]]; then
        echo "  → Preparing single_arm @ 16fps..."
        python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen_16fps.py
    fi

    if [[ "${EXPERIMENT}" == *"bimanual"* ]] || [[ "${EXPERIMENT}" == *"arm_bimanual"* ]]; then
        echo "  → Preparing bimanual @ 16fps..."
        python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen_16fps.py
    fi
else
    echo ""
    echo "=== [Step 1/3] Skipping data preparation (SKIP_PREP=1) ==="
fi

# ---- Step 2: Compute T5-XXL embeddings ----
if [[ "${SKIP_T5}" != "1" ]]; then
    echo ""
    echo "=== [Step 2/3] Computing T5-XXL embeddings ==="

    T5_DATASETS=""
    if [[ "${EXPERIMENT}" == *"single_arm"* ]] || [[ "${EXPERIMENT}" == *"arm_bimanual"* ]]; then
        T5_DATASETS="${T5_DATASETS} single_arm_16fps"
    fi
    if [[ "${EXPERIMENT}" == *"bimanual"* ]] || [[ "${EXPERIMENT}" == *"arm_bimanual"* ]]; then
        T5_DATASETS="${T5_DATASETS} bimanual_16fps"
    fi

    DATASETS="${T5_DATASETS}" bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
else
    echo ""
    echo "=== [Step 2/3] Skipping T5 embeddings (SKIP_T5=1) ==="
fi

# ---- Step 3: Launch training ----
echo ""
echo "=== [Step 3/3] Launching training ==="
echo "  Experiment: ${EXPERIMENT}"
echo "  NPROC: ${NPROC}"

NPROC="${NPROC}" bash finetuning/dreamgen/launch.sh "${EXPERIMENT}"
