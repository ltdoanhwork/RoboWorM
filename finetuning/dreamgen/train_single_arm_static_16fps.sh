#!/bin/bash
# =============================================================================
# Train DreamGen on DROID single-arm STATIC views (exterior_1 + exterior_2) @ 16fps.
#
# Uses the 2 static exterior cameras (not the dynamic wrist camera).
# Each view is a separate training sample sharing the same text prompt.
# Total training samples: ~260 episodes × 2 views = ~520 videos.
#
# Pipeline: prepare 16fps data → T5 embeddings → launch training
#
# Experiment: predict2_video2world_training_2b_single_arm_static_singleview_16fps
#
# Usage:
#   NPROC=4 bash finetuning/dreamgen/train_single_arm_static_16fps.sh
#   SKIP_PREP=1 NPROC=4 bash finetuning/dreamgen/train_single_arm_static_16fps.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT="predict2_video2world_training_2b_single_arm_static_singleview_16fps"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_T5="${SKIP_T5:-0}"
NPROC="${NPROC:-4}"

echo "============================================"
echo "DreamGen Single-arm STATIC Views 16fps"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Experiment:   ${EXPERIMENT}"
echo "  GPUs (NPROC): ${NPROC}"
echo "  Views:        exterior_1_left + exterior_2_left"
echo "============================================"

# ---- Step 1: Prepare 16fps static view data ----
if [[ "${SKIP_PREP}" != "1" ]]; then
    echo ""
    echo "=== [Step 1/3] Preparing static views @ 16fps ==="
    python finetuning/dreamgen/scripts/prepare_single_arm_static_dreamgen_16fps.py
else
    echo ""
    echo "=== [Step 1/3] Skipping data prep (SKIP_PREP=1) ==="
fi

# ---- Step 2: Compute T5-XXL embeddings ----
if [[ "${SKIP_T5}" != "1" ]]; then
    echo ""
    echo "=== [Step 2/3] Computing T5-XXL embeddings ==="
    DATASETS="single_arm_static_16fps" bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
else
    echo ""
    echo "=== [Step 2/3] Skipping T5 embeddings (SKIP_T5=1) ==="
fi

# ---- Step 3: Launch training ----
echo ""
echo "=== [Step 3/3] Launching training ==="
NPROC="${NPROC}" bash finetuning/dreamgen/launch.sh "${EXPERIMENT}"
