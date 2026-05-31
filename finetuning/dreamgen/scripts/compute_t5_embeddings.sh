#!/bin/bash
# Pre-compute T5-XXL embeddings for DreamGen singleview datasets.
#
# Must be run AFTER the prepare_*_dreamgen.py / caption scripts have populated
# datasets/{bimanual,single_arm,humanoid}_dreamgen/{videos,metas}/.
#
# Usage (from project root):
#   # All three datasets at once (default):
#   bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
#
#   # Single dataset:
#   DATASETS=humanoid bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
#
#   # Two datasets:
#   DATASETS="bimanual single_arm" bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh

set -ex

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

source "${DREAMGEN_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DREAMGEN_ROOT}:${PYTHONPATH}"

# Point constants.py to the local checkpoints dir (contains google-t5/t5-11b)
export COSMOS_PREDICT2_ARGS="--checkpoints ${PROJECT_ROOT}/checkpoints"

cd "${DREAMGEN_ROOT}"

DATASETS=${DATASETS:-"bimanual single_arm humanoid humanoid_multiview bimanual_multiview single_arm_16fps bimanual_16fps single_arm_static_16fps"}

for DATASET in ${DATASETS}; do
    DATASET_PATH="${PROJECT_ROOT}/datasets/${DATASET}_dreamgen"
    if [ ! -d "${DATASET_PATH}" ]; then
        echo "WARNING: ${DATASET_PATH} not found, skipping."
        case "${DATASET}" in
            bimanual)            echo "  → Run: python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen.py" ;;
            single_arm)          echo "  → Run: python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen.py" ;;
            humanoid)            echo "  → Run: bash scripts/run_caption_humanoid_singleview.sh" ;;
            humanoid_multiview)  echo "  → Run: python finetuning/dreamgen/scripts/prepare_humanoid_multiview_dreamgen.py" ;;
            bimanual_multiview)  echo "  → Run: python finetuning/dreamgen/scripts/prepare_bimanual_multiview_dreamgen.py" ;;
            single_arm_16fps)    echo "  → Run: python finetuning/dreamgen/scripts/prepare_single_arm_dreamgen_16fps.py" ;;
            bimanual_16fps)      echo "  → Run: python finetuning/dreamgen/scripts/prepare_bimanual_dreamgen_16fps.py" ;;
            single_arm_static_16fps) echo "  → Run: python finetuning/dreamgen/scripts/prepare_single_arm_static_dreamgen_16fps.py" ;;
        esac
        continue
    fi

    n_videos=$(ls "${DATASET_PATH}/videos/" 2>/dev/null | wc -l)
    n_metas=$(ls "${DATASET_PATH}/metas/" 2>/dev/null | wc -l)
    echo "========================================"
    echo "Computing T5 embeddings for: ${DATASET}"
    echo "  videos: ${n_videos}  metas: ${n_metas}"
    echo "  path:   ${DATASET_PATH}"
    echo "========================================"

    python -m scripts.get_t5_embeddings \
        --dataset_path "${DATASET_PATH}"

    echo "Done: ${DATASET_PATH}/t5_xxl/"
done

echo ""
echo "All T5 embeddings computed. Ready to train:"
echo "  NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_all_singleview"
