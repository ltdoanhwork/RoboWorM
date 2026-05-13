#!/bin/bash
# Pre-compute T5-XXL embeddings for all three singleview DreamGen datasets.
#
# Must be run AFTER the prepare_*_dreamgen.py scripts have populated
# datasets/{bimanual,single_arm,humanoid}_dreamgen/{videos,metas}/.
#
# Usage (from project root):
#   bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh
#
# Run a single dataset:
#   DATASETS=bimanual bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh

set -ex

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

source "${DREAMGEN_ROOT}/.venv/bin/activate"
export PYTHONPATH="${DREAMGEN_ROOT}:${PYTHONPATH}"

cd "${DREAMGEN_ROOT}"

DATASETS=${DATASETS:-"bimanual single_arm humanoid"}

for DATASET in ${DATASETS}; do
    DATASET_PATH="${PROJECT_ROOT}/datasets/${DATASET}_dreamgen"
    if [ ! -d "${DATASET_PATH}" ]; then
        echo "WARNING: ${DATASET_PATH} not found, skipping. Run prepare_${DATASET}_dreamgen.py first."
        continue
    fi

    echo "========================================"
    echo "Computing T5 embeddings for: ${DATASET}"
    echo "Dataset path: ${DATASET_PATH}"
    echo "========================================"

    python -m scripts.get_t5_embeddings \
        --dataset_path "${DATASET_PATH}"

    echo "Done: ${DATASET_PATH}/t5_xxl/"
done

echo ""
echo "All T5 embeddings computed."
