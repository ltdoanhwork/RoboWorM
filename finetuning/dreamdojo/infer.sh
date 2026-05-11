#!/bin/bash
set -ex

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DREAMDOJO_ROOT="${PROJECT_ROOT}/models/DreamDojo"

export PYTHONPATH="${DREAMDOJO_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=/mnt/amlfs-01/shared/shenyuang/cosmos_cache

source ${DREAMDOJO_ROOT}/.venv/bin/activate

cd "${DREAMDOJO_ROOT}"

python ${PROJECT_ROOT}/finetuning/dreamdojo/infer_dataset_sample.py "$@"
