#!/bin/bash
# PP B: Finetune Ctrl-World on bimanual/multiview at 5fps (down_sample=6) to
# reduce autoregressive drift (6x fewer rollout iterations than 30fps).
#
# Warm-starts from the 30fps checkpoint (default checkpoint-120000.pt), which
# already has the 14-D action encoder -> loads fully (strict=False is still used
# so any residual mismatch is tolerated). The visual backbone is reused; only the
# temporal cadence (5fps) is re-learned.
#
# Uses a SEPARATE dataset name (bimanual_multiview_5fps) and output dir so the
# 30fps assets are untouched.
#
# Usage:
#   NPROC=4 bash scripts/train/run_bimanual_multiview_ctrlworld_train_5fps.sh
#   SKIP_PREP=1 SKIP_META=1 NPROC=4 bash scripts/train/run_bimanual_multiview_ctrlworld_train_5fps.sh
#
# Env overrides:
#   NPROC (default 4), CUDA_VISIBLE_DEVICES (default 0,1,2,3), MAX_STEPS,
#   BATCH_SIZE, MASTER_PORT, WARM_CKPT (checkpoint to warm-start from),
#   LIMIT_EPISODES (debug)

set -e

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

export VENV="${ROOT}/models/Ctrl-World/.venv"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29502}"
DOWN_SAMPLE=6                 # 30fps -> 5fps (fixed for this variant)
DATASET_NAME="bimanual_multiview_5fps"

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
# Warm-start from the finetuned 30fps checkpoint (14-D action encoder already present)
# BROKEN (2026-10-05 refactor): the 30fps bimanual_multiview/checkpoint-120000.pt warm-start does not exist on disk (not in
# checkpoints/finetuned/ nor the HF backup). Set WARM_CKPT explicitly to run this script.
WARM_CKPT="${WARM_CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview/checkpoint-120000.pt}"

DATASET_ROOT="${ROOT}/models/Ctrl-World/dataset_example"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
OUTPUT_DIR="${ROOT}/checkpoints/finetuned/ctrlworld/${DATASET_NAME}"

MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

echo "=========================================="
echo "Ctrl-World Bimanual Multiview Finetune @5fps (PP B)"
echo "  down_sample=${DOWN_SAMPLE} (5fps)  nproc=${NPROC}  gpus=${CUDA_VISIBLE_DEVICES}"
echo "  warm_start=${WARM_CKPT}"
echo "  output=${OUTPUT_DIR}"
echo "=========================================="

# Step 0: prepare 5fps latents + annotations
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "[Step 0] Preparing 5fps latents + annotations..."
  LIMIT_ARG=""
  if [ -n "${LIMIT_EPISODES:-}" ]; then LIMIT_ARG="--limit-episodes ${LIMIT_EPISODES}"; fi
  python scripts/prepare/prepare_ctrlworld_bimanual_multiview.py \
    --subset both \
    --output-dir "${DATASET_ROOT}" \
    --svd-path "${SVD_MODEL_PATH}" \
    --down-sample "${DOWN_SAMPLE}" \
    --name "${DATASET_NAME}" ${LIMIT_ARG}
fi

# Step 1: meta-info
if [ "${SKIP_META:-0}" != "1" ]; then
  echo "[Step 1] Building meta-info..."
  cd "${ROOT}/models/Ctrl-World"
  python dataset_meta_info/create_meta_info_bimanual.py \
    --dataset-root "dataset_example/${DATASET_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --meta-out dataset_meta_info
  cd "${ROOT}"
fi

# Step 2: finetune
echo "[Step 2] Finetuning @5fps (warm-start from 30fps ckpt)..."
DEBUG_ARG=""
if [ "${DEBUG:-0}" = "1" ]; then DEBUG_ARG="--debug"; fi

cd "${ROOT}/models/Ctrl-World"
WANDB_MODE="${WANDB_MODE:-offline}" SWANLAB_MODE="${SWANLAB_MODE:-local}" accelerate launch \
  --num_processes "${NPROC}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_wm.py \
  --config bimanual \
  --svd_model_path "${SVD_MODEL_PATH}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --ckpt_path "${WARM_CKPT}" \
  --dataset_root_path "dataset_example" \
  --dataset_meta_info_path "dataset_meta_info" \
  --dataset_names "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --tag "${DATASET_NAME}" \
  --max_train_steps "${MAX_STEPS}" \
  --train_batch_size "${BATCH_SIZE}" \
  --no_load_strict ${DEBUG_ARG}

echo "Done. Checkpoints in ${OUTPUT_DIR}"
