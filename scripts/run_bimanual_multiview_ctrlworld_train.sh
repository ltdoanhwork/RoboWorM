#!/bin/bash
# Finetune Ctrl-World on datasets/bimanual/multiview (ALOHA, 3-view, 14-D qpos).
#
# Pipeline:
#   Step 0 (optional): prepare latents + annotations (SKIP_PREP=1 to skip)
#   Step 1 (optional): build meta-info (train/val samples + stat.json)
#   Step 2: train (warm-start from DROID checkpoint, strict=False)
#
# Usage:
#   NPROC=4 bash scripts/run_bimanual_multiview_ctrlworld_train.sh
#   SKIP_PREP=1 SKIP_META=1 DEBUG=1 NPROC=1 bash scripts/run_bimanual_multiview_ctrlworld_train.sh
#
# Env overrides:
#   DOWN_SAMPLE (default 1 = native 30fps), NPROC (default 4), CUDA_VISIBLE_DEVICES,
#   MAX_STEPS, BATCH_SIZE, MASTER_PORT, LIMIT_EPISODES
#
# Note: prep (Step 0) always runs single-GPU (VAE encode is sequential). Only
# training (Step 2) uses NPROC GPUs. To split, run once with SKIP_PREP=0 to build
# the dataset, then SKIP_PREP=1 SKIP_META=1 for subsequent multi-GPU train runs.

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

export VENV="${ROOT}/models/Ctrl-World/.venv"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
DOWN_SAMPLE="${DOWN_SAMPLE:-1}"
DATASET_NAME="bimanual_multiview"

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
CKPT_PATH="${ROOT}/checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt"

DATASET_ROOT="${ROOT}/models/Ctrl-World/dataset_example"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
OUTPUT_DIR="${ROOT}/checkpoints/finetuned/ctrlworld/${DATASET_NAME}"

MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

echo "=========================================="
echo "Ctrl-World Bimanual Multiview Finetune"
echo "  down_sample=${DOWN_SAMPLE}  nproc=${NPROC}  gpus=${CUDA_VISIBLE_DEVICES}"
echo "  output=${OUTPUT_DIR}"
echo "=========================================="

# Step 0: prepare latents + annotations
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "[Step 0] Preparing latents + annotations (LeRobot -> Ctrl-World train format)..."
  LIMIT_ARG=""
  if [ -n "${LIMIT_EPISODES:-}" ]; then LIMIT_ARG="--limit-episodes ${LIMIT_EPISODES}"; fi
  python scripts/prepare_ctrlworld_bimanual_multiview.py \
    --subset both \
    --output-dir "${DATASET_ROOT}" \
    --svd-path "${SVD_MODEL_PATH}" \
    --down-sample "${DOWN_SAMPLE}" \
    --name "${DATASET_NAME}" ${LIMIT_ARG}
fi

# Step 1: meta-info (train/val windows + stat.json)
if [ "${SKIP_META:-0}" != "1" ]; then
  echo "[Step 1] Building meta-info..."
  cd "${ROOT}/models/Ctrl-World"
  python dataset_meta_info/create_meta_info_bimanual.py \
    --dataset-root "dataset_example/${DATASET_NAME}" \
    --dataset-name "${DATASET_NAME}" \
    --meta-out dataset_meta_info
  cd "${ROOT}"
fi

# Step 2: train
echo "[Step 2] Training..."
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
  --ckpt_path "${CKPT_PATH}" \
  --dataset_root_path "dataset_example" \
  --dataset_meta_info_path "dataset_meta_info" \
  --dataset_names "${DATASET_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --tag "${DATASET_NAME}" \
  --max_train_steps "${MAX_STEPS}" \
  --train_batch_size "${BATCH_SIZE}" \
  --no_load_strict ${DEBUG_ARG}

echo "Done. Checkpoints in ${OUTPUT_DIR}"
