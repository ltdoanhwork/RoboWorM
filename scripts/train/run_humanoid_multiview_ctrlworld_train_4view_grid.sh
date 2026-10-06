#!/bin/bash
# Finetune Ctrl-World on datasets/humanoid/multiview (AIRBOT_MMK2, 36-D state) at
# 5fps as a 4-VIEW 2x2 GRID (2 static + 2 dynamic cameras).
#
# Grid layout (row-major, latent canvas 48x80 = 2*24 x 2*40):
#     [ TL=main external   TR=second external ]  (static, top row)
#     [ BL=cam_left_wrist  BR=cam_right_wrist ]  (dynamic, bottom row)
# where main   = cam_high_rgb  OR cam_head_rgb  (per task)
#       second = cam_third_view OR cam_front_rgb (per task)
#
# Warm-starts from the finetuned 3-view humanoid checkpoint (checkpoint-200000.pt),
# which already has the 36-D action encoder -> loads fully. The SVD UNet is
# view-agnostic (just sees a larger single image), so only the new spatial extent
# (48x80 vs 72x40) is re-learned. strict=False tolerates any residual mismatch.
#
# Uses a SEPARATE dataset name (humanoid_multiview_4view_grid) + output dir so the
# 3-view assets are untouched.
#
# Usage:
#   NPROC=3 CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/train/run_humanoid_multiview_ctrlworld_train_4view_grid.sh
#   SKIP_PREP=1 SKIP_META=1 NPROC=3 bash scripts/train/run_humanoid_multiview_ctrlworld_train_4view_grid.sh
#
# Env overrides:
#   NPROC (default 4), CUDA_VISIBLE_DEVICES (default 0,1,2,3), DOWN_SAMPLE (default 6),
#   MAX_STEPS, BATCH_SIZE (default 2; lower if the +33% grid VRAM OOMs),
#   MASTER_PORT, WARM_CKPT, LIMIT_EPISODES, DEBUG

set -e

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

export VENV="${ROOT}/models/Ctrl-World/.venv"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29504}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"          # 30fps -> 5fps
DATASET_NAME="humanoid_multiview_4view_grid"

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
# Warm-start from the finetuned 3-view humanoid checkpoint (36-D action encoder present)
WARM_CKPT="${WARM_CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview/checkpoint-200000.pt}"

DATASET_ROOT="${ROOT}/models/Ctrl-World/dataset_example"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
OUTPUT_DIR="${ROOT}/checkpoints/finetuned/ctrlworld/${DATASET_NAME}"

MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

echo "=========================================="
echo "Ctrl-World Humanoid Multiview Finetune @5fps — 4-VIEW 2x2 GRID"
echo "  down_sample=${DOWN_SAMPLE}  nproc=${NPROC}  gpus=${CUDA_VISIBLE_DEVICES}"
echo "  grid=2x2 [TL=main_ext TR=second_ext / BL=left_wrist BR=right_wrist]"
echo "  warm_start=${WARM_CKPT}"
echo "  output=${OUTPUT_DIR}"
echo "=========================================="

# Step 0: prepare 4-view grid latents + annotations
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "[Step 0] Preparing 4-view grid latents + annotations..."
  LIMIT_ARG=""
  if [ -n "${LIMIT_EPISODES:-}" ]; then LIMIT_ARG="--limit-episodes ${LIMIT_EPISODES}"; fi
  python scripts/prepare/prepare_ctrlworld_humanoid_multiview_grid.py \
    --subset both \
    --output-dir "${DATASET_ROOT}" \
    --svd-path "${SVD_MODEL_PATH}" \
    --down-sample "${DOWN_SAMPLE}" \
    --name "${DATASET_NAME}" ${LIMIT_ARG}
fi

# Step 1: meta-info (view-count agnostic; only reads state/video_length)
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
echo "[Step 2] Finetuning 4-view grid @5fps (warm-start from 3-view humanoid ckpt)..."
DEBUG_ARG=""
if [ "${DEBUG:-0}" = "1" ]; then DEBUG_ARG="--debug"; fi

cd "${ROOT}/models/Ctrl-World"
WANDB_MODE="${WANDB_MODE:-offline}" SWANLAB_MODE="${SWANLAB_MODE:-local}" accelerate launch \
  --num_processes "${NPROC}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_wm.py \
  --config humanoid_grid \
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
