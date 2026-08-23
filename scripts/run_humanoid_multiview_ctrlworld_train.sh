#!/bin/bash
# Finetune Ctrl-World on datasets/humanoid/multiview (AIRBOT_MMK2, 3-view, 36-D state) at 5fps.
#
# Mirrors the bimanual 5fps pipeline (down_sample=6 to reduce autoregressive drift).
# Warm-starts from the DROID Ctrl-World checkpoint (checkpoint-10000.pt) with
# strict=False -> action encoder is reinitialized for the 36-D state (same pattern
# used successfully for bimanual).
#
# Canonical 3 views: [cam_high_rgb|cam_head_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb].
#
# Usage:
#   NPROC=4 bash scripts/run_humanoid_multiview_ctrlworld_train.sh
#   SKIP_PREP=1 SKIP_META=1 NPROC=4 bash scripts/run_humanoid_multiview_ctrlworld_train.sh
#
# Env overrides:
#   NPROC (default 4), CUDA_VISIBLE_DEVICES (default 0,1,2,3), DOWN_SAMPLE (default 6),
#   MAX_STEPS, BATCH_SIZE, MASTER_PORT, WARM_CKPT, LIMIT_EPISODES, DEBUG

set -e

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

export VENV="${ROOT}/models/Ctrl-World/.venv"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-29503}"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"          # 30fps -> 5fps
DATASET_NAME="humanoid_multiview"

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
WARM_CKPT="${WARM_CKPT:-${ROOT}/checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt}"

DATASET_ROOT="${ROOT}/models/Ctrl-World/dataset_example"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
OUTPUT_DIR="${ROOT}/exp_logs/ctrlworld_logs/${DATASET_NAME}"

MAX_STEPS="${MAX_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-2}"

echo "=========================================="
echo "Ctrl-World Humanoid Multiview Finetune @5fps"
echo "  down_sample=${DOWN_SAMPLE}  nproc=${NPROC}  gpus=${CUDA_VISIBLE_DEVICES}"
echo "  warm_start=${WARM_CKPT}"
echo "  output=${OUTPUT_DIR}"
echo "=========================================="

# Step 0: prepare latents + annotations
if [ "${SKIP_PREP:-0}" != "1" ]; then
  echo "[Step 0] Preparing latents + annotations..."
  LIMIT_ARG=""
  if [ -n "${LIMIT_EPISODES:-}" ]; then LIMIT_ARG="--limit-episodes ${LIMIT_EPISODES}"; fi
  python scripts/prepare_ctrlworld_humanoid_multiview.py \
    --subset both \
    --output-dir "${DATASET_ROOT}" \
    --svd-path "${SVD_MODEL_PATH}" \
    --down-sample "${DOWN_SAMPLE}" \
    --name "${DATASET_NAME}" ${LIMIT_ARG}
fi

# Step 1: meta-info (reuse the bimanual meta builder; dataset-name is generic)
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
echo "[Step 2] Finetuning..."
DEBUG_ARG=""
if [ "${DEBUG:-0}" = "1" ]; then DEBUG_ARG="--debug"; fi

cd "${ROOT}/models/Ctrl-World"
WANDB_MODE="${WANDB_MODE:-offline}" SWANLAB_MODE="${SWANLAB_MODE:-local}" accelerate launch \
  --num_processes "${NPROC}" \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_wm.py \
  --config humanoid \
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
