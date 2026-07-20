#!/bin/bash
# Sweep late Ctrl-World humanoid_multiview checkpoints on the first N val episodes.
# Each checkpoint writes to its OWN save dir so results don't collide / skip.
# Distributes checkpoints across the 4 GPUs and runs them in parallel.
set -u

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"
source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT_DIR="${ROOT}/exp_logs/ctrlworld_logs/humanoid_multiview"
DATASET_NAME="humanoid_multiview"
DATASET_DIR="${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE=6
NUM_EPISODES="${NUM_EPISODES:-4}"
SWEEP_ROOT="${SWEEP_ROOT:-/tmp/ckpt_sweep_humanoid_mv}"

STEPS=(${STEPS:-150000 160000 170000 180000 190000 200000})
NGPU=4

mkdir -p "${SWEEP_ROOT}"

run_one() {
  local step="$1"; local gpu="$2"
  local ckpt="${CKPT_DIR}/checkpoint-${step}.pt"
  local out="${SWEEP_ROOT}/ckpt_${step}"
  local log="${SWEEP_ROOT}/ckpt_${step}.log"
  echo "[launch] step=${step} gpu=${gpu} -> ${out}"
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/infer_bimanual_multiview_ctrlworld.py \
    --config humanoid \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${ckpt}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${out}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --split val --num_episodes "${NUM_EPISODES}" > "${log}" 2>&1
  echo "[done] step=${step} gpu=${gpu}"
}

i=0
for step in "${STEPS[@]}"; do
  gpu=$(( i % NGPU ))
  run_one "${step}" "${gpu}" &
  i=$(( i + 1 ))
  # After filling all GPUs, wait for this wave to finish before next wave
  if [ $(( i % NGPU )) -eq 0 ]; then
    wait
  fi
done
wait
echo "ALL_SWEEP_DONE"
