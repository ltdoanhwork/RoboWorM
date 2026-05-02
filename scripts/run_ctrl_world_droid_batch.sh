#!/bin/bash
# run_ctrl_world_droid_batch.sh
#
# Full pipeline to run Ctrl-World inference on the DROID 20% dataset.

set -e

# ── Configuration ────────────────────────────────────────────────
PROJECT_ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
LEROBOT_ROOT="./datasets/droid_20p"
CTRL_WORLD_DATASET="./datasets/droid_20p_ctrl_world"
META_INFO_DIR="${CTRL_WORLD_DATASET}/meta_info"
MAX_EPISODES=100

# Model paths
SVD_MODEL="./checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL="./checkpoints/openai/clip-vit-base-patch32"
CTRL_WORLD_CKPT="./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"

cd "${PROJECT_ROOT}"

# ── Step 0: Activate environment ─────────────────────────────────
source models/DreamDojo/.venv/bin/activate

# Block ALL HuggingFace downloads - use local data only
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── Ctrl-World PYTHONPATH ─────────────────────────────────────────
# rollout_replay_traj.py does:
#   from models.pipeline_ctrl_world import ...     ← needs models/Ctrl-World on path
#   from methods.cache_strategy... import ...      ← needs project root on path
# Both must be in PYTHONPATH.
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}:${PYTHONPATH}"

# ── Step 1: Convert LeRobot DROID → Ctrl-World format ────────────
echo "================================================================="
echo "Step 1: Converting LeRobot DROID to Ctrl-World format..."
echo "================================================================="

PREPARE_ARGS="--lerobot_root ${LEROBOT_ROOT} --output_root ${CTRL_WORLD_DATASET} --split val"
if [ -n "${MAX_EPISODES}" ]; then
    PREPARE_ARGS="${PREPARE_ARGS} --max_episodes ${MAX_EPISODES}"
fi
python scripts/prepare_droid_for_ctrl_world.py ${PREPARE_ARGS}

# ── Step 2: Run Ctrl-World replay inference ──────────────────────
echo "================================================================="
echo "Step 2: Running Ctrl-World replay inference..."
echo "================================================================="

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 \
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path "${CTRL_WORLD_DATASET}" \
  --dataset_meta_info_path "${META_INFO_DIR}" \
  --dataset_names "." \
  --svd_model_path "${SVD_MODEL}" \
  --clip_model_path "${CLIP_MODEL}" \
  --ckpt_path "${CTRL_WORLD_CKPT}" \
  --task_type replay

echo "================================================================="
echo "Done! Check output in: synthetic_traj/"
echo "================================================================="
