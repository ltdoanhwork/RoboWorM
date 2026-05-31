#!/bin/bash
# Run original Ctrl-World rollout_replay_traj.py directly (GT actions, for quality comparison)
# Saves output to /tmp/ctrlworld_replay_test/
#
# Usage:
#   bash scripts/run_ctrlworld_replay_original.sh

set -e

PROJECT_ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
CTRL_WORLD_DIR="${PROJECT_ROOT}/models/Ctrl-World"

cd "${CTRL_WORLD_DIR}"

# Activate venv
source "${CTRL_WORLD_DIR}/.venv/bin/activate"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${CTRL_WORLD_DIR}:${PROJECT_ROOT}:${PYTHONPATH:-}"

# Paths
SVD_MODEL_PATH="${PROJECT_ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${PROJECT_ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT_PATH="${PROJECT_ROOT}/checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"

# Save to project tmp folder (persists on pfss)
SAVE_DIR="${PROJECT_ROOT}/tmp/ctrlworld_replay_test"
rm -rf "${SAVE_DIR}"
mkdir -p "${SAVE_DIR}"
rm -rf "${CTRL_WORLD_DIR}/synthetic_traj"
ln -sf "${SAVE_DIR}" "${CTRL_WORLD_DIR}/synthetic_traj"

echo "=========================================="
echo "Ctrl-World Replay (Original Script)"
echo "Dataset: dataset_example/droid_subset"
echo "Save dir: ${SAVE_DIR}"
echo "=========================================="

python scripts/rollout_replay_traj.py \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names droid_subset \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT_PATH}" \
    --task_type replay

echo ""
echo "Done! Output saved to: ${SAVE_DIR}/Rollouts_replay/video/"
