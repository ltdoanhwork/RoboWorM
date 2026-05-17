#!/bin/bash
# Run Ctrl-World POLICY-CONDITIONED inference on single_arm/multiview dataset (makovian & non_makovian)
#
# Uses pi05 policy to predict actions, then Ctrl-World generates video frames.
# Output saved in benchmark format for direct comparison with other models.
#
# Step 1: Prepare dataset (convert LeRoBot format -> Ctrl-World annotation format)
# Step 2: Run policy-conditioned inference (generate videos per episode)
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld_pi.sh [makovian|non_makovian|both]
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_multiview_ctrlworld_pi.sh both

set -e

SUBSET=${1:-both}

PROJECT_ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${PROJECT_ROOT}"

# Activate Ctrl-World venv (has all deps including openpi's)
source "${PROJECT_ROOT}/models/Ctrl-World/.venv/bin/activate"

# PYTHONPATH: Ctrl-World FIRST (resolves 'models' package), then openpi, then project root
export PYTHONPATH="${PROJECT_ROOT}/models/Ctrl-World:${PROJECT_ROOT}/models/Ctrl-World/openpi/src:${PROJECT_ROOT}/models/Ctrl-World/openpi/packages/openpi-client/src:${PROJECT_ROOT}/models/Ctrl-World/scripts:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Model paths
SVD_MODEL_PATH="./checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/openai/clip-vit-base-patch32"
CKPT_PATH="./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

# Policy checkpoint
PI_CKPT="${PI_CKPT:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/openpi/openpi-assets/checkpoints/pi05_droid}"
POLICY_TYPE="${POLICY_TYPE:-pi05}"

OUTPUT_BASE="${PROJECT_ROOT}/sampling_dataset/dense/single_arm/output/multiview/ctrlworld"
INPUT_BASE="${PROJECT_ROOT}/sampling_dataset/dense/single_arm/input/multiview/ctrlworld"

echo "=========================================================="
echo "Ctrl-World Policy-Conditioned Single-Arm Multiview Infer"
echo "Subset: ${SUBSET}"
echo "Policy: ${POLICY_TYPE} (${PI_CKPT})"
echo "=========================================================="

# Step 1: Prepare dataset (if not already done)
echo ""
echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
python scripts/prepare_ctrlworld_single_arm_multiview.py \
    --subset "${SUBSET}" \
    --output-dir "${DATASET_EXAMPLE_DIR}" \
    --dataset-base "${PROJECT_ROOT}/datasets/single_arm/multiview"

# Step 2: Run policy-conditioned inference
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running POLICY inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld_pi.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --pi_ckpt "${PI_CKPT}" \
        --policy_type "${POLICY_TYPE}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian"
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running POLICY inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld_pi.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --pi_ckpt "${PI_CKPT}" \
        --policy_type "${POLICY_TYPE}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian"
fi

echo ""
echo "=========================================================="
echo "Done! Results saved to:"
echo "  Output: ${OUTPUT_BASE}/{makovian,non_makovian}"
echo "  Input:  ${INPUT_BASE}/{makovian,non_makovian}"
echo "=========================================================="
