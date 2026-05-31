#!/bin/bash
# Run Ctrl-World inference on single_arm/multiview dataset (makovian & non_makovian)
#
# Step 1: Prepare dataset (convert LeRoBot format -> Ctrl-World annotation format)
# Step 2: Run inference (generate videos per episode)
#
# Usage:
#   bash scripts/run_single_arm_multiview_ctrlworld.sh [makovian|non_makovian|both]

set -e

SUBSET=${1:-both}

# Activate environment
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/Ctrl-World/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World:."
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# Paths
SVD_MODEL_PATH="./checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="./checkpoints/openai/clip-vit-base-patch32"
CKPT_PATH="./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"
DATASET_META_INFO="./models/Ctrl-World/dataset_meta_info"
DATASET_EXAMPLE_DIR="./models/Ctrl-World/dataset_example"

OUTPUT_BASE="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/sampling_dataset/dense/single_arm/output/multiview/ctrlworld"
INPUT_BASE="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/sampling_dataset/dense/single_arm/input/multiview/ctrlworld"

echo "=========================================="
echo "Ctrl-World Single-Arm Multiview Inference"
echo "Subset: ${SUBSET}"
echo "=========================================="

# Step 1: Prepare dataset
echo ""
echo "[Step 1] Preparing dataset (LeRoBot -> Ctrl-World format)..."
python scripts/prepare_ctrlworld_single_arm_multiview.py \
    --subset "${SUBSET}" \
    --output-dir "${DATASET_EXAMPLE_DIR}" \
    --dataset-base "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/datasets/single_arm/multiview"

# Step 2: Run inference
if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "makovian" ]; then
    echo ""
    echo "[Step 2a] Running inference on MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/makovian" \
        --input_save_dir "${INPUT_BASE}/makovian"
fi

if [ "${SUBSET}" = "both" ] || [ "${SUBSET}" = "non_makovian" ]; then
    echo ""
    echo "[Step 2b] Running inference on NON_MAKOVIAN subset..."
    python scripts/infer_single_arm_multiview_ctrlworld.py \
        --svd_model_path "${SVD_MODEL_PATH}" \
        --clip_model_path "${CLIP_MODEL_PATH}" \
        --ckpt_path "${CKPT_PATH}" \
        --subset non_makovian \
        --dataset_example_dir "${DATASET_EXAMPLE_DIR}" \
        --dataset_meta_info_path "${DATASET_META_INFO}" \
        --save_dir "${OUTPUT_BASE}/non_makovian" \
        --input_save_dir "${INPUT_BASE}/non_makovian"
fi

echo ""
echo "=========================================="
echo "Done! Results saved to: ${OUTPUT_BASE}"
echo "=========================================="
