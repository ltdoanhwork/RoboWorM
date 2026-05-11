#!/bin/bash
# ==============================================================================
# DreamDojo 14B Teacher Inference Benchmark (2-GPU Context Parallel)
# ==============================================================================
# Generates videos from batch_input.json using the 14B post-trained GR-1 model.
# Uses context parallelism to split temporal dimension across 2 GPUs.
#
# Prerequisites:
#   1. Convert DCP checkpoint to PT format (run once):
#      python scripts/convert_distcp_to_pt.py \
#        /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/checkpoints/dreamdojo/14B_G1_post-train/iter_000050000/model \
#        /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/checkpoints/dreamdojo/14B_G1_post-train/iter_000050000/
#
#   2. Ensure the venv is set up (bash install.sh or use existing cosmos venv)
# ==============================================================================
set -ex

# ---- Environment ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export TORCH_NCCL_ENABLE_MONITORING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

# ---- Virtual environment ----
VENV="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/cosmos-predict2.5/.venv/bin/activate"
source "${VENV}"

# ---- Paths ----
CKPT_DIR="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/checkpoints/dreamdojo/14B_G1_post-train/iter_000050000"
BATCH_INPUT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/VLA-Humanoid/output/dream_gen_benchmark/cosmos_predict2_2b_gr1_env/batch_input.json"
OUTPUT_DIR="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/dream_gen_benchmark/output_dreamdojo"

# ---- Model config ----
EXPERIMENT="dreamdojo_14b_480_640_gr1"
RESOLUTION="480,640"
GUIDANCE=0           # 0=no CFG, 3-7=moderate, 10+=strong
CHUNK_SIZE=12        # actions per chunk (12 = 1 second at 12 FPS)
ACTION_DIM=384       # full action space dimension
SEED=1
SAVE_FPS=5
NUM_GPUS=2           # context parallel size

# ---- Step 1: Convert checkpoint if needed ----
if [ ! -f "${CKPT_DIR}/model_ema_bf16.pt" ]; then
    echo "Converting DCP checkpoint to PT format..."
    python scripts/convert_distcp_to_pt.py \
        "${CKPT_DIR}/model" \
        "${CKPT_DIR}/"
    echo "Checkpoint conversion complete."
fi

# ---- Step 2: Run inference ----
echo "Starting 14B teacher inference with ${NUM_GPUS} GPUs..."
echo "Experiment: ${EXPERIMENT}"
echo "Checkpoint: ${CKPT_DIR}"
echo "Batch input: ${BATCH_INPUT}"
echo "Output dir: ${OUTPUT_DIR}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=12341 \
    -m cosmos_predict2._src.predict2.action.inference.inference_batch \
    --experiment="${EXPERIMENT}" \
    --ckpt_path "${CKPT_DIR}" \
    --batch_input "${BATCH_INPUT}" \
    --output_dir "${OUTPUT_DIR}" \
    --resolution "${RESOLUTION}" \
    --guidance ${GUIDANCE} \
    --chunk_size ${CHUNK_SIZE} \
    --action_dim ${ACTION_DIM} \
    --seed ${SEED} \
    --save_fps ${SAVE_FPS} \
    --context_parallel_size ${NUM_GPUS}

echo "Inference complete. Results saved to: ${OUTPUT_DIR}"
echo "Timing report: ${OUTPUT_DIR}/timing.json"
