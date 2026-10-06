#!/bin/bash
# Run Ctrl-World inference with FasterCache on bimanual/multiview (replay GT 14-D qpos).
#
# Mirrors scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld.sh (dense) exactly, only routing
# output to sampling_dataset/fastercache and adding the FasterCache flags.
#
# Output: sampling_dataset/fastercache/bimanual/{output,input}/multiview/ctrlworld/{category}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_bimanual_multiview_ctrlworld_fastercache.sh
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, FASTERCACHE_{START_STEP,MODEL_INTERVAL,BLOCK_INTERVAL}

set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_5fps/checkpoint-280000.pt}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_5fps}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
CONFIG="${CONFIG:-bimanual}"

# FasterCache parameters. NOTE (measured on bimanual multiview Ctrl-World):
# this model's custom SVD UNet exposes no `attn_processors`, so FasterCache's
# block-level cache attaches to 0 layers, and its model-level uncond-skip only
# fires under classifier-free guidance (guidance_scale>1) -- but dense runs at
# guidance_scale=1.0, so CFG is OFF and the skip never triggers. Net result:
# FasterCache == dense in compute (~1.0x) at PSNR ~dense. Kept runnable as a
# valid (net-zero-speedup) benchmark point per the acceleration-benchmark rule;
# do NOT delete/swap it. For real speedup+quality use WorldCache (thresh 0.04).
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/fastercache/bimanual/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/fastercache/bimanual/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
# Balanced subset for the timing pass: N episodes of EACH category. Plain
# NUM_EPISODES slices a stem-sorted list where every makovian_* precedes every
# non_makovian_*, so a small value silently yields makovian only.
if [ -n "${NUM_EPISODES_PER_CATEGORY:-}" ]; then EXTRA="${EXTRA} --num_episodes_per_category ${NUM_EPISODES_PER_CATEGORY}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================================="
echo "Ctrl-World + FasterCache | Bimanual Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT}  down_sample=${DOWN_SAMPLE}"
echo "  Params: start_step=${FASTERCACHE_START_STEP}, model_interval=${FASTERCACHE_MODEL_INTERVAL}, block_interval=${FASTERCACHE_BLOCK_INTERVAL}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer/infer_bimanual_multiview_ctrlworld.py \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --config "${CONFIG}" \
    --split "${SPLIT}" \
    --use_fastercache \
    --fastercache_start_step "${FASTERCACHE_START_STEP}" \
    --fastercache_model_interval "${FASTERCACHE_MODEL_INTERVAL}" \
    --fastercache_block_interval "${FASTERCACHE_BLOCK_INTERVAL}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
