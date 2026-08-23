#!/bin/bash
# Run Ctrl-World inference with WorldCache on bimanual/multiview (replay GT 14-D qpos).
#
# Mirrors scripts/run_bimanual_multiview_ctrlworld.sh (dense) exactly in logic /
# paths / checkpoint, only swapping the leading sampling_dataset/dense -> worldcache
# and adding the WorldCache flags (acceleration-benchmark + sampling-dataset-layout).
#
# Output: sampling_dataset/worldcache/bimanual/{output,input}/multiview/ctrlworld/{category}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_multiview_ctrlworld_worldcache.sh
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, WORLDCACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY}

set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/bimanual_multiview_5fps/checkpoint-280000.pt}"

DATASET_NAME="${DATASET_NAME:-bimanual_multiview_5fps}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
CONFIG="${CONFIG:-bimanual}"

# WorldCache parameters. TUNED optimum for bimanual multiview Ctrl-World:
# thresh=0.04 / ret_ratio=0.4 -> 1.27x speedup at PSNR 21.88 (dense 22.07, -0.19).
# This is the knee: at thresh>=0.06 quality collapses (PSNR 22->19.7->16) for
# almost no extra speed (194s floor). At thresh<=0.02 the cache never triggers
# (~1.0x). ret_ratio (dense warm-up steps) is what preserves quality here.
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.04}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.4}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-2.5}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi
if [ -n "${NUM_SHARDS:-}" ]; then EXTRA="${EXTRA} --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID:-0}"; fi

echo "=========================================================="
echo "Ctrl-World + WorldCache | Bimanual Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT}  down_sample=${DOWN_SAMPLE}"
echo "  Params: thresh=${WORLDCACHE_REL_L1_THRESH}, ret_ratio=${WORLDCACHE_RET_RATIO}, probe_depth=${WORLDCACHE_PROBE_DEPTH}, motion_sens=${WORLDCACHE_MOTION_SENSITIVITY}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer_bimanual_multiview_ctrlworld.py \
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
    --use_worldcache \
    --worldcache_rel_l1_thresh "${WORLDCACHE_REL_L1_THRESH}" \
    --worldcache_ret_ratio "${WORLDCACHE_RET_RATIO}" \
    --worldcache_probe_depth "${WORLDCACHE_PROBE_DEPTH}" \
    --worldcache_motion_sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
