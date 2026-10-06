#!/bin/bash
# Run Ctrl-World inference with WorldCache on humanoid/multiview (replay GT state).
#
# Mirrors scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld.sh (dense) exactly in logic /
# paths / checkpoint / --config humanoid, only swapping the leading
# sampling_dataset/dense -> worldcache and adding the WorldCache flags
# (acceleration-benchmark + sampling-dataset-layout rules).
#
# Output: sampling_dataset/worldcache/humanoid/{output,input}/multiview/ctrlworld/{category}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/ctrlworld/run_humanoid_multiview_ctrlworld_worldcache.sh
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   WORLDCACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH,MOTION_SENSITIVITY}

set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview/checkpoint-200000.pt}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
# 3-view strip by default; the 4-view grid wrapper passes CONFIG=humanoid_grid.
CONFIG="${CONFIG:-humanoid}"

# WorldCache parameters. TUNED optimum for humanoid multiview Ctrl-World
# (ckpt-200000, 2 val episodes; see .cursor/rules ctrlworld-humanoid-multiview-
# caching-insights). Dense = PSNR 21.27 / 117.5s. Knee at thresh 0.03-0.04:
#   t0.03 -> PSNR 21.29 (==dense) 1.14x ; t0.04 -> PSNR 21.20 (-0.07) 1.16x.
# Cliff at t>=0.05 (PSNR 20.5 -> 19.5) for no extra speed (~99s floor). DEFAULT
# t0.04 = the knee; use 0.03 for the quality-safe point. WorldCache is the winner
# (beats DiCache at equal threshold, same as bimanual).
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.04}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.4}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-2.5}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/input/multiview/ctrlworld}"

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
echo "Ctrl-World + WorldCache | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT}  down_sample=${DOWN_SAMPLE}"
echo "  Params: thresh=${WORLDCACHE_REL_L1_THRESH}, ret_ratio=${WORLDCACHE_RET_RATIO}, probe_depth=${WORLDCACHE_PROBE_DEPTH}, motion_sens=${WORLDCACHE_MOTION_SENSITIVITY}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer/infer_bimanual_multiview_ctrlworld.py \
    --config "${CONFIG}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${DATASET_DIR}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --split "${SPLIT}" \
    --use_worldcache \
    --worldcache_rel_l1_thresh "${WORLDCACHE_REL_L1_THRESH}" \
    --worldcache_ret_ratio "${WORLDCACHE_RET_RATIO}" \
    --worldcache_probe_depth "${WORLDCACHE_PROBE_DEPTH}" \
    --worldcache_motion_sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
