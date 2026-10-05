#!/bin/bash
# Run Ctrl-World inference with SiTo (spatio-temporal prune-once-hold) on
# humanoid/multiview (replay GT state, --config humanoid).
#
# Mirrors scripts/run_humanoid_multiview_ctrlworld.sh (dense) in logic / paths /
# checkpoint / --config humanoid, only routing output to sampling_dataset/sito
# and adding the SiTo ST-hold flags (acceleration-benchmark + sampling-dataset-layout).
#
# NOTE (ctrlworld-pruning-insights): on this SVD UNet token pruning has no free
# lunch — only SiTo spatiotemporal-hold can exceed ~1.1x, and pruning before the
# temporal block trades quality for speed. keep_ratio 0.9 / max_downsample_ratio 1
# is the quality-leaning point; lower keep_ratio (0.5) + max_downsample_ratio 4 is
# faster but lossy. Re-tuned per embodiment.
#
# Output: sampling_dataset/sito/humanoid/{output,input}/multiview/ctrlworld/{category}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_humanoid_multiview_ctrlworld_sito.sh
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   SITO_ST_KEEP_RATIO, SITO_MAX_DOWNSAMPLE_RATIO

set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "${ROOT}"

source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

SVD_MODEL_PATH="${ROOT}/checkpoints/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/openai/clip-vit-base-patch32"
CKPT="${CKPT:-${ROOT}/exp_logs/ctrlworld_logs/humanoid_multiview/checkpoint-200000.pt}"

DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/models/Ctrl-World/dataset_example/${DATASET_NAME}}"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"
SPLIT="${SPLIT:-all}"
# 3-view strip by default; the 4-view grid wrapper passes CONFIG=humanoid_grid.
CONFIG="${CONFIG:-humanoid}"

# SiTo spatio-temporal hold parameters (TUNED for humanoid multiview Ctrl-World,
# ckpt-200000, 2 val episodes; dense = PSNR 21.27 / 117.5s). Sweep:
#   k0.9/d1 -> 21.25 / 1.01x (no speedup) ; k0.8/d4 -> 20.87 / 1.05x ;
#   k0.7/d4 -> 20.49 / 1.26x (main table view smears — lossy) ;
#   k0.5/d4 -> 17.31 / 1.44x (CORRUPT).
# SiTo has NO clean fast point here (pruning before the temporal block smears the
# largest view). DEFAULT = k0.8/d4 (mild, 1.05x). Use k0.7/d4 for ~1.26x if you
# accept the smear. ITM ST-hold beats SiTo at equal speed on this model — prefer it.
SITO_ST_KEEP_RATIO="${SITO_ST_KEEP_RATIO:-0.8}"
SITO_MAX_DOWNSAMPLE_RATIO="${SITO_MAX_DOWNSAMPLE_RATIO:-4}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/sito/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/sito/humanoid/input/multiview/ctrlworld}"

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
echo "Ctrl-World + SiTo (ST-hold) | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  config=${CONFIG}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT}  down_sample=${DOWN_SAMPLE}"
echo "  Params: st_keep_ratio=${SITO_ST_KEEP_RATIO}, max_downsample_ratio=${SITO_MAX_DOWNSAMPLE_RATIO}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer_bimanual_multiview_ctrlworld.py \
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
    --use_sito \
    --sito_spatiotemporal_hold \
    --sito_st_keep_ratio "${SITO_ST_KEEP_RATIO}" \
    --sito_max_downsample_ratio "${SITO_MAX_DOWNSAMPLE_RATIO}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
