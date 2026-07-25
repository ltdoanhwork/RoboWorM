#!/bin/bash
# Run Ctrl-World inference with DiCache on humanoid/multiview (replay GT state).
#
# Mirrors scripts/run_humanoid_multiview_ctrlworld.sh (dense) exactly in logic /
# paths / checkpoint / --config humanoid, only routing output to
# sampling_dataset/dicache and adding the DiCache flags.
#
# Output: sampling_dataset/dicache/humanoid/{output,input}/multiview/ctrlworld/{category}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_multiview_ctrlworld_dicache.sh
#
# Env overrides: CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, NUM_EPISODES,
#   NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   DICACHE_{REL_L1_THRESH,RET_RATIO,PROBE_DEPTH}

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

# DiCache parameters. TUNED for humanoid multiview Ctrl-World (ckpt-200000, 2 val
# episodes; see .cursor/rules ctrlworld-humanoid-multiview-caching-insights).
# Dense = PSNR 21.27 / 117.5s. Knee at thresh 0.03: PSNR 21.17 (==dense) 1.11x.
# Degrades one step earlier than WorldCache: t0.04 -> 20.79 (-0.48), t0.05 -> 19.63
# (breaks). DEFAULT t0.03. DiCache is ~worse than WorldCache at equal speed here
# (same as bimanual); kept as a valid benchmark point — prefer WorldCache.
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.03}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.4}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-3}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dicache/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/dicache/humanoid/input/multiview/ctrlworld}"

EXTRA=""
if [ -n "${NUM_EPISODES:-}" ]; then EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"; fi
if [ -n "${NUM_INFERENCE_STEPS:-}" ]; then EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"; fi
if [ -n "${GUIDANCE:-}" ]; then EXTRA="${EXTRA} --guidance_scale ${GUIDANCE}"; fi

echo "=========================================================="
echo "Ctrl-World + DiCache | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}"
echo "  dataset=${DATASET_DIR}  split=${SPLIT}  down_sample=${DOWN_SAMPLE}"
echo "  Params: thresh=${DICACHE_REL_L1_THRESH}, ret_ratio=${DICACHE_RET_RATIO}, probe_depth=${DICACHE_PROBE_DEPTH}"
echo "  output=${OUTPUT_BASE}"
echo "=========================================================="

python scripts/infer_bimanual_multiview_ctrlworld.py \
    --config humanoid \
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
    --use_dicache \
    --dicache_rel_l1_thresh "${DICACHE_REL_L1_THRESH}" \
    --dicache_ret_ratio "${DICACHE_RET_RATIO}" \
    --dicache_probe_depth "${DICACHE_PROBE_DEPTH}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
