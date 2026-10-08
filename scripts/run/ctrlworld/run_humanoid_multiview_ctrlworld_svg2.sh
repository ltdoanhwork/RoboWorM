#!/bin/bash
# Ctrl-World inference with SVG2 (SAP / K-Means semantic-aware permutation) sparse
# attention on humanoid/multiview. Mirrors run_humanoid_multiview_ctrlworld.sh
# (dense), swapping sampling_dataset/dense -> svg2 and adding SVG2 flags.
#
# NOTE: env DOES have flashinfer (0.3.1) + triton (3.3.1) — verified 2026-08-17.
# SVG2 genuinely runs the flashinfer block-sparse path (`dynamic_block_sparse_fwd_
# flashinfer`; log prints "Installed SVG FlashInfer gated monkey patch"), NOT an
# SDPA fallback. Kept runnable as a valid benchmark point (acceleration-benchmark
# rule); it is still net-negative on Ctrl-World because the seq (T<=3840) is too
# short for block-sparse to beat dense SDPA. VENV overridable if needed.
#
# Output: sampling_dataset/svg2/humanoid/{output,input}/multiview/ctrlworld[...]/{category}/
#
# Env overrides: VENV, CKPT, DATASET_NAME, DATASET_DIR, DOWN_SAMPLE, SPLIT, CONFIG,
#   NUM_EPISODES, NUM_INFERENCE_STEPS, GUIDANCE, OUTPUT_ROOT, INPUT_ROOT,
#   NUM_SHARDS/SHARD_ID, SVG_{NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,FIRST_LAYERS_FP}
#
# Shared base of run_bimanual_multiview_ctrlworld_4view_grid_svg2.sh, run_humanoid_multiview_ctrlworld_4view_grid_svg2.sh (the paper cell(s)),
# which pass CONFIG=*_grid + their own CKPT/DATASET_NAME/OUTPUT_ROOT. The 3-view
# default CKPT below (humanoid_multiview/checkpoint-200000.pt) was removed on 2026-10-08;
# it is on HF doanh25032004/vgp_exp_logs (see checkpoints/CHECKPOINTS.md).

set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${ROOT}"

VENV="${VENV:-${ROOT}/models/Ctrl-World/.venv}"
source "${VENV}/bin/activate"
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
CONFIG="${CONFIG:-humanoid}"

# SVG2 parameters -- TUNED per model (see the 4view-grid sparse-attention rule).
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"
SVG_FIRST_LAYERS_FP="${SVG_FIRST_LAYERS_FP:-2}"

OUTPUT_BASE="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg2/humanoid/output/multiview/ctrlworld}"
INPUT_BASE="${INPUT_ROOT:-${ROOT}/sampling_dataset/svg2/humanoid/input/multiview/ctrlworld}"

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
echo "Ctrl-World + SVG2 (SAP) | Humanoid Multiview Inference"
echo "  ckpt=${CKPT}  config=${CONFIG}  venv=${VENV}"
echo "  Params: q_centroids=${SVG_NUM_Q_CENTROIDS}, k_centroids=${SVG_NUM_K_CENTROIDS}, top_p=${SVG_TOP_P_KMEANS}"
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
    --use_svg \
    --svg_pattern SVG2 \
    --svg_num_q_centroids "${SVG_NUM_Q_CENTROIDS}" \
    --svg_num_k_centroids "${SVG_NUM_K_CENTROIDS}" \
    --svg_top_p_kmeans "${SVG_TOP_P_KMEANS}" \
    --svg_first_layers_fp "${SVG_FIRST_LAYERS_FP}" ${EXTRA}

echo ""
echo "Done! Results saved to: ${OUTPUT_BASE}"
