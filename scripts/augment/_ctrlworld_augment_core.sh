#!/bin/bash
# Shared body for the three Ctrl-World multiview AUGMENT run scripts.
#
# NOT meant to be run directly — `scripts/run_{single_arm,bimanual,humanoid}_
# multiview_ctrlworld_augment.sh` set CONFIG_KEY (+ optionally SUBSET) and source
# this. Keeping one body avoids the three wrappers drifting apart, since the only
# real differences are the checkpoint, the prepared dataset dir and which infer
# script consumes the tree.
#
# Contract with the caller: ROOT (repo root), CONFIG_KEY (an augment_paths.py
# config), SUBSET (makovian|non_makovian|both) are already set.
#
# Steps: (2) recolor frame 0 of every view into an augment annotation tree,
#        (3) run the augment infer copy with --no-gt.
# Step 1 is "reuse the dense tree untouched" and needs no code.

: "${ROOT:?ROOT must be set by the wrapper}"
: "${CONFIG_KEY:?CONFIG_KEY must be set by the wrapper}"
SUBSET="${SUBSET:-both}"

PATHS="${ROOT}/scripts/augment/augment_paths.py"

# --- per-embodiment settings ------------------------------------------------
# Defaults are the checkpoints/datasets the DENSE outputs were produced with, so
# augment output is directly comparable to sampling_dataset/dense/.
case "${CONFIG_KEY}" in
  ctrlworld_single_arm_multiview)
    EMBODIMENT=single_arm
    INFER="${ROOT}/scripts/augment/infer_single_arm_multiview_ctrlworld_augment.py"
    CKPT="${CKPT:-${ROOT}/checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt}"
    # One prepared dir per category; the infer script appends it from --subset.
    DATASET_NAME=""
    CW_CONFIG=""
    ;;
  ctrlworld_bimanual_multiview)
    EMBODIMENT=bimanual
    INFER="${ROOT}/scripts/augment/infer_bimanual_multiview_ctrlworld_augment.py"
    CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_5fps/checkpoint-280000.pt}"
    DATASET_NAME="${DATASET_NAME:-bimanual_multiview_5fps}"
    CW_CONFIG="${CONFIG:-bimanual}"
    ;;
  ctrlworld_humanoid_multiview)
    EMBODIMENT=humanoid
    INFER="${ROOT}/scripts/augment/infer_bimanual_multiview_ctrlworld_augment.py"
    CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview/checkpoint-200000.pt}"
    DATASET_NAME="${DATASET_NAME:-humanoid_multiview}"
    CW_CONFIG="${CONFIG:-humanoid}"
    ;;
  ctrlworld_bimanual_multiview_4view_grid)
    # Same recolor pipeline, a DIFFERENT finetuned checkpoint/config (4 views,
    # 2x2 latent canvas) — dense keeps this as its own method dir alongside
    # ctrlworld_bimanual_multiview, not a replacement (see
    # .cursor/rules/ctrlworld-*-4view-grid-*-insights.mdc).
    EMBODIMENT=bimanual
    INFER="${ROOT}/scripts/augment/infer_bimanual_multiview_ctrlworld_augment.py"
    CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/bimanual_multiview_4view_grid/checkpoint-110000.pt}"
    DATASET_NAME="${DATASET_NAME:-bimanual_multiview_4view_grid}"
    CW_CONFIG="${CONFIG:-bimanual_grid}"
    ;;
  ctrlworld_humanoid_multiview_4view_grid)
    EMBODIMENT=humanoid
    INFER="${ROOT}/scripts/augment/infer_bimanual_multiview_ctrlworld_augment.py"
    CKPT="${CKPT:-${ROOT}/checkpoints/finetuned/ctrlworld/humanoid_multiview_4view_grid/checkpoint-140000.pt}"
    DATASET_NAME="${DATASET_NAME:-humanoid_multiview_4view_grid}"
    CW_CONFIG="${CONFIG:-humanoid_grid}"
    ;;
  *)
    echo "[augment-core] unsupported CONFIG_KEY '${CONFIG_KEY}'" >&2
    exit 1
    ;;
esac

# --- Step 2 (edit) config ---------------------------------------------------
# Qwen-Image-Edit (diffusers>=0.35, .venv_edit) recolors cleanly and preserves
# the background, so masking is OFF. Override EDIT_MODEL=ip2p USE_MASK=1
# EDIT_VENV=models/dreamgen/.venv for the old InstructPix2Pix+SAM path.
EDIT_MODEL="${EDIT_MODEL:-qwen}"
EDIT_SEED="${EDIT_SEED:-0}"
EDIT_IMAGE_GUIDANCE="${EDIT_IMAGE_GUIDANCE:-1.8}"
USE_MASK="${USE_MASK:-0}"
EDIT_VENV="${EDIT_VENV:-${ROOT}/.venv_edit}"
MANIFEST="${MANIFEST:-${ROOT}/augment_manifest.json}"

DENSE_EXAMPLE_DIR="${DENSE_EXAMPLE_DIR:-${ROOT}/models/Ctrl-World/dataset_example}"
AUGMENT_EXAMPLE_DIR="${AUGMENT_EXAMPLE_DIR:-${ROOT}/models/Ctrl-World/dataset_example_augment}"

# --- Step 3 (infer) config --------------------------------------------------
SVD_MODEL_PATH="${ROOT}/checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
CLIP_MODEL_PATH="${ROOT}/checkpoints/base/openai/clip-vit-base-patch32"
META_INFO="${ROOT}/models/Ctrl-World/dataset_meta_info"
DOWN_SAMPLE="${DOWN_SAMPLE:-6}"

# All paths come from augment_paths.py so the layout can never drift from dense.
OUTPUT_BASE="${OUTPUT_ROOT:-$(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base)}"
INPUT_BASE="${INPUT_ROOT:-$(python3 "${PATHS}" --config "${CONFIG_KEY}" --io input --base)}"

if [ "${SUBSET}" = "both" ]; then subsets=(makovian non_makovian); else subsets=("${SUBSET}"); fi

echo "=========================================================="
echo "Ctrl-World ${EMBODIMENT} multiview AUGMENT (recolor, no-gt)"
echo "  categories:  ${subsets[*]}"
echo "  manifest:    ${MANIFEST}"
echo "  edit:        ${EDIT_MODEL} (seed=${EDIT_SEED}, mask=${USE_MASK}, frame 0 only)"
echo "  ckpt:        ${CKPT}"
echo "  augment ds:  ${AUGMENT_EXAMPLE_DIR}${DATASET_NAME:+/${DATASET_NAME}}"
echo "  output:      ${OUTPUT_BASE}"
echo "=========================================================="

# ---- Step 2: recolor frame 0 -> augment annotation tree (.venv_edit) --------
if [ "${SKIP_EDIT:-0}" != "1" ]; then
  mask_flag=()
  [ "${USE_MASK}" = "1" ] && mask_flag=(--use-mask)
  limit_flag=()
  [ -n "${LIMIT:-}" ] && limit_flag=(--limit "${LIMIT}")

  for subset in "${subsets[@]}"; do
    echo ""
    echo "[Step 2/${subset}] Recoloring object in frame 0 of every view..."
    (
      VENV="${EDIT_VENV}"
      export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
      # shellcheck disable=SC1091
      source "${VENV}/bin/activate"
      export HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/huggingface}"
      export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
      python "${ROOT}/scripts/augment/edit_ctrlworld_views.py" \
        --config "${CONFIG_KEY}" --subset "${subset}" \
        --manifest "${MANIFEST}" \
        --dense-root "${DENSE_EXAMPLE_DIR}" --augment-root "${AUGMENT_EXAMPLE_DIR}" \
        --model "${EDIT_MODEL}" --seed "${EDIT_SEED}" \
        --image-guidance "${EDIT_IMAGE_GUIDANCE}" "${mask_flag[@]}" "${limit_flag[@]}"
    )
  done
fi

[ "${SKIP_INFER:-0}" = "1" ] && { echo "[augment-core] SKIP_INFER=1, stopping after edit."; return 0 2>/dev/null || exit 0; }

# ---- Step 3: inference on the augment tree (Ctrl-World venv) ---------------
# shellcheck disable=SC1091
source "${ROOT}/models/Ctrl-World/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/Ctrl-World:${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

EXTRA=""
[ -n "${NUM_EPISODES:-}" ] && EXTRA="${EXTRA} --num_episodes ${NUM_EPISODES}"
[ -n "${NUM_INFERENCE_STEPS:-}" ] && EXTRA="${EXTRA} --num_inference_steps ${NUM_INFERENCE_STEPS}"

if [ "${EMBODIMENT}" = "single_arm" ]; then
  # single_arm keeps one prepared dir per category, so one call per category and
  # the category is the last path segment of --save_dir.
  for subset in "${subsets[@]}"; do
    echo ""
    echo "[Step 3/${subset}] Ctrl-World inference on augment tree..."
    # shellcheck disable=SC2086
    python "${INFER}" \
      --svd_model_path "${SVD_MODEL_PATH}" \
      --clip_model_path "${CLIP_MODEL_PATH}" \
      --ckpt_path "${CKPT}" \
      --subset "${subset}" \
      --dataset_example_dir "${AUGMENT_EXAMPLE_DIR}" \
      --dataset_meta_info_path "${META_INFO}" \
      --save_dir "${OUTPUT_BASE}/${subset}" \
      --input_save_dir "${INPUT_BASE}/${subset}" \
      --no-gt ${EXTRA}
  done
else
  # bimanual/humanoid annotation stems embed the category, and the infer script
  # derives the `{category}/` path segment itself -> one call, split-less base.
  echo ""
  echo "[Step 3] Ctrl-World inference on augment tree (both categories)..."
  # shellcheck disable=SC2086
  python "${INFER}" \
    --config "${CW_CONFIG}" \
    --svd_model_path "${SVD_MODEL_PATH}" \
    --clip_model_path "${CLIP_MODEL_PATH}" \
    --ckpt_path "${CKPT}" \
    --dataset_dir "${AUGMENT_EXAMPLE_DIR}/${DATASET_NAME}" \
    --dataset_meta_info_path "${META_INFO}" \
    --dataset_name "${DATASET_NAME}" \
    --save_dir "${OUTPUT_BASE}" \
    --input_save_dir "${INPUT_BASE}" \
    --infer_down_sample "${DOWN_SAMPLE}" \
    --split all \
    --no-gt ${EXTRA}
fi

echo ""
echo "=========================================================="
echo "Done. Augment output: ${OUTPUT_BASE}"
echo "Compare against:      $(python3 "${PATHS}" --config "${CONFIG_KEY}" --io output --base --dense)"
echo "=========================================================="
