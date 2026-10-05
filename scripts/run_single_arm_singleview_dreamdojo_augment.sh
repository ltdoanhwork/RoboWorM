#!/usr/bin/env bash
# DreamDojo single_arm singleview AUGMENT run (object swapped in every frame).
#
# Mirrors run_single_arm_singleview_dreamdojo_dense.sh (same ckpt / infer script
# / autoregressive rollout), but:
#   Step 1: build a mini LeRobot dataset with the OBJECT swapped in every frame
#           of observation.images.global (scripts/augment/build_dreamdojo_edited_lerobot.py,
#           runs in the dreamgen venv which has diffusers), parquet/actions kept.
#   Step 2: run DreamDojo inference with --dataset-path pointing at that mini
#           dataset, routing output under sampling_dataset/augment/.
#
# DreamDojo uses prompt="" and only the first GT frame + actions; there is no GT
# object match, so judge quality on full_pred (per augment plan).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_arm_singleview_dreamdojo_augment.sh
#   SPLITS="makovian" bash scripts/run_single_arm_singleview_dreamdojo_augment.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Edit config (Step 1) ---
# Qwen-Image-Edit (diffusers>=0.35 in .venv_edit) = clean object swap, no smear,
# background preserved -> mask OFF by default. Override EDIT_MODEL=ip2p USE_MASK=1
# EDIT_VENV=models/dreamgen/.venv for the old InstructPix2Pix path.
EDIT_MODEL="${EDIT_MODEL:-qwen}"
EDIT_SEED="${EDIT_SEED:-0}"
EDIT_IMAGE_GUIDANCE="${EDIT_IMAGE_GUIDANCE:-1.8}"
USE_MASK="${USE_MASK:-0}"
EDIT_VENV="${EDIT_VENV:-${ROOT}/.venv_edit}"
MANIFEST="${MANIFEST:-${ROOT}/augment_manifest.json}"
SRC_ROOT="${SRC_ROOT:-${ROOT}/datasets/single_arm/singleview}"
LEROBOT_ROOT="${LEROBOT_ROOT:-${ROOT}/sampling_dataset/augment/_lerobot/single_arm/singleview}"

# --- Inference config (Step 2), same as dense ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/augment/single_arm/output/singleview/dreamdojo}"
SPLITS="${SPLITS:-makovian non_makovian}"
# Frame-0-only recolor means the un-edited tail would drive LAM back toward the
# original object, so disable LAM by default for augment (NO_LAM=0 to re-enable).
NO_LAM="${NO_LAM:-1}"
# Diffusion CFG scale for the rollout. Dense uses 0.0; raise it (e.g. 3-7) to make
# the model follow the (edited) frame-0 conditioning more strongly vs its prior.
INFER_GUIDANCE="${INFER_GUIDANCE:-0.0}"

mask_flag=()
[ "${USE_MASK}" = "1" ] && mask_flag=(--use-mask)

# ---- Step 1: build edited mini LeRobot datasets (edit venv) ----
for split in ${SPLITS}; do
  echo "=== [Step 1/${split}] Building edited mini LeRobot dataset ==="
  (
    export VENV="${EDIT_VENV}"
    export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
    source "${VENV}/bin/activate"
    export HF_HOME="${HF_HOME:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.cache/huggingface}"
    export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
    python "${ROOT}/scripts/augment/build_dreamdojo_edited_lerobot.py" \
      --manifest "${MANIFEST}" --subset "${split}" \
      --src "${SRC_ROOT}" --out "${LEROBOT_ROOT}" \
      --model "${EDIT_MODEL}" --seed "${EDIT_SEED}" \
      --image-guidance "${EDIT_IMAGE_GUIDANCE}" --fps 15 "${mask_flag[@]}"
  )
done

# ---- Step 2: DreamDojo inference on edited mini datasets (DreamDojo venv) ----
export VENV="${ROOT}/models/DreamDojo/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

for split in ${SPLITS}; do
  dataset_dir="${LEROBOT_ROOT}/${split}"
  if [ ! -d "${dataset_dir}/data" ]; then
    echo "Skipping ${split}: no edited data at ${dataset_dir}/data" >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  out_cache="${ROOT}/output/dreamdojo_augment_single_arm_sv_${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== [Step 2/${split}] DreamDojo augment inference -> ${save_base} ==="
  nolam_flag=()
  [ "${NO_LAM}" = "1" ] && nolam_flag=(--no-lam)
  python "${ROOT}/scripts/infer_single_arm_singleview_autoregressive.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_dir}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --save-fps 15 \
    --guidance "${INFER_GUIDANCE}" \
    --output-dir "${out_cache}" \
    --no-gt "${nolam_flag[@]}"
done

echo ""
echo "=========================================="
echo "Done! Augment results: ${SAVE_ROOT}"
echo "=========================================="
