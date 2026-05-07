#!/usr/bin/env bash
# =============================================================================
# run_bimanual_multiview_dreamgen_batch.sh
# Prepare 2×2 grid inputs + batch_input.json, then run DreamGen multiview (droid variant).
#
# Usage (from repo root):
#   DATASET_PATH=./datasets/bimanual/multiview/makovian/close_toolbox \
#     bash scripts/run_bimanual_multiview_dreamgen_batch.sh
#
# Environment overrides:
#   DATASET_PATH   (required) task dataset root under datasets/bimanual/multiview/...
#   CATEGORY       optional makovian|non_makovian if not inferable from DATASET_PATH
#   INPUT_DIR      default ./sampling_dataset/bimanual/input/multiview/dreamgen
#   OUTPUT_DIR     optional (see prepare_bimanual_multiview_dreamgen_batch.py --help)
#   BATCH_JSON     optional explicit batch_input.json path
#   FOUR_VIEWS     set to 1 for four camera quadrants (no black)
#   DYNAMIC_WRIST  left|right when FOUR_VIEWS unset (default left)
#   MAX_EPISODES, NUM_GPUS, MODEL_SIZE, MASTER_PORT, FPS
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "${DATASET_PATH}" ]]; then
  echo "[ERROR] Set DATASET_PATH to a task dataset root." >&2
  echo "Example: DATASET_PATH=./datasets/bimanual/multiview/makovian/close_toolbox bash $0" >&2
  exit 1
fi

DATASET_NAME="$(basename "${DATASET_PATH}")"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/bimanual/input/multiview/dreamgen}"
FOUR_VIEWS="${FOUR_VIEWS:-0}"
DYNAMIC_WRIST="${DYNAMIC_WRIST:-left}"

MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"

infer_category_from_path() {
  if [[ "${DATASET_PATH}" == *"/non_makovian/"* ]]; then
    echo non_makovian
  elif [[ "${DATASET_PATH}" == *"/makovian/"* ]]; then
    echo makovian
  else
    echo ""
  fi
}

CATEGORY_EFFECTIVE="${CATEGORY:-$(infer_category_from_path)}"
if [[ -z "${CATEGORY_EFFECTIVE}" ]]; then
  echo "[ERROR] Could not infer category from DATASET_PATH; set CATEGORY=makovian or non_makovian." >&2
  exit 1
fi

BATCH_JSON_RESOLVED="${BATCH_JSON:-${INPUT_DIR}/${CATEGORY_EFFECTIVE}/${DATASET_NAME}/batch_input.json}"

echo "============================================================"
echo " DreamGen multiview — bimanual (droid layout)"
echo "============================================================"
echo " Dataset   : ${DATASET_PATH}"
echo " Category  : ${CATEGORY_EFFECTIVE}"
echo " Input root: ${INPUT_DIR}"
echo " Batch JSON: ${BATCH_JSON_RESOLVED}"
echo " Four view : ${FOUR_VIEWS}  dynamic wrist: ${DYNAMIC_WRIST}"
echo " Model     : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo "============================================================"
echo ""

source "${REPO_ROOT}/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

PREPARE_ARGS=(
  --dataset_path "${DATASET_PATH}"
  --input_dir    "${INPUT_DIR}"
  --category     "${CATEGORY_EFFECTIVE}"
  --batch_json   "${BATCH_JSON_RESOLVED}"
)
if [[ -n "${OUTPUT_DIR:-}" ]]; then
  PREPARE_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ "${FOUR_VIEWS}" == "1" ]]; then
  PREPARE_ARGS+=(--four_views)
fi
PREPARE_ARGS+=(--dynamic_wrist "${DYNAMIC_WRIST}")
if [[ -n "${MAX_EPISODES:-}" ]]; then
  PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
fi
PREPARE_ARGS+=(--skip_existing)

python  "${REPO_ROOT}/scripts/prepare_bimanual_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"

if [[ ! -f "${BATCH_JSON_RESOLVED}" ]]; then
  echo "[ERROR] batch_input.json not found at ${BATCH_JSON_RESOLVED}" >&2
  exit 1
fi

AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
  echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
  exit 1
fi
if [[ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
  echo "[ERROR] NUM_GPUS=${NUM_GPUS} but only ${AVAILABLE_GPUS} GPU(s) available." >&2
  exit 1
fi

INFERENCE_ARGS=(
  -m examples.video2world_gr00t
  --model_size       "${MODEL_SIZE}"
  --gr00t_variant    droid
  --batch_input_json "${BATCH_JSON_RESOLVED}"
  --prompt_prefix    ""
  --num_gpus         "${NUM_GPUS}"
  --disable_guardrail
  --fps              "${FPS}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${INFERENCE_ARGS[@]}"
else
  python "${INFERENCE_ARGS[@]}"
fi

echo ""
echo "[Done] Inference used batch: ${BATCH_JSON_RESOLVED}"
