#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_batch.sh
# Prepare 2×2 grid + batch_input.json for one single-arm multiview category
# (makovian or non_makovian), then run DreamGen GR00T droid.
#
# Usage (repo root):
#   DATASET_PATH=./datasets/single_arm/multiview/makovian \
#     bash scripts/run_single_arm_multiview_dreamgen_batch.sh
#
# Longer videos: set USE_LVG=1 and NUM_CHUNKS (chains short clips; see docs).
#
# Environment:
#   DATASET_PATH   (required) e.g. .../single_arm/multiview/makovian
#   CATEGORY       optional; inferred from path if unset
#   INPUT_DIR      default ./sampling_dataset/dense/single_arm/input/multiview/dreamgen
#   OUTPUT_DIR     optional override for generated mp4 root
#   BATCH_JSON     optional explicit batch_input.json
#   MAX_EPISODES, MIN_EPISODE_INDEX (prep only), NUM_GPUS, MODEL_SIZE, MASTER_PORT, FPS
#   EPISODE_SOURCE meta|disk — default disk: only episodes that have all 3 mp4s (see prepare script)
#   USE_LVG        set to 1 for examples.video2world_gr00t_lvg + NUM_CHUNKS
#   NUM_CHUNKS     only when USE_LVG=1 (default 6)
#   SKIP_PREPARE   set to 1 to only run inference
#   FORCE_REGENERATE_OUTPUTS  set to 1 to re-run DreamGen even when episode mp4 + .txt exist
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATASET_PATH="${DATASET_PATH:-}"
if [[ -z "${DATASET_PATH}" ]]; then
  echo "[ERROR] Set DATASET_PATH to datasets/single_arm/multiview/makovian (or non_makovian)." >&2
  exit 1
fi

INPUT_DIR="${INPUT_DIR:-./sampling_dataset/dense/single_arm/input/multiview/dreamgen}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-12341}"
FPS="${FPS:-10}"
USE_LVG="${USE_LVG:-0}"
NUM_CHUNKS="${NUM_CHUNKS:-6}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"

infer_category_from_path() {
  if [[ "${DATASET_PATH}" == *"/non_makovian"* ]]; then
    echo non_makovian
  elif [[ "${DATASET_PATH}" == *"/makovian"* ]]; then
    echo makovian
  else
    echo ""
  fi
}

CATEGORY_EFFECTIVE="${CATEGORY:-$(infer_category_from_path)}"
if [[ -z "${CATEGORY_EFFECTIVE}" ]]; then
  echo "[ERROR] Set CATEGORY=makovian or non_makovian (not inferable from DATASET_PATH)." >&2
  exit 1
fi

BATCH_JSON_RESOLVED="${BATCH_JSON:-${INPUT_DIR}/${CATEGORY_EFFECTIVE}/batch_input.json}"

echo "============================================================"
echo " DreamGen multiview — single_arm (droid layout)"
echo "============================================================"
echo " Dataset    : ${DATASET_PATH}"
echo " Category   : ${CATEGORY_EFFECTIVE}"
echo " Batch JSON : ${BATCH_JSON_RESOLVED}"
echo " Model      : ${MODEL_SIZE}  GPUs: ${NUM_GPUS}  FPS: ${FPS}"
echo " USE_LVG    : ${USE_LVG}  NUM_CHUNKS: ${NUM_CHUNKS}"
echo " Skip done  : outputs with complete mp4 + .txt excluded (set FORCE_REGENERATE_OUTPUTS=1 to redo)"
echo "============================================================"
echo ""

source "${REPO_ROOT}/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  PREPARE_ARGS=(
    --dataset_path "${DATASET_PATH}"
    --input_dir    "${INPUT_DIR}"
    --category     "${CATEGORY_EFFECTIVE}"
    --batch_json   "${BATCH_JSON_RESOLVED}"
  )
  if [[ -n "${OUTPUT_DIR:-}" ]]; then
    PREPARE_ARGS+=(--output_dir "${OUTPUT_DIR}")
  fi
  if [[ -n "${MAX_EPISODES:-}" ]]; then
    PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
  fi
  if [[ -n "${MIN_EPISODE_INDEX:-}" ]]; then
    PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
  fi
  if [[ -n "${EPISODE_SOURCE:-}" ]]; then
    PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
  fi
  if [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]]; then
    PREPARE_ARGS+=(--force_regenerate_outputs)
  fi
  PREPARE_ARGS+=(--skip_existing)

  python "${REPO_ROOT}/scripts/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
fi

if [[ ! -f "${BATCH_JSON_RESOLVED}" ]]; then
  echo "[ERROR] Missing ${BATCH_JSON_RESOLVED}" >&2
  exit 1
fi

BATCH_FOR_INFER="${BATCH_JSON_RESOLVED}"
if [[ "${FORCE_REGENERATE_OUTPUTS}" != "1" ]]; then
  BATCH_TMP="$(mktemp --suffix=_dreamgen_batch.json)"
  cleanup_batch_tmp() { rm -f "${BATCH_TMP}"; }
  trap cleanup_batch_tmp EXIT
  ITEM_COUNT="$(python - "${BATCH_JSON_RESOLVED}" "${BATCH_TMP}" <<'PY'
import json, sys
from pathlib import Path

def output_done(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        txt = path.with_suffix(".txt")
        return txt.is_file() and txt.stat().st_size > 0
    except OSError:
        return False

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads(src.read_text(encoding="utf-8"))
pending = [x for x in items if not output_done(Path(x["output_video"]))]
dst.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(pending))
PY
  )"
  BATCH_FOR_INFER="${BATCH_TMP}"
else
  ITEM_COUNT="$(python -c "import json; print(len(json.load(open(r'''${BATCH_JSON_RESOLVED}''', encoding='utf-8'))))")"
fi

if [[ "${ITEM_COUNT}" -eq 0 ]]; then
  echo "[INFO] No pending episodes (all outputs exist or batch empty) — skipping inference."
  exit 0
fi
echo "[INFO] Inference items: ${ITEM_COUNT} (from ${BATCH_JSON_RESOLVED})"

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

if [[ "${USE_LVG}" == "1" ]]; then
  MODULE="examples.video2world_gr00t_lvg"
  INFERENCE_ARGS=(
    -m "${MODULE}"
    --model_size       "${MODEL_SIZE}"
    --gr00t_variant    droid
    --batch_input_json "${BATCH_FOR_INFER}"
    --prompt_prefix    ""
    --num_gpus         "${NUM_GPUS}"
    --disable_guardrail
    --num_chunks       "${NUM_CHUNKS}"
  )
else
  MODULE="examples.video2world_gr00t"
  INFERENCE_ARGS=(
    -m "${MODULE}"
    --model_size       "${MODEL_SIZE}"
    --gr00t_variant    droid
    --batch_input_json "${BATCH_FOR_INFER}"
    --prompt_prefix    ""
    --num_gpus         "${NUM_GPUS}"
    --disable_guardrail
    --fps              "${FPS}"
  )
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${INFERENCE_ARGS[@]}"
else
  python "${INFERENCE_ARGS[@]}"
fi

echo ""
echo "[Done] Inference batch: ${BATCH_FOR_INFER} (manifest: ${BATCH_JSON_RESOLVED})"
