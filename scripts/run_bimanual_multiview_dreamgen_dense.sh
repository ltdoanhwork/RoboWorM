#!/usr/bin/env bash
# DreamGen dense multiview inference — bimanual (4-view 2×2 grid).
# Uses the bimanual_multiview_seg16 fine-tuned 2B checkpoint (checkpoints_40_per, latest iter),
# the same run as run_bimanual_multiview_dreamgen_full.sh (all_multiview no longer exists).
# Runs both makovian and non_makovian splits across all task directories.
#
# IMPORTANT: two-phase design so the 2B model is loaded ONCE.
#   Phase 1 (CPU): prepare grids + per-task batch_input.json for ALL tasks.
#   Phase 2 (GPU): merge every batch_input.json into one combined json, then
#                  run examples.video2world a SINGLE time over all episodes.
#   (video2world loads the pipeline once and loops over all batch entries; the
#    old per-task loop reloaded the whole model each task = very slow.)
#
# Grid layout (--four_views):
#   TL: cam_high  |  TR: cam_low
#   BL: cam_left_wrist  |  BR: cam_right_wrist
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_bimanual_multiview_dreamgen_dense.sh
#
# Overrideable env vars:
#   CHECKPOINTS_DIR, DIT_PATH, NUM_GPUS, MODEL_SIZE, RESOLUTION, FPS,
#   MASTER_PORT, GUIDANCE, MAX_EPISODES, MULTIVIEW_ROOT, INPUT_ROOT, SAVE_ROOT,
#   PREPARE_ONLY (=1 to only build grids+json, no GPU)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (auto-detect latest from latest_checkpoint.txt) ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/bimanual_multiview_seg16/checkpoints_40_per}"
_LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
if [[ -f "${_LATEST_FILE}" ]]; then
  _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
else
  _LATEST_FNAME="iter_000062000.pt"
fi
ITER_NAME="${_LATEST_FNAME%.pt}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
MASTER_PORT="${MASTER_PORT:-29502}"
GUIDANCE="${GUIDANCE:-7}"
MAX_EPISODES="${MAX_EPISODES:-}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

# --- Data/output paths ---
MULTIVIEW_ROOT="${MULTIVIEW_ROOT:-${ROOT}/datasets/bimanual/multiview}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/multiview/dreamgen}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/multiview/dreamgen}"
MERGED_JSON="${INPUT_ROOT}/${ITER_NAME}_batch_input_all.json"

echo "============================================"
echo "DreamDojo Bimanual Multiview Dense (4 views)"
echo "  Checkpoint : ${DIT_PATH}"
echo "  Iter name  : ${ITER_NAME}"
echo "  Model      : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs       : ${NUM_GPUS}"
echo "  Guidance   : ${GUIDANCE}"
echo "  Data root  : ${MULTIVIEW_ROOT}"
echo "  Save root  : ${SAVE_ROOT}"
echo "  Episodes   : ${MAX_EPISODES:-all}"
echo "============================================"

if [[ "${PREPARE_ONLY}" != "1" && ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: prepare grids + per-task batch_input.json for ALL tasks (CPU only)
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE 1] Preparing 4-view grids + batch JSON for all tasks..."
BATCH_JSONS=()
for SPLIT in makovian non_makovian; do
  SPLIT_DIR="${MULTIVIEW_ROOT}/${SPLIT}"
  if [[ ! -d "${SPLIT_DIR}" ]]; then
    echo "[WARN] Missing split dir, skipping: ${SPLIT_DIR}"
    continue
  fi

  while IFS= read -r -d '' TASK_DIR; do
    if [[ ! -f "${TASK_DIR}/meta/episodes.jsonl" ]]; then
      echo "[WARN] No meta/episodes.jsonl in ${TASK_DIR}, skipping"
      continue
    fi

    TASK_NAME="$(basename "${TASK_DIR}")"
    INPUT_TASK_DIR="${INPUT_ROOT}/${SPLIT}/${TASK_NAME}"
    OUTPUT_DIR="${SAVE_ROOT}/${SPLIT}/${ITER_NAME}/${TASK_NAME}"
    BATCH_JSON="${INPUT_TASK_DIR}/batch_input.json"

    echo "  --- [${SPLIT}] ${TASK_NAME} ---"
    PREPARE_ARGS=(
      --dataset_path "${TASK_DIR}"
      --input_dir    "${INPUT_ROOT}"
      --category     "${SPLIT}"
      --output_dir   "${OUTPUT_DIR}"
      --batch_json   "${BATCH_JSON}"
      --four_views
      --no_skip_existing
    )
    if [[ -n "${MAX_EPISODES}" ]]; then
      PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
    fi
    # Tolerate per-task failures (e.g. empty meta/episodes.jsonl): skip, don't abort.
    if ! python "${ROOT}/scripts/prepare_bimanual_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"; then
      echo "[WARN] prepare failed for ${SPLIT}/${TASK_NAME}, skipping"
      continue
    fi

    if [[ -f "${BATCH_JSON}" ]]; then
      BATCH_JSONS+=("${BATCH_JSON}")
    fi
  done < <(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

if [[ "${#BATCH_JSONS[@]}" -eq 0 ]]; then
  echo "[ERROR] No batch_input.json produced; nothing to run." >&2
  exit 1
fi

# Merge all per-task batch JSONs into a single combined batch.
echo ""
echo "[PHASE 1] Merging ${#BATCH_JSONS[@]} batch JSON files -> ${MERGED_JSON}"
python - "${MERGED_JSON}" "${BATCH_JSONS[@]}" <<'PY'
import json, sys
out_path = sys.argv[1]
merged = []
for p in sys.argv[2:]:
    with open(p) as f:
        merged.extend(json.load(f))
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f"  merged {len(merged)} episodes from {len(sys.argv) - 2} tasks")
PY

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo ""
  echo "[PREPARE_ONLY] Done. Merged batch: ${MERGED_JSON}"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: run inference ONCE over the merged batch (model loads once)
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE 2] Running DreamGen inference once over merged batch..."
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m models.dreamgen.examples.video2world \
    --model_size       "${MODEL_SIZE}" \
    --resolution       "${RESOLUTION}" \
    --fps              "${FPS}" \
    --dit_path         "${DIT_PATH}" \
    --batch_input_json "${MERGED_JSON}" \
    --guidance         "${GUIDANCE}" \
    --num_gpus         "${NUM_GPUS}" \
    --disable_guardrail
else
  python \
    -m models.dreamgen.examples.video2world \
    --model_size       "${MODEL_SIZE}" \
    --resolution       "${RESOLUTION}" \
    --fps              "${FPS}" \
    --dit_path         "${DIT_PATH}" \
    --batch_input_json "${MERGED_JSON}" \
    --guidance         "${GUIDANCE}" \
    --num_gpus         "${NUM_GPUS}" \
    --disable_guardrail
fi

echo ""
echo "All done. Results under: ${SAVE_ROOT}"
