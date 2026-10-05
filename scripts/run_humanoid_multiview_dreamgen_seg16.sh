#!/usr/bin/env bash
# Inference for the humanoid_multiview_seg16 post-training checkpoint
# (DreamGen 2B, 4-view 2×2 grid, 480p @ 16fps).
#
# LENGTH-MATCHED autoregressive rollout (mirrors the singleview LVG scripts,
# e.g. run_humanoid_singleview_dreamgen_static16fps.sh):
#   - Phase 1 (CPU): build 2×2 grids + per-episode length-matched batch groups.
#     Each episode's num_chunks is derived so the generated video length
#     (@16fps) ≈ the source episode length (@30fps). Episodes are grouped by
#     num_chunks (batch_input_{N}chunks.json).
#   - Phase 2 (GPU): for each N-chunk group, merge that group across ALL
#     datasets/splits into one batch and run examples.video2world_lvg ONCE with
#     --num_chunks N (autoregressive, 1 conditional frame per chunk). The model
#     loads once per group.
#
# This replaces the old single-shot video2world logic, which always produced a
# fixed 93-frame (1-chunk) clip regardless of episode length.
#
# LVG chunk math @ 16fps: total = 93 + (num_chunks-1)*92 frames.
#   1ch=93 (~5.8s) · 2ch=185 (~11.6s) · 3ch=277 · 4ch=369 · 5ch=461 · 6ch=553
#
# Grid layout (auto-detected by prepare script, LAYOUT_B for humanoid):
#   TL: cam_high_rgb       |  TR: cam_third_view
#   BL: cam_left_wrist_rgb |  BR: cam_right_wrist_rgb
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_humanoid_multiview_dreamgen_seg16.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 MAX_EPISODES=3 bash scripts/run_humanoid_multiview_dreamgen_seg16.sh
#
# Overrideable env vars:
#   CHECKPOINTS_DIR, DIT_PATH, ITER_NAME, NUM_GPUS, MODEL_SIZE, RESOLUTION, FPS,
#   MASTER_PORT, GUIDANCE, MAX_EPISODES, DATASETS_ROOT, INPUT_ROOT, SAVE_ROOT,
#   SOURCE_FPS, MAX_CHUNKS, ASPECT_RATIO, FORCE_REGENERATE (=1 regen all),
#   PREPARE_ONLY (=1 to only build grids+json, no GPU)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (seg16 experiment; auto-detect latest from latest_checkpoint.txt) ---
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/multiview/humanoid_multiview_seg16/checkpoints_40_per}"
_LATEST_FILE="${CHECKPOINTS_DIR}/latest_checkpoint.txt"
if [[ -f "${_LATEST_FILE}" ]]; then
  _LATEST_FNAME="$(cat "${_LATEST_FILE}")"
else
  _LATEST_FNAME="iter_000030000.pt"
fi
ITER_NAME="${ITER_NAME:-${_LATEST_FNAME%.pt}}"
DIT_PATH="${DIT_PATH:-${CHECKPOINTS_DIR}/model/${_LATEST_FNAME}}"

# --- Inference settings (match training config: 480p, 16fps) ---
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
MASTER_PORT="${MASTER_PORT:-29503}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
MAX_EPISODES="${MAX_EPISODES:-}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- Length-matching (source humanoid multiview = 30fps) ---
SOURCE_FPS="${SOURCE_FPS:-30}"
MAX_CHUNKS="${MAX_CHUNKS:-6}"

# --- Data/output paths (seg16 gets its own output tag so runs don't collide) ---
DATASETS_ROOT="${DATASETS_ROOT:-${ROOT}/datasets/humanoid/multiview}"
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/multiview/dreamgen_seg16}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/output/multiview/dreamgen_seg16}"

echo "============================================"
echo "DreamGen Humanoid Multiview seg16 — LENGTH-MATCHED LVG (4 views)"
echo "  Checkpoint : ${DIT_PATH}"
echo "  Iter name  : ${ITER_NAME}"
echo "  Model      : ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | AR: ${ASPECT_RATIO}"
echo "  GPUs       : ${NUM_GPUS}"
echo "  Guidance   : ${GUIDANCE}"
echo "  Length     : source ${SOURCE_FPS}fps -> gen ${FPS}fps, max_chunks=${MAX_CHUNKS}"
echo "  Datasets   : ${DATASETS_ROOT}"
echo "  Save root  : ${SAVE_ROOT}"
echo "  Episodes   : ${MAX_EPISODES:-all}"
echo "============================================"

if [[ "${PREPARE_ONLY}" != "1" && ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] Checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Phase 1: prepare grids + per-episode length-matched batch groups (CPU only).
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE 1] Preparing 4-view grids + length-matched batch groups..."
PREPARE_ARGS=(
  --datasets-root "${DATASETS_ROOT}"
  --save-root     "${SAVE_ROOT}"
  --input-root    "${INPUT_ROOT}"
  --iter-name     "${ITER_NAME}"
  --group-by-length
  --source-fps    "${SOURCE_FPS}"
  --model-fps     "${FPS}"
  --max-chunks    "${MAX_CHUNKS}"
)
if [[ -n "${MAX_EPISODES}" ]]; then
  PREPARE_ARGS+=(--max-episodes "${MAX_EPISODES}")
fi
python "${ROOT}/scripts/prepare_multiview_humanoid_grid_batch.py" "${PREPARE_ARGS[@]}"

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo ""
  echo "[PREPARE_ONLY] Done. Grouped batches under: ${SAVE_ROOT}/*/${ITER_NAME}/*/batch_input_*chunks.json"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: for each num_chunks group, merge across all datasets/splits and run
# video2world_lvg ONCE with that --num_chunks (model loads once per group).
# ---------------------------------------------------------------------------
echo ""
echo "[PHASE 2] Running DreamGen LVG inference, grouped by length..."

# Discover which chunk counts exist.
mapfile -t CHUNK_COUNTS < <(
  find "${SAVE_ROOT}" -path "*/${ITER_NAME}/*" -name 'batch_input_*chunks.json' 2>/dev/null \
    | sed -E 's/.*batch_input_([0-9]+)chunks\.json/\1/' | sort -n -u
)
if [[ "${#CHUNK_COUNTS[@]}" -eq 0 ]]; then
  echo "[ERROR] No length-matched batch groups produced; nothing to run." >&2
  exit 1
fi
echo "  chunk groups found: ${CHUNK_COUNTS[*]}"

run_group() {
  local num_chunks="$1"
  local merged_json="$2"

  # Merge every dataset's N-chunk group, dropping already-generated episodes
  # unless FORCE_REGENERATE=1 (safely resumable).
  mapfile -t GROUP_JSONS < <(
    find "${SAVE_ROOT}" -path "*/${ITER_NAME}/*" -name "batch_input_${num_chunks}chunks.json" | sort
  )
  local pending
  pending="$(python3 - "${merged_json}" "${FORCE_REGENERATE}" "${GROUP_JSONS[@]}" <<'PY'
import json, sys
from pathlib import Path
out_path, force = sys.argv[1], sys.argv[2] == "1"
merged = []
for p in sys.argv[3:]:
    for item in json.load(open(p)):
        if force or not Path(item["output_video"]).is_file():
            merged.append(item)
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(len(merged))
PY
  )"

  if [[ "${pending}" -eq 0 ]]; then
    echo "  [SKIP] ${num_chunks}-chunk group: all episodes already generated"
    return
  fi
  echo "  === Running ${num_chunks}-chunk group: ${pending} episodes ==="

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${MASTER_PORT}" \
      -m models.dreamgen.examples.video2world_lvg \
      --model_size          "${MODEL_SIZE}" \
      --resolution          "${RESOLUTION}" \
      --fps                 "${FPS}" \
      --dit_path            "${DIT_PATH}" \
      --batch_input_json    "${merged_json}" \
      --guidance            "${GUIDANCE}" \
      --num_gpus            "${NUM_GPUS}" \
      --num_chunks          "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio        "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  else
    python -m models.dreamgen.examples.video2world_lvg \
      --model_size          "${MODEL_SIZE}" \
      --resolution          "${RESOLUTION}" \
      --fps                 "${FPS}" \
      --dit_path            "${DIT_PATH}" \
      --batch_input_json    "${merged_json}" \
      --guidance            "${GUIDANCE}" \
      --num_gpus            "${NUM_GPUS}" \
      --num_chunks          "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio        "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  fi
  MASTER_PORT=$((MASTER_PORT + 1))
  echo "  === Done ${num_chunks}-chunk group ==="
}

for num_chunks in "${CHUNK_COUNTS[@]}"; do
  merged_json="${INPUT_ROOT}/${ITER_NAME}_batch_input_${num_chunks}chunks_merged.json"
  mkdir -p "${INPUT_ROOT}"
  run_group "${num_chunks}" "${merged_json}"
done

echo ""
echo "All done. Results under: ${SAVE_ROOT}"
