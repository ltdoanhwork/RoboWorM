#!/usr/bin/env bash
# =============================================================================
# Run DreamGen (video2world_lvg) humanoid singleview — SHORT generation only,
# using the posttrained humanoid_singleview_16fps checkpoint (NOT all_singleview).
#
# Only generates 2-3 chunks per episode (~11.6s to ~17.3s) instead of full rollout.
#
# Chunk math @ 16fps:
#   2 chunks = 185 frames ≈ 11.6s
#   3 chunks = 277 frames ≈ 17.3s
#
# Threshold: episodes ≤ 10s (source @ 20fps) → 2 chunks, longer → 3 chunks
#
# Checkpoint:
#   exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps
#   default iter_000090000 (files on disk: 90k / 120k — 50k does NOT exist)
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
#   ITER_NAME=iter_000040000 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
#   MAX_EPISODES=10 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
#   THRESHOLD_SECONDS=8 bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
#   SPLITS="makovian" bash scripts/run_humanoid_singleview_dreamgen_static16fps.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (posttrained humanoid_singleview_16fps) ---
CKPT_BASE="${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  ls "${CKPT_BASE}/checkpoints_40_per/model/"*.pt 2>/dev/null >&2 || true
  exit 1
fi

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29520}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- Dataset / output ---
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense}"
SPLITS="${SPLITS:-makovian non_makovian}"
MAX_EPISODES="${MAX_EPISODES:-}"
THRESHOLD_SECONDS="${THRESHOLD_SECONDS:-10}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# Tag outputs so 16fps-checkpoint results don't collide with all_singleview ones.
OUT_ITER="${ITER_NAME}_static16fps"

echo "============================================"
echo "DreamGen Humanoid Singleview — SHORT (2-3 chunks) — static16fps ckpt"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Splits:     ${SPLITS}"
echo "  Threshold:  ${THRESHOLD_SECONDS}s (≤ → 2 chunks, > → 3 chunks)"
echo "  Episodes:   ${MAX_EPISODES:-all}"
echo "============================================"

# --- Step 1: Prepare short batch_input ---
if [[ "${SKIP_PREPARE}" != "1" ]]; then
  echo ""
  echo "[STEP 1] Preparing short batch_input.json..."
  PREPARE_ARGS=(
    --save-root "${SAVE_ROOT}"
    --splits ${SPLITS}
    --iter-name "${OUT_ITER}"
    --threshold-seconds "${THRESHOLD_SECONDS}"
    --skip-existing
  )
  if [[ -n "${MAX_EPISODES}" ]]; then
    PREPARE_ARGS+=(--max-episodes "${MAX_EPISODES}")
  fi
  python "${ROOT}/scripts/prepare_humanoid_singleview_dreamgen_short.py" "${PREPARE_ARGS[@]}"
fi

# --- Step 2: Run inference ---
echo ""
echo "[STEP 2] Running DreamGen LVG inference (short)..."

run_inference() {
  local batch_json="$1"
  local label="$2"
  local num_chunks="$3"

  if [[ ! -f "${batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found"
    return
  fi

  local item_count
  item_count="$(python3 -c "import json; print(len(json.load(open('${batch_json}'))))")"
  if [[ "${item_count}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: empty batch"
    return
  fi

  local batch_for_infer="${batch_json}"
  if [[ "${FORCE_REGENERATE}" != "1" ]]; then
    local batch_tmp
    batch_tmp="$(mktemp --suffix=_dreamgen_static16fps.json)"
    local pending
    pending="$(python3 - "${batch_json}" "${batch_tmp}" <<'PY'
import json, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
items = json.loads(src.read_text(encoding="utf-8"))
pending = [x for x in items if not Path(x["output_video"]).is_file()]
dst.write_text(json.dumps(pending, indent=2, ensure_ascii=False), encoding="utf-8")
print(len(pending))
PY
    )"
    if [[ "${pending}" -eq 0 ]]; then
      echo "  [SKIP] ${label}: all ${item_count} episodes already generated"
      rm -f "${batch_tmp}"
      return
    fi
    echo "  [INFO] ${label}: ${pending}/${item_count} episodes pending"
    batch_for_infer="${batch_tmp}"
  else
    echo "  [INFO] ${label}: ${item_count} episodes"
  fi

  echo "  === Running: ${label} (num_chunks=${num_chunks}) ==="
  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${MASTER_PORT}" \
      -m models.dreamgen.examples.video2world_lvg \
      --model_size "${MODEL_SIZE}" \
      --resolution "${RESOLUTION}" \
      --fps "${FPS}" \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${batch_for_infer}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --num_chunks "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  else
    python -m models.dreamgen.examples.video2world_lvg \
      --model_size "${MODEL_SIZE}" \
      --resolution "${RESOLUTION}" \
      --fps "${FPS}" \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${batch_for_infer}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --num_chunks "${num_chunks}" \
      --num_conditional_frames 1 \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --disable_prompt_refiner
  fi

  MASTER_PORT=$((MASTER_PORT + 1))

  if [[ "${batch_for_infer}" != "${batch_json}" ]]; then
    rm -f "${batch_for_infer}"
  fi
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${SAVE_ROOT}/humanoid/input/singleview/dreamgen/${split}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "  [SKIP] ${split}: input dir not found"
    continue
  fi

  for num_chunks in 2 3; do
    batch_json="${input_dir}/batch_input_short_${num_chunks}chunks.json"
    run_inference "${batch_json}" "humanoid/${split}" "${num_chunks}"
  done
done

echo ""
echo "Done. Results: ${SAVE_ROOT}/humanoid/output/singleview/dreamgen/{split}/${OUT_ITER}/"
