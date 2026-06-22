#!/usr/bin/env bash
# =============================================================================
# Run DreamGen (video2world_lvg) humanoid singleview — SHORT generation only,
# using the ORIGINAL (base) Cosmos-Predict2-2B Video2World checkpoint — i.e.
# the pretrained model BEFORE any humanoid posttraining. Use this to compare
# base-vs-posttrained generation on the same inputs.
#
# Base checkpoint:
#   checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt
#
# Same pipeline as run_humanoid_singleview_dreamgen_static16fps.sh, but:
#   - DIT_PATH points to the base .pt (no posttraining)
#   - Outputs are tagged "base_2b" so they DON'T collide with iter_* results
#
# Only generates 2-3 chunks per episode (~11.6s to ~17.3s).
#
# Output:
#   sampling_dataset/dense/humanoid/output/singleview/dreamgen/{split}/base_2b/
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamgen_base.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_humanoid_singleview_dreamgen_base.sh
#   MAX_EPISODES=10 SPLITS="makovian" bash scripts/run_humanoid_singleview_dreamgen_base.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (ORIGINAL base Cosmos-Predict2-2B-Video2World, 480p/16fps) ---
DIT_PATH="${DIT_PATH:-${ROOT}/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Base checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29530}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- Dataset / output ---
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense}"
SPLITS="${SPLITS:-makovian non_makovian}"
MAX_EPISODES="${MAX_EPISODES:-}"
THRESHOLD_SECONDS="${THRESHOLD_SECONDS:-10}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# Tag outputs so base-model results don't collide with posttrained iter_* ones.
OUT_ITER="${OUT_ITER:-base_2b}"

echo "============================================"
echo "DreamGen Humanoid Singleview — SHORT (2-3 chunks) — ORIGINAL base 2B ckpt"
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
    batch_tmp="$(mktemp --suffix=_dreamgen_base.json)"
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
