#!/usr/bin/env bash
# =============================================================================
# Run DreamGen (video2world_gr00t) humanoid singleview with Radial sparse attention.
#
# Usage:
#   bash scripts/run_humanoid_singleview_dreamgen_radial.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" bash scripts/run_humanoid_singleview_dreamgen_radial.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint ---
CKPT_BASE="${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/all_singleview"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-1}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29550}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- Radial parameters ---
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

# --- Dataset / output ---
INPUT_ROOT="${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen"
OUTPUT_ROOT="${ROOT}/sampling_dataset/radial/humanoid/output/singleview/dreamgen"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"

echo "============================================"
echo "DreamGen Humanoid Singleview + Radial"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  Radial:     decay=${RADIAL_DECAY_FACTOR}, block_size=${RADIAL_BLOCK_SIZE}, start_layer=${RADIAL_START_LAYER_IDX}, model_type=${RADIAL_MODEL_TYPE}"
echo "  Splits:     ${SPLITS}"
echo "============================================"

run_inference() {
  local src_batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local split="$4"

  if [[ ! -f "${src_batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found"
    return
  fi

  local method_batch_json
  method_batch_json="$(mktemp --suffix=_dreamgen_radial.json)"
  python3 - "${src_batch_json}" "${method_batch_json}" "${OUTPUT_ROOT}/${split}/${ITER_NAME}" <<'PY'
import json, sys
from pathlib import Path
src, dst_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
items = json.loads(Path(src).read_text())
for item in items:
    item["output_video"] = f"{out_dir}/{Path(item['output_video']).name}"
Path(dst_json).write_text(json.dumps(items, indent=2))
print(len(items))
PY

  local pending
  pending="$(python3 - "${method_batch_json}" <<'PY'
import json, sys
from pathlib import Path
items = json.loads(Path(sys.argv[1]).read_text())
pending = [x for x in items if not Path(x["output_video"]).is_file()]
if len(pending) < len(items):
    Path(sys.argv[1]).write_text(json.dumps(pending, indent=2))
print(len(pending))
PY
  )"

  if [[ "${pending}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: all episodes already generated"
    rm -f "${method_batch_json}"
    return
  fi

  mkdir -p "${OUTPUT_ROOT}/${split}/${ITER_NAME}"
  echo "  === Running: ${label} (${pending} pending, chunks=${num_chunks}) ==="

  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m models.dreamgen.examples.video2world_gr00t \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant gr1 \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${method_batch_json}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --fps "${FPS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --disable_guardrail \
    --use_radial \
    --radial_decay_factor "${RADIAL_DECAY_FACTOR}" \
    --radial_block_size "${RADIAL_BLOCK_SIZE}" \
    --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}" \
    --radial_model_type "${RADIAL_MODEL_TYPE}"

  MASTER_PORT=$((MASTER_PORT + 1))
  rm -f "${method_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  [[ ! -d "${input_dir}" ]] && { echo "  [SKIP] ${split}: not found"; continue; }
  for nc in ${NUM_CHUNKS}; do
    run_inference "${input_dir}/batch_input_short_${nc}chunks.json" "humanoid/${split}/short_${nc}chunks" "${nc}" "${split}"
  done
done

echo ""
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${ITER_NAME}/"
