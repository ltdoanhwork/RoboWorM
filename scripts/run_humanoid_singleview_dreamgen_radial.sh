#!/usr/bin/env bash
# =============================================================================
# DreamGen humanoid SINGLEVIEW + Radial Attention.
#
# Mirrors run_humanoid_singleview_dreamgen_static16fps.sh (dense) EXACTLY
# (video2world_lvg, humanoid_singleview_16fps, _static16fps tag, 2/3-chunk short
# rollout, num_conditional_frames=1) -- only routes output to sampling_dataset/radial
# and adds the Radial flags. Radial needs flashinfer (dreamgen .venv has it).
#
# Output: sampling_dataset/radial/humanoid/output/singleview/dreamgen/{split}/<iter>_static16fps/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_radial.sh
#
# Env overrides: VENV, ITER_NAME, DIT_PATH, GUIDANCE, FPS, MASTER_PORT, ASPECT_RATIO,
#   SPLITS, NUM_CHUNKS, FORCE_REGENERATE, DENSE_ROOT, OUTPUT_ROOT,
#   RADIAL_{DECAY_FACTOR,BLOCK_SIZE,START_LAYER_IDX,MODEL_TYPE}
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/${_PYVER}/site-packages/nvidia/cusparse/lib:${VENV}/lib/${_PYVER}/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CKPT_BASE="${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/humanoid_singleview_16fps"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
if [[ ! -f "${DIT_PATH}" ]]; then echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2; exit 1; fi
OUT_ITER="${ITER_NAME}_static16fps"

NUM_GPUS="${NUM_GPUS:-1}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29580}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- Radial parameters ---
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-1.0}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-4}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

DENSE_ROOT="${DENSE_ROOT:-${ROOT}/sampling_dataset/dense}"
INPUT_ROOT="${DENSE_ROOT}/humanoid/input/singleview/dreamgen"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/radial/humanoid/output/singleview/dreamgen}"

echo "============================================"
echo "DreamGen Humanoid Singleview + Radial (static16fps / LVG)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Radial:     decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start_layer=${RADIAL_START_LAYER_IDX} model_type=${RADIAL_MODEL_TYPE}"
echo "  Splits:     ${SPLITS} | chunks: ${NUM_CHUNKS}"
echo "============================================"

run_inference() {
  local src_batch_json="$1" label="$2" num_chunks="$3" split="$4"
  [[ -f "${src_batch_json}" ]] || { echo "  [SKIP] ${label}: batch file not found"; return; }
  local method_batch_json; method_batch_json="$(mktemp --suffix=_dreamgen_radial.json)"
  local pending
  pending="$(python3 - "${src_batch_json}" "${method_batch_json}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}" "${FORCE_REGENERATE}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out_dir, force = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
items = json.loads(Path(src).read_text())
for it in items:
    it["output_video"] = f"{out_dir}/{Path(it['output_video']).name}"
if not force:
    items = [x for x in items if not Path(x["output_video"]).is_file()]
Path(dst).write_text(json.dumps(items, indent=2))
print(len(items))
PY
  )"
  if [[ "${pending}" -eq 0 ]]; then echo "  [SKIP] ${label}: all episodes already generated"; rm -f "${method_batch_json}"; return; fi
  mkdir -p "${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  echo "  === Running: ${label} (${pending} pending, chunks=${num_chunks}) ==="

  local RUN=(python -m models.dreamgen.examples.video2world_lvg)
  [[ "${NUM_GPUS}" -gt 1 ]] && RUN=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m models.dreamgen.examples.video2world_lvg)
  "${RUN[@]}" \
    --model_size "${MODEL_SIZE}" --resolution "${RESOLUTION}" --fps "${FPS}" \
    --dit_path "${DIT_PATH}" --batch_input_json "${method_batch_json}" \
    --guidance "${GUIDANCE}" --num_gpus "${NUM_GPUS}" \
    --num_chunks "${num_chunks}" --num_conditional_frames 1 \
    --aspect_ratio "${ASPECT_RATIO}" --disable_guardrail --disable_prompt_refiner \
    --use_radial --radial_decay_factor "${RADIAL_DECAY_FACTOR}" --radial_block_size "${RADIAL_BLOCK_SIZE}" \
    --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}" --radial_model_type "${RADIAL_MODEL_TYPE}"

  MASTER_PORT=$((MASTER_PORT + 1)); rm -f "${method_batch_json}"; echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  [[ -d "${input_dir}" ]] || { echo "  [SKIP] ${split}: input dir not found"; continue; }
  for nc in ${NUM_CHUNKS}; do
    run_inference "${input_dir}/batch_input_short_${nc}chunks.json" "humanoid/${split}/short_${nc}chunks" "${nc}" "${split}"
  done
done

echo ""; echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
