#!/usr/bin/env bash
# =============================================================================
# run_bimanual_singleview_dreamgen_radial.sh
# DreamGen (video2world_gr00t_lvg, droid variant) bimanual SINGLEVIEW with
# Radial sparse attention.
#
# Mirrors the dense bimanual singleview pipeline: same checkpoint / resolution /
# fps / guidance / aspect / prompt-refiner + SHARED dense batch inputs. Only
# forces single-GPU, adds the Radial flags, and rewrites output_video into
# sampling_dataset/radial/.
#
# Output: sampling_dataset/radial/bimanual/output/singleview/dreamgen/{split}/{iter}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_radial.sh
#   RADIAL_DECAY_FACTOR=0.5 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_radial.sh
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000110000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
OUT_ITER="${OUT_ITER:-${ITER_NAME}_bimanual16fps}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# Sparse attention is single-GPU here (mirror cache scripts).
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
PROMPT_REFINE="${PROMPT_REFINE:-1}"

INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/radial/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- Radial parameters. TUNED for bimanual singleview 2B (iter 110k): ---
#   decay 0.2 / start_layer 1 -> 239s vs dense ~263s (~1.10x) but MILD distortion
#   (center arms/objects slightly warped). decay 1.0 (default) = 276s (~1.0x, clean,
#   no speedup). Radial is weak here; DiCache/WorldCache (~1.3x) are better. This
#   is the faster-but-slightly-lossy knee.
RADIAL_DECAY_FACTOR="${RADIAL_DECAY_FACTOR:-0.2}"
RADIAL_BLOCK_SIZE="${RADIAL_BLOCK_SIZE:-64}"
RADIAL_START_LAYER_IDX="${RADIAL_START_LAYER_IDX:-1}"
RADIAL_MODEL_TYPE="${RADIAL_MODEL_TYPE:-hunyuan}"

echo "============================================"
echo "DreamGen Bimanual Singleview — Radial (single-GPU)"
echo "  Checkpoint: ${DIT_PATH} | Iter tag: ${OUT_ITER}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE}"
echo "  Splits: ${SPLITS} | Chunks: ${NUM_CHUNKS} (max=${MAX_CHUNKS}) | MaxEps: ${MAX_EPISODES:-all}"
echo "  Radial: decay=${RADIAL_DECAY_FACTOR} block=${RADIAL_BLOCK_SIZE} start_layer=${RADIAL_START_LAYER_IDX} model_type=${RADIAL_MODEL_TYPE}"
echo "  Output: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

METHOD_FLAGS=(
  --use_radial
  --radial_decay_factor   "${RADIAL_DECAY_FACTOR}"
  --radial_block_size     "${RADIAL_BLOCK_SIZE}"
  --radial_start_layer_idx "${RADIAL_START_LAYER_IDX}"
  --radial_model_type     "${RADIAL_MODEL_TYPE}"
)

run_inference() {
  local src_batch_json="$1" label="$2" num_chunks="$3" split="$4"
  [[ ! -f "${src_batch_json}" ]] && { echo "  [SKIP] ${label}: batch file not found"; return; }
  local out_dir="${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  local cache_batch_json; cache_batch_json="$(mktemp --suffix=_dreamgen_radial.json)"
  local item_count pending
  item_count="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${src_batch_json}")"
  pending="$(python3 - "${src_batch_json}" "${cache_batch_json}" "${out_dir}" "${FORCE_REGENERATE}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out_dir, force, max_eps = sys.argv[1:6]
force = force == "1"; max_eps = int(max_eps) if max_eps else None
items = json.loads(Path(src).read_text())
for item in items:
    item["output_video"] = f"{out_dir}/{Path(item['output_video']).name}"
if not force:
    items = [x for x in items if not Path(x["output_video"]).is_file()]
if max_eps is not None:
    items = items[:max_eps]
Path(dst).write_text(json.dumps(items, indent=2))
print(len(items))
PY
  )"
  if [[ "${pending}" -eq 0 ]]; then
    echo "  [SKIP] ${label}: all ${item_count} episodes already generated"; rm -f "${cache_batch_json}"; return
  fi
  mkdir -p "${out_dir}"
  echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="
  local pr_flag=""; [[ "${PROMPT_REFINE}" == "1" ]] && pr_flag="--enable_prompt_refiner_per_chunk"
  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" --gr00t_variant droid --dit_path "${DIT_PATH}" \
    --batch_input_json "${cache_batch_json}" --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" --num_gpus "${NUM_GPUS}" --aspect_ratio "${ASPECT_RATIO}" \
    --disable_guardrail --no_skip_existing ${pr_flag} "${METHOD_FLAGS[@]}"
  rm -f "${cache_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  [[ ! -d "${input_dir}" ]] && { echo "  [SKIP] ${split}: input dir not found at ${input_dir}"; continue; }
  if [[ "${NUM_CHUNKS}" == "auto" ]]; then
    for batch_json in $(ls "${input_dir}"/batch_input_*chunks.json 2>/dev/null | grep -v '_17s\|_23s' | sort -t_ -k3 -n); do
      nc="$(echo "${batch_json}" | grep -oP '\d+(?=chunks)')"
      if [[ "${MAX_CHUNKS}" -gt 0 && "${nc}" -gt "${MAX_CHUNKS}" ]]; then continue; fi
      run_inference "${batch_json}" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  else
    for nc in ${NUM_CHUNKS}; do
      run_inference "${input_dir}/batch_input_${nc}chunks.json" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  fi
done

echo ""
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
