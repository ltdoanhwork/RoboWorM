#!/usr/bin/env bash
# =============================================================================
# Test DreamGen bimanual singleview using the bimanual_singleview_16fps checkpoint.
#
# Bimanual model was trained on cam_high view at 16fps (original 30fps → resampled).
# Input batch JSONs already exist from prepare_singleview_dreamgen_batch.py.
#
# Output:
#   sampling_dataset/dense/bimanual/output/singleview/dreamgen/{split}/{OUT_ITER}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh
#   CUDA_VISIBLE_DEVICES=0,1 ITER_NAME=iter_000050000 bash scripts/run_bimanual_singleview_dreamgen_16fps.sh
#   NUM_CHUNKS="2 3" SPLITS="makovian" bash scripts/run_bimanual_singleview_dreamgen_16fps.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

# --- Checkpoint (from bimanual_singleview_16fps training) ---
CKPT_BASE="${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/bimanual_singleview_16fps"
ITER_NAME="${ITER_NAME:-iter_000050000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  echo "Available checkpoints:" >&2
  ls "${CKPT_BASE}/checkpoints_40_per/model/"*.pt 2>/dev/null >&2 || true
  exit 1
fi

# --- Inference settings ---
NUM_GPUS="${NUM_GPUS:-2}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29560}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- Dataset / output ---
INPUT_ROOT="${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen"
OUTPUT_ROOT="${ROOT}/sampling_dataset/dense/bimanual/output/singleview/dreamgen"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

OUT_ITER="${ITER_NAME}_bimanual16fps"

echo "============================================"
echo "DreamGen Bimanual Singleview — bimanual_16fps ckpt"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
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

  local out_dir="${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  mkdir -p "${out_dir}"

  # Rewrite output paths to our output directory
  local tmp_batch_json
  tmp_batch_json="$(mktemp --suffix=_dreamgen_bimanual16fps.json)"
  python3 - "${src_batch_json}" "${tmp_batch_json}" "${out_dir}" <<'PY'
import json, sys
from pathlib import Path

src, dst_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
items = json.loads(Path(src).read_text())
for item in items:
    episode_name = Path(item["output_video"]).name
    item["output_video"] = f"{out_dir}/{episode_name}"
Path(dst_json).write_text(json.dumps(items, indent=2))
print(len(items))
PY

  local item_count
  item_count="$(python3 -c "import json; print(len(json.load(open('${tmp_batch_json}'))))")"

  # Skip already generated unless forced
  if [[ "${FORCE_REGENERATE}" != "1" ]]; then
    local pending
    pending="$(python3 -c "
import json
from pathlib import Path
items = json.loads(Path('${tmp_batch_json}').read_text())
pending = [x for x in items if not Path(x['output_video']).is_file()]
Path('${tmp_batch_json}').write_text(json.dumps(pending, indent=2))
print(len(pending))
")"
    if [[ "${pending}" -eq 0 ]]; then
      echo "  [SKIP] ${label}: all ${item_count} episodes done"
      rm -f "${tmp_batch_json}"
      return
    fi
    echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="
  else
    echo "  === Running: ${label} (${item_count} episodes, chunks=${num_chunks}) ==="
  fi

  if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun \
      --nproc_per_node="${NUM_GPUS}" \
      --master_port="${MASTER_PORT}" \
      -m models.dreamgen.examples.video2world_gr00t_lvg \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${tmp_batch_json}" \
      --num_chunks "${num_chunks}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --no_skip_existing \
      --enable_prompt_refiner_per_chunk
  else
    python -m models.dreamgen.examples.video2world_gr00t_lvg \
      --model_size "${MODEL_SIZE}" \
      --gr00t_variant droid \
      --dit_path "${DIT_PATH}" \
      --batch_input_json "${tmp_batch_json}" \
      --num_chunks "${num_chunks}" \
      --guidance "${GUIDANCE}" \
      --num_gpus "${NUM_GPUS}" \
      --aspect_ratio "${ASPECT_RATIO}" \
      --disable_guardrail \
      --no_skip_existing \
      --enable_prompt_refiner_per_chunk
  fi

  MASTER_PORT=$((MASTER_PORT + 1))
  rm -f "${tmp_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "  [SKIP] ${split}: input dir not found at ${input_dir}"
    continue
  fi

  for nc in ${NUM_CHUNKS}; do
    batch_json="${input_dir}/batch_input_${nc}chunks.json"
    run_inference "${batch_json}" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
  done
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"
