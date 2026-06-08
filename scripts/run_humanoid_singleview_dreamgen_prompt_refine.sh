#!/usr/bin/env bash
# =============================================================================
# Run DreamGen (video2world_gr00t_lvg) humanoid singleview with per-chunk
# prompt refinement. At each chunk boundary, the VLM re-describes the scene
# using the last generated frame, keeping the text prompt aligned with visual state.
#
# Output: sampling_dataset/prompt_refine/humanoid/output/singleview/dreamgen/{split}/{ITER_NAME}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_prompt_refine.sh
#   SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_humanoid_singleview_dreamgen_prompt_refine.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

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
MASTER_PORT="${MASTER_PORT:-29540}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"

# --- SiTo parameters ---
SITO_START_LAYER_IDX="${SITO_START_LAYER_IDX:-6}"
SITO_KEEP_LAST_N_DENSE="${SITO_KEEP_LAST_N_DENSE:-2}"

# --- Dataset / output ---
INPUT_ROOT="${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen"
OUTPUT_ROOT="${ROOT}/sampling_dataset/dense/humanoid/output/singleview/dreamgen"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-1 2 3 4 5 6 7 8 9 10 11 12 13 15 17 18 19 23 24 25 31}"

echo "============================================"
echo "DreamGen Humanoid Singleview + SiTo + Prompt Refinement per Chunk"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  SiTo:       start_layer=${SITO_START_LAYER_IDX}, keep_last_n_dense=${SITO_KEEP_LAST_N_DENSE}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
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

  # Rewrite output paths to prompt_refine/ directory
  local pr_batch_json
  pr_batch_json="$(mktemp --suffix=_dreamgen_prompt_refine.json)"
  python3 - "${src_batch_json}" "${pr_batch_json}" "${OUTPUT_ROOT}/${split}/${ITER_NAME}" <<'PY'
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
  item_count="$(python3 -c "import json; print(len(json.load(open('${pr_batch_json}'))))")"

  mkdir -p "${OUTPUT_ROOT}/${split}/${ITER_NAME}"
  echo "  === Running: ${label} (${item_count} episodes, chunks=${num_chunks}) ==="

  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant gr1 \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${pr_batch_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --disable_guardrail \
    --no_skip_existing \
    --enable_prompt_refiner_per_chunk \
    --use_sito \
    --sito_start_layer_idx "${SITO_START_LAYER_IDX}" \
    --sito_keep_last_n_dense "${SITO_KEEP_LAST_N_DENSE}"

  MASTER_PORT=$((MASTER_PORT + 1))
  rm -f "${pr_batch_json}"
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
    run_inference "${batch_json}" "humanoid/${split}/${nc}chunks" "${nc}" "${split}"
  done
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${ITER_NAME}/"
echo "============================================"
