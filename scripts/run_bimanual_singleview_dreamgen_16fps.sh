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
CKPT_BASE="${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps"
# Paper cell iter (iter_000050000 is not on disk).
ITER_NAME="${ITER_NAME:-iter_000110000}"
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
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
# "auto"  -> per-episode chunk count (matches GT length, recommended)
# "capped"-> fixed 3chunks_17s + 4chunks_23s (legacy; over-generates short episodes)
# "2 3"   -> explicit chunk counts
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
# Skip episodes whose required chunk count exceeds this (heavy autoregressive drift
# + long runtime). Set to 0 to disable the cap.
MAX_CHUNKS="${MAX_CHUNKS:-0}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
PROMPT_REFINE="${PROMPT_REFINE:-1}"

OUT_ITER="${ITER_NAME}_bimanual16fps"

echo "============================================"
echo "DreamGen Bimanual Singleview — bimanual_16fps ckpt"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS} (max=${MAX_CHUNKS})"
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

  # MAX_EPISODES: cap the batch. Without it a "1 episode" timing run generated all
  # 40 episodes of the split (40 x 3 chunks = 120 chunks) because this script had
  # no episode knob at all, unlike its sito/dicache siblings.
  if [[ -n "${MAX_EPISODES:-}" ]]; then
    python3 - "${tmp_batch_json}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
p, n = Path(sys.argv[1]), int(sys.argv[2])
items = json.loads(p.read_text())
p.write_text(json.dumps(items[:n], indent=2))
PY
    item_count="$(python3 -c "import json; print(len(json.load(open('${tmp_batch_json}'))))")"
  fi

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

  # Build prompt refiner flag
  local pr_flag=""
  if [[ "${PROMPT_REFINE}" == "1" ]]; then
    pr_flag="--enable_prompt_refiner_per_chunk"
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
      ${pr_flag}
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
      ${pr_flag}
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

  if [[ "${NUM_CHUNKS}" == "auto" ]]; then
    # Auto-discover all batch_input_*chunks.json files (original full-length, per-episode)
    for batch_json in $(ls "${input_dir}"/batch_input_*chunks.json 2>/dev/null | grep -v '_17s\|_23s' | sort -t_ -k3 -n); do
      nc="$(echo "${batch_json}" | grep -oP '\d+(?=chunks)')"
      if [[ "${MAX_CHUNKS}" -gt 0 && "${nc}" -gt "${MAX_CHUNKS}" ]]; then
        echo "  [SKIP] bimanual/${split}/${nc}chunks: exceeds MAX_CHUNKS=${MAX_CHUNKS}"
        continue
      fi
      run_inference "${batch_json}" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  elif [[ "${NUM_CHUNKS}" == "capped" ]]; then
    # Use capped files: 3 chunks (~17s) and 4 chunks (~23s)
    for suffix in "3chunks_17s" "4chunks_23s"; do
      batch_json="${input_dir}/batch_input_${suffix}.json"
      nc="${suffix%%chunks*}"
      run_inference "${batch_json}" "bimanual/${split}/${suffix}" "${nc}" "${split}"
    done
  else
    for nc in ${NUM_CHUNKS}; do
      batch_json="${input_dir}/batch_input_${nc}chunks.json"
      run_inference "${batch_json}" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  fi
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"
