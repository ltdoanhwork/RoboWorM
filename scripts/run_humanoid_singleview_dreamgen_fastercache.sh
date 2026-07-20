#!/usr/bin/env bash
# =============================================================================
# run_humanoid_singleview_dreamgen_fastercache.sh
# DreamGen (video2world_gr00t_lvg, gr1 variant) humanoid SINGLEVIEW with
# FasterCache acceleration.
#
# Mirrors the dense humanoid singleview pipeline (same checkpoint / resolution /
# fps / guidance / aspect / cond frames, same shared dense batch inputs), and only:
#   1. forces single-GPU (FasterCache v1 raises if num_gpus > 1, so no torchrun) and
#   2. adds the FasterCache flags and
#   3. rewrites each episode's output_video from the dense batch json into
#      sampling_dataset/fastercache/ (per sampling-dataset-layout rule).
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/fastercache/humanoid/output/singleview/dreamgen/{split}/{iter}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_fastercache.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_humanoid_singleview_dreamgen_fastercache.sh
#   FASTERCACHE_MODEL_INTERVAL=4 bash scripts/run_humanoid_singleview_dreamgen_fastercache.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable
# (config imports it at module load, before runtime sys.path injection).
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (posttrained humanoid_singleview_16fps; matches dense baseline) ---
CKPT_BASE="${CKPT_BASE:-${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
# Output tag mirrors the dense baseline sub-path (dense uses ${ITER_NAME}_static16fps)
# so cache outputs sit at the same relative path as dense for trivial comparison.
OUT_ITER="${OUT_ITER:-${ITER_NAME}_static16fps}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# --- Inference settings (identical to dense baseline) ---
# Cache backends are single-GPU only.
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-1}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"

# --- Dataset / output ---
# Inputs are the SHARED dense batch inputs; only output_video is rewritten to fastercache/.
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/fastercache/humanoid/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"

# --- FasterCache defaults (weakest of the 3 caches on this model — see below) ---
# Measured on episode_000200 (makovian, 2 chunks) single-GPU vs dense 259.9s:
#   model_interval 5 / block_interval 3 -> 226.7s (1.15x) with visible drift/mud.
# FasterCache's fixed-interval reuse both speeds up least AND degrades quality
# more than DiCache/WorldCache here. Kept as a valid benchmark data point; prefer
# DiCache (1.95x) or WorldCache (1.49x) for real speedup with quality.
FASTERCACHE_START_STEP="${FASTERCACHE_START_STEP:-0}"
FASTERCACHE_MODEL_INTERVAL="${FASTERCACHE_MODEL_INTERVAL:-5}"
FASTERCACHE_BLOCK_INTERVAL="${FASTERCACHE_BLOCK_INTERVAL:-3}"
FASTERCACHE_DEBUG="${FASTERCACHE_DEBUG:-1}"

echo "============================================"
echo "DreamGen Humanoid Singleview — FasterCache (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Cond frames: ${NUM_CONDITIONAL_FRAMES} | Aspect: ${ASPECT_RATIO}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  FasterCache: start=${FASTERCACHE_START_STEP} model_interval=${FASTERCACHE_MODEL_INTERVAL} block_interval=${FASTERCACHE_BLOCK_INTERVAL} debug=${FASTERCACHE_DEBUG}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

FASTERCACHE_FLAGS=(
  --fastercache-enabled
  --fastercache-start-step     "${FASTERCACHE_START_STEP}"
  --fastercache-model-interval "${FASTERCACHE_MODEL_INTERVAL}"
  --fastercache-block-interval "${FASTERCACHE_BLOCK_INTERVAL}"
)
if [[ "${FASTERCACHE_DEBUG}" == "1" ]]; then
  FASTERCACHE_FLAGS+=(--fastercache-debug)
fi

run_inference() {
  local src_batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local split="$4"

  if [[ ! -f "${src_batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found"
    return
  fi

  # Rewrite output paths: dense/ -> fastercache/
  local cache_batch_json
  cache_batch_json="$(mktemp --suffix=_dreamgen_fastercache.json)"
  python3 - "${src_batch_json}" "${cache_batch_json}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}" <<'PY'
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
  item_count="$(python3 -c "import json; print(len(json.load(open('${cache_batch_json}'))))")"

  # Filter already-done episodes
  local pending
  pending="$(python3 - "${cache_batch_json}" <<'PY'
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
    echo "  [SKIP] ${label}: all ${item_count} episodes already generated"
    rm -f "${cache_batch_json}"
    return
  fi

  mkdir -p "${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="

  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant gr1 \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${cache_batch_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
    --prompt_prefix "${PROMPT_PREFIX}" \
    --disable_guardrail \
    --no_skip_existing \
    "${FASTERCACHE_FLAGS[@]}"

  rm -f "${cache_batch_json}"
  echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  if [[ ! -d "${input_dir}" ]]; then
    echo "  [SKIP] ${split}: input dir not found at ${input_dir}"
    continue
  fi

  for nc in ${NUM_CHUNKS}; do
    batch_json="${input_dir}/batch_input_short_${nc}chunks.json"
    run_inference "${batch_json}" "humanoid/${split}/short_${nc}chunks" "${nc}" "${split}"
  done
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"
