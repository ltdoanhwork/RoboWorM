#!/usr/bin/env bash
# =============================================================================
# run_humanoid_singleview_dreamgen_dicache.sh
# DreamGen (video2world_gr00t_lvg, gr1 variant) humanoid SINGLEVIEW with DiCache
# acceleration.
#
# Mirrors the dense humanoid singleview pipeline (same checkpoint / resolution /
# fps / guidance / aspect / cond frames, same shared dense batch inputs), and only:
#   1. forces single-GPU (cache backends require context_parallel_size == 1, so
#      no torchrun) and
#   2. adds the DiCache flags and
#   3. rewrites each episode's output_video from the dense batch json into
#      sampling_dataset/dicache/ (per sampling-dataset-layout rule).
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/dicache/humanoid/output/singleview/dreamgen/{split}/{iter}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_dicache.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_humanoid_singleview_dreamgen_dicache.sh
#   DICACHE_REL_L1_THRESH=0.05 bash scripts/run_humanoid_singleview_dreamgen_dicache.sh
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
CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/humanoid_singleview_16fps}"
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
# Inputs are the SHARED dense batch inputs; only output_video is rewritten to dicache/.
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/dicache/humanoid/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"

# --- DiCache defaults (TUNED optimal for humanoid singleview 2B) ---
# Measured on episode_000200 (makovian, 2 chunks) single-GPU vs dense 259.9s:
#   thresh 0.08 -> 178.0s (1.46x)  quality ~ dense
#   thresh 0.15 -> 133.5s (1.95x)  quality ~ dense  <-- OPTIMAL (default, knee)
#   thresh 0.20 -> 126.9s (2.05x)  crate/bg corrupt (too aggressive)
DICACHE_NUM_STEPS="${DICACHE_NUM_STEPS:-35}"
DICACHE_REL_L1_THRESH="${DICACHE_REL_L1_THRESH:-0.15}"
DICACHE_RET_RATIO="${DICACHE_RET_RATIO:-0.2}"
DICACHE_PROBE_DEPTH="${DICACHE_PROBE_DEPTH:-2}"

echo "============================================"
echo "DreamGen Humanoid Singleview — DiCache (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Cond frames: ${NUM_CONDITIONAL_FRAMES} | Aspect: ${ASPECT_RATIO}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  DiCache:    thresh=${DICACHE_REL_L1_THRESH} ret=${DICACHE_RET_RATIO} probe=${DICACHE_PROBE_DEPTH} steps=${DICACHE_NUM_STEPS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

DICACHE_FLAGS=(
  --dicache-enabled
  --dicache-num-steps     "${DICACHE_NUM_STEPS}"
  --dicache-rel-l1-thresh "${DICACHE_REL_L1_THRESH}"
  --dicache-ret-ratio     "${DICACHE_RET_RATIO}"
  --dicache-probe-depth   "${DICACHE_PROBE_DEPTH}"
)

run_inference() {
  local src_batch_json="$1"
  local label="$2"
  local num_chunks="$3"
  local split="$4"

  if [[ ! -f "${src_batch_json}" ]]; then
    echo "  [SKIP] ${label}: batch file not found"
    return
  fi

  # Rewrite output paths: dense/ -> dicache/
  local cache_batch_json
  cache_batch_json="$(mktemp --suffix=_dreamgen_dicache.json)"
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
    "${DICACHE_FLAGS[@]}"

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
