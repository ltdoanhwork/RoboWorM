#!/usr/bin/env bash
# =============================================================================
# run_humanoid_singleview_dreamgen_worldcache.sh
# DreamGen (video2world_lvg) humanoid SINGLEVIEW with WorldCache acceleration.
#
# Mirrors the dense humanoid singleview pipeline EXACTLY (same entrypoint
# video2world_lvg, same checkpoint / resolution / fps / guidance / aspect / cond
# frames, same shared dense batch inputs), and only:
#   1. forces single-GPU (cache backends require context_parallel_size == 1, so
#      no torchrun) and
#   2. adds the WorldCache flags and
#   3. rewrites each episode's output_video from the dense batch json into
#      sampling_dataset/worldcache/ (per sampling-dataset-layout rule).
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/worldcache/humanoid/output/singleview/dreamgen/{split}/{iter}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_worldcache.sh
#   CUDA_VISIBLE_DEVICES=0 SPLITS="makovian" NUM_CHUNKS="2 3" bash scripts/run_humanoid_singleview_dreamgen_worldcache.sh
#   WORLDCACHE_REL_L1_THRESH=0.05 bash scripts/run_humanoid_singleview_dreamgen_worldcache.sh
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
# Inputs are the SHARED dense batch inputs; only output_video is rewritten to worldcache/.
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/humanoid/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/humanoid/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
# Optional comma/space-separated episode selector (e.g. "episode_000200,episode_000829"
# or bare ids "000200 000829"). Empty = all episodes in the batch json.
EPISODE_IDS="${EPISODE_IDS:-}"

# --- WorldCache defaults (TUNED optimal for humanoid singleview 2B) ---
# Re-measured on the CORRECT entrypoint (video2world_lvg, matching dense) against
# a matched dense control of 259.4s/episode over 3 makovian 2-chunk episodes.
# Quality below = fidelity to the matched dense prediction (PhyFid-DR).
#   thresh 0.15 ret 0.30 -> 136.0s (1.91x)  SSIM 0.6120 LPIPS 0.3332  (old default)
#   thresh 0.15 ret 0.25 -> 127.1s (2.04x)  SSIM 0.5859 LPIPS 0.3312  worst fidelity
#   thresh 0.12 ret 0.20 -> 121.8s (2.13x)  SSIM 0.6445 LPIPS 0.2866  <-- OPTIMAL
# The 0.12/0.20 point dominates the old 0.15/0.30 default on BOTH speed and
# fidelity: lowering the dense warm-up fraction only stays safe if the threshold
# is tightened with it (0.15/0.20 is the worst of the three, not the best).
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.12}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.2}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-4}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-4.0}"
WORLDCACHE_DYNAMIC_DECAY="${WORLDCACHE_DYNAMIC_DECAY:-1}"
# Set to 0 to drop ALL WorldCache flags and run the identical command line dense.
# This exists so a matched dense control can be measured through exactly this
# code path (same entrypoint / batch json / episode subset) for fair speedups.
WORLDCACHE_ENABLED="${WORLDCACHE_ENABLED:-1}"

echo "============================================"
echo "DreamGen Humanoid Singleview — WorldCache (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Guidance:   ${GUIDANCE}"
echo "  Cond frames: ${NUM_CONDITIONAL_FRAMES} | Aspect: ${ASPECT_RATIO}"
echo "  Splits:     ${SPLITS}"
echo "  Chunks:     ${NUM_CHUNKS}"
echo "  WorldCache: thresh=${WORLDCACHE_REL_L1_THRESH} ret=${WORLDCACHE_RET_RATIO} probe=${WORLDCACHE_PROBE_DEPTH} motion=${WORLDCACHE_MOTION_SENSITIVITY} decay=${WORLDCACHE_DYNAMIC_DECAY} steps=${WORLDCACHE_NUM_STEPS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

WORLDCACHE_FLAGS=()
if [[ "${WORLDCACHE_ENABLED}" == "1" ]]; then
  WORLDCACHE_FLAGS=(
    --worldcache-enabled
    --worldcache-num-steps          "${WORLDCACHE_NUM_STEPS}"
    --worldcache-rel-l1-thresh      "${WORLDCACHE_REL_L1_THRESH}"
    --worldcache-ret-ratio          "${WORLDCACHE_RET_RATIO}"
    --worldcache-probe-depth        "${WORLDCACHE_PROBE_DEPTH}"
    --worldcache-motion-sensitivity "${WORLDCACHE_MOTION_SENSITIVITY}"
  )
  if [[ "${WORLDCACHE_DYNAMIC_DECAY}" == "1" ]]; then
    WORLDCACHE_FLAGS+=(--worldcache-dynamic-decay)
  fi
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

  # Rewrite output paths: dense/ -> worldcache/, and optionally subset episodes
  local cache_batch_json
  cache_batch_json="$(mktemp --suffix=_dreamgen_worldcache.json)"
  EPISODE_IDS="${EPISODE_IDS}" python3 - "${src_batch_json}" "${cache_batch_json}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}" <<'PY'
import json, os, re, sys
from pathlib import Path

src, dst_json, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
items = json.loads(Path(src).read_text())
for item in items:
    episode_name = Path(item["output_video"]).name
    item["output_video"] = f"{out_dir}/{episode_name}"

sel_raw = os.environ.get("EPISODE_IDS", "").strip()
if sel_raw:
    wanted = set()
    for tok in re.split(r"[,\s]+", sel_raw):
        if not tok:
            continue
        wanted.add(tok if tok.startswith("episode_") else f"episode_{tok}")
    items = [x for x in items if Path(x["output_video"]).stem in wanted]
    if not items:
        print("0")
        Path(dst_json).write_text("[]")
        raise SystemExit(0)

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

  python -m models.dreamgen.examples.video2world_lvg \
    --model_size "${MODEL_SIZE}" \
    --resolution "${RESOLUTION}" \
    --fps "${FPS}" \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${cache_batch_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
    --disable_guardrail \
    --disable_prompt_refiner \
    "${WORLDCACHE_FLAGS[@]}"

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
