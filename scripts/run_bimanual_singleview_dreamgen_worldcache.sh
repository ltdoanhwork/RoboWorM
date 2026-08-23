#!/usr/bin/env bash
# =============================================================================
# run_bimanual_singleview_dreamgen_worldcache.sh
# DreamGen (video2world_gr00t_lvg, droid variant) bimanual SINGLEVIEW with
# WorldCache acceleration.
#
# Mirrors the dense bimanual singleview pipeline
# (run_bimanual_singleview_dreamgen_16fps.sh): same checkpoint / resolution /
# fps / guidance / aspect / prompt-refiner and the SAME shared dense batch
# inputs. It only:
#   1. forces single-GPU (cache backends require context_parallel_size == 1),
#   2. adds the WorldCache flags, and
#   3. rewrites each episode's output_video into sampling_dataset/worldcache/.
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/worldcache/bimanual/output/singleview/dreamgen/{split}/{iter}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamgen_worldcache.sh
#   WORLDCACHE_REL_L1_THRESH=0.20 bash scripts/run_bimanual_singleview_dreamgen_worldcache.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (posttrained bimanual_singleview_16fps; matches dense baseline) ---
CKPT_BASE="${CKPT_BASE:-${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/bimanual_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000110000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
OUT_ITER="${OUT_ITER:-${ITER_NAME}_bimanual16fps}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

# --- Inference settings (identical to dense baseline) ---
NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
PROMPT_REFINE="${PROMPT_REFINE:-1}"

# --- Dataset / output ---
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/worldcache/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- WorldCache defaults. TUNED for bimanual singleview 2B (iter 110k): ---
#   thresh 0.20 / ret 0.2 / probe 2 -> 201s vs dense ~263s (~1.31x), clean visuals.
#   thresh 0.15 / ret 0.3 / probe 4 -> 264s (~1.0x, barely triggers). Same knee as
#   bimanual multiview: high thresh + low ret + probe 2 is what makes WorldCache
#   competitive with DiCache on this DiT. decay off.
WORLDCACHE_NUM_STEPS="${WORLDCACHE_NUM_STEPS:-35}"
WORLDCACHE_REL_L1_THRESH="${WORLDCACHE_REL_L1_THRESH:-0.20}"
WORLDCACHE_RET_RATIO="${WORLDCACHE_RET_RATIO:-0.2}"
WORLDCACHE_PROBE_DEPTH="${WORLDCACHE_PROBE_DEPTH:-2}"
WORLDCACHE_MOTION_SENSITIVITY="${WORLDCACHE_MOTION_SENSITIVITY:-4.0}"
WORLDCACHE_DYNAMIC_DECAY="${WORLDCACHE_DYNAMIC_DECAY:-0}"

echo "============================================"
echo "DreamGen Bimanual Singleview — WorldCache (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS} | Guidance: ${GUIDANCE} | Aspect: ${ASPECT_RATIO}"
echo "  Splits:     ${SPLITS} | Chunks: ${NUM_CHUNKS} (max=${MAX_CHUNKS}) | MaxEps: ${MAX_EPISODES:-all}"
echo "  WorldCache: thresh=${WORLDCACHE_REL_L1_THRESH} ret=${WORLDCACHE_RET_RATIO} probe=${WORLDCACHE_PROBE_DEPTH} motion=${WORLDCACHE_MOTION_SENSITIVITY} decay=${WORLDCACHE_DYNAMIC_DECAY} steps=${WORLDCACHE_NUM_STEPS}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

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

  local cache_batch_json
  cache_batch_json="$(mktemp --suffix=_dreamgen_worldcache.json)"
  local item_count pending
  item_count="$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "${src_batch_json}")"
  pending="$(python3 - "${src_batch_json}" "${cache_batch_json}" "${out_dir}" "${FORCE_REGENERATE}" "${MAX_EPISODES}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out_dir, force, max_eps = sys.argv[1:6]
force = force == "1"
max_eps = int(max_eps) if max_eps else None
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
    echo "  [SKIP] ${label}: all ${item_count} episodes already generated"
    rm -f "${cache_batch_json}"
    return
  fi

  mkdir -p "${out_dir}"
  echo "  === Running: ${label} (${pending}/${item_count} pending, chunks=${num_chunks}) ==="

  local pr_flag=""
  [[ "${PROMPT_REFINE}" == "1" ]] && pr_flag="--enable_prompt_refiner_per_chunk"

  python -m models.dreamgen.examples.video2world_gr00t_lvg \
    --model_size "${MODEL_SIZE}" \
    --gr00t_variant droid \
    --dit_path "${DIT_PATH}" \
    --batch_input_json "${cache_batch_json}" \
    --num_chunks "${num_chunks}" \
    --guidance "${GUIDANCE}" \
    --num_gpus "${NUM_GPUS}" \
    --aspect_ratio "${ASPECT_RATIO}" \
    --disable_guardrail \
    --no_skip_existing \
    ${pr_flag} \
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

  if [[ "${NUM_CHUNKS}" == "auto" ]]; then
    for batch_json in $(ls "${input_dir}"/batch_input_*chunks.json 2>/dev/null | grep -v '_17s\|_23s' | sort -t_ -k3 -n); do
      nc="$(echo "${batch_json}" | grep -oP '\d+(?=chunks)')"
      if [[ "${MAX_CHUNKS}" -gt 0 && "${nc}" -gt "${MAX_CHUNKS}" ]]; then
        echo "  [SKIP] bimanual/${split}/${nc}chunks: exceeds MAX_CHUNKS=${MAX_CHUNKS}"
        continue
      fi
      run_inference "${batch_json}" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  else
    for nc in ${NUM_CHUNKS}; do
      run_inference "${input_dir}/batch_input_${nc}chunks.json" "bimanual/${split}/${nc}chunks" "${nc}" "${split}"
    done
  fi
done

echo ""
echo "============================================"
echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"
