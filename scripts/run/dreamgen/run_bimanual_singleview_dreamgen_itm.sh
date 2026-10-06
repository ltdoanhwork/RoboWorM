#!/usr/bin/env bash
# =============================================================================
# run_bimanual_singleview_dreamgen_dicache.sh
# DreamGen (video2world_gr00t_lvg, droid variant) bimanual SINGLEVIEW with ITM (token merging)
# acceleration.
#
# Mirrors the dense bimanual singleview pipeline
# (run_bimanual_singleview_dreamgen_16fps.sh): same checkpoint / resolution /
# fps / guidance / aspect / prompt-refiner and the SAME shared dense batch
# inputs. It only:
#   1. forces single-GPU (cache backends require context_parallel_size == 1, so
#      no torchrun) and
#   2. adds the DiCache flags and
#   3. rewrites each episode's output_video from the dense batch json into
#      sampling_dataset/dicache/ (per sampling-dataset-layout rule).
#
# Output (method = top-level dir, per sampling-dataset-layout rule):
#   sampling_dataset/dicache/bimanual/output/singleview/dreamgen/{split}/{iter}/
#
# Usage (repo root):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_dicache.sh
#   CUDA_VISIBLE_DEVICES=0 MAX_EPISODES=1 SPLITS=makovian NUM_CHUNKS=2 FORCE_REGENERATE=1 \
#     bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_dicache.sh
#   DICACHE_REL_L1_THRESH=0.08 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_dicache.sh
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
# Repo root must precede models/dreamgen so `methods.cache_strategy` is importable
# (config imports it at module load, before runtime sys.path injection).
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

# --- Checkpoint (posttrained bimanual_singleview_16fps; matches dense baseline) ---
CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000110000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
# Output tag mirrors the dense baseline sub-path (dense uses ${ITER_NAME}_bimanual16fps)
# so cache outputs sit at the same relative path as dense for trivial comparison.
OUT_ITER="${OUT_ITER:-${ITER_NAME}_bimanual16fps}"

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
PROMPT_REFINE="${PROMPT_REFINE:-1}"

# --- Dataset / output ---
# Inputs are the SHARED dense batch inputs; only output_video is rewritten to dicache/.
INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/itm/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
# "auto" -> discover all batch_input_*chunks.json (per-episode, matches GT length).
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- ITM defaults (DreamGen 2B DiT, see dreamgen pruning insights rule): ---
#   ITM is typically net-negative on small 2B DiT; kept at least-bad "mid" config
#   (start 11, keep_last 4, compress 0.20). Valid benchmark point per accel rule.
ITM_COMPRESS_RATIO="${ITM_COMPRESS_RATIO:-0.20}"
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-11}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-4}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-1}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-2}"

echo "============================================"
echo "DreamGen Bimanual Singleview — ITM (single-GPU)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  Model:      ${MODEL_SIZE} | Resolution: ${RESOLUTION} | FPS: ${FPS}"
echo "  GPUs:       ${NUM_GPUS} | Guidance: ${GUIDANCE} | Aspect: ${ASPECT_RATIO}"
echo "  Splits:     ${SPLITS} | Chunks: ${NUM_CHUNKS} (max=${MAX_CHUNKS}) | MaxEps: ${MAX_EPISODES:-all}"
echo "  ITM:        compress=${ITM_COMPRESS_RATIO} start=${ITM_START_LAYER_IDX} keep_last=${ITM_KEEP_LAST_N_DENSE} prune_from=${ITM_PRUNE_FROM_STEP} merge_from=${ITM_MERGE_FROM_STEP}"
echo "  Output:     ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

DICACHE_FLAGS=(
  --use_itm
  --itm_compress_ratio      "${ITM_COMPRESS_RATIO}"
  --itm_start_layer_idx     "${ITM_START_LAYER_IDX}"
  --itm_keep_last_n_dense   "${ITM_KEEP_LAST_N_DENSE}"
  --itm_prune_from_step     "${ITM_PRUNE_FROM_STEP}"
  --itm_merge_from_step     "${ITM_MERGE_FROM_STEP}"
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

  local out_dir="${OUTPUT_ROOT}/${split}/${OUT_ITER}"

  # Rewrite output paths dense/->dicache/, filter pending, optionally cap MAX_EPISODES.
  local cache_batch_json
  cache_batch_json="$(mktemp --suffix=_dreamgen_itm.json)"
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
