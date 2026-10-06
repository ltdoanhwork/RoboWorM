#!/usr/bin/env bash
# =============================================================================
# run_bimanual_singleview_dreamgen_svg2.sh
# DreamGen (video2world_gr00t_lvg, droid variant) bimanual SINGLEVIEW with
# Sparse VideoGen 2 (SVG2, k-means semantic-aware) sparse attention.
#
# Mirrors dense bimanual singleview (shared dense batch inputs, single-GPU),
# adds SVG2 flags, rewrites output into sampling_dataset/svg2/.
#
# Output: sampling_dataset/svg2/bimanual/output/singleview/dreamgen/{split}/{iter}/
#
# NOTE (fix): SVG2 needs flashinfer>=0.3 (VariableBlockSparseAttentionWrapper).
# The stock dreamgen .venv has 0.2.5 which OOMs (dense (BH,S,S) SDPA fallback,
# ~57 GiB). This script defaults to .venv_svg (a copy of .venv with flashinfer
# 0.3.1) and routes JIT caches to workspace .jitcache (home quota is too small).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_svg2.sh
#   SVG_DENSE_STEP_FRAC=0.2 bash scripts/run/dreamgen/run_bimanual_singleview_dreamgen_svg2.sh
# =============================================================================
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv_svg}"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"
# JIT compile caches -> workspace (home dir quota is too small for triton/flashinfer).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.jitcache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT}/.jitcache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ROOT}/.jitcache/flashinfer}"
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-${ROOT}/.jitcache/flashinfer}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.jitcache/xdg}"
export TMPDIR="${TMPDIR:-${ROOT}/.jitcache/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.jitcache/mpl}"
mkdir -p "${ROOT}/.jitcache/triton" "${ROOT}/.jitcache/torchinductor" "${ROOT}/.jitcache/flashinfer" "${ROOT}/.jitcache/xdg" "${ROOT}/.jitcache/tmp" "${ROOT}/.jitcache/mpl"

CKPT_BASE="${CKPT_BASE:-${ROOT}/checkpoints/finetuned/dreamgen/posttraining/singleview/bimanual_singleview_16fps}"
ITER_NAME="${ITER_NAME:-iter_000110000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
OUT_ITER="${OUT_ITER:-${ITER_NAME}_bimanual16fps}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2
  exit 1
fi

NUM_GPUS=1
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
PROMPT_REFINE="${PROMPT_REFINE:-1}"

INPUT_ROOT="${INPUT_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/input/singleview/dreamgen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg2/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- SVG2 parameters. FIXED + measured on bimanual singleview 2B (iter 110k): ---
# SVG2 now RUNS (was OOM with flashinfer 0.2.5). Warm-cache ~314s vs dense ~263s
# (~0.84x, net-NEGATIVE) but visually clean. The k-means cluster+permute+block-
# sparse overhead exceeds attention saved on this small 2B DiT. Valid benchmark
# point; use DiCache/WorldCache (~1.3x) for real speedup.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-1}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.3}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.9}"
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.10}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-10}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-1}"

echo "============================================"
echo "DreamGen Bimanual Singleview — SVG2 (single-GPU, .venv_svg/flashinfer0.3)"
echo "  Checkpoint: ${DIT_PATH} | Iter tag: ${OUT_ITER}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE}"
echo "  Splits: ${SPLITS} | Chunks: ${NUM_CHUNKS} (max=${MAX_CHUNKS}) | MaxEps: ${MAX_EPISODES:-all}"
echo "  SVG2: start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} qC=${SVG_NUM_Q_CENTROIDS} kC=${SVG_NUM_K_CENTROIDS} topp=${SVG_TOP_P_KMEANS}"
echo "  Output: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

METHOD_FLAGS=(
  --use_svg
  --svg_variant          svg2
  --svg_start_layer_idx  "${SVG_START_LAYER_IDX}"
  --svg_dense_step_frac  "${SVG_DENSE_STEP_FRAC}"
  --svg_num_q_centroids  "${SVG_NUM_Q_CENTROIDS}"
  --svg_num_k_centroids  "${SVG_NUM_K_CENTROIDS}"
  --svg_top_p_kmeans     "${SVG_TOP_P_KMEANS}"
  --svg_min_kc_ratio     "${SVG_MIN_KC_RATIO}"
  --svg_kmeans_iter_init "${SVG_KMEANS_ITER_INIT}"
  --svg_kmeans_iter_step "${SVG_KMEANS_ITER_STEP}"
)

run_inference() {
  local src_batch_json="$1" label="$2" num_chunks="$3" split="$4"
  [[ ! -f "${src_batch_json}" ]] && { echo "  [SKIP] ${label}: batch file not found"; return; }
  local out_dir="${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  local cache_batch_json; cache_batch_json="$(mktemp --suffix=_dreamgen_svg2.json)"
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
