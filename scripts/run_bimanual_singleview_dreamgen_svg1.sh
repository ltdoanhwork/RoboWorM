#!/usr/bin/env bash
# =============================================================================
# run_bimanual_singleview_dreamgen_svg1.sh
# DreamGen (video2world_gr00t_lvg, droid variant) bimanual SINGLEVIEW with
# Sparse VideoGen 1 (SVG1) sparse attention.
#
# Mirrors dense bimanual singleview (shared dense batch inputs, single-GPU),
# adds SVG1 flags, rewrites output into sampling_dataset/svg1/.
#
# Output: sampling_dataset/svg1/bimanual/output/singleview/dreamgen/{split}/{iter}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamgen_svg1.sh
#   SVG_SPARSITY=0.5 bash scripts/run_bimanual_singleview_dreamgen_svg1.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv}"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"
# SVG1 uses flex_attention (torch.compile/inductor), NOT flashinfer VariableBlock,
# so it stays on the standard dreamgen .venv (flashinfer 0.3.1 in .venv_svg breaks
# SVG1's flex_attention compile). JIT caches -> workspace (home quota too small).
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT}/.jitcache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${ROOT}/.jitcache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${ROOT}/.jitcache/flashinfer}"
export FLASHINFER_JIT_DIR="${FLASHINFER_JIT_DIR:-${ROOT}/.jitcache/flashinfer}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.jitcache/xdg}"
export TMPDIR="${TMPDIR:-${ROOT}/.jitcache/tmp}"
# Force in-process inductor compile so the compile worker inherits the cache env
# above (subprocess workers otherwise fall back to the quota-limited home cache).
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.jitcache/mpl}"
mkdir -p "${ROOT}/.jitcache/triton" "${ROOT}/.jitcache/torchinductor" "${ROOT}/.jitcache/flashinfer" "${ROOT}/.jitcache/xdg" "${ROOT}/.jitcache/tmp" "${ROOT}/.jitcache/mpl"

CKPT_BASE="${CKPT_BASE:-${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/bimanual_singleview_16fps}"
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg1/bimanual/output/singleview/dreamgen}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-auto}"
MAX_CHUNKS="${MAX_CHUNKS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- SVG1 parameters. MEASURED on bimanual singleview 2B (iter 110k): ---
# SVG1 is NET-NEGATIVE here: default (sparsity 0.25) = 341s, aggressive
# (sparsity 0.5, start 1) = 324s, both SLOWER than dense ~263s though visually
# clean. Spatial-attn sparsity has little to attack on this 2B DiT and its
# overhead dominates. Kept runnable as a benchmark point; use DiCache/WorldCache
# (~1.3x) for real speedup. Default kept at the clean point.
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-4}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.1}"
SVG_SPARSITY="${SVG_SPARSITY:-0.25}"
SVG_NUM_SAMPLED_ROWS="${SVG_NUM_SAMPLED_ROWS:-64}"
SVG_SAMPLE_MSE_MAX_ROW="${SVG_SAMPLE_MSE_MAX_ROW:-3000}"

echo "============================================"
echo "DreamGen Bimanual Singleview — SVG1 (single-GPU)"
echo "  Checkpoint: ${DIT_PATH} | Iter tag: ${OUT_ITER}"
echo "  Model: ${MODEL_SIZE} | Res: ${RESOLUTION} | FPS: ${FPS} | Guidance: ${GUIDANCE}"
echo "  Splits: ${SPLITS} | Chunks: ${NUM_CHUNKS} (max=${MAX_CHUNKS}) | MaxEps: ${MAX_EPISODES:-all}"
echo "  SVG1: start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} sparsity=${SVG_SPARSITY}"
echo "  Output: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
echo "============================================"

METHOD_FLAGS=(
  --use_svg
  --svg_variant           svg1
  --svg_start_layer_idx   "${SVG_START_LAYER_IDX}"
  --svg_dense_step_frac   "${SVG_DENSE_STEP_FRAC}"
  --svg_sparsity          "${SVG_SPARSITY}"
  --svg_num_sampled_rows  "${SVG_NUM_SAMPLED_ROWS}"
  --svg_sample_mse_max_row "${SVG_SAMPLE_MSE_MAX_ROW}"
)

run_inference() {
  local src_batch_json="$1" label="$2" num_chunks="$3" split="$4"
  [[ ! -f "${src_batch_json}" ]] && { echo "  [SKIP] ${label}: batch file not found"; return; }
  local out_dir="${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  local cache_batch_json; cache_batch_json="$(mktemp --suffix=_dreamgen_svg1.json)"
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
