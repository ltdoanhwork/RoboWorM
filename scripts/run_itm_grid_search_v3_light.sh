#!/usr/bin/env bash
# Grid search v3: ultra-light configs. Tiết kiệm nhẹ nhẹ thôi.
# Triết lý: ít layers, ít tokens bị ảnh hưởng, late diffusion steps only.
#
# Save to same grid_search_v2/ for comparison.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_itm_grid_search_v3_light.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINTS_DIR="${ROOT}/checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000"
EXPERIMENT="dreamdojo_2b_480_640_gr1"
NUM_FRAMES=49
NUM_SAMPLES="${NUM_SAMPLES:-2}"
GRID_ROOT="${ROOT}/tmp_2/grid_search_v2"
SPLIT="${SPLIT:-makovian}"
TIMING_LOG="${GRID_ROOT}/timing_log_v3.txt"

mkdir -p "${GRID_ROOT}"
echo "============================================================" | tee "${TIMING_LOG}"
echo "  Grid Search v3 (light) | GPU ${CUDA_VISIBLE_DEVICES} | ${NUM_SAMPLES} samples" | tee -a "${TIMING_LOG}"
echo "  Started: $(date)" | tee -a "${TIMING_LOG}"
echo "============================================================" | tee -a "${TIMING_LOG}"
echo "" | tee -a "${TIMING_LOG}"

if [ "${SKIP_META_REPAIR:-0}" != "1" ]; then
  python3 "${ROOT}/scripts/repair_humanoid_singleview_split_meta.py" --workspace "${ROOT}"
fi

run_config() {
  local config_name="$1"
  shift
  local extra_args=("$@")
  local save_dir="${GRID_ROOT}/${config_name}/${SPLIT}"
  local out_cache="${ROOT}/output/grid_v3_${config_name}_${SPLIT}"
  mkdir -p "${save_dir}" "${out_cache}"

  echo "--- Config: ${config_name} ---" | tee -a "${TIMING_LOG}"
  echo "  Args: ${extra_args[*]:-none}" | tee -a "${TIMING_LOG}"
  local start_time=$(date +%s)

  python -m models.DreamDojo.examples.action_conditioned \
    -o "${out_cache}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${save_dir}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${NUM_SAMPLES}" \
    --dataset-path "${ROOT}/datasets/humanoid/singleview/${SPLIT}" \
    --data-split full \
    --single-base-index \
    "${extra_args[@]}"

  local end_time=$(date +%s)
  local elapsed=$((end_time - start_time))
  local s_per_it=$(echo "scale=2; ${elapsed}/${NUM_SAMPLES}" | bc)
  echo "  Total time: ${elapsed}s ($(echo "scale=1; ${elapsed}/60" | bc)min)" | tee -a "${TIMING_LOG}"
  echo "  s/it: ${s_per_it}" | tee -a "${TIMING_LOG}"
  echo "" | tee -a "${TIMING_LOG}"
}

# ============================================================
# SiTo: chỉ 2 layers (start=24, keep=0), prune 10%
# Gần như không overhead, chỉ nhẹ nhẹ 2 layers cuối
# ============================================================
run_config "sito_2layers_10pct" \
  --use-sito \
  --sito-start-layer-idx 24 \
  --sito-keep-last-n-dense 0 \
  --sito-prune-ratio 0.10

# ============================================================
# SiTo: 4 layers (start=22, keep=0), prune 10%
# ============================================================
run_config "sito_4layers_10pct" \
  --use-sito \
  --sito-start-layer-idx 22 \
  --sito-keep-last-n-dense 0 \
  --sito-prune-ratio 0.10

# ============================================================
# SiTo: 4 layers (start=22, keep=0), prune 20%
# ============================================================
run_config "sito_4layers_20pct" \
  --use-sito \
  --sito-start-layer-idx 22 \
  --sito-keep-last-n-dense 0 \
  --sito-prune-ratio 0.20

# ============================================================
# ITM: chỉ 4 layers (start=22, keep=0), compress 10%, late steps only
# Merge chỉ từ step 10+ (đã qua giai đoạn structure)
# ============================================================
run_config "itm_4layers_10pct_late" \
  --use-itm \
  --itm-start-layer-idx 22 \
  --itm-keep-last-n-dense 0 \
  --itm-compress-ratio 0.10 \
  --itm-prune-from-step 10 \
  --itm-merge-from-step 12

# ============================================================
# ITM: 6 layers (start=20, keep=0), compress 10%, late steps
# ============================================================
run_config "itm_6layers_10pct_late" \
  --use-itm \
  --itm-start-layer-idx 20 \
  --itm-keep-last-n-dense 0 \
  --itm-compress-ratio 0.10 \
  --itm-prune-from-step 10 \
  --itm-merge-from-step 12

# ============================================================
# ITM: 6 layers, compress 15%, merge from step 15 (half way)
# Chỉ merge nửa sau của diffusion (steps 15-34)
# ============================================================
run_config "itm_6layers_15pct_half" \
  --use-itm \
  --itm-start-layer-idx 20 \
  --itm-keep-last-n-dense 0 \
  --itm-compress-ratio 0.15 \
  --itm-prune-from-step 15 \
  --itm-merge-from-step 17

echo "============================================================" | tee -a "${TIMING_LOG}"
echo "  Grid Search v3 (light) Complete" | tee -a "${TIMING_LOG}"
echo "  Finished: $(date)" | tee -a "${TIMING_LOG}"
echo "============================================================" | tee -a "${TIMING_LOG}"
echo ""
echo "Compare: python scripts/compare_grid_results.py --grid-root tmp_2/grid_search_v2"
