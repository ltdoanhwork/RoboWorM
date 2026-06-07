#!/usr/bin/env bash
# Grid search v2b: additional configs to run on a separate GPU.
# Results saved to same grid_search_v2/ dir for easy comparison.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_itm_grid_search_v2b.sh
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
TIMING_LOG="${GRID_ROOT}/timing_log_v2b.txt"

mkdir -p "${GRID_ROOT}"
echo "============================================================" | tee "${TIMING_LOG}"
echo "  Grid Search v2b | GPU ${CUDA_VISIBLE_DEVICES} | ${SPLIT} | ${NUM_SAMPLES} samples" | tee -a "${TIMING_LOG}"
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
  local out_cache="${ROOT}/output/grid_v2b_${config_name}_${SPLIT}"
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
# Baseline: Dense no CFG (fair comparison for SiTo)
# ============================================================
run_config "dense_no_cfg"

# ============================================================
# ITM: merge_mlp + high compress + max aggressive
# Push token reduction to the max
# ============================================================
run_config "itm_merge_mlp_high_compress" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 0 \
  --itm-merge-from-step 0 \
  --itm-compress-ratio 0.35 \
  --itm-merge-mlp

# ============================================================
# ITM: conservative from README (start=20, keep=4, compress=0.20, late steps)
# Adapted from DreamGen 14B "Conservative" preset to 28-block
# ============================================================
run_config "itm_readme_conservative" \
  --use-itm \
  --itm-start-layer-idx 14 \
  --itm-keep-last-n-dense 4 \
  --itm-compress-ratio 0.20 \
  --itm-prune-from-step 2 \
  --itm-merge-from-step 4

# ============================================================
# ITM: balanced from README (start=10, keep=2, compress=0.25)
# Adapted from DreamGen 14B "Balanced" preset to 28-block
# ============================================================
run_config "itm_readme_balanced" \
  --use-itm \
  --itm-start-layer-idx 10 \
  --itm-keep-last-n-dense 2 \
  --itm-compress-ratio 0.25 \
  --itm-prune-from-step 1 \
  --itm-merge-from-step 2

# ============================================================
# SiTo: very minimal (start=20, keep=2, low prune, low noise)
# Only 6 layers pruned (20-25), almost no overhead
# ============================================================
run_config "sito_minimal" \
  --use-sito \
  --sito-start-layer-idx 20 \
  --sito-keep-last-n-dense 2 \
  --sito-prune-ratio 0.15 \
  --sito-noise-alpha 0.02

# ============================================================
# SiTo: from README DreamGen 2B (start=16, keep=4, prune=0.25)
# ============================================================
run_config "sito_readme_2b" \
  --use-sito \
  --sito-start-layer-idx 16 \
  --sito-keep-last-n-dense 4 \
  --sito-prune-ratio 0.25 \
  --sito-noise-alpha 0.1

# ============================================================
# SiTo: from README (start=18, keep=2)
# Only 8 layers pruned, auto prune ratio
# ============================================================
run_config "sito_start18" \
  --use-sito \
  --sito-start-layer-idx 18 \
  --sito-keep-last-n-dense 2

echo "============================================================" | tee -a "${TIMING_LOG}"
echo "  Grid Search v2b Complete" | tee -a "${TIMING_LOG}"
echo "  Finished: $(date)" | tee -a "${TIMING_LOG}"
echo "============================================================" | tee -a "${TIMING_LOG}"
echo ""
echo "Results: ${GRID_ROOT}/"
echo "Timing:  ${TIMING_LOG}"
echo ""
echo "Compare all v2:"
echo "  python scripts/compare_grid_results.py --grid-root tmp_2/grid_search_v2"
