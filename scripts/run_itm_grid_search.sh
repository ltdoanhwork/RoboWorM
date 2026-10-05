#!/usr/bin/env bash
# Grid search: compare dense baseline vs multiple ITM configs on a few episodes.
# Logs wall-clock time + metrics for each config.
# Output: tmp_2/grid_search/{config_name}/{makovian,non_makovian}/iter_*/
#
# Usage:
#   bash scripts/run_itm_grid_search.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=5 bash scripts/run_itm_grid_search.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
GRID_ROOT="${ROOT}/tmp_2/grid_search"
SPLIT="${SPLIT:-makovian}"
TIMING_LOG="${GRID_ROOT}/timing_log.txt"

mkdir -p "${GRID_ROOT}"
echo "=== ITM Grid Search: ${SPLIT}, ${NUM_SAMPLES} samples ===" | tee "${TIMING_LOG}"
echo "Started: $(date)" | tee -a "${TIMING_LOG}"
echo "" | tee -a "${TIMING_LOG}"

if [ "${SKIP_META_REPAIR:-0}" != "1" ]; then
  python3 "${ROOT}/scripts/repair_humanoid_singleview_split_meta.py" --workspace "${ROOT}"
fi

run_config() {
  local config_name="$1"
  shift
  local extra_args=("$@")
  local save_dir="${GRID_ROOT}/${config_name}/${SPLIT}"
  local out_cache="${ROOT}/output/grid_${config_name}_${SPLIT}"
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
# Config 0: Dense baseline (no ITM, no guidance)
# ============================================================
run_config "dense_baseline"

# ============================================================
# Config 1: Current ITM (aggressive) - your original params
# start_layer=4, keep_last_n_dense=2, prune_from=1, merge_from=2
# ============================================================
run_config "itm_aggressive" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 1 \
  --itm-merge-from-step 2

# ============================================================
# Config 2: ITM conservative layers
# start_layer=8, keep_last_n_dense=4 → only 16 layers affected (8-23)
# ============================================================
run_config "itm_conservative_layers" \
  --use-itm \
  --itm-start-layer-idx 8 \
  --itm-keep-last-n-dense 4 \
  --itm-prune-from-step 1 \
  --itm-merge-from-step 2

# ============================================================
# Config 3: ITM delayed steps
# Keep original layer range but delay prune/merge to preserve early structure
# prune_from=5, merge_from=8 → first 5 steps fully dense
# ============================================================
run_config "itm_delayed_steps" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 5 \
  --itm-merge-from-step 8

# ============================================================
# Config 4: ITM balanced (both conservative layers + delayed steps)
# start_layer=6, keep_last=4, prune_from=3, merge_from=5
# ============================================================
run_config "itm_balanced" \
  --use-itm \
  --itm-start-layer-idx 6 \
  --itm-keep-last-n-dense 4 \
  --itm-prune-from-step 3 \
  --itm-merge-from-step 5

# ============================================================
# Config 5: ITM low compress ratio (explicit 0.15 instead of auto 0.35)
# ============================================================
run_config "itm_low_compress" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 1 \
  --itm-merge-from-step 2 \
  --itm-compress-ratio 0.15

echo "=== Grid Search Complete ===" | tee -a "${TIMING_LOG}"
echo "Finished: $(date)" | tee -a "${TIMING_LOG}"
echo ""
echo "Results in: ${GRID_ROOT}/"
echo "Timing log: ${TIMING_LOG}"
echo ""
echo "To compare metrics:"
echo "  find ${GRID_ROOT} -name 'all_summary.json' | xargs -I{} sh -c 'echo \"=== {} ===\"; cat {}'"
