#!/usr/bin/env bash
# Grid search v2: configs inspired by README_prunning.md recommendations.
# Focus on finding something FASTER than dense while keeping quality.
#
# Key insight from v1: ITM always needs CFG (2 passes) so compare against dense+CFG.
# SiTo doesn't need CFG so compare against dense (no CFG).
#
# Usage:
#   bash scripts/run_itm_grid_search_v2.sh
#   CUDA_VISIBLE_DEVICES=1 NUM_SAMPLES=5 bash scripts/run_itm_grid_search_v2.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CHECKPOINTS_DIR="${ROOT}/checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000"
EXPERIMENT="dreamdojo_2b_480_640_gr1"
NUM_FRAMES=49
NUM_SAMPLES="${NUM_SAMPLES:-2}"
GRID_ROOT="${ROOT}/tmp_2/grid_search_v2"
SPLIT="${SPLIT:-makovian}"
TIMING_LOG="${GRID_ROOT}/timing_log.txt"

mkdir -p "${GRID_ROOT}"
echo "============================================================" | tee "${TIMING_LOG}"
echo "  Grid Search v2 | GPU ${CUDA_VISIBLE_DEVICES} | ${SPLIT} | ${NUM_SAMPLES} samples" | tee -a "${TIMING_LOG}"
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
  local out_cache="${ROOT}/output/grid_v2_${config_name}_${SPLIT}"
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
# Baseline: Dense + CFG (fair comparison for ITM)
# ============================================================
FORCE_GUIDANCE=1.5 run_config "dense_cfg"

# ============================================================
# ITM: max aggressive (prune=0, merge=0, merge from first step)
# From README: "Setting prune_from_step 0 merge_from_step 0 applies full merge
# from the very first step (max aggression)"
# ============================================================
run_config "itm_max_aggressive" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 0 \
  --itm-merge-from-step 0

# ============================================================
# ITM: with merge_mlp (largest VRAM savings, from README)
# ============================================================
run_config "itm_merge_mlp" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 1 \
  --itm-merge-from-step 2 \
  --itm-merge-mlp

# ============================================================
# ITM: merge_mlp + merge_crossattn (max token reduction)
# ============================================================
run_config "itm_merge_all" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 0 \
  --itm-merge-from-step 0 \
  --itm-merge-mlp \
  --itm-merge-crossattn

# ============================================================
# ITM: higher compress ratio (0.35 = auto default for >=2048 tokens)
# ============================================================
run_config "itm_high_compress" \
  --use-itm \
  --itm-start-layer-idx 4 \
  --itm-keep-last-n-dense 2 \
  --itm-prune-from-step 0 \
  --itm-merge-from-step 0 \
  --itm-compress-ratio 0.35

# ============================================================
# SiTo: very high start layer (start=16, only 10 layers affected)
# From README: DreamGen 2B uses start_layer_idx=16
# ============================================================
run_config "sito_start16" \
  --use-sito \
  --sito-start-layer-idx 16 \
  --sito-keep-last-n-dense 4 \
  --sito-prune-ratio 0.25

# ============================================================
# SiTo: start=12, prune=0.4 (from README DreamGen 2B config)
# ============================================================
run_config "sito_start12_prune04" \
  --use-sito \
  --sito-start-layer-idx 12 \
  --sito-keep-last-n-dense 2 \
  --sito-prune-ratio 0.4

# ============================================================
# SiTo: start=20, very late (only 4 layers: 20-25, dense tail 26-27)
# Minimal overhead, minimal speedup, highest quality
# ============================================================
run_config "sito_start20" \
  --use-sito \
  --sito-start-layer-idx 20 \
  --sito-keep-last-n-dense 4 \
  --sito-prune-ratio 0.25 \
  --sito-noise-alpha 0.05

echo "============================================================" | tee -a "${TIMING_LOG}"
echo "  Grid Search v2 Complete" | tee -a "${TIMING_LOG}"
echo "  Finished: $(date)" | tee -a "${TIMING_LOG}"
echo "============================================================" | tee -a "${TIMING_LOG}"
echo ""
echo "Results: ${GRID_ROOT}/"
echo "Timing:  ${TIMING_LOG}"
echo ""
echo "Compare all:"
echo "  python scripts/compare_grid_results.py --grid-root tmp_2/grid_search tmp_2/grid_search_sito tmp_2/grid_search_v2"
