#!/usr/bin/env bash
# Grid search: compare dense baseline vs multiple SiTo configs on a few episodes.
# Logs wall-clock time, s/it (seconds per iteration), and metrics for each config.
# Output: tmp_2/grid_search_sito/{config_name}/{makovian}/iter_*/
#
# Usage:
#   bash scripts/run_sito_grid_search.sh
#   CUDA_VISIBLE_DEVICES=0 NUM_SAMPLES=5 bash scripts/run_sito_grid_search.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
GRID_ROOT="${ROOT}/tmp_2/grid_search_sito"
SPLIT="${SPLIT:-makovian}"
TIMING_LOG="${GRID_ROOT}/timing_log.txt"

mkdir -p "${GRID_ROOT}"
echo "============================================================" | tee "${TIMING_LOG}"
echo "  SiTo Grid Search: ${SPLIT}, ${NUM_SAMPLES} samples" | tee -a "${TIMING_LOG}"
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
  local out_cache="${ROOT}/output/grid_sito_${config_name}_${SPLIT}"
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
# Config 0: Dense baseline (no acceleration)
# ============================================================
run_config "dense_baseline"

# ============================================================
# SiTo configs — vary layer range, prune_ratio, patch size
# Default auto prune_ratio for 2048 tokens = 0.45
# Max prune_ratio for patch 2x2 = 0.75 (1 - 1/4)
# ============================================================

# Config 1: SiTo aggressive (defaults: start=4, keep_last=2, auto prune ~0.45)
run_config "sito_aggressive" \
  --use-sito \
  --sito-start-layer-idx 4 \
  --sito-keep-last-n-dense 2

# Config 2: SiTo conservative layers (start=8, keep_last=4 → only 16 layers)
run_config "sito_conservative_layers" \
  --use-sito \
  --sito-start-layer-idx 8 \
  --sito-keep-last-n-dense 4

# Config 3: SiTo low prune ratio (explicit 0.25 vs auto 0.45)
run_config "sito_low_prune" \
  --use-sito \
  --sito-start-layer-idx 4 \
  --sito-keep-last-n-dense 2 \
  --sito-prune-ratio 0.25

# Config 4: SiTo larger patches (3x3 → finer selection, max prune = 0.89)
# with moderate prune ratio
run_config "sito_patch3x3" \
  --use-sito \
  --sito-start-layer-idx 4 \
  --sito-keep-last-n-dense 2 \
  --sito-patch-h 3 \
  --sito-patch-w 3 \
  --sito-prune-ratio 0.35

# Config 5: SiTo balanced (conservative layers + low prune)
run_config "sito_balanced" \
  --use-sito \
  --sito-start-layer-idx 6 \
  --sito-keep-last-n-dense 4 \
  --sito-prune-ratio 0.30

# Config 6: SiTo tuned noise/sim (higher sim_beta, lower noise for more deterministic)
run_config "sito_high_sim" \
  --use-sito \
  --sito-start-layer-idx 4 \
  --sito-keep-last-n-dense 2 \
  --sito-noise-alpha 0.02 \
  --sito-sim-beta 2.0

echo "============================================================" | tee -a "${TIMING_LOG}"
echo "  Grid Search Complete" | tee -a "${TIMING_LOG}"
echo "  Finished: $(date)" | tee -a "${TIMING_LOG}"
echo "============================================================" | tee -a "${TIMING_LOG}"
echo ""
echo "Results in: ${GRID_ROOT}/"
echo "Timing log: ${TIMING_LOG}"
echo ""
echo "To compare metrics:"
echo "  find ${GRID_ROOT} -name 'all_summary.json' | sort | xargs -I{} sh -c 'echo \"=== {} ===\"; cat {}'"
