#!/usr/bin/env bash
# =============================================================================
# run_pruning_benchmark.sh
#
# Benchmarks SiTo and ITM token-pruning methods on the DreamGen 14B GR1 model.
#
# Tested on:
#   - model_size: 14B, gr00t_variant: gr1, resolution: 480p, fps: 16
#   - Token geometry: DiT grid (T=24, H=30, W=40) → tokens/frame = 1200
#   - Auto prune_ratio for 1200 tokens: 0.35 (SiTo) / 0.25 (ITM)
#   - Model has 36 blocks; default start_layer_idx = 6
#
# Usage:
#   bash scripts/run_pruning_benchmark.sh [NUM_SAMPLES] [MASTER_PORT]
#
# Examples:
#   bash scripts/run_pruning_benchmark.sh          # 5 samples, port 29501
#   bash scripts/run_pruning_benchmark.sh 3 29502  # 3 samples, port 29502
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_SAMPLES=${1:-5}
MASTER_PORT=${2:-29501}

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_JSON="./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json"
BENCH_OUTPUT_ROOT="./output/pruning_benchmark"
LOG_DIR="${BENCH_OUTPUT_ROOT}/logs"
SUMMARY_CSV="${BENCH_OUTPUT_ROOT}/summary.csv"

export PYTHONPATH="${ROOT_DIR}/models/dreamgen"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NUM_GPUS=1

TORCHRUN="torchrun --master_port=${MASTER_PORT}"
SCRIPT="models.dreamgen.examples.video2world_gr00t"
COMMON_ARGS=(
    --model_size 14B
    --gr00t_variant gr1
    --disable_guardrail
    --num_gpus "${NUM_GPUS}"
)

mkdir -p "${LOG_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Create a 5-sample subset JSON from a full batch_input.json
SUBSET_JSON="${BENCH_OUTPUT_ROOT}/batch_input_subset_${NUM_SAMPLES}.json"
mkdir -p "${BENCH_OUTPUT_ROOT}"
python3 - <<PYEOF
import json, sys
with open("${INPUT_JSON}") as f:
    data = json.load(f)
print(f"Total samples in input: {len(data)}", file=sys.stderr)
subset = data[:${NUM_SAMPLES}]
with open("${SUBSET_JSON}", "w") as f:
    json.dump(subset, f, indent=2)
print(f"Subset of {len(subset)} samples written to ${SUBSET_JSON}", file=sys.stderr)
PYEOF

# Write CSV header
echo "config_id,method,start_layer_idx,keep_last_n_dense,compress_or_prune_ratio,prune_from_step,merge_from_step,extra_flags,avg_self_attn_s,exit_code" > "${SUMMARY_CSV}"

# run_config <config_id> <method> <output_subdir> <extra torchrun args...>
run_config() {
    local CONFIG_ID="$1"
    local METHOD="$2"
    local SUBDIR="$3"
    shift 3
    local EXTRA_ARGS=("$@")

    local OUT_DIR="${BENCH_OUTPUT_ROOT}/${SUBDIR}"
    local LOG_FILE="${LOG_DIR}/${CONFIG_ID}.log"

    # Rewrite output_video paths in subset JSON to point to this config's subdir
    local CFG_JSON="${BENCH_OUTPUT_ROOT}/${CONFIG_ID}_input.json"
    python3 - <<PYEOF2
import json, os, pathlib
with open("${SUBSET_JSON}") as f:
    data = json.load(f)
out_dir = pathlib.Path("${OUT_DIR}")
out_dir.mkdir(parents=True, exist_ok=True)
for item in data:
    base = os.path.basename(item["output_video"])
    item["output_video"] = str(out_dir / base)
with open("${CFG_JSON}", "w") as f:
    json.dump(data, f, indent=2)
PYEOF2

    echo ""
    echo "======================================================================="
    echo " Running: ${CONFIG_ID}"
    echo " Method:  ${METHOD}"
    echo " Output:  ${OUT_DIR}"
    echo " Log:     ${LOG_FILE}"
    echo " Args:    ${EXTRA_ARGS[*]}"
    echo "======================================================================="

    local EXIT_CODE=0
    ${TORCHRUN} -m "${SCRIPT}" \
        "${COMMON_ARGS[@]}" \
        --batch_input_json "${CFG_JSON}" \
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}" || EXIT_CODE=$?

    # Extract avg self-attn from log
    local AVG_SA
    AVG_SA=$(grep -oP "Average DiT Self-Attention: \K[0-9.]+" "${LOG_FILE}" | tail -1 || echo "N/A")

    echo "  → avg_self_attn = ${AVG_SA}s  (exit=${EXIT_CODE})"

    # Append to CSV (extra columns parsed from config_id naming convention)
    echo "${CONFIG_ID},${METHOD},${AVG_SA},${EXIT_CODE}" >> "${SUMMARY_CSV}"
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================
# The 3 "models" here are the same 14B checkpoint but with different abstraction
# levels of pruning:
#   M1 = baseline (dense)
#   M2 = SiTo (spatial token pruning)
#   M3 = ITM  (importance token merge driven by CFG heat maps)
#
# Token geometry reminder (14B / 480p / 16fps / patch_spatial=2):
#   DiT grid:          T=24, H=30, W=40
#   tokens/frame:      1200   → auto SiTo ratio = 0.35, auto ITM ratio = 0.25
#   max prune ratio:   0.75   (with patch_h=patch_w=2, guaranteed ≥1 anchor/patch)
#   36 blocks total:   default start_layer_idx = 6  (recommended ≥ 10 for quality)
# ============================================================================

# ============================================================================
# M1 — Baseline (dense, no pruning)
# ============================================================================
run_config \
    "m1_baseline" "baseline" "m1_baseline" \
    # no extra flags

# ============================================================================
# M2 — SiTo (self-attention token pruning)
#
# SiTo parameters for 14B / 36-block / 480p:
#   start_layer_idx : first layer to prune; higher → better quality, less speedup
#   keep_last_n_dense : tail layers always dense; ≥2 recommended for sharpness
#   prune_ratio : fraction of tokens removed/frame (auto=0.35 for 1200 tokens)
#   noise_alpha : randomness in anchor selection (lower = more deterministic)
#   sim_beta    : weight of cosine similarity in anchor scoring (higher = more similar anchors kept)
#
# 7 combinations: vary start_layer_idx [10,14,18,22] and prune_ratio [0.20,0.25,0.30]
# ============================================================================

# SiTo-A: Conservative start + low ratio
run_config \
    "m2_sito_a" "sito" "m2_sito_a" \
    --use_sito \
    --sito_start_layer_idx 22 \
    --sito_keep_last_n_dense 4 \
    --sito_prune_ratio 0.20 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.05 \
    --sito_sim_beta 1.0

# SiTo-B: Conservative start + medium ratio
run_config \
    "m2_sito_b" "sito" "m2_sito_b" \
    --use_sito \
    --sito_start_layer_idx 20 \
    --sito_keep_last_n_dense 4 \
    --sito_prune_ratio 0.25 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.05 \
    --sito_sim_beta 1.0

# SiTo-C: Balanced start + medium ratio (best quality/speed balance)
run_config \
    "m2_sito_c" "sito" "m2_sito_c" \
    --use_sito \
    --sito_start_layer_idx 16 \
    --sito_keep_last_n_dense 2 \
    --sito_prune_ratio 0.25 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.1 \
    --sito_sim_beta 1.0

# SiTo-D: Balanced start + higher ratio
run_config \
    "m2_sito_d" "sito" "m2_sito_d" \
    --use_sito \
    --sito_start_layer_idx 14 \
    --sito_keep_last_n_dense 2 \
    --sito_prune_ratio 0.30 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.1 \
    --sito_sim_beta 1.2

# SiTo-E: Aggressive start + low ratio  (tradeoff: many layers pruned but gently)
run_config \
    "m2_sito_e" "sito" "m2_sito_e" \
    --use_sito \
    --sito_start_layer_idx 10 \
    --sito_keep_last_n_dense 2 \
    --sito_prune_ratio 0.20 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.1 \
    --sito_sim_beta 1.0

# SiTo-F: Aggressive start + medium ratio
run_config \
    "m2_sito_f" "sito" "m2_sito_f" \
    --use_sito \
    --sito_start_layer_idx 10 \
    --sito_keep_last_n_dense 2 \
    --sito_prune_ratio 0.30 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.1 \
    --sito_sim_beta 1.2

# SiTo-G: Aggressive start + high ratio (max speedup, use for ablation only)
run_config \
    "m2_sito_g" "sito" "m2_sito_g" \
    --use_sito \
    --sito_start_layer_idx 8 \
    --sito_keep_last_n_dense 2 \
    --sito_prune_ratio 0.35 \
    --sito_patch_h 2 --sito_patch_w 2 \
    --sito_noise_alpha 0.1 \
    --sito_sim_beta 1.2

# ============================================================================
# M3 — ITM (Importance Token Merge, driven by CFG heatmap)
#
# ITM parameters for 14B / 36-block / 480p:
#   start_layer_idx   : first layer eligible for ITM; higher → better quality
#   keep_last_n_dense : tail layers always dense
#   compress_ratio    : fraction of tokens merged per frame (auto=0.25 for 1200)
#   prune_from_step   : diffusion step index where ITM first activates
#   merge_from_step   : step where full mean-pool merge activates (≥ prune_from_step)
#   merge_mlp (opt.)  : also merge MLP for more VRAM savings  (quality tradeoff)
#
# Key difference from SiTo:
#   - ITM uses the CFG-error heat map to identify unimportant background tokens
#   - Merged tokens are mean-pooled into anchor (dst) tokens, unlike SiTo which just drops
#   - Requires guidance > 0 (already the case with default guidance=7)
#
# 7 combinations: vary start_layer_idx, compress_ratio, prune/merge step
# ============================================================================

# ITM-A: Conservative (high quality)
run_config \
    "m3_itm_a" "itm" "m3_itm_a" \
    --use_itm \
    --itm_start_layer_idx 22 \
    --itm_keep_last_n_dense 4 \
    --itm_compress_ratio 0.20 \
    --itm_prune_from_step 2 \
    --itm_merge_from_step 4

# ITM-B: Conservative + slightly higher compression
run_config \
    "m3_itm_b" "itm" "m3_itm_b" \
    --use_itm \
    --itm_start_layer_idx 20 \
    --itm_keep_last_n_dense 4 \
    --itm_compress_ratio 0.25 \
    --itm_prune_from_step 2 \
    --itm_merge_from_step 4

# ITM-C: Balanced (recommended default)
run_config \
    "m3_itm_c" "itm" "m3_itm_c" \
    --use_itm \
    --itm_start_layer_idx 16 \
    --itm_keep_last_n_dense 2 \
    --itm_compress_ratio 0.25 \
    --itm_prune_from_step 1 \
    --itm_merge_from_step 2

# ITM-D: Balanced + earlier activation
run_config \
    "m3_itm_d" "itm" "m3_itm_d" \
    --use_itm \
    --itm_start_layer_idx 16 \
    --itm_keep_last_n_dense 2 \
    --itm_compress_ratio 0.25 \
    --itm_prune_from_step 0 \
    --itm_merge_from_step 1

# ITM-E: Aggressive + higher compression, merge from step 0
run_config \
    "m3_itm_e" "itm" "m3_itm_e" \
    --use_itm \
    --itm_start_layer_idx 12 \
    --itm_keep_last_n_dense 2 \
    --itm_compress_ratio 0.30 \
    --itm_prune_from_step 0 \
    --itm_merge_from_step 0

# ITM-F: Aggressive + also merge MLP (max VRAM reduction)
run_config \
    "m3_itm_f" "itm" "m3_itm_f" \
    --use_itm \
    --itm_start_layer_idx 10 \
    --itm_keep_last_n_dense 2 \
    --itm_compress_ratio 0.30 \
    --itm_prune_from_step 0 \
    --itm_merge_from_step 0 \
    --itm_merge_mlp

# ITM-G: Max aggression (early start, max merge, for ablation only)
run_config \
    "m3_itm_g" "itm" "m3_itm_g" \
    --use_itm \
    --itm_start_layer_idx 8 \
    --itm_keep_last_n_dense 2 \
    --itm_compress_ratio 0.35 \
    --itm_prune_from_step 0 \
    --itm_merge_from_step 0

# ============================================================================
# Finalize — print summary table
# ============================================================================
echo ""
echo "======================================================================="
echo " BENCHMARK COMPLETE"
echo " Results root : ${BENCH_OUTPUT_ROOT}"
echo " Timing CSV   : ${SUMMARY_CSV}"
echo " Logs         : ${LOG_DIR}/"
echo "======================================================================="
echo ""

# Pretty-print self-attn timing summary from all logs
python3 - <<'PYEOF3'
import os, re, glob

log_dir = "./output/pruning_benchmark/logs"
logs = sorted(glob.glob(f"{log_dir}/*.log"))

print(f"{'Config':<22} {'Avg Self-Attn (s)':>20} {'Avg Denoise (s)':>18} {'Total Time (s)':>16}")
print("-" * 80)
for lf in logs:
    config_id = os.path.splitext(os.path.basename(lf))[0]
    content = open(lf).read()

    def extract(pattern, txt):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else float("nan")

    sa   = extract(r"Average DiT Self-Attention: ([0-9.]+)", content)
    dn   = extract(r"Average denoise time: ([0-9.]+)", content)
    tot  = extract(r"Total inference time: ([0-9.]+)", content)
    print(f"{config_id:<22} {sa:>20.4f} {dn:>18.4f} {tot:>16.2f}")
PYEOF3
