#!/usr/bin/env bash
# WorldCache threshold benchmark script
# Tests rel_l1_thresh at 0.03, 0.05, 0.08, 0.10 with only 3 samples each (single chunk)
# for quick wall-time and skip-ratio comparison.
#
# Usage (from video_gen_physics root):
#   bash bench_worldcache_thresh.sh 2>&1 | tee output/worldcache_bench/bench.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source ./models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/DreamDojo:${PYTHONPATH:-}"

CKPT_DIR="./checkpoints/dreamdojo/2B_GR1_post-train"
DATASET="./checkpoints/dreamdojo/datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot"
EXPERIMENT="dreamdojo_2b_480_640_gr1"
OUT_BASE="./output/worldcache_bench"
NUM_SAMPLES=3    # 3 samples for quick wall-time comparison
NUM_FRAMES=37    # default chunk size; 1 long sequence per sample

mkdir -p "${OUT_BASE}"

# -----------------------------------------------------------------------
# Helper: run one configuration and capture key metrics
# -----------------------------------------------------------------------
run_config() {
    local label="$1"       # e.g. "thresh_0.03"
    local extra_args="$2"  # worldcache CLI args

    local out_dir="${OUT_BASE}/${label}"
    mkdir -p "${out_dir}"

    echo ""
    echo "========================================================"
    echo " Starting: ${label}"
    echo " Time: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo "========================================================"

    local start_s
    start_s=$(date +%s)

    # shellcheck disable=SC2086
    python -m models.DreamDojo.examples.action_conditioned \
        -o "${out_dir}/logs" \
        --checkpoints-dir "${CKPT_DIR}" \
        --experiment "${EXPERIMENT}" \
        --save-dir "${out_dir}" \
        --num-frames "${NUM_FRAMES}" \
        --num-samples "${NUM_SAMPLES}" \
        --dataset-path "${DATASET}" \
        --data-split test \
        --deterministic-uniform-sampling \
        ${extra_args} \
        2>&1 | tee "${out_dir}/run.log"

    local end_s
    end_s=$(date +%s)
    local elapsed=$(( end_s - start_s ))

    echo ""
    echo "[BENCH] ${label}: done in ${elapsed}s"

    # Extract all WorldCache skip ratios
    local skip_line
    skip_line=$(grep -oE '\[WorldCache\] Skipped [0-9]+/[0-9]+ \([0-9.]+%\)' "${out_dir}/run.log" | tail -n 5 | tr '\n' ' ' || true)
    if [[ -n "${skip_line}" ]]; then
        echo "[BENCH] ${label}: skip_ratios => ${skip_line}"
    else
        echo "[BENCH] ${label}: (no WorldCache skip ratio — baseline or error)"
        skip_line="N/A"
    fi

    # Extract it/s from tqdm progress bar (last 5 readings)
    local speed_line
    speed_line=$(grep -oE '[0-9]+\.[0-9]+it/s' "${out_dir}/run.log" | tail -n 5 | tr '\n' ' ' || true)
    if [[ -n "${speed_line}" ]]; then
        echo "[BENCH] ${label}: last it/s => ${speed_line}"
    else
        speed_line="N/A"
    fi

    # Summary line
    local summary="${label}  elapsed=${elapsed}s  skip=[${skip_line}]  speed=[${speed_line}]"
    echo "${summary}"
    echo "${summary}" >> "${OUT_BASE}/summary.txt"
    echo "---"
}

# -----------------------------------------------------------------------
# Clear previous summary
# -----------------------------------------------------------------------
{
  echo "# WorldCache threshold benchmark"
  echo "# Run at: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "# NUM_SAMPLES=${NUM_SAMPLES}  NUM_FRAMES=${NUM_FRAMES}"
  echo ""
} > "${OUT_BASE}/summary.txt"

# -----------------------------------------------------------------------
# 1. BASELINE — dense, no cache
# -----------------------------------------------------------------------
run_config "baseline" ""

# -----------------------------------------------------------------------
# 2. thresh=0.03  (original CLI value — now expected to be too strict)
# -----------------------------------------------------------------------
run_config "thresh_0.03" "\
  --worldcache-enabled \
  --worldcache-num-steps 35 \
  --worldcache-rel-l1-thresh 0.03 \
  --worldcache-probe-depth 4 \
  --worldcache-ret-ratio 0.5"

# -----------------------------------------------------------------------
# 3. thresh=0.05
# -----------------------------------------------------------------------
run_config "thresh_0.05" "\
  --worldcache-enabled \
  --worldcache-num-steps 35 \
  --worldcache-rel-l1-thresh 0.05 \
  --worldcache-probe-depth 4 \
  --worldcache-ret-ratio 0.5"

# -----------------------------------------------------------------------
# 4. thresh=0.08  (primary candidate for corrected normalization)
# -----------------------------------------------------------------------
run_config "thresh_0.08" "\
  --worldcache-enabled \
  --worldcache-num-steps 35 \
  --worldcache-rel-l1-thresh 0.08 \
  --worldcache-probe-depth 4 \
  --worldcache-ret-ratio 0.5"

# -----------------------------------------------------------------------
# 5. thresh=0.10
# -----------------------------------------------------------------------
run_config "thresh_0.10" "\
  --worldcache-enabled \
  --worldcache-num-steps 35 \
  --worldcache-rel-l1-thresh 0.10 \
  --worldcache-probe-depth 4 \
  --worldcache-ret-ratio 0.5"

# -----------------------------------------------------------------------
# Final summary
# -----------------------------------------------------------------------
echo ""
echo "========================================================"
echo " BENCHMARK COMPLETE — SUMMARY"
echo "========================================================"
cat "${OUT_BASE}/summary.txt"
