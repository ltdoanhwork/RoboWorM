#!/bin/bash
# =============================================================================
# DreamGen Benchmark - Long Video Generation (LVG) 
# Regenerate all benchmark videos using LVG pipeline
# =============================================================================
set -e

# --- Configuration ---
PROJ_DIR="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/VLA-Humanoid"
BENCHMARK_DIR="${PROJ_DIR}/output/dream_gen_benchmark"
VENV_PATH="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/cosmos-predict2.5/.venv/bin/activate"

# Number of GPUs
NUM_GPUS_2B=${NUM_GPUS_2B:-1}
NUM_GPUS_14B=${NUM_GPUS_14B:-1}

# LVG settings
NUM_CHUNKS=${NUM_CHUNKS:-6}

# Smoke test: only process first 1 item per folder
SMOKE_TEST=${SMOKE_TEST:-0}

# Mode: "2b", "14b", or "all"
MODE=${MODE:-"all"}

# --- Setup ---
cd "${PROJ_DIR}"
source "${VENV_PATH}"
export PYTHONPATH="${PROJ_DIR}/cosmos_predict2:${PYTHONPATH}"

echo "============================================="
echo "DreamGen Benchmark LVG Regeneration"
echo "============================================="
echo "Project dir:   ${PROJ_DIR}"
echo "Benchmark dir: ${BENCHMARK_DIR}"
echo "NUM_GPUS_2B:   ${NUM_GPUS_2B}"
echo "NUM_GPUS_14B:  ${NUM_GPUS_14B}"
echo "NUM_CHUNKS:    ${NUM_CHUNKS}"
echo "SMOKE_TEST:    ${SMOKE_TEST}"
echo "MODE:          ${MODE}"
echo "============================================="

# --- Helper: Create smoke test batch_input.json with only 1 item ---
create_smoke_batch() {
    local input_json="$1"
    local smoke_json="$2"
    python3 -c "
import json, sys
with open('${input_json}') as f:
    data = json.load(f)
with open('${smoke_json}', 'w') as f:
    json.dump(data[:1], f, indent=2)
print(f'Created smoke test with 1/{len(data)} items', file=sys.stderr)
"
}

get_batch_json() {
    local folder="$1"
    local original="${BENCHMARK_DIR}/${folder}/batch_input.json"
    
    if [ "${SMOKE_TEST}" = "1" ]; then
        local smoke="${BENCHMARK_DIR}/${folder}/batch_input_smoke.json"
        create_smoke_batch "${original}" "${smoke}"
        echo "${smoke}"
    else
        echo "${original}"
    fi
}

# =============================================================================
# 2B Models (using video2world_lvg.py)
# =============================================================================
run_2b_models() {
    echo ""
    echo "============================================="
    echo "Running 2B Models (video2world_lvg.py)"
    echo "============================================="

    local folders_2b=(
        "cosmos_predict2_2b_gr1_behavior"
        "cosmos_predict2_2b_gr1_env"
        "cosmos_predict2_2b_gr1_object"
    )

    for folder in "${folders_2b[@]}"; do
        local batch_json
        batch_json=$(get_batch_json "${folder}")
        
        if [ ! -f "${batch_json}" ]; then
            echo "SKIP: ${folder} — batch_input.json not found"
            continue
        fi

        echo ""
        echo ">>> Processing: ${folder}"
        echo "    Batch JSON: ${batch_json}"
        echo ""

        if [ "${NUM_GPUS_2B}" -gt 1 ]; then
            torchrun --nproc_per_node=${NUM_GPUS_2B} \
                "${PROJ_DIR}/examples/video2world_lvg.py" \
                --model_size 2B \
                --batch_input_json "${batch_json}" \
                --disable_guardrail \
                --disable_prompt_refiner \
                --num_gpus ${NUM_GPUS_2B} \
                --num_chunks ${NUM_CHUNKS}
        else
            python "${PROJ_DIR}/examples/video2world_lvg.py" \
                --model_size 2B \
                --batch_input_json "${batch_json}" \
                --disable_guardrail \
                --disable_prompt_refiner \
                --num_gpus ${NUM_GPUS_2B} \
                --num_chunks ${NUM_CHUNKS}
        fi

        echo ">>> Done: ${folder}"
    done
}

# =============================================================================
# 14B GR00T Models (using video2world_gr00t_lvg.py)
# =============================================================================
run_14b_models() {
    echo ""
    echo "============================================="
    echo "Running 14B GR00T Models (video2world_gr00t_lvg.py)"
    echo "============================================="

    local folders_14b=(
        "cosmos_predict2_14b_gr1_behavior"
        "cosmos_predict2_14b_gr1_env"
        "cosmos_predict2_14b_gr1_object"
    )

    for folder in "${folders_14b[@]}"; do
        local batch_json
        batch_json=$(get_batch_json "${folder}")
        
        if [ ! -f "${batch_json}" ]; then
            echo "SKIP: ${folder} — batch_input.json not found"
            continue
        fi

        echo ""
        echo ">>> Processing: ${folder}"
        echo "    Batch JSON: ${batch_json}"
        echo ""

        if [ "${NUM_GPUS_14B}" -gt 1 ]; then
            torchrun --nproc_per_node=${NUM_GPUS_14B} \
                -m examples.video2world_gr00t_lvg \
                --model_size 14B \
                --gr00t_variant gr1 \
                --batch_input_json "${batch_json}" \
                --disable_guardrail \
                --num_gpus ${NUM_GPUS_14B} \
                --num_chunks ${NUM_CHUNKS}
        else
            python -m examples.video2world_gr00t_lvg \
                --model_size 14B \
                --gr00t_variant gr1 \
                --batch_input_json "${batch_json}" \
                --disable_guardrail \
                --num_gpus ${NUM_GPUS_14B} \
                --num_chunks ${NUM_CHUNKS}
        fi

        echo ">>> Done: ${folder}"
    done
}

# =============================================================================
# Main
# =============================================================================
case "${MODE}" in
    2b)
        run_2b_models
        ;;
    14b)
        run_14b_models
        ;;
    all)
        run_2b_models
        run_14b_models
        ;;
    *)
        echo "ERROR: Invalid MODE='${MODE}'. Use '2b', '14b', or 'all'."
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "All benchmark regenerations complete!"
echo "============================================="
