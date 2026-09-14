#!/bin/bash
# Sweep multiview consistency metrics over all configurations under data_for_web_2fps.
#
# Layout:
#   {DATA_ROOT}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/...
#     ctrlworld:  .../{task_type}/episode_*/{view1,view2}
#     dreamgen:   .../{task_type}/episode_*/...  OR  .../iter_*/{task}/episode_*/...
#   {DATA_ROOT}/tmp/lowquality_multiview/episode_*/{view1,view2,...}

set -euo pipefail

HOST="127.0.0.1"
PORT="6111"
FRAME_BATCH_SIZE=8
VIEWS="view1,view2"

CHECKPOINT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/checkpoint/vggt_omega_1b_512.pt"
DATA_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/data/preprocessing/data_for_web_2fps"
OUT_ROOT="/pfss/mlde/workspaces/mlde_wsp_MGPATH/thanhnv/WorldModelBenchmark/output/data_for_web_2fps"

# Optional filters (empty = run all)
VIDEO_TYPES=(${VIDEO_TYPES:-augment dense worldcache tmp})
ROBOT_TYPES=(${ROBOT_TYPES:-bimanual single_arm humanoid})
MODEL_TYPES=(${MODEL_TYPES:-ctrlworld dreamgen})
TASK_TYPES=(${TASK_TYPES:-makovian non_makovian})

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

has_episode_children() {
    local dir="$1"
    local child
    for child in "$dir"/*/; do
        [ -d "${child}view1" ] && [ -d "${child}view2" ] && return 0
    done
    return 1
}

# Print episode-root directories (parents of episode_*/view1 folders).
find_episode_roots() {
    local dir="$1"
    [ -d "$dir" ] || return 0

    if has_episode_children "$dir"; then
        echo "$dir"
        return 0
    fi

    local child
    for child in "$dir"/*/; do
        [ -d "$child" ] || continue
        find_episode_roots "$child"
    done
}

rel_path() {
    local abs="$1"
    echo "${abs#"${DATA_ROOT}/"}"
}

run_metrics() {
    local episodes_root="$1"
    local out_csv="$2"
    local rel
    rel="$(rel_path "$episodes_root")"

    mkdir -p "$(dirname "$out_csv")"

    echo "======================================================================"
    echo "🚀 episodes_root: ${rel}"
    echo "   out_csv:       ${out_csv#${OUT_ROOT}/}"
    echo "======================================================================"

    if ! uv run python -m multiview_consistency.run_video_metrics \
        --episodes_root "$episodes_root" \
        --out_csv "$out_csv" \
        --views "$VIEWS" \
        --frame_batch_size "$FRAME_BATCH_SIZE" \
        --host "$HOST" \
        --port "$PORT" ;then
        echo "⚠️  Failed: ${rel}" >&2
    else
        echo "✅ Done: ${rel}"
    fi
    echo ""
}

sweep_multiview_tree() {
    local multiview_dir="$1"
    local model_type="$2"

    local task_type_dir
    for task_type in "${TASK_TYPES[@]}"; do
        task_type_dir="${multiview_dir}/${model_type}/${task_type}"
        [ -d "$task_type_dir" ] || continue

        local root
        while IFS= read -r root; do
            [ -n "$root" ] || continue
            local rel out_csv
            rel="$(rel_path "$root")"
            out_csv="${OUT_ROOT}/${rel}.csv"
            run_metrics "$root" "$out_csv"
        done < <(find_episode_roots "$task_type_dir")
    done
}

sweep_tmp() {
    local tmp_dir="${DATA_ROOT}/tmp/lowquality_multiview"
    [ -d "$tmp_dir" ] || return 0

    local out_csv="${OUT_ROOT}/tmp/lowquality_multiview.csv"
    run_metrics "$tmp_dir" "$out_csv"
}

# ----------------------------------------------------------------------
# 1. Start model service (background)
# ----------------------------------------------------------------------
echo "======================================================================"
echo "🤖 Starting Model Service (VGGT-Omega)..."
echo "======================================================================"

uv run python -m multiview_consistency.model_service \
    --host "$HOST" \
    --port "$PORT" \
    --dtype float16 \
    --omega_checkpoint "$CHECKPOINT" \
    --omega_image_resolution 512 \
    --omega_patch_size 16 \
    --no-compile &

SERVICE_PID=$!

cleanup() {
    echo ""
    echo "🛑 Stopping Model Service (PID: $SERVICE_PID)..."
    kill "$SERVICE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "⏳ Waiting for Model Service on port ${PORT}..."
until python3 -c "
from multiprocessing.connection import Client
c = Client(('$HOST', int('$PORT')), authkey=b'vggt_omega')
c.send({'cmd': 'ping'})
print(c.recv())
c.close()
" 2>/dev/null; do
    sleep 2
done
echo "✅ Model Service ready!"
echo ""

# ----------------------------------------------------------------------
# 2. Sweep all configurations
# ----------------------------------------------------------------------
for video_type in "${VIDEO_TYPES[@]}"; do
    if [ "$video_type" = "tmp" ]; then
        sweep_tmp
        continue
    fi

    for robot_type in "${ROBOT_TYPES[@]}"; do
        multiview_dir="${DATA_ROOT}/${video_type}/${robot_type}/output/multiview"
        [ -d "$multiview_dir" ] || continue

        for model_type in "${MODEL_TYPES[@]}"; do
            [ -d "${multiview_dir}/${model_type}" ] || continue
            echo "----------------------------------------------------------------------"
            echo "📂 ${video_type} / ${robot_type} / ${model_type}"
            echo "----------------------------------------------------------------------"
            sweep_multiview_tree "$multiview_dir" "$model_type"
        done
    done
done

echo "🎉 Finished sweep of data_for_web_2fps!"
