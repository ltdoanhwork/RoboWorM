#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_dreamgen_svg2.sh
# DreamGen multiview (single_arm, droid layout) with SVG2 sparse attention.
# Exact mirror of the dicache flow; single-GPU; output -> sampling_dataset/svg2/.
# SVG2 needs .venv_svg (flashinfer 0.3.1 block-sparse wrapper) — default here.
#
# Env: VENV, DATASET_ROOT, INPUT_DIR, OUTPUT_ROOT, CATEGORY, MODEL_SIZE, FPS,
#   MAX_EPISODES, MIN_EPISODE_INDEX, EPISODE_SOURCE, FORCE_REGENERATE_OUTPUTS,
#   SVG_{START_LAYER_IDX,DENSE_STEP_FRAC,NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,MIN_KC_RATIO,KMEANS_ITER_INIT,KMEANS_ITER_STEP}
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VENV="${VENV:-${ROOT}/models/dreamgen/.venv_svg}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-./datasets/single_arm/multiview}"
INPUT_DIR="${INPUT_DIR:-./sampling_dataset/svg2/single_arm/input/multiview/dreamgen}"  # per-method (avoid concurrent-prepare races on shared dense json)
OUTPUT_ROOT="${OUTPUT_ROOT:-./sampling_dataset/svg2/single_arm/output/multiview/dreamgen}"
CATEGORY_FILTER="${CATEGORY:-}"
MODEL_SIZE="${MODEL_SIZE:-14B}"
FPS="${FPS:-10}"
FORCE_REGENERATE_OUTPUTS="${FORCE_REGENERATE_OUTPUTS:-0}"
MAX_EPISODES="${MAX_EPISODES:-}"
MIN_EPISODE_INDEX="${MIN_EPISODE_INDEX:-}"
EPISODE_SOURCE="${EPISODE_SOURCE:-}"
NUM_GPUS=1

SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-1}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.3}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"  # 0.7 (not 0.9) is the real speed lever; keep centroids >= q32/k64
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-10}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-1}"

CATEGORIES=()
if [[ -n "${CATEGORY_FILTER}" ]]; then CATEGORIES+=("${CATEGORY_FILTER}"); else CATEGORIES=(makovian non_makovian); fi

echo "============================================================"
echo " DreamGen multiview — single_arm — SVG2 (single-GPU)"
echo " Venv=${VENV}"
echo " Model=${MODEL_SIZE} FPS=${FPS} | SVG2 start=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} q_c=${SVG_NUM_Q_CENTROIDS} k_c=${SVG_NUM_K_CENTROIDS}"
echo " Output: ${OUTPUT_ROOT}"
echo "============================================================"

for CAT in "${CATEGORIES[@]}"; do
  DS_PATH="${DATASET_ROOT}/${CAT}"
  [[ -d "${DS_PATH}" ]] || { echo "[WARN] Skip missing: ${DS_PATH}"; continue; }
  [[ -f "${DS_PATH}/meta/episodes.jsonl" ]] || { echo "[WARN] No meta/episodes.jsonl in ${DS_PATH}, skip"; continue; }
  BATCH_JSON="${INPUT_DIR}/${CAT}/batch_input.json"
  OUTPUT_DIR="${OUTPUT_ROOT}/${CAT}"
  echo "========== ${CAT} =========="
  PREPARE_ARGS=(
    --dataset_path "${DS_PATH}" --input_dir "${INPUT_DIR}" --category "${CAT}"
    --batch_json "${BATCH_JSON}" --output_dir "${OUTPUT_DIR}" --skip_existing
  )
  [[ -n "${MAX_EPISODES}" ]] && PREPARE_ARGS+=(--max_episodes "${MAX_EPISODES}")
  [[ -n "${MIN_EPISODE_INDEX}" ]] && PREPARE_ARGS+=(--min_episode_index "${MIN_EPISODE_INDEX}")
  [[ -n "${EPISODE_SOURCE}" ]] && PREPARE_ARGS+=(--episode_source "${EPISODE_SOURCE}")
  [[ "${FORCE_REGENERATE_OUTPUTS}" == "1" ]] && PREPARE_ARGS+=(--force_regenerate_outputs)
  python "${ROOT}/scripts/prepare_single_arm_multiview_dreamgen_batch.py" "${PREPARE_ARGS[@]}"
  [[ -f "${BATCH_JSON}" ]] || { echo "[WARN] No batch at ${BATCH_JSON}, skip inference"; continue; }

  python -m examples.video2world_gr00t \
    --model_size "${MODEL_SIZE}" --gr00t_variant droid \
    --batch_input_json "${BATCH_JSON}" --prompt_prefix "" \
    --num_gpus "${NUM_GPUS}" --disable_guardrail --fps "${FPS}" \
    --use_svg --svg_variant svg2 \
    --svg_start_layer_idx "${SVG_START_LAYER_IDX}" --svg_dense_step_frac "${SVG_DENSE_STEP_FRAC}" \
    --svg_num_q_centroids "${SVG_NUM_Q_CENTROIDS}" --svg_num_k_centroids "${SVG_NUM_K_CENTROIDS}" \
    --svg_top_p_kmeans "${SVG_TOP_P_KMEANS}" --svg_min_kc_ratio "${SVG_MIN_KC_RATIO}" \
    --svg_kmeans_iter_init "${SVG_KMEANS_ITER_INIT}" --svg_kmeans_iter_step "${SVG_KMEANS_ITER_STEP}"
  echo ""
done

echo "[Done] SVG2 single_arm multiview — ${CATEGORIES[*]}"
