#!/usr/bin/env bash
# =============================================================================
# DreamGen humanoid SINGLEVIEW + SVG2 sparse attention.
#
# Mirrors run_humanoid_singleview_dreamgen_static16fps.sh (dense) EXACTLY
# (video2world_lvg, humanoid_singleview_16fps, _static16fps tag, 2/3-chunk short
# rollout, num_conditional_frames=1) -- only routes output to sampling_dataset/svg2
# and adds the SVG2 flags. SVG2 needs the .venv_svg env (flashinfer 0.3.1 block-
# sparse wrapper); the default .venv (flashinfer 0.2.5) falls into a dense-mask
# fallback. Override VENV to change.
#
# Output: sampling_dataset/svg2/humanoid/output/singleview/dreamgen/{split}/<iter>_static16fps/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_humanoid_singleview_dreamgen_svg2.sh
#
# Env overrides: VENV, ITER_NAME, DIT_PATH, GUIDANCE, FPS, MASTER_PORT, ASPECT_RATIO,
#   SPLITS, NUM_CHUNKS, FORCE_REGENERATE, DENSE_ROOT, OUTPUT_ROOT,
#   SVG_{START_LAYER_IDX,DENSE_STEP_FRAC,NUM_Q_CENTROIDS,NUM_K_CENTROIDS,TOP_P_KMEANS,MIN_KC_RATIO,KMEANS_ITER_INIT,KMEANS_ITER_STEP}
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export VENV="${VENV:-${ROOT}/models/dreamgen/.venv_svg}"
_PYVER="$(basename "$(ls -d ${VENV}/lib/python3.* 2>/dev/null | head -1)")"
export LD_LIBRARY_PATH="${VENV}/lib/${_PYVER}/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/${_PYVER}/site-packages/nvidia/cusparse/lib:${VENV}/lib/${_PYVER}/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}:${ROOT}/models/dreamgen:${PYTHONPATH:-}"

CKPT_BASE="${ROOT}/exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps"
ITER_NAME="${ITER_NAME:-iter_000090000}"
DIT_PATH="${DIT_PATH:-${CKPT_BASE}/checkpoints_40_per/model/${ITER_NAME}.pt}"
if [[ ! -f "${DIT_PATH}" ]]; then echo "ERROR: Checkpoint not found at ${DIT_PATH}" >&2; exit 1; fi
OUT_ITER="${ITER_NAME}_static16fps"

NUM_GPUS="${NUM_GPUS:-1}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
GUIDANCE="${GUIDANCE:-7}"
MASTER_PORT="${MASTER_PORT:-29570}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
SPLITS="${SPLITS:-makovian non_makovian}"
NUM_CHUNKS="${NUM_CHUNKS:-2 3}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"

# --- SVG2 parameters (small centroids for stability; re-tune per model) ---
SVG_START_LAYER_IDX="${SVG_START_LAYER_IDX:-1}"
SVG_DENSE_STEP_FRAC="${SVG_DENSE_STEP_FRAC:-0.3}"
SVG_NUM_Q_CENTROIDS="${SVG_NUM_Q_CENTROIDS:-32}"
SVG_NUM_K_CENTROIDS="${SVG_NUM_K_CENTROIDS:-64}"
SVG_TOP_P_KMEANS="${SVG_TOP_P_KMEANS:-0.7}"  # 0.7 (not 0.9) is the real speed lever; keep centroids >= q32/k64
SVG_MIN_KC_RATIO="${SVG_MIN_KC_RATIO:-0.15}"
SVG_KMEANS_ITER_INIT="${SVG_KMEANS_ITER_INIT:-10}"
SVG_KMEANS_ITER_STEP="${SVG_KMEANS_ITER_STEP:-1}"

DENSE_ROOT="${DENSE_ROOT:-${ROOT}/sampling_dataset/dense}"
INPUT_ROOT="${DENSE_ROOT}/humanoid/input/singleview/dreamgen"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/sampling_dataset/svg2/humanoid/output/singleview/dreamgen}"

echo "============================================"
echo "DreamGen Humanoid Singleview + SVG2 (static16fps / LVG)"
echo "  Checkpoint: ${DIT_PATH}"
echo "  Venv:       ${VENV}"
echo "  Iter tag:   ${OUT_ITER}"
echo "  SVG2:       start_layer=${SVG_START_LAYER_IDX} dense_frac=${SVG_DENSE_STEP_FRAC} q_c=${SVG_NUM_Q_CENTROIDS} k_c=${SVG_NUM_K_CENTROIDS}"
echo "  Splits:     ${SPLITS} | chunks: ${NUM_CHUNKS}"
echo "============================================"

run_inference() {
  local src_batch_json="$1" label="$2" num_chunks="$3" split="$4"
  [[ -f "${src_batch_json}" ]] || { echo "  [SKIP] ${label}: batch file not found"; return; }
  local method_batch_json; method_batch_json="$(mktemp --suffix=_dreamgen_svg2.json)"
  local pending
  pending="$(python3 - "${src_batch_json}" "${method_batch_json}" "${OUTPUT_ROOT}/${split}/${OUT_ITER}" "${FORCE_REGENERATE}" <<'PY'
import json, sys
from pathlib import Path
src, dst, out_dir, force = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
items = json.loads(Path(src).read_text())
for it in items:
    it["output_video"] = f"{out_dir}/{Path(it['output_video']).name}"
if not force:
    items = [x for x in items if not Path(x["output_video"]).is_file()]
Path(dst).write_text(json.dumps(items, indent=2))
print(len(items))
PY
  )"
  if [[ "${pending}" -eq 0 ]]; then echo "  [SKIP] ${label}: all episodes already generated"; rm -f "${method_batch_json}"; return; fi
  mkdir -p "${OUTPUT_ROOT}/${split}/${OUT_ITER}"
  echo "  === Running: ${label} (${pending} pending, chunks=${num_chunks}) ==="

  local RUN=(python -m models.dreamgen.examples.video2world_lvg)
  [[ "${NUM_GPUS}" -gt 1 ]] && RUN=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" -m models.dreamgen.examples.video2world_lvg)
  "${RUN[@]}" \
    --model_size "${MODEL_SIZE}" --resolution "${RESOLUTION}" --fps "${FPS}" \
    --dit_path "${DIT_PATH}" --batch_input_json "${method_batch_json}" \
    --guidance "${GUIDANCE}" --num_gpus "${NUM_GPUS}" \
    --num_chunks "${num_chunks}" --num_conditional_frames 1 \
    --aspect_ratio "${ASPECT_RATIO}" --disable_guardrail --disable_prompt_refiner \
    --use_svg --svg_variant svg2 \
    --svg_start_layer_idx "${SVG_START_LAYER_IDX}" --svg_dense_step_frac "${SVG_DENSE_STEP_FRAC}" \
    --svg_num_q_centroids "${SVG_NUM_Q_CENTROIDS}" --svg_num_k_centroids "${SVG_NUM_K_CENTROIDS}" \
    --svg_top_p_kmeans "${SVG_TOP_P_KMEANS}" --svg_min_kc_ratio "${SVG_MIN_KC_RATIO}" \
    --svg_kmeans_iter_init "${SVG_KMEANS_ITER_INIT}" --svg_kmeans_iter_step "${SVG_KMEANS_ITER_STEP}"

  MASTER_PORT=$((MASTER_PORT + 1)); rm -f "${method_batch_json}"; echo "  === Done: ${label} ==="
}

for split in ${SPLITS}; do
  input_dir="${INPUT_ROOT}/${split}"
  [[ -d "${input_dir}" ]] || { echo "  [SKIP] ${split}: input dir not found"; continue; }
  for nc in ${NUM_CHUNKS}; do
    run_inference "${input_dir}/batch_input_short_${nc}chunks.json" "humanoid/${split}/short_${nc}chunks" "${nc}" "${split}"
  done
done

echo ""; echo "Done! Results: ${OUTPUT_ROOT}/{split}/${OUT_ITER}/"
