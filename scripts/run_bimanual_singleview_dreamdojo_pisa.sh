#!/usr/bin/env bash
# DreamDojo bimanual (ALOHA) singleview inference with PISA (piecewise sparse attention).
# Mirrors run_bimanual_singleview_dreamdojo_full_episode.sh / _dicache.sh exactly
# (same ckpt / dataset / inference script), only adding the PISA flags and
# routing output to the pisa/ method tree.
#
# Output: sampling_dataset/pisa/bimanual/output/singleview/dreamdojo/{split}/iter_*/{task}__episode_*/
#
# Usage:
#   bash scripts/run_bimanual_singleview_dreamdojo_pisa.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_bimanual_singleview_dreamdojo_pisa.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
CHUNK_SIZE="${CHUNK_SIZE:-12}"
SAVE_FPS="${SAVE_FPS:-10}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/pisa/bimanual/output/singleview/dreamdojo}"
NUM_EPISODES="${NUM_EPISODES:-}"
GUIDANCE="${GUIDANCE:-1.5}"
SPLITS="${SPLITS:-makovian non_makovian}"
SAVE_MODE="${SAVE_MODE:-video_only}"   # video_only | full (full also writes gt/merged/metrics.json)

# PISA parameters (tuned; see dreamdojo-bimanual-singleview-sparse-attention-insights)
# density 0.2 is the only faster point: 377.21s vs dense 390.08s (1.03x) but ALREADY
# SSIM-lossy (0.686 vs 0.710). density 0.35 is slower AND still below dense — there
# is NO ==dense PISA config. BREAK POINT is density <= 0.10: PSNR 17.34, SSIM 0.659.
# Speed saturates ~1.07x (Amdahl ceiling ~1.22x; self-attn is 18% of gen time).
PISA_DENSITY="${PISA_DENSITY:-0.2}"
PISA_BLOCK_SIZE="${PISA_BLOCK_SIZE:-64}"
PISA_START_LAYER_IDX="${PISA_START_LAYER_IDX:-2}"

get_task_paths() {
  local split="$1"
  local split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  local paths=""
  for d in $(find "${split_dir}" -maxdepth 1 -mindepth 1 -type d | sort); do
    if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
      if [ -n "${paths}" ]; then
        paths="${paths},"
      fi
      paths="${paths}${d}"
    fi
  done
  echo "${paths}"
}

ep_flag=()
if [[ -n "${NUM_EPISODES}" ]]; then
  ep_flag=(--num-episodes "${NUM_EPISODES}")
fi
save_flag=()
if [[ "${SAVE_MODE}" == "full" ]]; then
  save_flag=(--save-full)
fi

for split in ${SPLITS}; do
  split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  if [ ! -d "${split_dir}" ]; then
    echo "Skipping ${split}: directory not found at ${split_dir}" >&2
    continue
  fi

  dataset_paths="$(get_task_paths "${split}")"
  if [ -z "${dataset_paths}" ]; then
    echo "Skipping ${split}: no task directories with data/ found" >&2
    continue
  fi

  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  echo "=== DreamDojo PISA bimanual SV: ${split} -> ${save_base} ==="
  python "${ROOT}/scripts/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${dataset_paths}" \
    --save-dir "${save_base}" \
    --chunk-size "${CHUNK_SIZE}" \
    --save-fps "${SAVE_FPS}" \
    --output-dir "${ROOT}/output/dreamdojo_pisa_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    --use-pisa \
    --pisa-density "${PISA_DENSITY}" \
    --pisa-block-size "${PISA_BLOCK_SIZE}" \
    --pisa-start-layer-idx "${PISA_START_LAYER_IDX}" \
    "${ep_flag[@]}" "${save_flag[@]}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}"
echo "=========================================================="
