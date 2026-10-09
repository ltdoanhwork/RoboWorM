#!/usr/bin/env bash
# Full-episode single-segment autoregressive rollout for bimanual singleview DreamDojo.
# Pure autoregressive from first GT frame — no GT reset between chunks.
# Generates video as long as the input episode.
#
# Output: sampling_dataset/dense/bimanual/output/singleview/dreamdojo/{split}/iter_*/episode_*/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh
#   NUM_EPISODES=10 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_full_episode.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
NUM_EPISODES="${NUM_EPISODES:-}"
# Guidance of the dense reference. 0 = the infer-script default this launcher has
# always used. Every method launcher of this cell reads the same GUIDANCE variable
# with the same default.
GUIDANCE="${GUIDANCE:-0}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"

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

# SAVE_FULL=1 adds --save-full (PSNR/SSIM/LPIPS). The sparse-attention scripts
# for this cell already pass it, so without this the dense baseline would have no
# quality numbers to pair against theirs.
full_flag=()
if [ "${SAVE_FULL:-0}" = "1" ]; then full_flag=(--save-full); fi
# GT_ONLY=1 backfills full_gt.mp4 next to predictions that were written video-only,
# without running diffusion -- needed because the reference-based (vs_gt) half of the
# DR grid has no reference for this cell otherwise. Loads no checkpoint, needs no GPU.
if [ "${GT_ONLY:-0}" = "1" ]; then full_flag=(--gt-only); fi

# This script fans the two splits onto GPU 0 and GPU 1 concurrently. That is
# wrong whenever fewer than two devices are visible: the second split lands on a
# device that does not exist (hard failure), or -- if CUDA_VISIBLE_DEVICES pins a
# single card -- both splits share it and every timing is inflated by contention.
# Count what is actually visible and fall back to sequential. SEQUENTIAL=1 forces
# it regardless, which is what the timing harness wants even on a multi-GPU box.
n_visible=2
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  n_visible=$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")
else
  n_visible=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  [[ "${n_visible}" -lt 1 ]] && n_visible=1
fi
SEQUENTIAL="${SEQUENTIAL:-0}"
if [[ "${SEQUENTIAL}" != "1" && "${n_visible}" -lt 2 ]]; then
  SEQUENTIAL=1
  echo "[info] only ${n_visible} GPU visible -> running splits SEQUENTIALLY" \
       "(set SEQUENTIAL=0 and expose 2 GPUs to parallelise)"
fi

pids=()
gpu=0

for split in makovian non_makovian; do
  task_paths="$(get_task_paths "${split}")"
  if [ -z "${task_paths}" ]; then
    echo "Skipping ${split}: no task directories found." >&2
    continue
  fi
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}"
  # In sequential mode keep whatever device the caller pinned; only the parallel
  # path needs to hand each split its own index.
  if [[ "${SEQUENTIAL}" == "1" ]]; then
    split_dev="${CUDA_VISIBLE_DEVICES:-0}"
  else
    split_dev="${gpu}"
  fi
  echo "=== Full-episode single-segment rollout: ${split} (GPU ${split_dev}) -> ${save_base} ==="
  CUDA_VISIBLE_DEVICES="${split_dev}" python "${ROOT}/scripts/infer/infer_bimanual_singleview_single_segment.py" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --dataset-path "${task_paths}" \
    --save-dir "${save_base}" \
    --save-fps 10 \
    --output-dir "${OUT_CACHE_ROOT}/dreamdojo_full_ep_bimanual_sv_${split}" \
    --guidance "${GUIDANCE}" \
    "${full_flag[@]}" \
    "${ep_flag[@]}" &
  pids+=($!)
  gpu=$((gpu + 1))
  # Sequential mode: wait here so the next split does not share the device.
  if [[ "${SEQUENTIAL}" == "1" ]]; then
    wait "${pids[-1]}"
    pids=()
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done
