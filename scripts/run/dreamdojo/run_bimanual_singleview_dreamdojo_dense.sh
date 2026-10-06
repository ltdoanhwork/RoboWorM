#!/usr/bin/env bash
# DreamDojo dense action-conditioned inference on bimanual singleview splits.
# Runs datasets/bimanual/singleview/makovian and .../non_makovian with data_split full.
#
# Outputs under SAVE_ROOT/{makovian,non_makovian}/<iter_*>/
#
# Usage:
#   bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_dense.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_bimanual_singleview_dreamdojo_dense.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/bimanual/output/singleview/dreamdojo}"
DETERMINISTIC="${DETERMINISTIC:-0}"

get_task_paths() {
  local split="$1"
  local split_dir="${ROOT}/datasets/bimanual/singleview/${split}"
  local paths=""
  for d in $(find "${split_dir}" -maxdepth 1 -mindepth 1 -type d | sort); do
    # Skip tasks with no parquet files
    if find "$d/data" -name "*.parquet" 2>/dev/null | grep -q .; then
      if [ -n "${paths}" ]; then
        paths="${paths},"
      fi
      paths="${paths}${d}"
    else
      echo "Skipping empty task: $(basename $d)" >&2
    fi
  done
  echo "${paths}"
}

count_split_samples() {
  local split="$1"
  local paths="$2"
  python3 <<PY
import io
import sys

sys.path.insert(0, "${ROOT}/models/DreamDojo")
_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from groot_dreams.dataloader import MultiVideoActionDataset

    paths = "${paths}".split(",")
    d = MultiVideoActionDataset(
        dataset_path=paths,
        num_frames=int("${NUM_FRAMES}"),
        data_split="full",
        restrict_len=None,
        deterministic_uniform_sampling=False,
    )
    n = len(d)
finally:
    sys.stdout = _stdout
print(n)
PY
}

det_flag=()
if [[ "${DETERMINISTIC}" == "1" ]]; then
  det_flag=(--deterministic-uniform-sampling)
fi

for split in makovian non_makovian; do
  task_paths="$(get_task_paths "${split}")"
  if [ -z "${task_paths}" ]; then
    echo "Skipping ${split}: no task directories found." >&2
    continue
  fi
  n="$(count_split_samples "${split}" "${task_paths}")"
  if [[ -n "${NUM_SAMPLES_OVERRIDE:-}" ]]; then
    n="${NUM_SAMPLES_OVERRIDE}"
  fi
  if [ "${n}" -eq 0 ]; then
    echo "Skipping ${split}: dataset length is 0." >&2
    continue
  fi
  out_cache="${ROOT}/output/dreamdojo_dense_bimanual_sv_${split}"
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== DreamDojo dense bimanual SV: ${split} (num-samples=${n}) -> ${save_base} ==="
  python -m models.DreamDojo.examples.action_conditioned \
    -o "${out_cache}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${n}" \
    --dataset-path "${task_paths}" \
    --data-split full \
    "${det_flag[@]}"
done
