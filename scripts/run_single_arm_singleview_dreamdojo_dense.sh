#!/usr/bin/env bash
# DreamDojo dense action-conditioned inference on LeRobot single-arm singleview splits.
# Runs datasets/single_arm/singleview/makovian and .../non_makovian with data_split full.
#
# Outputs under SAVE_ROOT/{makovian,non_makovian}/<iter_*>/ (iter dir matches checkpoint).
#
# Usage:
#   bash scripts/run_single_arm_singleview_dreamdojo_dense.sh
#   CHECKPOINTS_DIR=.../checkpoints EXPERIMENT=dreamdojo_2b_480_640_aloha \
#     CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamdojo_dense.sh
#
# Optional: NUM_SAMPLES_OVERRIDE=100  (otherwise num-samples = full dataset length)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/exp_logs/dreamdojo_logs/dreamdojo/exp_aloha_resume40k/aloha_bimanual_singleview_full/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_aloha}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/dense/single_arm/output/singleview/dreamdojo}"
# Match finetuning.md batch examples; set to false to match humanoid dense script.
DETERMINISTIC="${DETERMINISTIC:-1}"

count_split_samples() {
  local split="$1"
  python3 <<PY
import io
import sys

sys.path.insert(0, "${ROOT}/models/DreamDojo")
_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    from groot_dreams.dataloader import MultiVideoActionDataset

    p = "${ROOT}/datasets/single_arm/singleview/${split}"
    d = MultiVideoActionDataset(
        dataset_path=p,
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
  n="$(count_split_samples "${split}")"
  if [[ -n "${NUM_SAMPLES_OVERRIDE:-}" ]]; then
    n="${NUM_SAMPLES_OVERRIDE}"
  fi
  if [ "${n}" -eq 0 ]; then
    echo "Skipping ${split}: dataset length is 0." >&2
    continue
  fi
  out_cache="${ROOT}/output/dreamdojo_dense_single_arm_sv_${split}"
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== DreamDojo dense single-arm SV: ${split} (num-samples=${n}) -> ${save_base} ==="
  python -m models.DreamDojo.examples.action_conditioned \
    -o "${out_cache}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${n}" \
    --dataset-path "${ROOT}/datasets/single_arm/singleview/${split}" \
    --data-split full \
    "${det_flag[@]}"
done
