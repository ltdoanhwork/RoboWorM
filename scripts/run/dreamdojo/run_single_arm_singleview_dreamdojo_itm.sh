#!/usr/bin/env bash
# DreamDojo action-conditioned inference with ITM for single_arm/singleview.
# Output: sampling_dataset/itm/single_arm/output/singleview/dreamdojo/{makovian,non_makovian}/<iter_*>/
#
# Usage:
#   bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_itm.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_single_arm_singleview_dreamdojo_itm.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

# 2B_DROID_post-train never existed in this repo; mirror run_single_arm_singleview_dreamdojo_dense.sh
# (ckpt + experiment + num_frames) so the method is comparable to dense.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_single_arm_sv}"
NUM_FRAMES="${NUM_FRAMES:-97}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/itm/single_arm/output/singleview/dreamdojo}"

# ITM parameters
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-4}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-2}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-1}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-2}"

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
        single_base_index=True,
    )
    n = len(d)
finally:
    sys.stdout = _stdout
print(n)
PY
}

for split in makovian non_makovian; do
  n="$(count_split_samples "${split}")"
  if [ "${n}" -eq 0 ]; then
    echo "Skipping ${split}: dataset length is 0." >&2
    continue
  fi
  out_cache="${ROOT}/output/dreamdojo_itm_single_arm_${split}"
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== DreamDojo ITM: single_arm/${split} (num-samples=${n}) -> ${save_base} ==="
  python -m models.DreamDojo.examples.action_conditioned \
    -o "${out_cache}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${n}" \
    --dataset-path "${ROOT}/datasets/single_arm/singleview/${split}" \
    --data-split full \
    --single-base-index \
    --use-itm \
    --itm-start-layer-idx "${ITM_START_LAYER_IDX}" \
    --itm-keep-last-n-dense "${ITM_KEEP_LAST_N_DENSE}" \
    --itm-prune-from-step "${ITM_PRUNE_FROM_STEP}" \
    --itm-merge-from-step "${ITM_MERGE_FROM_STEP}"
done

echo ""
echo "=========================================================="
echo "Done! Results saved to: ${SAVE_ROOT}/{makovian,non_makovian}"
echo "=========================================================="
