#!/usr/bin/env bash
# DreamDojo action-conditioned inference with ITM (Iterative Token Merging).
# Runs datasets/humanoid/singleview/makovian and .../non_makovian separately.
# Output: sampling_dataset/itm/humanoid/output/singleview/dreamdojo/{makovian,non_makovian}/<iter_*>/
# (was <repo>/tmp_2/ before 2026-10-09).
#
# NOTE: SiTo/ITM are wired only into models/DreamDojo/examples/action_conditioned.py,
# so this launcher cannot use the dense entrypoint
# (scripts/infer/infer_humanoid_singleview_full_episode.py has no --use-itm). ITM also
# needs CFG: action_conditioned.py raises guidance 0 -> 1.5 when ITM is on
# (override with FORCE_GUIDANCE), whereas dense runs at guidance 0.
#
# Usage:
#   bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_itm.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run/dreamdojo/run_humanoid_singleview_dreamdojo_itm.sh
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000}"
EXPERIMENT="${EXPERIMENT:-dreamdojo_2b_480_640_gr1}"
NUM_FRAMES="${NUM_FRAMES:-49}"
_SAVE_ROOT_ENV="${SAVE_ROOT:-}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/sampling_dataset/itm/humanoid/output/singleview/dreamdojo}"
# Intermediate --output-dir: OUTPUT_ROOT if set, else <SAVE_ROOT>/_output when SAVE_ROOT
# was given (keeps scratch runs self-contained), else the historical ${ROOT}/output.
OUT_CACHE_ROOT="${OUTPUT_ROOT:-${_SAVE_ROOT_ENV:+${_SAVE_ROOT_ENV}/_output}}"
OUT_CACHE_ROOT="${OUT_CACHE_ROOT:-${ROOT}/output}"

# ITM parameters
ITM_START_LAYER_IDX="${ITM_START_LAYER_IDX:-4}"
ITM_KEEP_LAST_N_DENSE="${ITM_KEEP_LAST_N_DENSE:-2}"
ITM_PRUNE_FROM_STEP="${ITM_PRUNE_FROM_STEP:-1}"
ITM_MERGE_FROM_STEP="${ITM_MERGE_FROM_STEP:-2}"

if [ "${SKIP_META_REPAIR:-0}" != "1" ]; then
  echo "=== Repairing makovian/non_makovian meta (SKIP_META_REPAIR=1 to skip) ==="
  python3 "${ROOT}/scripts/prepare/repair_humanoid_singleview_split_meta.py" --workspace "${ROOT}"
fi

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

    p = "${ROOT}/datasets/humanoid/singleview/${split}"
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
    echo "Skipping ${split}: dataset length is 0 (check missing parquets / paths)." >&2
    continue
  fi
  out_cache="${OUT_CACHE_ROOT}/dreamdojo_itm_humanoid_${split}"
  save_base="${SAVE_ROOT}/${split}"
  mkdir -p "${save_base}" "${out_cache}"
  echo "=== DreamDojo ITM: ${split} (num-samples=${n}) -> ${save_base} ==="
  python -m models.DreamDojo.examples.action_conditioned \
    -o "${out_cache}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${save_base}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${n}" \
    --dataset-path "${ROOT}/datasets/humanoid/singleview/${split}" \
    --data-split full \
    --single-base-index \
    --use-itm \
    --itm-start-layer-idx "${ITM_START_LAYER_IDX}" \
    --itm-keep-last-n-dense "${ITM_KEEP_LAST_N_DENSE}" \
    --itm-prune-from-step "${ITM_PRUNE_FROM_STEP}" \
    --itm-merge-from-step "${ITM_MERGE_FROM_STEP}"
done
