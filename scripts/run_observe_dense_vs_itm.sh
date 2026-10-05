#!/usr/bin/env bash
# Compare dense+CFG baseline vs ITM (both use guidance=1.5, so fair comparison).
#
# Usage:
#   Terminal 1 (GPU 0 - dense with CFG):
#     bash scripts/run_observe_dense_vs_itm.sh dense_cfg
#
#   Terminal 2 (GPU 1 - ITM):
#     bash scripts/run_observe_dense_vs_itm.sh itm
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "${ROOT}/models/DreamDojo/.venv/bin/activate"
export PYTHONPATH="${ROOT}/models/DreamDojo:${PYTHONPATH:-}"

CHECKPOINTS_DIR="${ROOT}/checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000"
EXPERIMENT="dreamdojo_2b_480_640_gr1"
NUM_FRAMES=49
NUM_SAMPLES="${NUM_SAMPLES:-10}"
SPLIT="${SPLIT:-makovian}"
SAVE_ROOT="${ROOT}/tmp_2/observe"

MODE="${1:-}"

if [ -z "${MODE}" ]; then
  echo "Usage: bash $0 <dense_cfg|itm>"
  echo ""
  echo "  Terminal 1:  bash $0 dense_cfg   (dense + guidance=1.5, GPU 0)"
  echo "  Terminal 2:  bash $0 itm         (ITM + guidance=1.5, GPU 1)"
  exit 1
fi

if [ "${SKIP_META_REPAIR:-0}" != "1" ]; then
  python3 "${ROOT}/scripts/repair_humanoid_singleview_split_meta.py" --workspace "${ROOT}"
fi

if [ "${MODE}" = "dense_cfg" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  SAVE_DIR="${SAVE_ROOT}/dense_cfg/${SPLIT}"
  OUT_CACHE="${ROOT}/output/observe_dense_cfg_${SPLIT}"
  mkdir -p "${SAVE_DIR}" "${OUT_CACHE}"

  export FORCE_GUIDANCE=1.5
  echo "=== Dense + CFG (guidance=1.5) | GPU ${CUDA_VISIBLE_DEVICES} | ${NUM_SAMPLES} samples ==="
  time python -m models.DreamDojo.examples.action_conditioned \
    -o "${OUT_CACHE}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${SAVE_DIR}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${NUM_SAMPLES}" \
    --dataset-path "${ROOT}/datasets/humanoid/singleview/${SPLIT}" \
    --data-split full \
    --single-base-index \
    --deterministic-uniform-sampling

elif [ "${MODE}" = "itm" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  SAVE_DIR="${SAVE_ROOT}/itm_ultralight/${SPLIT}"
  OUT_CACHE="${ROOT}/output/observe_itm_ultralight_${SPLIT}"
  mkdir -p "${SAVE_DIR}" "${OUT_CACHE}"

  echo "=== ITM aggressive (guidance=1.5) | GPU ${CUDA_VISIBLE_DEVICES} | ${NUM_SAMPLES} samples ==="
  echo "  Config: start=4, keep=2, compress=0.25, prune_from=0, merge_from=0"
  time python -m models.DreamDojo.examples.action_conditioned \
    -o "${OUT_CACHE}" \
    --checkpoints-dir "${CHECKPOINTS_DIR}" \
    --experiment "${EXPERIMENT}" \
    --save-dir "${SAVE_DIR}" \
    --num-frames "${NUM_FRAMES}" \
    --num-samples "${NUM_SAMPLES}" \
    --dataset-path "${ROOT}/datasets/humanoid/singleview/${SPLIT}" \
    --data-split full \
    --single-base-index \
    --use-itm \
    --itm-start-layer-idx 4 \
    --itm-keep-last-n-dense 2 \
    --itm-prune-from-step 0 \
    --itm-merge-from-step 0 \
    --itm-compress-ratio 0.25 \
    --deterministic-uniform-sampling

else
  echo "Unknown mode: ${MODE}"
  echo "Usage: bash $0 <dense_cfg|itm>"
  exit 1
fi
