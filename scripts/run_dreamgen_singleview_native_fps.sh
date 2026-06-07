#!/usr/bin/env bash
# =============================================================================
# Run DreamGen singleview with NATIVE fps conditioning matching training.
#
# Key fix: The default pipeline passes random fps in [16,32] as conditioning,
# but training used actual native fps (bimanual=30, single_arm=15, humanoid=20).
# This script monkey-patches the pipeline to pass correct fps conditioning
# while keeping state_t=24 (architecture matches training).
#
# Output: tmp/2/{embodiment}/{split}/fps_{native}_native/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_dreamgen_singleview_native_fps.sh
#   CUDA_VISIBLE_DEVICES=0 EMBODIMENT=single_arm bash scripts/run_dreamgen_singleview_native_fps.sh
#   CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/run_dreamgen_singleview_native_fps.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- Environment ---
export VENV="${ROOT}/models/dreamgen/.venv"
export LD_LIBRARY_PATH="${VENV}/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${VENV}/lib/python3.12/site-packages/nvidia/cusparse/lib:${VENV}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
source "${VENV}/bin/activate"
export PYTHONPATH="${ROOT}/models/dreamgen:${ROOT}:${PYTHONPATH:-}"

# --- Settings ---
EMBODIMENT="${EMBODIMENT:-all}"
SPLIT="${SPLIT:-both}"
MAX_EPISODES="${MAX_EPISODES:-3}"
SAVE_ROOT="${SAVE_ROOT:-${ROOT}/tmp/2}"
NUM_GPUS="${NUM_GPUS:-1}"
ITER_NAME="${ITER_NAME:-iter_000090000}"
GUIDANCE="${GUIDANCE:-7}"
USE_PROMPT_REFINEMENT="${USE_PROMPT_REFINEMENT:-0}"

# Build optional flags
EXTRA_FLAGS=""
if [[ "${USE_PROMPT_REFINEMENT}" == "1" ]]; then
  EXTRA_FLAGS="--use-prompt-refinement"
fi

echo "============================================"
echo "DreamGen Native FPS Conditioning Test"
echo "  Embodiment:  ${EMBODIMENT}"
echo "  Split:       ${SPLIT}"
echo "  Max eps:     ${MAX_EPISODES}"
echo "  GPUs:        ${NUM_GPUS}"
echo "  Iter:        ${ITER_NAME}"
echo "  Save root:   ${SAVE_ROOT}"
echo "  Prompt refine: ${USE_PROMPT_REFINEMENT}"
echo "============================================"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-29515}" \
    "${ROOT}/scripts/infer_dreamgen_singleview_native_fps.py" \
    --embodiment "${EMBODIMENT}" \
    --split "${SPLIT}" \
    --max-episodes "${MAX_EPISODES}" \
    --save-root "${SAVE_ROOT}" \
    --iter-name "${ITER_NAME}" \
    --guidance "${GUIDANCE}" \
    --num-gpus "${NUM_GPUS}" \
    ${EXTRA_FLAGS}
else
  python "${ROOT}/scripts/infer_dreamgen_singleview_native_fps.py" \
    --embodiment "${EMBODIMENT}" \
    --split "${SPLIT}" \
    --max-episodes "${MAX_EPISODES}" \
    --save-root "${SAVE_ROOT}" \
    --iter-name "${ITER_NAME}" \
    --guidance "${GUIDANCE}" \
    --num-gpus "${NUM_GPUS}" \
    ${EXTRA_FLAGS}
fi
