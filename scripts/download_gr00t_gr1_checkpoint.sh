#!/usr/bin/env bash
set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT}/checkpoints}"
DOWNLOAD_SCRIPT="${ROOT}/models/dreamgen/scripts/download_checkpoints.py"

# video2world_gr00t.py currently resolves GR1 to:
# checkpoints/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-480p-16fps.pt
MODEL_SIZE="${MODEL_SIZE:-14B}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${MODEL_SIZE}" != "14B" ]]; then
  echo "[ERROR] GR00T GR1 checkpoint only supports MODEL_SIZE=14B, got: ${MODEL_SIZE}" >&2
  exit 1
fi

if [[ "${RESOLUTION}" != "480" || "${FPS}" != "16" ]]; then
  echo "[WARN] video2world_gr00t currently loads 480p-16fps checkpoints." >&2
  echo "       You set RESOLUTION=${RESOLUTION}, FPS=${FPS}." >&2
fi

EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

python "${DOWNLOAD_SCRIPT}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --model_sizes "${MODEL_SIZE}" \
  --model_types video2world sample_gr00t_dreams_gr1 \
  --resolution "${RESOLUTION}" \
  --fps "${FPS}" \
  "${EXTRA_ARGS[@]}"

echo
echo "[DONE] Expected GR1 checkpoint path:"
echo "  ${CHECKPOINT_DIR}/nvidia/Cosmos-Predict2-14B-Sample-GR00T-Dreams-GR1/model-480p-16fps.pt"
