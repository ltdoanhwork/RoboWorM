#!/usr/bin/env bash
# =============================================================================
# run_single_arm_multiview_video2world_2b_tmp.sh
# Run Cosmos **base** Video2World **2B** (not GR00T) on the same multiview batch
# JSON produced by prepare_single_arm_multiview_dreamgen_batch.py — for lower-
# fidelity / baseline comparison ("video tệ hơn" vs 14B GR00T Dreams DROID).
#
# Default: **LVG** (`examples.video2world_lvg`), `NUM_CHUNKS=4`, 480p @ 16fps
# (pipeline must match DiT: model-480p-16fps.pt). Set USE_LVG=0 for one-shot
# `examples.video2world` (shorter clips).
#
# Default checkpoint: Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt
# Outputs only: every output_video path is rewritten to under ./tmp/...
#
# From repo root:
#   bash scripts/run_single_arm_multiview_video2world_2b_tmp.sh
#
# Env:
#   BATCH_JSON     input batch (default: makovian dense batch)
#   TMP_OUT        output dir (default: ./tmp/multiview_video2world_2b_480p16fps_lvg)
#   DIT_PATH       DiT .pt (default: checkpoints/nvidia/.../model-480p-16fps.pt)
#   USE_LVG        1 (default) = video2world_lvg + NUM_CHUNKS; 0 = video2world
#   NUM_CHUNKS     LVG chunks (default 4)
#   RESOLUTION, FPS  must match checkpoint (defaults 480, 16)
#   NUM_GPUS, MASTER_PORT, GUIDANCE, SEED
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BATCH_JSON="${BATCH_JSON:-${REPO_ROOT}/sampling_dataset/dense/single_arm/input/multiview/dreamgen/makovian/batch_input.json}"
USE_LVG="${USE_LVG:-1}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
RESOLUTION="${RESOLUTION:-480}"
FPS="${FPS:-16}"

if [[ "${USE_LVG}" == "1" ]]; then
  TMP_OUT="${TMP_OUT:-${REPO_ROOT}/tmp/multiview_video2world_2b_480p16fps_lvg}"
else
  TMP_OUT="${TMP_OUT:-${REPO_ROOT}/tmp/multiview_video2world_2b_480p16fps}"
fi

DIT_PATH="${DIT_PATH:-${REPO_ROOT}/checkpoints/nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt}"
MODEL_SIZE="${MODEL_SIZE:-2B}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-12342}"
GUIDANCE="${GUIDANCE:-7}"
SEED="${SEED:-0}"

if [[ ! -f "${DIT_PATH}" ]]; then
  echo "[ERROR] DiT checkpoint not found: ${DIT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${BATCH_JSON}" ]]; then
  echo "[ERROR] batch_input.json not found: ${BATCH_JSON}" >&2
  exit 1
fi

mkdir -p "${TMP_OUT}"
export REPO_ROOT SOURCE_BATCH="${BATCH_JSON}" TMP_OUT

# Rewrite output_video basename into TMP_OUT (keep same grid inputs / prompts).
BATCH_REDIRECT="$(python - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["TMP_OUT"])
out_root.mkdir(parents=True, exist_ok=True)
src = Path(os.environ["SOURCE_BATCH"])
data = json.loads(src.read_text(encoding="utf-8"))
for item in data:
    name = Path(item["output_video"]).name
    item["output_video"] = str(out_root / name)
dst = out_root / "batch_input_redirected.json"
dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(dst)
PY
)"

echo "============================================================"
echo " Video2World 2B (non-GR00T) → tmp"
echo "============================================================"
echo " DiT        : ${DIT_PATH}"
echo " Batch in   : ${BATCH_JSON}"
echo " Redirected : ${BATCH_REDIRECT}"
echo " Model      : ${MODEL_SIZE} ${RESOLUTION}p ${FPS}fps  GPUs: ${NUM_GPUS}"
echo " USE_LVG    : ${USE_LVG}  NUM_CHUNKS: ${NUM_CHUNKS}"
echo "============================================================"

source "${REPO_ROOT}/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="${REPO_ROOT}/models/dreamgen${PYTHONPATH:+:$PYTHONPATH}"

AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)
if [[ "${AVAILABLE_GPUS}" -lt 1 ]]; then
  echo "[ERROR] No CUDA GPUs visible to PyTorch." >&2
  exit 1
fi
if [[ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
  echo "[ERROR] NUM_GPUS=${NUM_GPUS} but only ${AVAILABLE_GPUS} GPU(s)." >&2
  exit 1
fi

COMMON=(
  --model_size       "${MODEL_SIZE}"
  --dit_path         "${DIT_PATH}"
  --batch_input_json "${BATCH_REDIRECT}"
  --num_gpus         "${NUM_GPUS}"
  --aspect_ratio     "16:9"
  --num_conditional_frames 1
  --guidance         "${GUIDANCE}"
  --seed             "${SEED}"
  --disable_guardrail
  --disable_prompt_refiner
)

if [[ "${USE_LVG}" == "1" ]]; then
  INFER=(
    -m examples.video2world_lvg
    "${COMMON[@]}"
    --resolution       "${RESOLUTION}"
    --fps              "${FPS}"
    --num_chunks       "${NUM_CHUNKS}"
  )
else
  INFER=(
    -m examples.video2world
    "${COMMON[@]}"
    --resolution       "${RESOLUTION}"
    --fps              "${FPS}"
  )
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${INFER[@]}"
else
  python "${INFER[@]}"
fi

echo ""
echo "[Done] Videos under: ${TMP_OUT}"
