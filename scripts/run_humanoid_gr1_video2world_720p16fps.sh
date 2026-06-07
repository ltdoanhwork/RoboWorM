#!/usr/bin/env bash
# Run Cosmos Predict2 14B Video2World at 720p / 16fps on the GR1-100 single-view batch,
# writing outputs to a dedicated folder so runs do not overwrite ./dense/ generations.
#
# Defaults assume you already have batch_input.json with PNG conditioning (see README).
#
# Usage (from repo root):
#   bash ./scripts/run_humanoid_gr1_video2world_720p16fps.sh
#
# Override examples:
#   OUT_TAG=my720_run bash ./scripts/run_humanoid_gr1_video2world_720p16fps.sh
#   SOURCE_BATCH=./sampling_dataset/.../dense/batch_input.json OUT_TAG=720p_16fps_v2 bash ./scripts/run_humanoid_gr1_video2world_720p16fps.sh
set -euo pipefail

ROOT="/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"
cd "$ROOT"

SOURCE_BATCH="${SOURCE_BATCH:-${ROOT}/sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-GR1/dense/batch_input.json}"
OUT_TAG="${OUT_TAG:-720p_16fps}"
DIT_PATH="${DIT_PATH:-${ROOT}/checkpoints/nvidia/Cosmos-Predict2-14B-Video2World/model-720p-16fps.pt}"

OUTPUT_VIDEO_ROOT="${OUTPUT_VIDEO_ROOT:-${ROOT}/output/dreamgen_singleview/PhysicalAI-Robotics-GR00T-GR1/${OUT_TAG}}"
BATCH_OUT="${BATCH_OUT:-${ROOT}/sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-GR1/${OUT_TAG}/batch_input.json}"

export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NUM_GPUS="${NUM_GPUS:-1}"

if [[ ! -f "$SOURCE_BATCH" ]]; then
  echo "[ERROR] SOURCE_BATCH not found: $SOURCE_BATCH" >&2
  exit 1
fi
if [[ ! -f "$DIT_PATH" ]]; then
  echo "[ERROR] DiT checkpoint not found: $DIT_PATH" >&2
  echo "        Download with:" >&2
  echo "        python ./models/dreamgen/scripts/download_checkpoints.py \\" >&2
  echo "          --checkpoint_dir ./checkpoints --model_sizes 14B --model_types video2world \\" >&2
  echo "          --resolution 720 --fps 16" >&2
  exit 1
fi

mkdir -p "$(dirname "$BATCH_OUT")"
mkdir -p "$OUTPUT_VIDEO_ROOT"

export SOURCE_BATCH OUTPUT_VIDEO_ROOT BATCH_OUT

python - <<PY
import json
import os
from pathlib import Path

src = Path(os.environ["SOURCE_BATCH"])
out_root = Path(os.environ["OUTPUT_VIDEO_ROOT"])
dst = Path(os.environ["BATCH_OUT"])

items = json.loads(src.read_text(encoding="utf-8"))
for it in items:
    p = Path(it["output_video"])
    it["output_video"] = str(out_root / p.name)

dst.write_text(json.dumps(items, indent=4, ensure_ascii=False), encoding="utf-8")
print(f"[prepare] wrote {len(items)} entries -> {dst}")
print(f"[prepare] output_video root -> {out_root}")
PY

# shellcheck disable=SC1091
source "${ROOT}/models/DreamDojo/.venv/bin/activate"

torchrun -m models.dreamgen.examples.video2world \
  --model_size 14B \
  --resolution 720 \
  --fps 16 \
  --dit_path "$DIT_PATH" \
  --batch_input_json "$BATCH_OUT" \
  --disable_guardrail \
  --num_gpus "$NUM_GPUS"
