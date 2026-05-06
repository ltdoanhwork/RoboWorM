#!/usr/bin/env bash
# Caption all PhysicalAI-Robotics-GR00T-Teleop-GR1 single-view clips with
# Qwen3-VL-30B-A3B-Instruct-FP8 and produce DreamGen batch_input.json files.
#
# NOTE: Qwen3-VL FP8 needs vLLM >= 0.11 and transformers >= 4.57.
#       The default DreamDojo/.venv ships transformers 4.51 / no vllm, so
#       run this from an env that has the newer stack (e.g. a dedicated
#       qwen3vl venv). Override PYTHON= to point at it.
set -euo pipefail

ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
cd "$ROOT"

PYTHON=${PYTHON:-python}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}

SRC_ROOT=${SRC_ROOT:-./datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1}
OUT_ROOT=${OUT_ROOT:-./sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1}
OUTPUT_VIDEO_ROOT=${OUTPUT_VIDEO_ROOT:-./output/dreamgen_singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1}

SUBSETS=${SUBSETS:-"DreamDojo-HV_Eval EgoDex_Eval"}
# Tạm tắt: In-lab_Eval GR1_robot
LIMIT=${LIMIT:-0}      # per-subset cap; 0 = all
EXTRA=${EXTRA:-}       # e.g. --overwrite

$PYTHON -m scripts.video2prompt \
  --src_root "$SRC_ROOT" \
  --out_root "$OUT_ROOT" \
  --subsets $SUBSETS \
  --output_video_root "$OUTPUT_VIDEO_ROOT" \
  --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
  --limit_per_subset "$LIMIT" \
  $EXTRA
