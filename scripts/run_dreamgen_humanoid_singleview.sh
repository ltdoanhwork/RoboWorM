#!/usr/bin/env bash
# Run DreamGen video2world_gr00t (gr1 single-view) on every subset whose
# batch_input.json was produced by scripts/video2prompt.py.
#
# Inputs come from:
#   ./sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1/<subset>/batch_input.json
set -euo pipefail

ROOT=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
cd "$ROOT"

# source "$ROOT/models/dreamgen/.venv/bin/activate"
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NUM_GPUS=${NUM_GPUS:-1}

INPUT_ROOT=${INPUT_ROOT:-./sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1}
SUBSETS=${SUBSETS:-"DreamDojo-HV_Eval EgoDex_Eval"}
# Tạm tắt: In-lab_Eval GR1_robot
MODEL_SIZE=${MODEL_SIZE:-14B}
GR00T_VARIANT=${GR00T_VARIANT:-gr1}

for subset in $SUBSETS; do
  BATCH_JSON="$INPUT_ROOT/$subset/batch_input.json"
  if [ ! -f "$BATCH_JSON" ]; then
    echo "[skip] $BATCH_JSON not found"
    continue
  fi
  echo "=== DreamGen on $subset ==="
  torchrun -m models.dreamgen.examples.video2world_gr00t \
    --model_size "$MODEL_SIZE" \
    --gr00t_variant "$GR00T_VARIANT" \
    --batch_input_json "$BATCH_JSON" \
    --disable_guardrail \
    --num_gpus "$NUM_GPUS"
done
