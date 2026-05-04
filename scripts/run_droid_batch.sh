#!/bin/bash
# run_droid_batch.sh
set -euo pipefail

# Ensure we are in the root directory
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

# Activate the environment
source models/DreamDojo/.venv/bin/activate

# Setup PYTHONPATH for DreamGen
export PYTHONPATH="$(pwd)/models/dreamgen:${PYTHONPATH:-}"
export MASTER_PORT="${MASTER_PORT:-12341}"

AVAILABLE_GPUS=$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "ERROR: No CUDA GPUs are visible to PyTorch." >&2
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-${AVAILABLE_GPUS}}"
if [ "${NUM_GPUS}" -gt "${AVAILABLE_GPUS}" ]; then
  echo "ERROR: NUM_GPUS=${NUM_GPUS}, but only ${AVAILABLE_GPUS} CUDA GPU(s) are visible." >&2
  echo "Set CUDA_VISIBLE_DEVICES or lower NUM_GPUS." >&2
  exit 1
fi

echo "Using ${NUM_GPUS} CUDA GPU(s)."

# 1. Prepare the batch input JSON (extract screenshots and prompts from DROID dataset)
echo "Step 1: Preparing batch input JSON..."
if [ -n "${PREPARE_ARGS:-}" ]; then
  python scripts/prepare_droid_batch.py ${PREPARE_ARGS}
else
  MAX_EPISODES="${MAX_EPISODES:-100}"
  if [ "${MAX_EPISODES}" -gt 0 ]; then
    echo "Preparing ${MAX_EPISODES} episode(s). Set MAX_EPISODES=0 for all local episodes."
    python scripts/prepare_droid_batch.py --max_episodes "${MAX_EPISODES}"
  else
    echo "Preparing all local episodes. This can take a long time."
    python scripts/prepare_droid_batch.py
  fi
fi

# 2. Run inference using torchrun and the DreamGen droid variant
echo "Step 2: Running inference..."
INFERENCE_ARGS=(
  -m examples.video2world_gr00t
  --model_size 14B
  --gr00t_variant droid
  --batch_input_json ./sampling_dataset/droid_20p/batch_input.json
  --num_gpus "${NUM_GPUS}"
  --disable_guardrail
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" "${INFERENCE_ARGS[@]}"
else
  python "${INFERENCE_ARGS[@]}"
fi

