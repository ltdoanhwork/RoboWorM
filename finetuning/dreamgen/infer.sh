#!/bin/bash
# DreamGen (Cosmos-Predict2 Video2World) inference on post-trained checkpoints.
#
# Usage:
#   bash finetuning/dreamgen/infer.sh \
#       --ckpt_path <path_to_checkpoint.pt> \
#       --input_path <first_frame_image.png_or_jpg> \
#       --prompt "A description of the task." \
#       --save_path output/generated_video.mp4
#
# Examples:
#
#   # Bimanual singleview post-trained checkpoint
#   bash finetuning/dreamgen/infer.sh \
#       --ckpt_path exp_logs/dreamgen_logs/posttraining/singleview/bimanual_singleview/checkpoints/model/iter_000005000.pt \
#       --input_path /path/to/first_frame.png \
#       --prompt "A bimanual robot performing the task: close cardboard box." \
#       --save_path output/dreamgen_bimanual_infer.mp4
#
#   # Single-arm singleview post-trained checkpoint
#   bash finetuning/dreamgen/infer.sh \
#       --ckpt_path exp_logs/dreamgen_logs/posttraining/singleview/single_arm_singleview/checkpoints/model/iter_000005000.pt \
#       --input_path /path/to/first_frame.png \
#       --prompt "A robot arm performing a manipulation task." \
#       --save_path output/dreamgen_single_arm_infer.mp4
#
#   # Humanoid GR1 singleview post-trained checkpoint
#   bash finetuning/dreamgen/infer.sh \
#       --ckpt_path exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview/checkpoints/model/iter_000005000.pt \
#       --input_path /path/to/first_frame.png \
#       --prompt "A GR-1 humanoid robot performing mug robot task." \
#       --save_path output/dreamgen_humanoid_infer.mp4
#
# Optional flags forwarded to examples/video2world.py:
#   --load_ema              Use EMA weights (often better quality)
#   --model_size 14B        Use 14B model instead of 2B (default)
#   --num_gpus <N>          Number of GPUs (default: 1)
#   --guidance <float>      CFG guidance scale
#   --seed <int>            Random seed

set -ex

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DREAMGEN_ROOT="${PROJECT_ROOT}/models/dreamgen"

export PYTHONPATH="${DREAMGEN_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_MODULE_LOADING=LAZY
export LD_PRELOAD=""

source "${DREAMGEN_ROOT}/.venv/bin/activate"

cd "${DREAMGEN_ROOT}"

python examples/video2world.py \
    --model_size 2B \
    --disable_guardrail \
    --disable_prompt_refiner \
    "$@"
