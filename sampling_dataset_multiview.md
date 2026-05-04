# Run Multi-view Video Generation in DreamGen

This document provides instructions and examples on how to run multi-view video generation using the Cosmos-Predict2-Multiview model from the **root directory** (without `cd`).

## Environment Setup

To run DreamGen scripts from the root directory, you must add `models/dreamgen` to your `PYTHONPATH` so that internal imports like `from examples...` can be resolved:

```bash
# From the root: /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen
```

## Running the Multi-view Script

The multi-view pipeline generates **7 views** with **29 frames** at **720p**.

Here is the command to run multi-view inference from the root directory:

```bash
torchrun --nproc_per_node=1 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot pick the lid and put it on the pot The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lid and put it on the pot" \
  --input_path models/dreamgen/assets/sample_gr00t_dreams_droid/episode_000408.png \
  --prompt_prefix "" \
  --num_gpus 1 \
  --disable_guardrail \
  --save_path output/generated_video_droid.mp4

```

### Parameter Explanations
- `--model_size 2B`: Uses the 2-Billion parameter multi-view checkpoint.
- `--input_path`: Path to your initial image (relative to the root).
- `--num_conditional_frames 1`: Uses only the first frame (the image) as the condition.
- `--n_views 7`: Generates 7 different camera views.
- `--save_path`: The output location for the generated `mp4` video.

### Multi-GPU Support
To run with Context Parallelism using multiple GPUs:

```bash
torchrun --nproc_per_node=2 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot pick the lid and put it on the pot The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lid and put it on the pot" \
  --input_path models/dreamgen/assets/sample_gr00t_dreams_droid/episode_000408.png \
  --prompt_prefix "" \
  --num_gpus 2 \
  --disable_guardrail \
  --save_path output/generated_video_droid.mp4
  
```
# Đảm bảo đang ở thư mục root: /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics

source models/DreamDojo/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)/models/dreamgen

torchrun --nproc_per_node=2 --master_port=12341 \
  -m examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant droid \
  --prompt "A multi-view video shows that a robot picks up a pomegranate with its left hand and puts it in the storage box, then picks up a mango with its right hand and puts it in the storage box. The video is split into four views: top-left (front view), top-right (head view), bottom-left (wrist view), and bottom-right (black screen)." \
  --input_path models/dreamgen/airbot_multiview_input.png \
  --prompt_prefix "" \
  --num_gpus 2 \
  --disable_guardrail \
  --save_path output/generated_video_airbot_droid.mp4
