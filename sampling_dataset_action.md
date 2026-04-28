# Action-Conditioned Video Generation Scripts

This document contains instructions for running video generation conditioned on robot action sequences.

## 1. Prepare Action Data
Use the extraction script to convert LeRobot `.parquet` files to the JSON format expected by the model.

### EgoDex_Eval
```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
python scripts/extract_actions_for_eval.py \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/actions \
  --output_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input_action.json
```

## 2. Run Inference
Run the action-conditioned model (2B) using the prepared batch input.

### EgoDex_Eval
```bash
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1

torchrun --nproc_per_node=${NUM_GPUS} -m models.dreamgen.examples.video2world_action \
  --model_size 2B \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input_action.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}
```

> [!NOTE]
> The action-conditioned model currently supports the 2B variant. The action space is mapped to 7 dimensions (6D EE pose + 1D gripper) extracted from the GR1 dataset.
