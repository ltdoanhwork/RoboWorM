# Create batch input for dreamgen
## Dataset: PhysicalAI-Robotics-GR00T-GR1/gr1

export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/gr1 \
  --metadata_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/metadata.csv \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense/batch_input.json

### Dense 

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}

### Sparse 