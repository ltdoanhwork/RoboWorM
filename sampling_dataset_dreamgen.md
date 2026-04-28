# Create batch input for dreamgen
## Dataset: PhysicalAI-Robotics-GR00T-GR1/gr1

### Dense 
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/gr1 \
  --metadata_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/metadata.csv \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense/batch_input.json

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

### pisa
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/gr1 \
  --metadata_path ./datasets/PhysicalAI-Robotics-GR00T-GR1/metadata.csv \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/pisa \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/pisa/batch_input.json


torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-GR1/gr1/dense/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_pisa \
  --pisa_density 0.2 \
  --pisa_block_size 64 \
  --pisa_start_layer_idx 2

## Dataset: PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval

### Dense 
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval \
  --dataset_type lerobot \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/dense \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/dense/batch_input.json

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t_lvg \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/dense/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --num_chunks 4

## Dataset: PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval

### Dense 
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval \
  --dataset_type lerobot \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/dense \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/dense/batch_input.json

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/dense/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}

## Dataset: PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval

### Dense 
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json_csv \
  --dataset_path ./datasets/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval \
  --dataset_type lerobot \
  --save_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/dense \
  --output_path ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/dense/batch_input.json

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/models/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./sampling_dataset/PhysicalAI-Robotics-GR00T-Teleop-GR1/In-lab_Eval/dense/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}