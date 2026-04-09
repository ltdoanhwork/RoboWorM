
# video_gen_physics


## Ctrl world

export CUDA_VISIBLE_DEVICES=0 
export PYTHONPATH="./models/Ctrl-World" 
python models/Ctrl-World/scripts/rollout_replay_traj.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt



CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/openai/clip-vit-base-patch32 --ckpt_path ${path to ctrl-world ckpt} --task_type keyboard --keyboard lllrrr

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace

## dreamgen base 
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json \
  --dataset_path ./datasets/dream_gen_benchmark/gr1_behavior \
  --save_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior \
  --output_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json

### dreamgen base - one gpu
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}

### dreamgen base - 4 gpu
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0,1,2,3 
export NUM_GPUS=4
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS}