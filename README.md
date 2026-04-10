
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


### Default
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

### Use with Pisa
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_pisa --pisa_density 0.5 --pisa_block_size 32

### Use-svg v1
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_svg --svg_pattern SVG1 --svg_sparsity 0.25

### Use-svg v2
export VENV=/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv
export LD_LIBRARY_PATH=$VENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$VENV/lib/python3.10/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_svg --svg_pattern SVG2 --svg_num_q_centroids 50 --svg_num_k_centroids 200


### use with radial attention (padding)
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial \
  --radial_pad_small_layers 


### use with radial attention (no padding)

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial


### use with radial attention (no padding)

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/Ctrl-World"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python models/Ctrl-World/scripts/rollout_interact_pi.py \
  --dataset_root_path ./models/Ctrl-World/dataset_example \
  --dataset_meta_info_path ./models/Ctrl-World/dataset_meta_info \
  --dataset_names ./models/Ctrl-World/droid_subset \
  --svd_model_path ./checkpoints/stabilityai/stable-video-diffusion-img2vid \
  --clip_model_path ./checkpoints/openai/clip-vit-base-patch32 \
  --ckpt_path ./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt \
  --task_type pickplace \
  --use_radial \
  --radial_decay_factor 0.5 \
  --radial_model_type hunyuan \
  --radial_first_layers_fp 0
## dreamgen base 
source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
python -m scripts.prepare_batch_input_json \
  --dataset_path ./datasets/dream_gen_benchmark/gr1_env \
  --save_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env \
  --output_path ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env/batch_input.json

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

### dreamgen - pisa
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/methods/sparse_attention/piecewise-sparse-attention
uv pip install -e .

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0 
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_env/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_pisa \
  --pisa_density 0.2 \
  --pisa_block_size 64 \
  --pisa_start_layer_idx 2

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0,1
export NUM_GPUS=2
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_pisa \
  --pisa_density 0.1 \
  --pisa_block_size 64 \
  --pisa_use_bias

### dreamgen - radial
Radial is not supported in the default `DreamDojo/.venv`. It requires a separate env with `flashinfer`.

cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
bash ./scripts/bootstrap_radial_env.sh

source /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/DreamDojo/.venv/bin/activate
export PYTHONPATH="./models/dreamgen"
export CUDA_VISIBLE_DEVICES=0
export NUM_GPUS=1
torchrun -m models.dreamgen.examples.video2world_gr00t \
  --model_size 14B \
  --gr00t_variant gr1 \
  --batch_input_json ./output/dream_gen_benchmark/cosmos_predict2_14b_gr1_behavior/batch_input.json \
  --disable_guardrail \
  --num_gpus ${NUM_GPUS} \
  --use_radial \
  --radial_decay_factor 1.0 \
  --radial_block_size 64 \
  --radial_start_layer_idx 4 \
  --radial_model_type hunyuan




