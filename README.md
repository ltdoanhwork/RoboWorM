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