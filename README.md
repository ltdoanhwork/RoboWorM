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


CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 python scripts/rollout_interact_pi.py  --dataset_root_path dataset_example --dataset_meta_info_path dataset_meta_info --dataset_names droid_subset --svd_model_path ${path to svd folder} --clip_model_path /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics/checkpoints/openai/clip-vit-base-patch32 --ckpt_path ${path to ctrl-world ckpt} --pi_ckpt ${path to ctrl-world ckpt} --task_type ${pickplace}