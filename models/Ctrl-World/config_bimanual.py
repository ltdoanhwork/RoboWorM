"""
Ctrl-World training config for datasets/bimanual/multiview (ALOHA).

Mirrors config.wm_args but with the bimanual-specific settings:
  - action_dim = 14 (observation.state qpos: 6 joints + gripper per arm)
  - down_sample = 1 (state arrays are frame-aligned with latents at prep time;
    30fps kept natively for this experiment)
  - dataset_names / dataset_cfgs = bimanual_multiview
"""

import torch
from dataclasses import dataclass


@dataclass
class wm_args_bimanual:
    # model paths (overridden by CLI / launch script)
    svd_model_path = "./checkpoints/base/stabilityai/stable-video-diffusion-img2vid"
    clip_model_path = "./checkpoints/base/openai/clip-vit-base-patch32"
    # warm-start from DROID Ctrl-World checkpoint (loaded strict=False in train_wm.py)
    ckpt_path = "./checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt"

    # dataset parameters
    dataset_root_path = "dataset_example"
    dataset_names = "bimanual_multiview"
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"
    num_workers = 4
    # state arrays are already frame-aligned with latent frames -> no extra mapping
    down_sample = 1
    skip_step = 1

    # logs
    debug = False
    tag = "bimanual_multiview"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "bimanual_multiview"

    # training
    learning_rate = 1e-5
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    train_batch_size = 2
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 100000
    checkpointing_steps = 10000
    validation_steps = 2500
    max_grad_norm = 1.0
    video_num = 4

    # model params
    motion_bucket_id = 127
    fps = 7  # micro-conditioning scalar (not real fps)
    guidance_scale = 1.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    num_frames = 5
    num_history = 6
    action_dim = 14  # bimanual qpos
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16

    # unused rollout/policy fields (kept for API parity with training-time code)
    data_stat_path = None
    val_model_path = ckpt_path
    history_idx = [0, 0, -12, -9, -6, -3]
    save_dir = "synthetic_traj"

    def __post_init__(self):
        if self.dataset_meta_info_path and self.dataset_names:
            name_split = self.dataset_names.split("+")[0].split("/")[-1]
            self.data_stat_path = f"{self.dataset_meta_info_path}/{name_split}/stat.json"
