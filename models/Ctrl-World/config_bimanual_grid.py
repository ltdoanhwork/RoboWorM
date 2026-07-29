"""
Ctrl-World training config for the bimanual/multiview 4-VIEW 2x2 GRID variant.

Mirrors config_bimanual.wm_args_bimanual but:
  - num_views = 4 arranged in a 2x2 grid (grid_rows=2, grid_cols=2)
        [ TL=cam_high        TR=cam_low         ]  (static, top row)
        [ BL=cam_left_wrist  BR=cam_right_wrist ]  (dynamic, bottom row)
    -> latent canvas 48x80 (2*24 x 2*40), vs the 3-view strip's 72x40.
  - dataset_names / dataset_cfgs = bimanual_multiview_4view_grid
  - down_sample = 1 internally (state frame-aligned at prep time; prep ran at 5fps).

The SVD UNet is view-agnostic (sees one larger image), so the checkpoint from the
3-view model warm-starts fine (train_wm.py loads strict=False for this config).
"""

import torch
from dataclasses import dataclass


@dataclass
class wm_args_bimanual_grid:
    # model paths (overridden by CLI / launch script)
    svd_model_path = "./checkpoints/stabilityai/stable-video-diffusion-img2vid"
    clip_model_path = "./checkpoints/openai/clip-vit-base-patch32"
    ckpt_path = "./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"

    # dataset parameters
    dataset_root_path = "dataset_example"
    dataset_names = "bimanual_multiview_4view_grid"
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"
    num_workers = 4
    # state arrays are already frame-aligned with latent frames -> no extra mapping
    down_sample = 1
    skip_step = 1

    # ---- 4-view 2x2 grid layout (read by dataset / train / infer) ----
    num_views = 4
    grid_rows = 2
    grid_cols = 2

    # logs
    debug = False
    tag = "bimanual_multiview_4view_grid"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "bimanual_multiview_4view_grid"

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
