"""
Ctrl-World training config for the humanoid/multiview 4-VIEW 2x2 GRID variant.

Mirrors config_humanoid.wm_args_humanoid but:
  - num_views = 4 arranged in a 2x2 grid (grid_rows=2, grid_cols=2)
        [ TL=main external   TR=second external ]  (static, top row)
        [ BL=left wrist      BR=right wrist     ]  (dynamic, bottom row)
    -> latent canvas 48x80 (2*24 x 2*40), vs the 3-view strip's 72x40.
  - dataset_names / dataset_cfgs = humanoid_multiview_4view_grid
  - action_dim = 36 (full observation.state), as in the 3-view humanoid config.

Reuses the dimension-agnostic bimanual training dataset class (reads
observation.state.qpos of any dim) which is also grid-aware via num_views/grid_*.

The SVD UNet is view-agnostic (sees one larger image), so the 3-view humanoid
checkpoint warm-starts fine (train_wm.py loads strict=False for this config).
"""

import torch
from dataclasses import dataclass


@dataclass
class wm_args_humanoid_grid:
    svd_model_path = "./checkpoints/stabilityai/stable-video-diffusion-img2vid"
    clip_model_path = "./checkpoints/openai/clip-vit-base-patch32"
    ckpt_path = "./checkpoints/ctrl_world/Ctrl-World/checkpoint-10000.pt"

    dataset_root_path = "dataset_example"
    dataset_names = "humanoid_multiview_4view_grid"
    dataset_meta_info_path = "dataset_meta_info"
    dataset_cfgs = dataset_names
    prob = [1.0]
    annotation_name = "annotation"
    num_workers = 4
    down_sample = 1
    skip_step = 1

    # ---- 4-view 2x2 grid layout (read by dataset / train / infer) ----
    num_views = 4
    grid_rows = 2
    grid_cols = 2

    debug = False
    tag = "humanoid_multiview_4view_grid"
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "humanoid_multiview_4view_grid"

    learning_rate = 1e-5
    gradient_accumulation_steps = 1
    mixed_precision = "fp16"
    train_batch_size = 2
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 200000
    checkpointing_steps = 10000
    validation_steps = 2500
    max_grad_norm = 1.0
    video_num = 4

    motion_bucket_id = 127
    fps = 7
    guidance_scale = 1.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    num_frames = 5
    num_history = 6
    action_dim = 36  # humanoid full state
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16

    data_stat_path = None
    val_model_path = ckpt_path
    history_idx = [0, 0, -12, -9, -6, -3]
    save_dir = "synthetic_traj"

    def __post_init__(self):
        if self.dataset_meta_info_path and self.dataset_names:
            name_split = self.dataset_names.split("+")[0].split("/")[-1]
            self.data_stat_path = f"{self.dataset_meta_info_path}/{name_split}/stat.json"
