from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


AIRBOT_DATASETS = (
    "Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate",
    "Airbot_MMK2_doodled_line",
    "Airbot_MMK2_move_pan",
    "Airbot_MMK2_storage_bottle_part",
    "AIRBOT_MMK2_boxs_storage",
    "AIRBOT_MMK2_mobile_car",
)

AIRBOT_GROUP_A = (
    "observation.images.cam_front_rgb",
    "observation.images.cam_head_rgb",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)

AIRBOT_GROUP_B = (
    "observation.images.cam_high_rgb",
    "observation.images.cam_third_view",
    "observation.images.cam_left_wrist_rgb",
    "observation.images.cam_right_wrist_rgb",
)

LEJU_GROUP = (
    "observation.images.camera_head_rgb",
    "observation.images.camera_left_wrist_rgb",
    "observation.images.camera_right_wrist_rgb",
    None,
)


@dataclass
class FinetuneConfig:
    # Data paths.
    dataset_root: str = "datasets/humanoid/multiview/non_makovian"
    processed_root: str = "datasets/processed/ctrl_world_robo_multiview"
    manifest_path: str = "datasets/processed/ctrl_world_robo_multiview/manifest.json"
    include_pattern: str = "Airbot_MMK2*,AIRBOT_MMK2*"
    exclude_pattern: str = ""
    max_episodes_per_dataset: int | None = None
    val_ratio: float = 0.1

    # Source and processed data schema.
    action_key: str = "eef_sim_pose_action"
    state_key: str = "eef_sim_pose_state"
    action_dim: int = 12
    n_views: int = 4
    source_fps: int = 30
    fps: int = 5
    downsample_stride: int = 6
    image_height: int = 192
    image_width: int = 320
    num_history: int = 6
    num_frames: int = 5
    start_frame_interval: int = 1
    skip_existing: bool = True

    # Model paths.
    svd_model_path: str = "checkpoints/stabilityai/stable-video-diffusion-img2vid"
    clip_model_path: str = "checkpoints/openai/clip-vit-base-patch32"
    init_ckpt_path: str | None = None
    ckpt_path: str | None = None

    # Training.
    output_dir: str = "datasets/processed/ctrl_world_robo_multiview/checkpoints"
    train_batch_size: int = 1
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    max_train_steps: int = 100000
    checkpointing_steps: int = 5000
    validation_steps: int = 1000
    log_steps: int = 100
    max_grad_norm: float = 1.0
    num_workers: int = 4
    shuffle: bool = True
    seed: int = 42

    # SVD/Ctrl-World generation parameters.
    motion_bucket_id: int = 127
    guidance_scale: float = 1.0
    num_inference_steps: int = 30
    decode_chunk_size: int = 8
    text_cond: bool = True
    frame_level_cond: bool = True
    his_cond_zero: bool = False
    use_embodiment_embedding: bool = True
    num_embodiments: int = 16
    dtype: str = "bfloat16"

    @property
    def sequence_length(self) -> int:
        return self.num_history + self.num_frames

    @property
    def latent_slot_height(self) -> int:
        return self.image_height // 8

    @property
    def latent_slot_width(self) -> int:
        return self.image_width // 8

    @property
    def latent_height(self) -> int:
        return self.n_views * self.latent_slot_height

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        return torch.float32

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sequence_length"] = self.sequence_length
        data["latent_slot_height"] = self.latent_slot_height
        data["latent_slot_width"] = self.latent_slot_width
        data["latent_height"] = self.latent_height
        return data


def add_config_arguments(parser, defaults: FinetuneConfig | None = None):
    cfg = defaults or FinetuneConfig()
    field_map = {f.name: f for f in fields(cfg)}
    for name, f in field_map.items():
        default = getattr(cfg, name)
        arg = f"--{name.replace('_', '-')}"
        if isinstance(default, bool):
            parser.add_argument(arg, default=default, action=argparse.BooleanOptionalAction)
        else:
            parser.add_argument(arg, default=default, type=_arg_type(name, default), required=False)
    return parser


def config_from_args(args) -> FinetuneConfig:
    base = FinetuneConfig()
    values = {}
    for f in fields(base):
        if hasattr(args, f.name):
            values[f.name] = getattr(args, f.name)
    return FinetuneConfig(**values)


def _arg_type(name, default):
    if name in {"max_episodes_per_dataset"}:
        return int
    if default is None:
        return str
    if isinstance(default, int):
        return int
    if isinstance(default, float):
        return float
    return str
