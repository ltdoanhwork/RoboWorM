# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from copy import deepcopy

import attrs

from cosmos_predict2.conditioner import ReMapkey, TextAttr, TextConditioner
from cosmos_predict2.configs.base.config_pisa import build_pisa_parameters
from cosmos_predict2.configs.base.config_radial import build_radial_parameters
from cosmos_predict2.configs.base.config_svg import build_svg_parameters
from cosmos_predict2.configs.base.defaults.ema import EMAConfig
from cosmos_predict2.models.text2image_dit import MiniTrainDIT
from cosmos_predict2.tokenizers_local.tokenizer import CosmosImageTokenizer, TokenizerInterface
from imaginaire.auxiliary.text_encoder import (
    CosmosTextEncoderConfig,
)
from imaginaire.config import make_freezable
from imaginaire.constants import (
    CHECKPOINTS_DIR,
    CosmosPredict2Video2WorldModelSize,
    get_cosmos_predict2_text2image_tokenizer,
)
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict


@make_freezable
@attrs.define(slots=False)
class SolverTimestampConfig:
    nfe: int = 35
    t_min: float = 0.002
    t_max: float = 80.0
    order: float = 7.0
    is_forward: bool = False  # whether generate forward or backward timestamps


@make_freezable
@attrs.define(slots=False)
class CosmosGuardrailConfig:
    checkpoint_dir: str = CHECKPOINTS_DIR
    offload_model_to_cpu: bool = True
    enabled: bool = True


@make_freezable
@attrs.define(slots=False)
class Text2ImagePipelineConfig:
    adjust_video_noise: bool
    conditioner: LazyDict[TextConditioner]
    net: LazyDict[MiniTrainDIT]
    tokenizer: LazyDict[TokenizerInterface]
    guardrail_config: CosmosGuardrailConfig
    precision: str
    rectified_flow_t_scaling_factor: float
    rectified_flow_loss_weight_uniform: bool
    resize_online: bool
    resolution: str
    ema: EMAConfig
    sigma_data: float = 1.0
    state_ch: int = 16
    state_t: int = 24
    text_encoder: CosmosTextEncoderConfig = attrs.field(factory=CosmosTextEncoderConfig)
    input_video_key: str = "video"
    input_image_key: str = "images"
    timestamps: SolverTimestampConfig = attrs.field(factory=SolverTimestampConfig)


# Cosmos Predict2 Text2Image 0.6B
_PREDICT2_TEXT2IMAGE_NET_0P6B = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=1280,
    num_blocks=20,
    num_heads=20,
    mlp_ratio=4.0,
    # cross attention settings
    crossattn_emb_channels=1024,
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

_PREDICT2_TEXT2IMAGE_PIPELINE_0P6B = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(TextConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=_PREDICT2_TEXT2IMAGE_NET_0P6B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="1024",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_text2image_tokenizer(model_size="0.6B"),
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Config for using fast tokenizer
_PREDICT2_TEXT2IMAGE_PIPELINE_0P6B_FAST_TOKENIZER = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(TextConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=_PREDICT2_TEXT2IMAGE_NET_0P6B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="1024",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    tokenizer=L(CosmosImageTokenizer)(
        name="tokenizer",
        vae_pth=get_cosmos_predict2_text2image_tokenizer(model_size="0.6B", fast_tokenizer=True),
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Cosmos Predict2 Text2Image 2B
_PREDICT2_TEXT2IMAGE_NET_2B = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=2048,
    num_blocks=28,
    num_heads=16,
    mlp_ratio=4.0,
    atten_backend="minimal_a2a",
    # cross attention settings
    crossattn_emb_channels=1024,
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

_PREDICT2_TEXT2IMAGE_PIPELINE_2B = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(TextConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=_PREDICT2_TEXT2IMAGE_NET_2B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="1024",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_text2image_tokenizer(model_size="2B"),
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=True,
    ),
)

# Cosmos Predict2 Text2Image 14B
_PREDICT2_TEXT2IMAGE_NET_14B = L(MiniTrainDIT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    concat_padding_mask=True,
    # attention settings
    model_channels=5120,
    num_blocks=36,
    num_heads=40,
    mlp_ratio=4.0,
    # cross attention settings
    crossattn_emb_channels=1024,
    # positional embedding settings
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    min_fps=1,
    max_fps=30,
    use_adaln_lora=True,
    adaln_lora_dim=256,
    rope_h_extrapolation_ratio=4.0,
    rope_w_extrapolation_ratio=4.0,
    rope_t_extrapolation_ratio=1.0,
    extra_per_block_abs_pos_emb=False,
    extra_h_extrapolation_ratio=1.0,
    extra_w_extrapolation_ratio=1.0,
    extra_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False,
)

_PREDICT2_TEXT2IMAGE_PIPELINE_14B = Text2ImagePipelineConfig(
    adjust_video_noise=True,
    conditioner=L(TextConditioner)(
        fps=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="fps",
            output_key="fps",
        ),
        padding_mask=L(ReMapkey)(
            dropout_rate=0.0,
            dtype=None,
            input_key="padding_mask",
            output_key="padding_mask",
        ),
        text=L(TextAttr)(
            dropout_rate=0.2,
            input_key=["t5_text_embeddings"],
        ),
    ),
    net=_PREDICT2_TEXT2IMAGE_NET_14B,
    precision="bfloat16",
    rectified_flow_t_scaling_factor=1.0,
    rectified_flow_loss_weight_uniform=True,
    resize_online=True,
    resolution="1024",
    ema=L(EMAConfig)(enabled=False),  # defaults to inference
    sigma_data=1.0,
    state_ch=16,
    state_t=24,
    tokenizer=L(TokenizerInterface)(
        chunk_duration=81,
        load_mean_std=False,
        name="tokenizer",
        vae_pth=get_cosmos_predict2_text2image_tokenizer(model_size="14B"),
    ),
    guardrail_config=CosmosGuardrailConfig(
        checkpoint_dir=CHECKPOINTS_DIR,
        offload_model_to_cpu=True,
        enabled=True,
    ),
)


@dataclasses.dataclass(frozen=True)
class _Text2ImagePipelineConfig:
    model_size: CosmosPredict2Video2WorldModelSize
    fast_tokenizer: bool = dataclasses.field(default=False, kw_only=True)


_PREDICT2_TEXT2IMAGE_PIPELINES: dict[_Text2ImagePipelineConfig, Text2ImagePipelineConfig] = {
    _Text2ImagePipelineConfig("0.6B"): _PREDICT2_TEXT2IMAGE_PIPELINE_0P6B,
    _Text2ImagePipelineConfig("2B"): _PREDICT2_TEXT2IMAGE_PIPELINE_2B,
    _Text2ImagePipelineConfig("14B"): _PREDICT2_TEXT2IMAGE_PIPELINE_14B,
    _Text2ImagePipelineConfig("0.6B", fast_tokenizer=True): _PREDICT2_TEXT2IMAGE_PIPELINE_0P6B_FAST_TOKENIZER,
}


def get_cosmos_predict2_text2image_pipeline(
    *,
    model_size: CosmosPredict2Video2WorldModelSize,
    fast_tokenizer: bool = False,
    pisa: bool = False,
    pisa_density: float = 0.5,
    pisa_block_size: int = 64,
    pisa_start_layer_idx: int = 0,
    pisa_use_bias: bool = False,
    radial: bool = False,
    radial_decay_factor: float = 1.0,
    radial_block_size: int = 64,
    radial_start_layer_idx: int = 4,
    radial_model_type: str = "hunyuan",
    svg: bool = False,
    svg_variant: str = "svg1",
    svg_start_layer_idx: int = 4,
    svg_dense_step_frac: float = 0.1,
    svg_sparsity: float = 0.25,
    svg_num_sampled_rows: int = 64,
    svg_sample_mse_max_row: int = 3000,
    svg_num_q_centroids: int = 50,
    svg_num_k_centroids: int = 200,
    svg_top_p_kmeans: float = 0.9,
    svg_min_kc_ratio: float = 0.0,
    svg_kmeans_iter_init: int = 5,
    svg_kmeans_iter_step: int = 2,
) -> Text2ImagePipelineConfig:
    if sum(bool(enabled) for enabled in (pisa, radial, svg)) > 1:
        raise ValueError("PISA, Radial, and SVG cannot be enabled at the same time.")
    key = _Text2ImagePipelineConfig(model_size, fast_tokenizer=fast_tokenizer)
    config = deepcopy(_PREDICT2_TEXT2IMAGE_PIPELINES[key])
    if pisa:
        config.net.pisa_parameters = build_pisa_parameters(
            config.net.num_blocks,
            density=pisa_density,
            block_size=pisa_block_size,
            start_layer_idx=pisa_start_layer_idx,
            use_bias=pisa_use_bias,
        )
    if radial:
        config.net.radial_parameters = build_radial_parameters(
            config.net.num_blocks,
            decay_factor=radial_decay_factor,
            block_size=radial_block_size,
            start_layer_idx=radial_start_layer_idx,
            model_type=radial_model_type,
        )
    if svg:
        config.net.svg_parameters = build_svg_parameters(
            config.net.num_blocks,
            variant=svg_variant,
            start_layer_idx=svg_start_layer_idx,
            dense_step_frac=svg_dense_step_frac,
            sparsity=svg_sparsity,
            num_sampled_rows=svg_num_sampled_rows,
            sample_mse_max_row=svg_sample_mse_max_row,
            num_q_centroids=svg_num_q_centroids,
            num_k_centroids=svg_num_k_centroids,
            top_p_kmeans=svg_top_p_kmeans,
            min_kc_ratio=svg_min_kc_ratio,
            kmeans_iter_init=svg_kmeans_iter_init,
            kmeans_iter_step=svg_kmeans_iter_step,
        )
    return config
