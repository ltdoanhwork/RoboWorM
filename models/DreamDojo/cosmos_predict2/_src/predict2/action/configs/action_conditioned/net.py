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

import copy

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.predict2.action.networks.action_conditioned_minimal_v1_lvg_dit import (
    ActionChunkConditionedMinimalV1LVGDiT,
    ActionConditionedMinimalV1LVGDiT,
)

# from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT
from cosmos_predict2._src.predict2.networks.minimal_v4_dit import SACConfig

COSMOS_V1_7B_NET_MININET: LazyDict = L(ActionConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    self_attention_backend="minimal_a2a",
    cross_attention_backend="minimal_a2a",
    svg_parameters=None,
    sito_parameters=None,
    use_sito=False,
    sito_start_layer_idx=None,
    sito_keep_last_n_dense=2,
    sito_prune_ratio=None,
    sito_patch_h=2,
    sito_patch_w=2,
    sito_noise_alpha=0.1,
    sito_sim_beta=1.0,
    itm_parameters=None,
    use_itm=False,
    itm_start_layer_idx=None,
    itm_keep_last_n_dense=2,
    itm_compress_ratio=None,
    itm_prune_from_step=1,
    itm_merge_from_step=2,
    itm_merge_attn=True,
    itm_merge_crossattn=False,
    itm_merge_mlp=False,
    svg_variant=None,
    svg_start_layer_idx=None,
    svg_dense_step_frac=None,
    svg_sparsity=None,
    svg_num_sampled_rows=None,
    svg_sample_mse_max_row=None,
    svg_num_q_centroids=None,
    svg_num_k_centroids=None,
    svg_top_p_kmeans=None,
    svg_min_kc_ratio=None,
    svg_kmeans_iter_init=None,
    svg_kmeans_iter_step=None,
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_2B_NET_MININET.model_channels = 2048
COSMOS_V1_2B_NET_MININET.num_heads = 16
COSMOS_V1_2B_NET_MININET.num_blocks = 28
COSMOS_V1_2B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
COSMOS_V1_14B_NET_MININET.model_channels = 5120
COSMOS_V1_14B_NET_MININET.num_heads = 40
COSMOS_V1_14B_NET_MININET.num_blocks = 36
COSMOS_V1_14B_NET_MININET.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET.rope_t_extrapolation_ratio = 1.0

mini_net = copy.deepcopy(COSMOS_V1_7B_NET_MININET)
mini_net.model_channels = 1024
mini_net.num_heads = 8
mini_net.num_blocks = 2
mini_net.rope_t_extrapolation_ratio = 1.0


COSMOS_V1_7B_NET_MININET_ACTION_CHUNK: LazyDict = L(ActionChunkConditionedMinimalV1LVGDiT)(
    max_img_h=240,
    max_img_w=240,
    max_frames=128,
    in_channels=16,
    out_channels=16,
    patch_spatial=2,
    patch_temporal=1,
    model_channels=4096,
    num_blocks=28,
    num_heads=32,
    concat_padding_mask=True,
    pos_emb_cls="rope3d",
    pos_emb_learnable=True,
    pos_emb_interpolation="crop",
    use_adaln_lora=True,
    adaln_lora_dim=256,
    atten_backend="minimal_a2a",
    self_attention_backend="minimal_a2a",
    cross_attention_backend="minimal_a2a",
    svg_parameters=None,
    sito_parameters=None,
    use_sito=False,
    sito_start_layer_idx=None,
    sito_keep_last_n_dense=2,
    sito_prune_ratio=None,
    sito_patch_h=2,
    sito_patch_w=2,
    sito_noise_alpha=0.1,
    sito_sim_beta=1.0,
    itm_parameters=None,
    use_itm=False,
    itm_start_layer_idx=None,
    itm_keep_last_n_dense=2,
    itm_compress_ratio=None,
    itm_prune_from_step=1,
    itm_merge_from_step=2,
    itm_merge_attn=True,
    itm_merge_crossattn=False,
    itm_merge_mlp=False,
    svg_variant=None,
    svg_start_layer_idx=None,
    svg_dense_step_frac=None,
    svg_sparsity=None,
    svg_num_sampled_rows=None,
    svg_sample_mse_max_row=None,
    svg_num_q_centroids=None,
    svg_num_k_centroids=None,
    svg_top_p_kmeans=None,
    svg_min_kc_ratio=None,
    svg_kmeans_iter_init=None,
    svg_kmeans_iter_step=None,
    extra_per_block_abs_pos_emb=True,
    rope_h_extrapolation_ratio=1.0,
    rope_w_extrapolation_ratio=1.0,
    rope_t_extrapolation_ratio=2.0,
    sac_config=SACConfig(),
)
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK = copy.deepcopy(COSMOS_V1_7B_NET_MININET_ACTION_CHUNK)
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.model_channels = 2048
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.num_heads = 16
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.num_blocks = 28
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.extra_per_block_abs_pos_emb = False
COSMOS_V1_2B_NET_MININET_ACTION_CHUNK.rope_t_extrapolation_ratio = 1.0

COSMOS_V1_14B_NET_MININET_ACTION_CHUNK = copy.deepcopy(COSMOS_V1_7B_NET_MININET_ACTION_CHUNK)
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.model_channels = 5120
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.num_heads = 40
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.num_blocks = 36
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.extra_per_block_abs_pos_emb = False
COSMOS_V1_14B_NET_MININET_ACTION_CHUNK.rope_t_extrapolation_ratio = 1.0


def register_net():
    cs = ConfigStore.instance()
    cs.store(group="net", package="model.config.net", name="mini_net", node=mini_net)
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_2B_action_conditioned", node=COSMOS_V1_2B_NET_MININET
    )
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_7B_action_conditioned", node=COSMOS_V1_7B_NET_MININET
    )
    cs.store(
        group="net", package="model.config.net", name="cosmos_v1_14B_action_conditioned", node=COSMOS_V1_14B_NET_MININET
    )

    # action chunk conditioned
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_2B_action_chunk_conditioned",
        node=COSMOS_V1_2B_NET_MININET_ACTION_CHUNK,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_7B_action_chunk_conditioned",
        node=COSMOS_V1_7B_NET_MININET_ACTION_CHUNK,
    )
    cs.store(
        group="net",
        package="model.config.net",
        name="cosmos_v1_14B_action_chunk_conditioned",
        node=COSMOS_V1_14B_NET_MININET_ACTION_CHUNK,
    )
