from __future__ import annotations

from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

cosmos_cuda = types.ModuleType("cosmos_cuda")
cosmos_cuda.__version__ = "1.4.1"
sys.modules.setdefault("cosmos_cuda", cosmos_cuda)

import pytest
import torch
import torch.nn as nn

from methods.cache_strategy.DiCache.runtime import apply_dicache
from methods.cache_strategy.WorldCache.runtime import apply_worldcache
from methods.cache_strategy.common import (
    DiCacheConfig,
    WorldCacheConfig,
    reset_cache_runtime,
)


class MockTEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, timesteps: torch.Tensor):
        emb = timesteps.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_size)
        adaln = timesteps.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_size * 3)
        return emb, adaln


class SumActionEmbed(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=-1, keepdim=True).repeat(1, 1, self.out_dim)


class MockBlock(nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb,
        rope_emb_L_1_1_D=None,
        adaln_lora_B_T_3D=None,
        extra_per_block_pos_emb=None,
        **kwargs,
    ) -> torch.Tensor:
        del crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, extra_per_block_pos_emb, kwargs
        return x_B_T_H_W_D + emb_B_T_D[:, :, None, None, :] + self.bias


class MockFinalLayer(nn.Module):
    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        adaln_lora_B_T_3D=None,
    ) -> torch.Tensor:
        del adaln_lora_B_T_3D
        return x_B_T_H_W_D + emb_B_T_D[:, :, None, None, :]


class MockCacheModel(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.use_crossattn_projection = False
        self.extra_image_context_dim = None
        self.use_wan_fp32_strategy = False
        self.t_embedder = MockTEmbedder(hidden_size)
        self.t_embedding_norm = nn.Identity()
        self.action_embedder_B_D = SumActionEmbed(hidden_size)
        self.action_embedder_B_3D = SumActionEmbed(hidden_size * 3)
        self._num_action_per_latent_frame = 2
        self.blocks = nn.ModuleList([MockBlock(0.1), MockBlock(0.2), MockBlock(0.3)])
        self.final_layer = MockFinalLayer()
        self.self_attention_backend = "minimal_a2a"
        self.cross_attention_backend = "minimal_a2a"
        self.has_svg_backend = False
        self.has_sito_backend = False
        self.has_itm_backend = False

    def prepare_embedded_sequence(self, x_B_C_T_H_W: torch.Tensor, fps=None, padding_mask=None):
        del fps, padding_mask
        return x_B_C_T_H_W.permute(0, 2, 3, 4, 1).to(torch.float32), None, None

    def unpatchify(self, x_B_T_H_W_D: torch.Tensor) -> torch.Tensor:
        return x_B_T_H_W_D.permute(0, 4, 1, 2, 3)


def _build_inputs():
    x = torch.ones(1, 4, 3, 1, 1)
    timesteps = torch.ones(1, 3)
    crossattn = torch.zeros(1, 1, 4)
    return x, timesteps, crossattn


@pytest.mark.parametrize(
    ("apply_fn", "config"),
    [
        (apply_worldcache, WorldCacheConfig(num_steps=3, rel_l1_thresh=1e6, ret_ratio=0.0, probe_depth=1)),
        (apply_dicache, DiCacheConfig(num_steps=3, rel_l1_thresh=1e6, ret_ratio=0.0, probe_depth=1)),
    ],
)
def test_cache_runtime_consumes_action_in_compute_and_cache_paths(apply_fn, config):
    model = MockCacheModel()
    apply_fn(model, config)
    x, timesteps, crossattn = _build_inputs()

    action_zero = torch.zeros(1, 4, 2)
    action_one = torch.ones(1, 4, 2)

    out_first = model(x, timesteps, crossattn, action=action_zero)
    _ = model(x, timesteps, crossattn, action=action_zero)
    out_third = model(x, timesteps, crossattn, action=action_one)

    assert out_first.shape == x.shape
    assert out_third.shape == x.shape
    assert not torch.allclose(out_first, out_third)


def test_worldcache_reset_logic_uses_cfg_mode():
    sequential_model = MockCacheModel()
    sequential_config = WorldCacheConfig(num_steps=35, parallel_cfg=False)
    apply_worldcache(sequential_model, sequential_config)
    target_steps = reset_cache_runtime(sequential_model, sequential_config, num_steps=7)
    assert target_steps == 14
    assert sequential_model.worldcache_num_steps == 14
    assert sequential_model.cnt == 0
    assert sequential_model.previous_input == [None, None]

    parallel_model = MockCacheModel()
    parallel_config = WorldCacheConfig(num_steps=35, parallel_cfg=True)
    apply_worldcache(parallel_model, parallel_config)
    target_steps = reset_cache_runtime(parallel_model, parallel_config, num_steps=7)
    assert target_steps == 7
    assert parallel_model.worldcache_num_steps == 7
    assert parallel_model.previous_output == [None, None]


def test_dicache_reset_logic_uses_cfg_double_pass():
    model = MockCacheModel()
    config = DiCacheConfig(num_steps=35)
    apply_dicache(model, config)
    target_steps = reset_cache_runtime(model, config, num_steps=9)
    assert target_steps == 18
    assert model.dicache_num_steps == 18
    assert model.cnt == 0
    assert model.previous_input == [None, None]
