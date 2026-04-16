from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass(frozen=True)
class WorldCacheConfig:
    num_steps: int = 35
    rel_l1_thresh: float = 0.5
    ret_ratio: float = 0.2
    probe_depth: int = 8
    motion_sensitivity: float = 5.0
    flow_enabled: bool = False
    flow_scale: float = 0.5
    hf_enabled: bool = False
    hf_thresh: float = 0.01
    saliency_enabled: bool = False
    saliency_weight: float = 5.0
    osi_enabled: bool = False
    dynamic_decay: bool = False
    aduc_enabled: bool = False
    aduc_start: float = 0.5
    parallel_cfg: bool = False


@dataclass(frozen=True)
class DiCacheConfig:
    num_steps: int = 35
    rel_l1_thresh: float = 0.5
    ret_ratio: float = 0.2
    probe_depth: int = 8


@dataclass(frozen=True)
class FasterCacheConfig:
    start_step: int = 1
    model_interval: int = 2
    block_interval: int = 3


CacheRuntimeConfig = WorldCacheConfig | DiCacheConfig | FasterCacheConfig


def get_cache_backend_name(config: CacheRuntimeConfig | None) -> str | None:
    if config is None:
        return None
    if isinstance(config, WorldCacheConfig):
        return "worldcache"
    if isinstance(config, DiCacheConfig):
        return "dicache"
    if isinstance(config, FasterCacheConfig):
        return "fastercache"
    raise TypeError(f"Unsupported cache config type: {type(config)!r}")


def _fresh_slot_list(value):
    return [value, value]


def initialize_common_cache_state(model) -> None:
    model.cnt = 0
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = _fresh_slot_list(None)
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = _fresh_slot_list(None)
    model.previous_input = _fresh_slot_list(None)
    model.resume_flag = [False, False]


def initialize_worldcache_state(model, num_steps: int) -> None:
    model.worldcache_num_steps = num_steps
    model.worldcache_step_skipped_count = 0
    model.probe_residual_cache = _fresh_slot_list(None)
    model.previous_output = _fresh_slot_list(None)
    initialize_common_cache_state(model)


def initialize_dicache_state(model, num_steps: int) -> None:
    model.dicache_num_steps = num_steps
    initialize_common_cache_state(model)


def reset_worldcache_state(model, config: WorldCacheConfig, num_steps: int) -> int:
    target_steps = num_steps if config.parallel_cfg else num_steps * 2
    initialize_worldcache_state(model, num_steps=target_steps)
    return target_steps


def reset_dicache_state(model, _config: DiCacheConfig, num_steps: int) -> int:
    target_steps = num_steps * 2
    initialize_dicache_state(model, num_steps=target_steps)
    return target_steps


def reset_cache_runtime(model, config: CacheRuntimeConfig | None, num_steps: int) -> int | None:
    if config is None:
        return None
    if isinstance(config, WorldCacheConfig):
        return reset_worldcache_state(model, config, num_steps=num_steps)
    if isinstance(config, DiCacheConfig):
        return reset_dicache_state(model, config, num_steps=num_steps)
    if isinstance(config, FasterCacheConfig):
        raise ValueError("[FasterCache] FasterCache runtime is not available in this workspace.")
    raise TypeError(f"Unsupported cache config type: {type(config)!r}")


def _maybe_embed_action(
    model,
    t_embedding_B_T_D: torch.Tensor,
    adaln_lora_B_T_3D: torch.Tensor | None,
    kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    action = kwargs.pop("action", None)
    if action is None:
        return t_embedding_B_T_D, adaln_lora_B_T_3D

    if not hasattr(model, "action_embedder_B_D") or model.action_embedder_B_D is None:
        return t_embedding_B_T_D, adaln_lora_B_T_3D
    if not hasattr(model, "action_embedder_B_3D") or model.action_embedder_B_3D is None:
        return t_embedding_B_T_D, adaln_lora_B_T_3D

    if hasattr(model, "_num_action_per_latent_frame"):
        num_actions = action.shape[1]
        action = rearrange(action, "b t d -> b 1 (t d)")
        action = rearrange(
            action,
            "b 1 (t d) -> b t d",
            t=num_actions // model._num_action_per_latent_frame,
        )
        action_emb_B_D = model.action_embedder_B_D(action)
        action_emb_B_3D = model.action_embedder_B_3D(action)

        zero_pad_action_emb_B_D = torch.zeros_like(action_emb_B_D[:, :1, :], device=action_emb_B_D.device)
        zero_pad_action_emb_B_3D = torch.zeros_like(action_emb_B_3D[:, :1, :], device=action_emb_B_3D.device)
        action_emb_B_D = torch.cat([zero_pad_action_emb_B_D, action_emb_B_D], dim=1)
        action_emb_B_3D = torch.cat([zero_pad_action_emb_B_3D, action_emb_B_3D], dim=1)
    else:
        action = rearrange(action, "b t d -> b 1 (t d)")
        action_emb_B_D = model.action_embedder_B_D(action)
        action_emb_B_3D = model.action_embedder_B_3D(action)

    t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D
    if adaln_lora_B_T_3D is not None:
        adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D

    return t_embedding_B_T_D, adaln_lora_B_T_3D
