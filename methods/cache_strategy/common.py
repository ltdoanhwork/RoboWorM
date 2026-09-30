from __future__ import annotations

import inspect
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
    start_step: int = 0
    model_interval: int = 5
    block_interval: int = 3
    debug: bool = False


@dataclass(frozen=True)
class ScalingCacheConfig:
    """ScalingCache (difference-scaling + dynamic-interval). Implemented in AM_DiT/src/scl."""

    num_steps: int = 35
    first_enhance: int = 10
    last_enhance: int = 2
    error_rate: float = 1.0
    fresh_threshold: int = 2
    dynamic_cache: bool = True
    use_alpha: bool = False
    alpha_dict_path: str | None = None
    update_alpha: bool = False
    granularity: str = "block"  # "block" (memory-safe) or "submodule" (faithful, ~3x memory)


@dataclass(frozen=True)
class BSCConfig:
    """BSC (Block Skip-eligibility Caching) -- ours. Implemented in AM_DiT.

    Algorithm core: ``AM_DiT/src/bsc/`` (gate, groups, metrics -- backbone-agnostic).
    DiT adaptation: ``AM_DiT/methods/cache_strategy/BSC/{dit,dreamgen}.py``, reached from
    here as ``bsc.backend`` because ``AM_DiT/src`` is what this process has on PYTHONPATH
    (same route as ``scl.backend``).

    Unlike the step-level baselines above, BSC gates *groups of DiT blocks* and predicts a
    skipped group's output from its cached anchor, so the knobs are per-block/per-group
    rather than per-step. See ``AM_DiT/docs/ALGORITHM.md``.

    Step-count convention: DreamGen runs classifier-free guidance sequentially, so
    ``net.forward`` fires twice per denoise step. ``warmup_steps`` and
    ``min_full_interval`` are therefore in net-forward units (two per denoise step) --
    the same doubling ``reset_worldcache_state`` applies.
    """

    # --- gate (AM_DiT docs/ALGORITHM.md 3.1, 3.4) ---
    gate_mode: str = "group"          # "group" | "perblock_pooled" | "step3" | "drift"
    rcs_threshold: float = 0.1        # tau: threshold on the (predicted-output) drift
    copy_threshold: float = 0.03
    min_full_interval: int = 3        # max reuse horizon, in net forwards
    warmup_steps: int = 3
    output_gate: bool = True          # gate on s*E_I (output space), not raw E_I
    # --- groups (3.1) ---
    group_size: int = 4
    group_predict: str = "io"         # "io" = members copy, leader maps I->O; "block"
    group_bounds: str | None = None   # non-uniform, e.g. "0-1,2-5,6-9"; None = uniform
    # --- protection ---
    protect_first_n: int = 0
    protect_last_frac: float = 0.0
    protect_from_block: int | None = None
    max_skip_run: int = 0
    # --- skip-eligibility (3.2) ---
    skip_eligibility: bool = False
    elig_s_dev: float = 0.5
    elig_cv: float = 0.35
    elig_warmup_fits: int = 2
    s_ema_beta: float = 0.8
    # --- predictor (3.5) ---
    predictor_mode: str = "linear_s1"
    s_clamp: float | None = None
    s_floor: float | None = None
    decay_rate: float = 0.5
    # --- adaptive cadence (3.6) ---
    adaptive_interval: bool = False
    adaptive_rate: float = 1.0
    error_budget: float | None = None
    mfi_max: int = 12
    cv_interval: float = 0.0
    depth_interval_scale: float = 0.0
    step_copy_threshold: float = 0.0
    # --- SNR-phase gate (3.3) ---
    snr_skip_frac: float = 0.0
    snr_skip_t: float = 0.0


CacheRuntimeConfig = WorldCacheConfig | DiCacheConfig | FasterCacheConfig | ScalingCacheConfig | BSCConfig


def get_cache_backend_name(config: CacheRuntimeConfig | None) -> str | None:
    if config is None:
        return None
    if isinstance(config, WorldCacheConfig):
        return "worldcache"
    if isinstance(config, DiCacheConfig):
        return "dicache"
    if isinstance(config, FasterCacheConfig):
        return "fastercache"
    if isinstance(config, ScalingCacheConfig):
        return "scalingcache"
    if isinstance(config, BSCConfig):
        return "bsc"
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


def prepare_cache_runtime_model(model) -> None:
    signature = inspect.signature(model.forward)
    model._cache_runtime_uses_condition_mask = "condition_video_input_mask_B_C_T_H_W" in signature.parameters


def _maybe_concat_condition_mask(
    model,
    x_B_C_T_H_W: torch.Tensor,
    *,
    is_video: bool,
    condition_video_input_mask_B_C_T_H_W: torch.Tensor | None,
) -> torch.Tensor:
    if not getattr(model, "_cache_runtime_uses_condition_mask", False):
        return x_B_C_T_H_W

    extra_view_channels = getattr(model, "view_condition_dim", 0) if getattr(model, "concat_view_embedding", False) else 0
    expected_channels = getattr(model, "in_channels", x_B_C_T_H_W.shape[1])
    needs_condition_mask = x_B_C_T_H_W.shape[1] + 1 + extra_view_channels == expected_channels
    if not needs_condition_mask:
        return x_B_C_T_H_W

    if is_video:
        if condition_video_input_mask_B_C_T_H_W is None:
            raise ValueError("condition_video_input_mask_B_C_T_H_W is required for cache-enabled video-conditioned models.")
        condition_channel = condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)
    else:
        batch, _channels, time, height, width = x_B_C_T_H_W.shape
        condition_channel = torch.zeros(
            (batch, 1, time, height, width),
            dtype=x_B_C_T_H_W.dtype,
            device=x_B_C_T_H_W.device,
        )

    return torch.cat([x_B_C_T_H_W, condition_channel], dim=1)


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
        from methods.cache_strategy.FasterCache.runtime import reset_fastercache_state

        return reset_fastercache_state(model, config, num_steps=num_steps)
    if isinstance(config, ScalingCacheConfig):
        from scl.backend import reset_scalingcache_state

        return reset_scalingcache_state(model, config, num_steps=num_steps)
    if isinstance(config, BSCConfig):
        # Per-episode reset is NOT optional. The DeepCache baseline produced a whole
        # table of numbers that had to be retracted twice because its cache was never
        # cleared between episodes: step 0 of every later episode hit an anchor computed
        # from a different episode, which flattered compute (a warm cache it never paid
        # for) and corrupted quality (a stale tensor from elsewhere) simultaneously.
        # See AM_DiT/docs/INSIGHTS.md section 47. `reset_bsc_state` therefore RAISES if
        # it cannot find the runtime rather than returning quietly.
        from bsc.backend import reset_bsc_state

        return reset_bsc_state(model, config, num_steps=num_steps)
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
