from __future__ import annotations

from typing import Any

from methods.cache_strategy.common import (
    CacheRuntimeConfig,
    DiCacheConfig,
    FasterCacheConfig,
    WorldCacheConfig,
)


def build_cache_runtime_config(setup_args: Any) -> CacheRuntimeConfig | None:
    if setup_args.worldcache_enabled:
        return WorldCacheConfig(
            num_steps=setup_args.worldcache_num_steps,
            rel_l1_thresh=setup_args.worldcache_rel_l1_thresh,
            ret_ratio=setup_args.worldcache_ret_ratio,
            probe_depth=setup_args.worldcache_probe_depth,
            motion_sensitivity=setup_args.worldcache_motion_sensitivity,
            flow_enabled=setup_args.worldcache_flow_enabled,
            flow_scale=setup_args.worldcache_flow_scale,
            hf_enabled=setup_args.worldcache_hf_enabled,
            hf_thresh=setup_args.worldcache_hf_thresh,
            saliency_enabled=setup_args.worldcache_saliency_enabled,
            saliency_weight=setup_args.worldcache_saliency_weight,
            osi_enabled=setup_args.worldcache_osi_enabled,
            dynamic_decay=setup_args.worldcache_dynamic_decay,
            aduc_enabled=setup_args.worldcache_aduc_enabled,
            aduc_start=setup_args.worldcache_aduc_start,
            parallel_cfg=setup_args.worldcache_parallel_cfg,
        )
    if setup_args.dicache_enabled:
        return DiCacheConfig(
            num_steps=setup_args.dicache_num_steps,
            rel_l1_thresh=setup_args.dicache_rel_l1_thresh,
            ret_ratio=setup_args.dicache_ret_ratio,
            probe_depth=setup_args.dicache_probe_depth,
        )
    if setup_args.fastercache_enabled:
        return FasterCacheConfig(
            start_step=setup_args.fastercache_start_step,
            model_interval=setup_args.fastercache_model_interval,
            block_interval=setup_args.fastercache_block_interval,
        )
    return None


def validate_runtime_backend_args(
    setup_args: Any,
    inference_args: Any,
) -> None:
    enabled_sparse = [
        name
        for name, enabled in (
            ("SVG", setup_args.use_svg),
            ("SiTo", setup_args.use_sito),
            ("ITM", setup_args.use_itm),
        )
        if enabled
    ]
    enabled_cache = [
        name
        for name, enabled in (
            ("WorldCache", setup_args.worldcache_enabled),
            ("DiCache", setup_args.dicache_enabled),
            ("FasterCache", setup_args.fastercache_enabled),
        )
        if enabled
    ]

    if setup_args.use_svg and setup_args.context_parallel_size > 1:
        raise ValueError("[SVG] DreamDojo SVG v1 is single-GPU only; set context_parallel_size=1.")
    if setup_args.use_sito and setup_args.context_parallel_size > 1:
        raise ValueError("[SiTo] DreamDojo SiTo v1 is single-GPU only; set context_parallel_size=1.")
    if setup_args.use_itm and setup_args.context_parallel_size > 1:
        raise ValueError("[ITM] DreamDojo ITM v1 is single-GPU only; set context_parallel_size=1.")
    if setup_args.use_sito and setup_args.use_svg:
        raise ValueError("[SiTo] SiTo and SVG are mutually exclusive in DreamDojo.")
    if setup_args.use_itm and inference_args.guidance <= 0:
        raise ValueError("[ITM] DreamDojo ITM requires classifier-free guidance with `guidance > 0`.")
    if len(enabled_sparse) > 1:
        raise ValueError("[ITM] DreamDojo sparse/token backends are mutually exclusive: SVG, SiTo, ITM.")
    if len(enabled_cache) > 1:
        raise ValueError("[Cache] DreamDojo cache backends are mutually exclusive: WorldCache, DiCache, FasterCache.")
    if enabled_cache and setup_args.context_parallel_size > 1:
        raise ValueError("[Cache] DreamDojo cache backends are v1 single-GPU only; set context_parallel_size=1.")
    if enabled_cache and enabled_sparse:
        raise ValueError(
            "[Cache] DreamDojo cache backends are mutually exclusive with sparse/token backends: "
            + ", ".join(enabled_sparse)
            + "."
        )
