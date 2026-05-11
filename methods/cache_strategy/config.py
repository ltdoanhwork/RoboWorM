import argparse
from typing import Any

from methods.cache_strategy.common import (
    CacheRuntimeConfig,
    DiCacheConfig,
    FasterCacheConfig,
    WorldCacheConfig,
)


def add_dreamgen_cache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("DreamGen Cache Backends")
    
    # FasterCache
    group.add_argument("--fastercache-enabled", action="store_true", help="Enable FasterCache acceleration.")
    group.add_argument("--fastercache-debug", action="store_true", help="Enable debug logs for FasterCache.")
    
    # WorldCache
    group.add_argument("--worldcache-enabled", action="store_true", help="Enable WorldCache acceleration for DiT inference.")
    group.add_argument("--worldcache-num-steps", type=int, default=35, help="Total denoising steps (must match --num-steps in inference).")
    group.add_argument("--worldcache-rel-l1-thresh", type=float, default=0.03, help="Relative L1 threshold for cache hit decision.")
    group.add_argument("--worldcache-ret-ratio", type=float, default=0.4, help="Fraction of initial steps to always compute (warm-up).")
    group.add_argument("--worldcache-probe-depth", type=int, default=4, help="Number of leading DiT blocks used as the probe.")
    group.add_argument("--worldcache-motion-sensitivity", type=float, default=5.0, help="Alpha in dynamic threshold.")
    group.add_argument("--worldcache-flow-enabled", action="store_true", help="Warp cached features using optical-flow before reuse.")
    group.add_argument("--worldcache-flow-scale", type=float, default=0.5, help="Down-scale factor for flow estimation (0-1).")
    group.add_argument("--worldcache-hf-enabled", action="store_true", help="Enable spectral (high-frequency) drift guard.")
    group.add_argument("--worldcache-hf-thresh", type=float, default=0.01, help="HF drift above this forces a full compute step.")
    group.add_argument("--worldcache-saliency-enabled", action="store_true", help="Weight drift by channel-variance saliency map.")
    group.add_argument("--worldcache-saliency-weight", type=float, default=5.0, help="Beta multiplier for saliency-guided thresholding.")
    group.add_argument("--worldcache-osi-enabled", action="store_true", help="Online System Identification for optimal gamma.")
    group.add_argument("--worldcache-dynamic-decay", action="store_true", help="Increase threshold over time.")
    group.add_argument("--worldcache-aduc-enabled", action="store_true", help="Adaptive Unconditional Caching.")
    group.add_argument("--worldcache-aduc-start", type=float, default=0.5, help="Step ratio after which AdUC activates.")
    group.add_argument("--worldcache-parallel-cfg", action="store_true", help="Batch cond+uncond in B=2.")

    # DiCache
    group.add_argument("--dicache-enabled", action="store_true", help="Enable DiCache acceleration.")
    group.add_argument("--dicache-num-steps", type=int, default=35, help="Denoising steps.")
    group.add_argument("--dicache-rel-l1-thresh", type=float, default=0.08, help="Relative threshold.")
    group.add_argument("--dicache-ret-ratio", type=float, default=0.2, help="Retention ratio.")
    group.add_argument("--dicache-probe-depth", type=int, default=2, help="Probe depth.")


def build_cache_runtime_config(setup_args: Any) -> CacheRuntimeConfig | None:
    if getattr(setup_args, "fastercache_enabled", False) or getattr(setup_args, "use_fastercache", False):
        from methods.cache_strategy.common import FasterCacheConfig
        return FasterCacheConfig(
            start_step=getattr(setup_args, "fastercache_start_step", 0),
            model_interval=getattr(setup_args, "fastercache_model_interval", 5),
            block_interval=getattr(setup_args, "fastercache_block_interval", 3),
            debug=getattr(setup_args, "fastercache_debug", False),
        )
    if getattr(setup_args, "worldcache_enabled", False):
        return WorldCacheConfig(
            num_steps=getattr(setup_args, "worldcache_num_steps", 35),
            rel_l1_thresh=getattr(setup_args, "worldcache_rel_l1_thresh", 0.03),
            ret_ratio=getattr(setup_args, "worldcache_ret_ratio", 0.4),
            probe_depth=getattr(setup_args, "worldcache_probe_depth", 4),
            motion_sensitivity=getattr(setup_args, "worldcache_motion_sensitivity", 5.0),
            flow_enabled=getattr(setup_args, "worldcache_flow_enabled", False),
            flow_scale=getattr(setup_args, "worldcache_flow_scale", 0.5),
            hf_enabled=getattr(setup_args, "worldcache_hf_enabled", False),
            hf_thresh=getattr(setup_args, "worldcache_hf_thresh", 0.01),
            saliency_enabled=getattr(setup_args, "worldcache_saliency_enabled", False),
            saliency_weight=getattr(setup_args, "worldcache_saliency_weight", 5.0),
            osi_enabled=getattr(setup_args, "worldcache_osi_enabled", False),
            dynamic_decay=getattr(setup_args, "worldcache_dynamic_decay", False),
            aduc_enabled=getattr(setup_args, "worldcache_aduc_enabled", False),
            aduc_start=getattr(setup_args, "worldcache_aduc_start", 0.5),
            parallel_cfg=getattr(setup_args, "worldcache_parallel_cfg", False),
        )
    if getattr(setup_args, "dicache_enabled", False):
        return DiCacheConfig(
            num_steps=getattr(setup_args, "dicache_num_steps", 35),
            rel_l1_thresh=getattr(setup_args, "dicache_rel_l1_thresh", 0.08),
            ret_ratio=getattr(setup_args, "dicache_ret_ratio", 0.2),
            probe_depth=getattr(setup_args, "dicache_probe_depth", 2),
        )
    return None
