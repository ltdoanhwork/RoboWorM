import argparse
from typing import Any

from methods.cache_strategy.common import (
    CacheRuntimeConfig,
    DiCacheConfig,
    FasterCacheConfig,
    ScalingCacheConfig,
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

    # ScalingCache (implemented in AM_DiT/src/scl)
    group.add_argument("--scalingcache-enabled", action="store_true", help="Enable ScalingCache (difference-scaling + dynamic-interval) from AM_DiT/src/scl.")
    group.add_argument("--scalingcache-num-steps", type=int, default=35, help="Total denoising steps (must match inference --num-steps).")
    group.add_argument("--scalingcache-first-enhance", type=int, default=10, help="Warm-up steps always fully computed.")
    group.add_argument("--scalingcache-last-enhance", type=int, default=2, help="Final steps always fully computed.")
    group.add_argument("--scalingcache-error-rate", type=float, default=1.0, help="Scales the adaptive error threshold (dynamic mode).")
    group.add_argument("--scalingcache-fresh-threshold", type=int, default=2, help="Fixed refresh period when --scalingcache-no-dynamic.")
    group.add_argument("--scalingcache-no-dynamic", action="store_true", help="Use fixed periodic refresh instead of error-driven scheduling.")
    group.add_argument("--scalingcache-use-alpha", action="store_true", help="Use offline-fitted scaling coefficients.")
    group.add_argument("--scalingcache-alpha-dict-path", type=str, default=None, help="Path to alpha_dict_*.pth (when --scalingcache-use-alpha).")
    group.add_argument("--scalingcache-update-alpha", action="store_true", help="Offline alpha-fitting pass (force Scaling, fit alphas).")
    group.add_argument("--scalingcache-granularity", type=str, default="block", choices=["block", "submodule"], help="Cache granularity: 'block' (memory-safe, default) or 'submodule' (faithful, ~3x memory).")


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
    if getattr(setup_args, "scalingcache_enabled", False):
        return ScalingCacheConfig(
            num_steps=getattr(setup_args, "scalingcache_num_steps", 35),
            first_enhance=getattr(setup_args, "scalingcache_first_enhance", 10),
            last_enhance=getattr(setup_args, "scalingcache_last_enhance", 2),
            error_rate=getattr(setup_args, "scalingcache_error_rate", 1.0),
            fresh_threshold=getattr(setup_args, "scalingcache_fresh_threshold", 2),
            dynamic_cache=not getattr(setup_args, "scalingcache_no_dynamic", False),
            use_alpha=getattr(setup_args, "scalingcache_use_alpha", False),
            alpha_dict_path=getattr(setup_args, "scalingcache_alpha_dict_path", None),
            update_alpha=getattr(setup_args, "scalingcache_update_alpha", False),
            granularity=getattr(setup_args, "scalingcache_granularity", "block"),
        )
    return None
