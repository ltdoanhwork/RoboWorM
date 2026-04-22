from __future__ import annotations

from methods.cache_strategy.common import WorldCacheConfig
from methods.cache_strategy.WorldCache.ctrl_world_runtime import CtrlWorldWorldCacheRuntime


def enable_worldcache(
    unet,
    *,
    num_steps: int,
    rel_l1_thresh: float = 0.03,
    ret_ratio: float = 0.4,
    probe_depth: int = 3,
    motion_sensitivity: float = 5.0,
    hf_enabled: bool = False,
    hf_thresh: float = 0.01,
    saliency_enabled: bool = False,
    saliency_weight: float = 5.0,
    osi_enabled: bool = False,
    dynamic_decay: bool = False,
) -> None:
    config = WorldCacheConfig(
        num_steps=num_steps,
        rel_l1_thresh=rel_l1_thresh,
        ret_ratio=ret_ratio,
        probe_depth=probe_depth,
        motion_sensitivity=motion_sensitivity,
        flow_enabled=False,
        flow_scale=0.5,
        hf_enabled=hf_enabled,
        hf_thresh=hf_thresh,
        saliency_enabled=saliency_enabled,
        saliency_weight=saliency_weight,
        osi_enabled=osi_enabled,
        dynamic_decay=dynamic_decay,
        aduc_enabled=False,
        aduc_start=0.5,
        parallel_cfg=False,
    )
    unet._ctrl_cache_runtime = CtrlWorldWorldCacheRuntime(config=config, total_steps=num_steps)
    print(
        "[WorldCache] Enabled on Ctrl-World "
        f"(steps={num_steps}, rel_l1_thresh={rel_l1_thresh}, ret_ratio={ret_ratio}, "
        f"probe_depth={probe_depth}, motion_sensitivity={motion_sensitivity}, "
        f"hf={hf_enabled}, saliency={saliency_enabled}, osi={osi_enabled}, decay={dynamic_decay})"
    )


def disable_worldcache(unet) -> None:
    if hasattr(unet, "_ctrl_cache_runtime"):
        delattr(unet, "_ctrl_cache_runtime")
    print("[WorldCache] Disabled on Ctrl-World.")
