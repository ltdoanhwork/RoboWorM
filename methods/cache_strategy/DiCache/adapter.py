from __future__ import annotations

from methods.cache_strategy.common import DiCacheConfig
from methods.cache_strategy.DiCache.ctrl_world_runtime import CtrlWorldDiCacheRuntime


def enable_dicache(
    unet,
    *,
    num_steps: int,
    rel_l1_thresh: float = 0.08,
    ret_ratio: float = 0.2,
    probe_depth: int = 2,
) -> None:
    config = DiCacheConfig(
        num_steps=num_steps,
        rel_l1_thresh=rel_l1_thresh,
        ret_ratio=ret_ratio,
        probe_depth=probe_depth,
    )
    unet._ctrl_cache_runtime = CtrlWorldDiCacheRuntime(config=config, total_steps=num_steps)
    print(
        "[DiCache] Enabled on Ctrl-World "
        f"(steps={num_steps}, rel_l1_thresh={rel_l1_thresh}, "
        f"ret_ratio={ret_ratio}, probe_depth={probe_depth})"
    )


def disable_dicache(unet) -> None:
    if hasattr(unet, "_ctrl_cache_runtime"):
        delattr(unet, "_ctrl_cache_runtime")
    print("[DiCache] Disabled on Ctrl-World.")
