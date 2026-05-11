from __future__ import annotations

from methods.cache_strategy.FasterCache.processor import SVDFasterCacheAttnProcessor


def enable_fastercache(
    unet,
    *,
    start_step: int = 0,
    model_interval: int = 5,
    block_interval: int = 3,
    first_layers_fp: int = 2,
) -> None:
    runtime_state = {
        "start_step": start_step,
        "model_interval": model_interval,
        "block_interval": block_interval,
        "first_layers_fp": first_layers_fp,
        "current_step_idx": -1,
        "resolved_start_step": 0,
        "delta_lf": None,
        "delta_hf": None,
        "block_histories": {},
        "block_alpha": 0.3,
    }
    processors = {}
    for layer_idx, name in enumerate(unet.attn_processors.keys()):
        processors[name] = SVDFasterCacheAttnProcessor(
            runtime_state=runtime_state,
            layer_idx=layer_idx,
            first_layers_fp=first_layers_fp,
        )

    unet._fastercache_state = runtime_state
    unet.set_attn_processor(processors)
    print(
        "[FasterCache] Enabled on "
        f"{len(processors)} attention layers (start_step={start_step}, "
        f"model_interval={model_interval}, block_interval={block_interval})"
    )


def disable_fastercache(unet) -> None:
    if hasattr(unet, "_fastercache_state"):
        delattr(unet, "_fastercache_state")
    unet.set_default_attn_processor()
    print("[FasterCache] Disabled - restored default attention processors.")
