from __future__ import annotations

from .processor import SVDSiToAttnProcessor


def enable_sito(
    unet,
    *,
    start_layer_idx: int = 0,
    prune_ratio: float | None = None,
    patch_h: int = 2,
    patch_w: int = 2,
    noise_alpha: float = 0.1,
    sim_beta: float = 1.0,
    max_downsample_ratio: int = 4,
) -> None:
    processors = {}
    for layer_idx, name in enumerate(unet.attn_processors.keys()):
        processors[name] = SVDSiToAttnProcessor(
            layer_idx=layer_idx,
            start_layer_idx=start_layer_idx,
            prune_ratio=prune_ratio,
            patch_h=patch_h,
            patch_w=patch_w,
            noise_alpha=noise_alpha,
            sim_beta=sim_beta,
            max_downsample_ratio=max_downsample_ratio,
        )
    unet.set_attn_processor(processors)
    print(
        "[SiTo] Enabled on "
        f"{len(processors)} attention layers (start_layer_idx={start_layer_idx}, "
        f"max_downsample_ratio={max_downsample_ratio})"
    )


def disable_sito(unet) -> None:
    unet.set_default_attn_processor()
    print("[SiTo] Disabled - restored default attention processors.")
