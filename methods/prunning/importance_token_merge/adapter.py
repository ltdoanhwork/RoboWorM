from __future__ import annotations

from .processor import SVDITMAttnProcessor


def enable_itm(
    unet,
    *,
    itm_state: dict,
    start_layer_idx: int = 0,
    compress_ratio: float | None = None,
    prune_from_step: int = 1,
    merge_from_step: int = 2,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    max_downsample_ratio: int = 4,
) -> None:
    if not merge_attn:
        raise ValueError("[ITM] Ctrl-World requires `itm_merge_attn=True`.")
    if merge_crossattn:
        raise ValueError("[ITM] Ctrl-World v1 does not support `itm_merge_crossattn=True`.")
    if merge_mlp:
        raise ValueError("[ITM] Ctrl-World v1 does not support `itm_merge_mlp=True`.")
    itm_state.setdefault("prune_from_step", prune_from_step)
    itm_state.setdefault("merge_from_step", merge_from_step)
    itm_state.setdefault("merge_attn", merge_attn)
    itm_state.setdefault("merge_crossattn", merge_crossattn)
    itm_state.setdefault("merge_mlp", merge_mlp)
    itm_state.setdefault("phase", "disabled")
    itm_state.setdefault("merge_mode", "mean")
    itm_state.setdefault("heat_map", None)

    processors = {}
    for layer_idx, name in enumerate(unet.attn_processors.keys()):
        processors[name] = SVDITMAttnProcessor(
            itm_state=itm_state,
            layer_idx=layer_idx,
            start_layer_idx=start_layer_idx,
            compress_ratio=compress_ratio,
            prune_from_step=prune_from_step,
            merge_from_step=merge_from_step,
            max_downsample_ratio=max_downsample_ratio,
        )
    unet._itm_state = itm_state
    unet.set_attn_processor(processors)
    print(
        "[ITM] Enabled on "
        f"{len(processors)} attention layers (start_layer_idx={start_layer_idx}, "
        f"max_downsample_ratio={max_downsample_ratio})"
    )


def disable_itm(unet) -> None:
    if hasattr(unet, "_itm_state"):
        delattr(unet, "_itm_state")
    unet.set_default_attn_processor()
    print("[ITM] Disabled - restored default attention processors.")
