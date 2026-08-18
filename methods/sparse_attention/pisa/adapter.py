"""
PISA adapter for Ctrl-World SVD UNet.

Isolated entirely within methods/sparse_attention/pisa/ — no dependency on
core Ctrl-World model files (ctrl_world.py, unet_spatio_temporal_condition.py, etc.).

API
---
    enable_pisa(unet, density, block_size, start_layer_idx)
    disable_pisa(unet)
"""

import torch

try:
    from .piecewise_sparse_attn_tma import piecewise_sparse_attention
except ImportError as e:
    raise ImportError(
        "[PISA] Could not import vendored piecewise_sparse_attn_tma.\n"
        "Make sure its Triton dependencies are installed.\n"
        f"Original error: {e}"
    )

from .processor import SVDPISAAttnProcessor


def enable_pisa(
    unet,
    density: float = 0.5,
    block_size: int = 16,
    start_layer_idx: int = 0,
) -> None:
    """
    Inject PISA sparse attention into all attention layers of an SVD UNet.

    Parameters
    ----------
    unet : UNetSpatioTemporalConditionModel
        The UNet inside CrtlWorld / CtrlWorldDiffusionPipeline.
    density : float
        Fraction of attention blocks computed with exact attention (0–1).
        Lower = faster but less precise. Recommended: 0.3–0.7.
    block_size : int
        Token block size for piecewise decomposition. Must be ≤ sequence length.
        Recommended: 16 for SVD spatial, 8 for temporal.
    start_layer_idx : int
        Skip the first N layers (keep them at full attention).
        Layers with very short sequences (e.g. temporal T=25) are skipped
        automatically inside the processor regardless of this setting.
    """
    # Enumerate attention modules directly via named_modules and assign the
    # processor in-place. We deliberately do NOT use unet.attn_processors /
    # set_attn_processor here: on the Ctrl-World custom
    # UNetSpatioTemporalConditionModel that property returns an EMPTY dict (its
    # Attention modules don't expose get_processor the way the diffusers property
    # expects), so the old code silently injected PISA into ZERO layers and ran
    # pure dense (verified via CW_ATTN_PROFILE: T=3840 attn1 stayed at the dense
    # 1.49 ms/call). named_modules finds every real Attention module.
    new_processors = {}
    layer_idx = 0
    for name, module in unet.named_modules():
        if hasattr(module, "processor") and hasattr(module, "to_q"):
            proc = SVDPISAAttnProcessor(
                attn_fn=piecewise_sparse_attention,
                density=density,
                block_size=block_size,
                layer_idx=layer_idx,
                start_layer_idx=start_layer_idx,
            )
            module.processor = proc
            new_processors[name] = proc
            layer_idx += 1
    print(
        f"[PISA] Enabled on {len(new_processors)} attention layers "
        f"(density={density}, block_size={block_size}, "
        f"start_layer_idx={start_layer_idx})"
    )


def disable_pisa(unet) -> None:
    """
    Restore the UNet's default PyTorch SDPA attention processors.
    Safe to call even if PISA was never enabled.
    """
    unet.set_default_attn_processor()
    print("[PISA] Disabled — restored default attention processors.")
