"""
Radial Attention Adapter for Ctrl-World SVD UNet.

Public API:
    enable_radial_attention(unet, ...)
    disable_radial_attention(unet)
"""

from __future__ import annotations

import os
import sys
import warnings

# Path to the upstream radial-attention package
_RADIAL_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "radial-attention"))

def _ensure_radial_on_path() -> None:
    """Add radial-attention to sys.path so its internal imports work."""
    if _RADIAL_SRC not in sys.path:
        sys.path.insert(0, _RADIAL_SRC)

def enable_radial_attention(
    unet,
    *,
    decay_factor: float = 1.0,
    block_size: int = 64,
    pad_small_layers: bool = True,
    first_layers_fp: int = 2,
    # Geometry
    num_views: int = 3,
    H_per_view: int = 24,          
    W: int = 40,
    model_type: str = "hunyuan",
) -> None:
    """
    Inject Radial Attention into all diffusers Attention layers of `unet`.
    """
    _ensure_radial_on_path()

    from .processor import SVDRadialAttnProcessor

    # Set class-level shared state
    SVDRadialAttnProcessor.decay_factor = decay_factor
    SVDRadialAttnProcessor.block_size = block_size
    SVDRadialAttnProcessor.pad_small_layers = pad_small_layers
    SVDRadialAttnProcessor.first_layers_fp = first_layers_fp
    
    SVDRadialAttnProcessor.num_views = num_views
    SVDRadialAttnProcessor.H_per_view = H_per_view
    SVDRadialAttnProcessor.W = W
    SVDRadialAttnProcessor.model_type = model_type

    # Enumerate attention modules via named_modules (NOT unet.attn_processors —
    # that property returns EMPTY on the Ctrl-World custom UNet, so the old code
    # set 0 processors and ran pure dense; verified via CW_ATTN_PROFILE).
    attn_modules = [
        (name, module) for name, module in unet.named_modules()
        if hasattr(module, "processor") and hasattr(module, "to_q")
    ]
    num_layers = len(attn_modules)
    for idx, (name, module) in enumerate(attn_modules):
        module.processor = SVDRadialAttnProcessor(layer_idx=idx, num_layers=num_layers)

    # Summary
    H_total = num_views * H_per_view
    T_l0 = H_total * W
    print(
        f"[Radial] Enabled on {num_layers} layers.\n"
        f"[Radial] Config: decay={decay_factor}, block_size={block_size}, pad_small={pad_small_layers}, "
        f"first_layers_fp={first_layers_fp}/{num_layers}, L0 T={T_l0}, model_type={model_type}"
    )

def disable_radial_attention(unet) -> None:
    unet.set_default_attn_processor()
    print("[Radial] Disabled — restored default SDPA processor.")
