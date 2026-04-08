"""
PISA (Piecewise Sparse Attention) adapter for Ctrl-World SVD UNet.

Usage:
    from methods.sparse_attention.pisa import enable_pisa, disable_pisa
    enable_pisa(ctrl_world_model.unet, density=0.5, block_size=16)
"""
from .adapter import enable_pisa, disable_pisa

__all__ = ["enable_pisa", "disable_pisa"]
