"""
PISA (Piecewise Sparse Attention) adapter for Ctrl-World SVD UNet.

Usage:
    from methods.sparse_attention.pisa import enable_pisa, disable_pisa
    enable_pisa(ctrl_world_model.unet, density=0.5, block_size=16)
"""
__all__ = ["enable_pisa", "disable_pisa"]


def enable_pisa(*args, **kwargs):
    from .adapter import enable_pisa as _enable_pisa
    return _enable_pisa(*args, **kwargs)


def disable_pisa(*args, **kwargs):
    from .adapter import disable_pisa as _disable_pisa
    return _disable_pisa(*args, **kwargs)
