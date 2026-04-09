"""
Sparse-VideoGen (SVG) adapter for Ctrl-World SVD UNet.

Public API
----------
    enable_svg   — inject SVG1 (stripe sparse) or SVG2 (SAP) attention
    disable_svg  — restore default diffusers SDPA
"""

from .adapter import enable_svg, disable_svg

__all__ = ["enable_svg", "disable_svg"]
