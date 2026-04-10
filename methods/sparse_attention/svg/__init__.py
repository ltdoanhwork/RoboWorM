"""
Sparse-VideoGen (SVG) surfaces for Ctrl-World and DreamGen.

The Ctrl-World adapter is exposed lazily so DreamGen can import
`methods.sparse_attention.svg.config` without paying adapter import costs.
"""


def enable_svg(*args, **kwargs):
    from .adapter import enable_svg as _enable_svg

    return _enable_svg(*args, **kwargs)


def disable_svg(*args, **kwargs):
    from .adapter import disable_svg as _disable_svg

    return _disable_svg(*args, **kwargs)


__all__ = ["enable_svg", "disable_svg"]
