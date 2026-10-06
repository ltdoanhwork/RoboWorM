"""FVD (Unterthiner et al., 2018) -- I3D Frechet distance, shared real pool.

``from metrics.fvd import ...`` keeps working for every pre-refactor caller: the
implementation is in :mod:`metrics.fvd.core`, the shared real-pool protocol in
:mod:`metrics.fvd.pool`. Setup (I3D weights, pool construction, per-view protocol,
sample-size caveat) is in ``metrics/fvd/README.md``.
"""
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .pool import (RESTRICT_CAMERAS, TILE_CAMERA, SharedRealPool, budgeted,  # noqa: F401
                   mean_over_common_cameras, ref_identity, split_half)

__all__ = list(_core_all) + ["RESTRICT_CAMERAS", "TILE_CAMERA", "SharedRealPool", "budgeted",
                             "mean_over_common_cameras", "ref_identity", "split_half"]
