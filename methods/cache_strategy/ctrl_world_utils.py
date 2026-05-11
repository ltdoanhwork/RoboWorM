from __future__ import annotations


CTRL_WORLD_BACKEND_FLAGS = (
    "use_pisa",
    "use_svg",
    "use_radial",
    "use_sito",
    "use_itm",
    "use_fastercache",
    "use_dicache",
    "use_worldcache",
)


def validate_ctrl_world_backend_args(args) -> None:
    enabled = [flag for flag in CTRL_WORLD_BACKEND_FLAGS if bool(getattr(args, flag, False))]
    if len(enabled) > 1:
        raise ValueError(
            "[Cache] Ctrl-World acceleration backends are mutually exclusive: "
            "PISA, SVG, Radial, SiTo, ITM, FasterCache, DiCache, and WorldCache."
        )
