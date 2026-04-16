from __future__ import annotations


def add_worldcache_args(parser) -> None:
    parser.add_argument(
        "--use_worldcache",
        action="store_true",
        help="Enable WorldCache v1 acceleration for Ctrl-World.",
    )
    parser.add_argument(
        "--worldcache_rel_l1_thresh",
        type=float,
        default=0.03,
        help="Base relative L1 threshold for WorldCache skip decisions.",
    )
    parser.add_argument(
        "--worldcache_ret_ratio",
        type=float,
        default=0.4,
        help="Fraction of denoising steps to keep dense before WorldCache becomes eligible.",
    )
    parser.add_argument(
        "--worldcache_probe_depth",
        type=int,
        default=3,
        help="Number of attention-bearing UNet stages used for the WorldCache probe.",
    )
    parser.add_argument(
        "--worldcache_motion_sensitivity",
        type=float,
        default=5.0,
        help="Velocity sensitivity for dynamic WorldCache thresholding.",
    )
    parser.add_argument(
        "--worldcache_hf_enabled",
        action="store_true",
        help="Enable high-frequency drift guard for WorldCache.",
    )
    parser.add_argument(
        "--worldcache_hf_thresh",
        type=float,
        default=0.01,
        help="High-frequency drift threshold for WorldCache.",
    )
    parser.add_argument(
        "--worldcache_saliency_enabled",
        action="store_true",
        help="Enable saliency-weighted probe drift for WorldCache.",
    )
    parser.add_argument(
        "--worldcache_saliency_weight",
        type=float,
        default=5.0,
        help="Saliency weighting factor for WorldCache probe drift.",
    )
    parser.add_argument(
        "--worldcache_osi_enabled",
        action="store_true",
        help="Enable optimal scalar interpolation gamma for WorldCache output extrapolation.",
    )
    parser.add_argument(
        "--worldcache_dynamic_decay",
        action="store_true",
        help="Relax the WorldCache threshold toward late denoising steps.",
    )
