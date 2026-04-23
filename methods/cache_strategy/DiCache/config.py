from __future__ import annotations


def add_dicache_args(parser) -> None:
    parser.add_argument(
        "--use_dicache",
        action="store_true",
        help="Enable DiCache v1 acceleration for Ctrl-World.",
    )
    parser.add_argument(
        "--dicache_rel_l1_thresh",
        type=float,
        default=0.08,
        help="Relative L1 threshold for DiCache skip decisions.",
    )
    parser.add_argument(
        "--dicache_ret_ratio",
        type=float,
        default=0.2,
        help="Fraction of denoising steps to keep dense before DiCache becomes eligible.",
    )
    parser.add_argument(
        "--dicache_probe_depth",
        type=int,
        default=2,
        help="Number of attention-bearing UNet stages used for the DiCache probe.",
    )
