"""
Config dataclass for PISA sparse attention.

Can be mixed into wm_args or used standalone via argparse.
This file has zero dependency on any Ctrl-World model code.
"""

from dataclasses import dataclass


@dataclass
class PISAConfig:
    """
    PISA sparse attention configuration.

    Fields
    ------
    use_pisa : bool
        Master switch. Set to True to enable PISA at inference time.
    pisa_density : float
        Fraction of attention blocks computed with exact attention.
        Range (0, 1]. Lower = faster but less accurate.
        Recommended starting point: 0.5
    pisa_block_size : int
        Token block size for piecewise decomposition.
        Must be ≤ sequence length at runtime (auto-clamped).
        Recommended: 16 for SVD spatial attention.
    pisa_start_layer_idx : int
        Skip the first N attention layers (keep them at full precision).
        Early layers often have small T and contribute little to speedup.
        Recommended: 2-4 for SVD UNet.
    """
    use_pisa: bool = False
    pisa_density: float = 0.5
    pisa_block_size: int = 16
    pisa_start_layer_idx: int = 2


def add_pisa_args(parser):
    """
    Convenience function: add PISA flags to an argparse.ArgumentParser.

    Usage:
        from methods.sparse_attention.pisa.config import add_pisa_args
        add_pisa_args(parser)
        args = parser.parse_args()
        # Then: enable_pisa(unet, args.pisa_density, ...)
    """
    group = parser.add_argument_group("PISA sparse attention")
    group.add_argument(
        "--use_pisa",
        action="store_true",
        default=False,
        help="Enable PISA sparse attention (training-free, inference-only).",
    )
    group.add_argument(
        "--pisa_density",
        type=float,
        default=0.5,
        help="Fraction of attention blocks computed exactly. Lower = faster. (default: 0.5)",
    )
    group.add_argument(
        "--pisa_block_size",
        type=int,
        default=16,
        help="Token block size for piecewise decomposition. (default: 16)",
    )
    group.add_argument(
        "--pisa_start_layer_idx",
        type=int,
        default=2,
        help="Skip first N attention layers from PISA. (default: 2)",
    )
    return parser
