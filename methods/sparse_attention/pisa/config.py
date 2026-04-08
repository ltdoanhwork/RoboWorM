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

    Tuned for Ctrl-World SVD UNet (72×40 latent, 3-view stacked):
      - L0 spatial self-attn: T=2880  ← main bottleneck, block_size=64 gives 45 blocks
      - L1 spatial self-attn: T=720
      - L2 spatial self-attn: T=180
      - Temporal / cross-attn: automatically skipped by processor

    Fields
    ------
    use_pisa : bool
        Master switch. Set to True to enable PISA at inference time.
    pisa_density : float
        Fraction of attention blocks computed with exact attention.
        Range (0, 1]. Lower = faster but less accurate.
        Default 0.4: ~2.5-3× speedup with low quality degradation.
    pisa_block_size : int
        Token block size for piecewise decomposition.
        2880 / 64 = 45 blocks at L0 — a good trade-off for SVD.
        Must be ≤ sequence length at runtime (auto-clamped).
    pisa_start_layer_idx : int
        Skip the first N attention layers.
        0 = apply PISA from L0 (the bottleneck), which is recommended.
    """
    use_pisa: bool = False
    pisa_density: float = 0.4       # conservative default: speed vs quality sweet spot
    pisa_block_size: int = 64       # 2880/64=45 blocks at L0 (72×40 latent)
    pisa_start_layer_idx: int = 0   # apply from L0 — that's where the bottleneck is


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
