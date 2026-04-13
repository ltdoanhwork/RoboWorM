"""
Config dataclass for PISA sparse attention.

Can be mixed into wm_args or used standalone via argparse.
This file has zero dependency on any Ctrl-World model code.
"""

from dataclasses import dataclass
from typing import Any


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
    pisa_use_bias : bool
        Enable the auxiliary bias term used by the TMA-flavored kernel.
    """
    use_pisa: bool = False
    pisa_density: float = 0.4       # conservative default: speed vs quality sweet spot
    pisa_block_size: int = 64       # 2880/64=45 blocks at L0 (72×40 latent)
    pisa_start_layer_idx: int = 0   # apply from L0 — that's where the bottleneck is
    pisa_use_bias: bool = False


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
    group.add_argument(
        "--pisa_use_bias",
        action="store_true",
        default=False,
        help="Enable the bias-enhanced PISA kernel variant.",
    )
    return parser


def _dreamdojo_pisa_defaults(num_blocks: int) -> dict[str, Any]:
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")

    # DreamDojo teacher inference uses S=4800 tokens at 480x640 / 13 frames after VAE + patchify:
    # latent (T, H, W) = (4, 60, 80) -> patchify with patch_spatial=2 gives (4, 30, 40).
    # Flattening order is t-h-w, so one spatial row is 40 tokens. We use block_size=80 so each
    # chunk aligns to 2 contiguous rows, giving NT=60 chunks. This is a better fit than 128
    # for DreamDojo's shorter sequence length while keeping chunk-scoring overhead modest.
    if num_blocks <= 28:
        return {
            "density": 0.45,
            "block_size": 80,
            "start_layer_idx": 10,
            "use_bias": False,
        }

    return {
        "density": 0.40,
        "block_size": 80,
        "start_layer_idx": 12,
        "use_bias": False,
    }


def build_dreamdojo_pisa_parameters(
    num_blocks: int,
    *,
    density: float | None = None,
    block_size: int | None = None,
    start_layer_idx: int | None = None,
    use_bias: bool | None = None,
) -> list[dict[str, Any] | None]:
    defaults = _dreamdojo_pisa_defaults(num_blocks=num_blocks)

    resolved_density = defaults["density"] if density is None else density
    resolved_block_size = defaults["block_size"] if block_size is None else block_size
    resolved_start_layer_idx = defaults["start_layer_idx"] if start_layer_idx is None else start_layer_idx
    resolved_use_bias = defaults["use_bias"] if use_bias is None else use_bias

    if not 0 < resolved_density <= 1:
        raise ValueError(f"`density` must be in (0, 1], got {resolved_density}.")
    if resolved_block_size <= 0:
        raise ValueError(f"`block_size` must be positive, got {resolved_block_size}.")
    if resolved_start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {resolved_start_layer_idx}.")

    return [
        None
        if layer_idx < resolved_start_layer_idx
        else {
            "density": resolved_density,
            "block_size": resolved_block_size,
            "use_bias": resolved_use_bias,
        }
        for layer_idx in range(num_blocks)
    ]
