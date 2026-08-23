import argparse


def add_radial_args(parser: argparse.ArgumentParser) -> None:
    """Add Radial Attention CLI arguments to an existing parser."""
    try:
        group = parser.add_argument_group("Radial Attention Options")
    except ValueError:
        # If the group already exists, the user's parser implementation might raise.
        group = parser

    group.add_argument(
        "--use_radial",
        action="store_true",
        help="Enable Radial Attention in the SVD UNet."
    )
    group.add_argument(
        "--radial_decay_factor",
        type=float,
        default=1.0,
        help="Decay factor for radial mask."
    )
    group.add_argument(
        "--radial_block_size",
        type=int,
        default=64,
        help="Block size for Radial Attention (default 64, flashinfer relies on this)."
    )
    group.add_argument(
        "--radial_pad_small_layers",
        action="store_true",
        default=True,
        help="Pad small non-divisible layers to be strictly sparse. If unset, fall back to SDPA."
    )
    group.add_argument(
        "--radial_first_layers_fp",
        type=int,
        default=2,
        help="Keep the first N layers at full precision / full attention."
    )
    group.add_argument(
        "--radial_num_views",
        type=int,
        default=3,
        help="Number of stacked views in the UNet spatial frame (for SVD, usually 3)."
    )
    group.add_argument(
        "--radial_h_per_view",
        type=int,
        default=24,
        help="Latent height per view (for 192px height, 192/8 = 24)."
    )
    group.add_argument(
        "--radial_w",
        type=int,
        default=40,
        help="Latent width per view (for 320px width, 320/8 = 40)."
    )
    group.add_argument(
        "--radial_model_type",
        type=str,
        default="hunyuan",
        choices=["hunyuan", "wan"],
        help="Underlying radial sparsity mask strategy (wan has larger sink/window). "
    )


def add_dreamgen_radial_args(parser: argparse.ArgumentParser) -> None:
    """Add DreamGen-focused Radial backend flags to an argparse parser."""
    group = parser.add_argument_group("Radial sparse attention")
    group.add_argument(
        "--use_radial",
        action="store_true",
        default=False,
        help="Enable Radial sparse self-attention in DreamGen (inference-only, single-GPU only).",
    )
    group.add_argument(
        "--radial_decay_factor",
        type=float,
        default=1.0,
        help="Decay factor for the Radial sparsity mask. (default: 1.0)",
    )
    group.add_argument(
        "--radial_block_size",
        type=int,
        default=64,
        choices=[64, 128],
        help="Radial block size. Only 64 and 128 are supported. (default: 64)",
    )
    group.add_argument(
        "--radial_start_layer_idx",
        type=int,
        default=4,
        help="Skip the first N self-attention blocks before enabling Radial. (default: 4)",
    )
    group.add_argument(
        "--radial_model_type",
        type=str,
        default="hunyuan",
        choices=["hunyuan", "wan"],
        help="Radial mask profile to use for DreamGen video self-attention. (default: hunyuan)",
    )
    return parser


def build_dreamdojo_radial_parameters(
    num_blocks: int,
    *,
    decay_factor: float,
    block_size: int,
    start_layer_idx: int,
    model_type: str = "hunyuan",
):
    """Per-block Radial parameter list for the DreamDojo config-driven backend."""
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    if decay_factor <= 0:
        raise ValueError(f"`decay_factor` must be positive, got {decay_factor}.")
    if block_size not in (64, 128):
        raise ValueError(f"`block_size` must be 64 or 128, got {block_size}.")
    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if model_type not in {"hunyuan", "wan"}:
        raise ValueError(f"`model_type` must be 'hunyuan' or 'wan', got {model_type!r}.")

    return [
        None
        if layer_idx < start_layer_idx
        else {
            "decay_factor": decay_factor,
            "block_size": block_size,
            "model_type": model_type,
        }
        for layer_idx in range(num_blocks)
    ]
