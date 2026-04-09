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
    
