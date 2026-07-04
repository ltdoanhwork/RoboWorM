from __future__ import annotations

from argparse import ArgumentParser


def add_sito_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("SiTo sparse attention")
    group.add_argument(
        "--use_sito",
        action="store_true",
        default=False,
        help="Enable SiTo token pruning for self-attention at inference time.",
    )
    group.add_argument(
        "--sito_prune_ratio",
        type=float,
        default=None,
        help="Override SiTo prune ratio. Default: auto based on sequence length.",
    )
    group.add_argument(
        "--sito_patch_h",
        type=int,
        default=2,
        help="SiTo patch height used when choosing anchor tokens. Default: 2.",
    )
    group.add_argument(
        "--sito_patch_w",
        type=int,
        default=2,
        help="SiTo patch width used when choosing anchor tokens. Default: 2.",
    )
    group.add_argument(
        "--sito_noise_alpha",
        type=float,
        default=0.1,
        help="Noise mixing term used when selecting SiTo anchor tokens. Default: 0.1.",
    )
    group.add_argument(
        "--sito_sim_beta",
        type=float,
        default=1.0,
        help="Similarity weighting term used when selecting SiTo anchor tokens. Default: 1.0.",
    )
    group.add_argument(
        "--sito_start_layer_idx",
        type=int,
        default=None,
        help="First transformer/attention layer index eligible for SiTo. Default depends on model depth.",
    )
    group.add_argument(
        "--sito_keep_last_n_dense",
        type=int,
        default=2,
        help="Keep the last N eligible self-attention layers dense. Default: 2.",
    )
    group.add_argument(
        "--sito_group_mode",
        type=str,
        default="per_frame",
        choices=["per_frame", "temporal", "full_2d"],
        help=(
            "Token grouping strategy. 'per_frame' (default, v1) prunes spatially "
            "within each frame. 'temporal' prunes temporally-redundant tokens "
            "across the time/view axis and shares one prune-once plan across all "
            "blocks (recommended for DreamGen multiview). 'full_2d' for square grids."
        ),
    )
    group.add_argument(
        "--sito_max_downsample_ratio",
        type=int,
        default=4,
        help="Ctrl-World only: maximum spatial downsample ratio eligible for SiTo. Default: 4.",
    )
    group.add_argument(
        "--sito_plan_recompute_every",
        type=int,
        default=0,
        help=(
            "Ctrl-World only: reuse the SiTo prune plan and only recompute it "
            "every N attention calls per layer (0 = recompute every call, the "
            "original behavior). Values like 5 remove the expensive prepare() "
            "step from most denoise steps and are what makes SiTo actually "
            "faster than dense."
        ),
    )
    group.add_argument(
        "--sito_spatiotemporal_hold",
        action="store_true",
        default=False,
        help=(
            "Ctrl-World only [EXPERIMENTAL, quality-lossy]: prune spatial tokens "
            "ONCE at each TransformerSpatioTemporalModel and hold them "
            "compressed through BOTH the spatial and temporal blocks, then "
            "unmerge. This DOES speed up (~1.14x at keep=0.5) because it targets "
            "~38%% of UNet, BUT pruning before the temporal block corrupts the "
            "temporal dynamics: PSNR collapses (e.g. 9-17 vs ~21 dense) even at "
            "light pruning. Not recommended for quality-sensitive benchmarking; "
            "kept only as a measured data point."
        ),
    )
    group.add_argument(
        "--sito_st_keep_ratio",
        type=float,
        default=0.5,
        help="Spatio-temporal hold: fraction of spatial tokens to keep. Default 0.5.",
    )
    return parser


def add_dreamgen_sito_args(parser: ArgumentParser) -> ArgumentParser:
    return add_sito_args(parser)
