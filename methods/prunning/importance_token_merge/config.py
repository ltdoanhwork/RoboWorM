from __future__ import annotations

from argparse import ArgumentParser


def add_itm_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("Importance Token Merge")
    group.add_argument("--use_itm", action="store_true", default=False, help="Enable importance token merge.")
    group.add_argument(
        "--itm_compress_ratio",
        type=float,
        default=None,
        help="Fraction of tokens to merge/prune. Default: auto by token count.",
    )
    group.add_argument(
        "--itm_prune_from_step",
        type=int,
        default=1,
        help="First diffusion step that uses prune-only ITM.",
    )
    group.add_argument(
        "--itm_merge_from_step",
        type=int,
        default=2,
        help="First diffusion step that uses merge ITM.",
    )
    group.add_argument(
        "--itm_merge_attn",
        action="store_true",
        default=True,
        help="Apply ITM around self-attention. Enabled by default.",
    )
    group.add_argument(
        "--no_itm_merge_attn",
        action="store_false",
        dest="itm_merge_attn",
        help="Disable ITM around self-attention.",
    )
    group.add_argument(
        "--itm_merge_crossattn",
        action="store_true",
        default=False,
        help="Apply ITM around cross-attention.",
    )
    group.add_argument(
        "--itm_merge_mlp",
        action="store_true",
        default=False,
        help="Apply ITM around MLP blocks.",
    )
    group.add_argument(
        "--itm_start_layer_idx",
        type=int,
        default=None,
        help="First layer eligible for ITM. Default depends on model depth.",
    )
    group.add_argument(
        "--itm_keep_last_n_dense",
        type=int,
        default=2,
        help="Keep the last N layers dense even when ITM is enabled.",
    )
    group.add_argument(
        "--itm_max_downsample_ratio",
        type=int,
        default=4,
        help="Ctrl-World only: maximum eligible spatial downsample ratio.",
    )
    group.add_argument(
        "--itm_block_hold",
        action="store_true",
        default=False,
        help=(
            "Ctrl-World only: prune tokens ONCE at each BasicTransformerBlock "
            "input and keep them compressed through the whole block "
            "(self-attn + cross-attn + feed-forward), unmerging at the block "
            "output. This captures feed-forward savings (not just attention) "
            "and is the mode that actually makes ITM faster than dense."
        ),
    )
    group.add_argument(
        "--itm_plan_recompute_every",
        type=int,
        default=0,
        help=(
            "Ctrl-World block-hold only: reuse the merge plan and only recompute "
            "it every N attention calls per block (0 = every call). ITM's "
            "prepare() is ~12ms/block, so caching it (e.g. N=25 = once per "
            "interact step) is required for block-hold to beat dense."
        ),
    )
    group.add_argument(
        "--itm_spatiotemporal_hold",
        action="store_true",
        default=False,
        help=(
            "Ctrl-World only [EXPERIMENTAL, quality-lossy]: same spatio-temporal "
            "prune-once-hold as SiTo's --sito_spatiotemporal_hold, exposed on the "
            "ITM path. Prunes spatial tokens once at each "
            "TransformerSpatioTemporalModel and holds them compressed through the "
            "spatial AND temporal blocks. Real speedup (~1.04x at keep 0.8, "
            "~1.16x at keep 0.5) but PSNR drops (16.4 / 13.7)."
        ),
    )
    group.add_argument(
        "--itm_st_keep_ratio",
        type=float,
        default=0.8,
        help="ITM spatio-temporal hold: fraction of spatial tokens to keep. Default 0.8.",
    )
    return parser


def add_dreamgen_itm_args(parser: ArgumentParser) -> ArgumentParser:
    return add_itm_args(parser)
