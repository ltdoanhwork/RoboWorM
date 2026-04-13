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
    return parser


def add_dreamgen_itm_args(parser: ArgumentParser) -> ArgumentParser:
    return add_itm_args(parser)
