from __future__ import annotations


def add_fastercache_args(parser) -> None:
    parser.add_argument(
        "--use_fastercache",
        action="store_true",
        help="Enable FasterCache v1 acceleration.",
    )
    parser.add_argument(
        "--fastercache_start_step",
        type=int,
        default=0,
        help="First denoising step where FasterCache becomes eligible. Use 0 for ceil(0.3 * num_steps).",
    )
    parser.add_argument(
        "--fastercache_model_interval",
        type=int,
        default=5,
        help="Anchor interval for full model CFG passes.",
    )
    parser.add_argument(
        "--fastercache_block_interval",
        type=int,
        default=3,
        help="Anchor interval for block-level self-attention refresh.",
    )
