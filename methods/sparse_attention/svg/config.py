"""
Configuration helpers for Sparse-VideoGen (SVG).

This module keeps the legacy Ctrl-World `add_svg_args` surface and adds a
DreamGen-specific `add_dreamgen_svg_args` surface that matches the config-driven
backend integration used by PISA and Radial.
"""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class SVGConfig:
    use_svg: bool = False
    svg_pattern: str = "SVG1"
    svg_sparsity: float = 0.25
    svg_num_sampled_rows: int = 64
    svg_first_layers_fp: int = 2
    svg_sample_mse_max_row: int = 3000
    svg_num_q_centroids: int = 50
    svg_num_k_centroids: int = 200
    svg_top_p_kmeans: float = 0.9
    svg_min_kc_ratio: float = 0.0
    svg_kmeans_iter_init: int = 5
    svg_kmeans_iter_step: int = 2


def add_svg_args(parser: ArgumentParser) -> None:
    g = parser.add_argument_group("SVG sparse attention (optional, training-free)")

    g.add_argument(
        "--use_svg",
        action="store_true",
        default=False,
        help="Enable Sparse-VideoGen sparse attention.",
    )
    g.add_argument(
        "--svg_pattern",
        type=str,
        default="SVG1",
        choices=["SVG1", "SVG2"],
        help="SVG1: stripe sparse + MSE profiling. SVG2: semantic-aware permutation (needs flashinfer).",
    )
    g.add_argument(
        "--svg_sparsity",
        type=float,
        default=0.25,
        help="Fraction of tokens attended per row [0,1]. Lower = faster. Default 0.25.",
    )
    g.add_argument(
        "--svg_num_sampled_rows",
        type=int,
        default=64,
        help="Rows sampled for MSE profiling per attention call (SVG1 only).",
    )
    g.add_argument(
        "--svg_first_layers_fp",
        type=int,
        default=2,
        help="First N layers use full SDPA regardless. Default 2.",
    )
    g.add_argument(
        "--svg_num_q_centroids",
        type=int,
        default=50,
        help="K-Means query centroids (SVG2/SAP only).",
    )
    g.add_argument(
        "--svg_num_k_centroids",
        type=int,
        default=200,
        help="K-Means key centroids (SVG2/SAP only).",
    )
    g.add_argument(
        "--svg_top_p_kmeans",
        type=float,
        default=0.9,
        help="Top-p threshold for block selection (SVG2/SAP only).",
    )
    g.add_argument(
        "--svg_kmeans_iter_init",
        type=int,
        default=5,
        help="K-Means iterations on first step (SVG2/SAP only).",
    )
    g.add_argument(
        "--svg_kmeans_iter_step",
        type=int,
        default=2,
        help="K-Means iterations on subsequent steps (SVG2/SAP only).",
    )


def add_dreamgen_svg_args(parser: ArgumentParser) -> ArgumentParser:
    group = parser.add_argument_group("SVG sparse attention")
    group.add_argument(
        "--use_svg",
        action="store_true",
        default=False,
        help="Enable Sparse-VideoGen sparse self-attention in DreamGen.",
    )
    group.add_argument(
        "--svg_variant",
        type=str,
        default="svg1",
        choices=["svg1", "svg2"],
        help="SVG backend variant. svg1 uses sparse profile routing; svg2 uses semantic-aware permutation.",
    )
    group.add_argument(
        "--svg_start_layer_idx",
        type=int,
        default=4,
        help="Skip the first N self-attention blocks before enabling SVG.",
    )
    group.add_argument(
        "--svg_dense_step_frac",
        type=float,
        default=0.1,
        help="Fraction of early denoising steps that stay dense before SVG activates.",
    )
    group.add_argument(
        "--svg_sparsity",
        type=float,
        default=0.25,
        help="SVG1 sparsity ratio. Lower is sparser.",
    )
    group.add_argument(
        "--svg_num_sampled_rows",
        type=int,
        default=64,
        help="SVG1 online-profiling sample rows.",
    )
    group.add_argument(
        "--svg_sample_mse_max_row",
        type=int,
        default=3000,
        help="SVG1 max row range when sampling MSE profiling rows.",
    )
    group.add_argument(
        "--svg_num_q_centroids",
        type=int,
        default=50,
        help="SVG2 query K-Means centroid count.",
    )
    group.add_argument(
        "--svg_num_k_centroids",
        type=int,
        default=200,
        help="SVG2 key K-Means centroid count.",
    )
    group.add_argument(
        "--svg_top_p_kmeans",
        type=float,
        default=0.9,
        help="SVG2 top-p threshold when selecting sparse centroid blocks.",
    )
    group.add_argument(
        "--svg_min_kc_ratio",
        type=float,
        default=0.0,
        help="SVG2 minimum fraction of key centroids preserved per query centroid.",
    )
    group.add_argument(
        "--svg_kmeans_iter_init",
        type=int,
        default=5,
        help="SVG2 K-Means iterations for the first sparse denoise step.",
    )
    group.add_argument(
        "--svg_kmeans_iter_step",
        type=int,
        default=2,
        help="SVG2 K-Means iterations for subsequent sparse denoise steps.",
    )
    return parser
