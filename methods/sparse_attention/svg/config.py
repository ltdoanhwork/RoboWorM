"""
Configuration helpers for Sparse-VideoGen (SVG).

This module keeps the legacy Ctrl-World `add_svg_args` surface and adds a
DreamGen-specific `add_dreamgen_svg_args` surface that matches the config-driven
backend integration used by PISA and Radial.
"""

from __future__ import annotations

import math
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any


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


def _dreamdojo_svg_defaults(num_blocks: int, variant: str) -> dict[str, Any]:
    if num_blocks <= 0:
        raise ValueError(f"`num_blocks` must be positive, got {num_blocks}.")
    if variant not in {"svg1", "svg2"}:
        raise ValueError(f"`variant` must be 'svg1' or 'svg2', got {variant!r}.")

    if variant == "svg1":
        return {
            "start_layer_idx": math.ceil(0.30 * num_blocks),
            "dense_step_frac": 0.20,
            "sparsity": 0.40,
            "num_sampled_rows": 32,
            "sample_mse_max_row": 1200,
        }

    return {
        "start_layer_idx": math.ceil(0.35 * num_blocks),
        "dense_step_frac": 0.25,
        "num_q_centroids": 24,
        "num_k_centroids": 96,
        "top_p_kmeans": 0.90,
        "min_kc_ratio": 0.10,
        "kmeans_iter_init": 8,
        "kmeans_iter_step": 2,
    }


def build_dreamdojo_svg_parameters(
    num_blocks: int,
    *,
    variant: str,
    start_layer_idx: int | None = None,
    dense_step_frac: float | None = None,
    sparsity: float | None = None,
    num_sampled_rows: int | None = None,
    sample_mse_max_row: int | None = None,
    num_q_centroids: int | None = None,
    num_k_centroids: int | None = None,
    top_p_kmeans: float | None = None,
    min_kc_ratio: float | None = None,
    kmeans_iter_init: int | None = None,
    kmeans_iter_step: int | None = None,
) -> list[dict[str, Any] | None]:
    defaults = _dreamdojo_svg_defaults(num_blocks=num_blocks, variant=variant)

    start_layer_idx = defaults["start_layer_idx"] if start_layer_idx is None else start_layer_idx
    dense_step_frac = defaults["dense_step_frac"] if dense_step_frac is None else dense_step_frac

    if start_layer_idx < 0:
        raise ValueError(f"`start_layer_idx` must be non-negative, got {start_layer_idx}.")
    if not 0 <= dense_step_frac < 1:
        raise ValueError(f"`dense_step_frac` must be in [0, 1), got {dense_step_frac}.")

    parameters: list[dict[str, Any] | None] = []
    for layer_idx in range(num_blocks):
        if layer_idx < start_layer_idx:
            parameters.append(None)
            continue

        params: dict[str, Any] = {
            "variant": variant,
            "layer_idx": layer_idx,
            "dense_step_frac": dense_step_frac,
        }
        if variant == "svg1":
            resolved_sparsity = defaults["sparsity"] if sparsity is None else sparsity
            resolved_num_sampled_rows = defaults["num_sampled_rows"] if num_sampled_rows is None else num_sampled_rows
            resolved_sample_mse_max_row = (
                defaults["sample_mse_max_row"] if sample_mse_max_row is None else sample_mse_max_row
            )
            if not 0 < resolved_sparsity <= 1:
                raise ValueError(f"`sparsity` must be in (0, 1], got {resolved_sparsity}.")
            if resolved_num_sampled_rows <= 0:
                raise ValueError(
                    f"`num_sampled_rows` must be positive, got {resolved_num_sampled_rows}."
                )
            if resolved_sample_mse_max_row <= 0:
                raise ValueError(
                    f"`sample_mse_max_row` must be positive, got {resolved_sample_mse_max_row}."
                )
            params.update(
                {
                    "sparsity": resolved_sparsity,
                    "num_sampled_rows": resolved_num_sampled_rows,
                    "sample_mse_max_row": resolved_sample_mse_max_row,
                }
            )
        else:
            resolved_num_q_centroids = defaults["num_q_centroids"] if num_q_centroids is None else num_q_centroids
            resolved_num_k_centroids = defaults["num_k_centroids"] if num_k_centroids is None else num_k_centroids
            resolved_top_p_kmeans = defaults["top_p_kmeans"] if top_p_kmeans is None else top_p_kmeans
            resolved_min_kc_ratio = defaults["min_kc_ratio"] if min_kc_ratio is None else min_kc_ratio
            resolved_kmeans_iter_init = (
                defaults["kmeans_iter_init"] if kmeans_iter_init is None else kmeans_iter_init
            )
            resolved_kmeans_iter_step = (
                defaults["kmeans_iter_step"] if kmeans_iter_step is None else kmeans_iter_step
            )
            if resolved_num_q_centroids <= 0 or resolved_num_k_centroids <= 0:
                raise ValueError("Centroid counts must be positive for DreamDojo SVG2.")
            if not 0 < resolved_top_p_kmeans <= 1:
                raise ValueError(
                    f"`top_p_kmeans` must be in (0, 1], got {resolved_top_p_kmeans}."
                )
            if not 0 <= resolved_min_kc_ratio <= 1:
                raise ValueError(
                    f"`min_kc_ratio` must be in [0, 1], got {resolved_min_kc_ratio}."
                )
            if resolved_kmeans_iter_init < 0 or resolved_kmeans_iter_step < 0:
                raise ValueError("K-Means iteration counts must be non-negative for DreamDojo SVG2.")
            params.update(
                {
                    "num_q_centroids": resolved_num_q_centroids,
                    "num_k_centroids": resolved_num_k_centroids,
                    "top_p_kmeans": resolved_top_p_kmeans,
                    "min_kc_ratio": resolved_min_kc_ratio,
                    "kmeans_iter_init": resolved_kmeans_iter_init,
                    "kmeans_iter_step": resolved_kmeans_iter_step,
                }
            )
        parameters.append(params)

    return parameters
