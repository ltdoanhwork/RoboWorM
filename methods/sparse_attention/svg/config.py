"""
Configuration for Sparse-VideoGen (SVG) adapter on Ctrl-World SVD UNet.

Two modes:
  SVG1 — Stripe sparse attention with online MSE profiling.
         Fast, no extra dependencies beyond svg package.
  SVG2 — Semantic Aware Permutation (SAP): K-Means clustering + block sparse.
         Better quality retention, requires flashinfer + cuvs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from argparse import ArgumentParser


@dataclass
class SVGConfig:
    """
    SVG sparse attention configuration — tuned for Ctrl-World.

    Ctrl-World SVD UNet attention shapes (72×40 latent, 3-view stacked):
      - L0 spatial self-attn: T=2880  ← main bottleneck
      - L1 spatial self-attn: T=720
      - L2 spatial self-attn: T=180
      - Temporal / cross-attn: automatically skipped by processor

    SVG1 params
    -----------
    use_svg : bool
        Master switch. Set True to enable SVG at inference.
    svg_pattern : str
        "SVG1" (stripe sparse + MSE profiling) or
        "SVG2" (semantic aware permutation, requires flashinfer).
    svg_sparsity : float
        Fraction of tokens attended per row (density). Range (0, 1].
        Lower = faster but less accurate. Default 0.25 ≈ 1.5-2× speedup.
    svg_num_sampled_rows : int
        Number of query rows sampled to estimate MSE per attention call.
        Higher = more accurate profiling but more overhead.
    svg_first_layers_fp : int
        First N attention layers use full SDPA (not sparse).
        Early layers serve as structural anchors; sparing them preserves quality.

    SVG2 additional params
    ----------------------
    svg_num_q_centroids : int
        K-Means clusters for query tokens.
    svg_num_k_centroids : int
        K-Means clusters for key tokens.
    svg_top_p_kmeans : float
        Top-p threshold for block selection (how many K-K pairs to keep).
    svg_min_kc_ratio : float
        Minimum fraction of key blocks to keep per query block.
    svg_kmeans_iter_init : int
        K-Means iterations on first denoising step.
    svg_kmeans_iter_step : int
        K-Means iterations on subsequent steps (warm-started).
    """

    use_svg: bool = False
    svg_pattern: str = "SVG1"                # "SVG1" or "SVG2"

    # SVG1 params
    svg_sparsity: float = 0.25               # 25% of tokens attended (75% sparse)
    svg_num_sampled_rows: int = 64           # MSE profiling: rows sampled per call
    svg_first_layers_fp: int = 2             # keep first 2 layers as full attn
    svg_sample_mse_max_row: int = 3000       # cap mask rows to avoid OOM

    # SVG2 params (SAP)
    svg_num_q_centroids: int = 50
    svg_num_k_centroids: int = 200
    svg_top_p_kmeans: float = 0.9
    svg_min_kc_ratio: float = 0.0
    svg_kmeans_iter_init: int = 5
    svg_kmeans_iter_step: int = 2


def add_svg_args(parser: ArgumentParser) -> None:
    """
    Register SVG CLI arguments onto an existing ArgumentParser.
    Mirror the style of add_pisa_args() for consistency.
    """
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
    # SVG1
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
    # SVG2
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
