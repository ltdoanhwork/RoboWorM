"""
SVG Adapter for Ctrl-World SVD UNet.

Public API:
    enable_svg(unet, ...)   — inject SVG attention (SVG1 or SVG2)
    disable_svg(unet)       — restore default diffusers SDPA processor

Design rules (same as PISA adapter):
  - Core model files are NOT modified.
  - Injection is done via unet.set_attn_processor() after model load.
  - SVG source (Sparse-VideoGen/) is added to sys.path on demand.
  - disable_svg restores unet.set_default_attn_processor() cleanly.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Optional

# Path to the upstream Sparse-VideoGen package
_SVG_SRC = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "Sparse-VideoGen")
)


def _ensure_svg_on_path() -> None:
    """Add Sparse-VideoGen to sys.path so `import svg` works."""
    if _SVG_SRC not in sys.path:
        sys.path.insert(0, _SVG_SRC)


def enable_svg(
    unet,
    *,
    pattern: str = "SVG1",
    # Geometry — MUST match actual inference shapes
    num_views: int = 3,
    H_per_view: int = 24,          # 192 px / 8 = 24
    W: int = 40,                   # 320 px / 8 = 40
    # SVG1 params
    sparsity: float = 0.25,
    num_sampled_rows: int = 64,
    first_layers_fp: int = 2,
    sample_mse_max_row: int = 3000,
    plan_recompute_every: int = 0,
    # SVG2 / SAP params
    num_q_centroids: int = 50,
    num_k_centroids: int = 200,
    top_p_kmeans: float = 0.9,
    min_kc_ratio: float = 0.0,
    kmeans_iter_init: int = 5,
    kmeans_iter_step: int = 2,
    logging_file: Optional[str] = None,
) -> None:
    """
    Inject SVG sparse attention into all diffusers Attention layers of `unet`.

    Args:
        unet:               The SVD UNet module (UNetSpatioTemporalConditionModel).
        pattern:            "SVG1" (stripe sparse) or "SVG2" (SAP/K-Means).
        num_views:          Camera views stacked vertically (default 3).
        H_per_view:         Latent height per view in pixels/8 (default 24).
        W:                  Latent width in pixels/8 (default 40).
        sparsity:           SVG1 only — fraction of tokens attended per row.
        num_sampled_rows:   SVG1 only — rows sampled for MSE profiling.
        first_layers_fp:    First N layers kept at full SDPA.
        sample_mse_max_row: Cap on precomputed mask rows (memory control).
        num_q_centroids:    SVG2 only — K-Means clusters for queries.
        num_k_centroids:    SVG2 only — K-Means clusters for keys.
        top_p_kmeans:       SVG2 only — top-p block selection threshold.
        min_kc_ratio:       SVG2 only — min fraction of key blocks per query block.
        kmeans_iter_init:   SVG2 only — K-Means iters on first step.
        kmeans_iter_step:   SVG2 only — K-Means iters on subsequent steps.
        logging_file:       SVG2 only — optional path to write density logs.
    """
    _ensure_svg_on_path()

    # ── pick processor class ────────────────────────────────────────────────
    if pattern == "SVG1":
        from .processor_svg1 import SVDSVGAttnProcessor as ProcessorCls
    elif pattern == "SVG2":
        from .processor_svg2 import SVDSAPAttnProcessor as ProcessorCls
    else:
        raise ValueError(f"[SVG] Unknown pattern '{pattern}'. Choose 'SVG1' or 'SVG2'.")

    # ── set class-level shared state ────────────────────────────────────────
    # Masks are NOT precomputed here — they are built lazily per T value
    # inside processor._get_masks_for_T() on the first call for each layer size.
    ProcessorCls.sparsity = sparsity
    ProcessorCls.num_sampled_rows = num_sampled_rows
    ProcessorCls.first_layers_fp = first_layers_fp
    # Hold the SVG1 MSE mask-selection plan after the first call (prune-once-hold).
    # SVG1 only; harmless attribute on SVG2.
    ProcessorCls.plan_recompute_every = plan_recompute_every
    # Geometry: stored so the processor can reconstruct the view-stripe mask
    # for T=T_full and a fallback wider band for smaller T.
    ProcessorCls.num_views = num_views
    ProcessorCls.H_per_view = H_per_view
    ProcessorCls.W = W

    if pattern == "SVG2":
        ProcessorCls.num_q_centroids = num_q_centroids
        ProcessorCls.num_k_centroids = num_k_centroids
        ProcessorCls.top_p_kmeans = top_p_kmeans
        ProcessorCls.min_kc_ratio = min_kc_ratio
        ProcessorCls.kmeans_iter_init = kmeans_iter_init
        ProcessorCls.kmeans_iter_step = kmeans_iter_step
        ProcessorCls.logging_file = logging_file

    # ── build per-layer processor and inject via named_modules ──────────────
    # NOT unet.attn_processors — that property returns EMPTY on the Ctrl-World
    # custom UNet, so the old code injected 0 processors and ran pure dense
    # (verified via CW_ATTN_PROFILE). named_modules finds every Attention module.
    attn_modules = [
        (name, module) for name, module in unet.named_modules()
        if hasattr(module, "processor") and hasattr(module, "to_q")
    ]
    num_layers = len(attn_modules)
    for idx, (name, module) in enumerate(attn_modules):
        module.processor = ProcessorCls(layer_idx=idx, num_layers=num_layers)

    # ── summary ─────────────────────────────────────────────────────────────
    H_total = num_views * H_per_view
    T_l0 = H_total * W
    print(
        f"[SVG] Enabled {pattern} on {num_layers} attention layers.\n"
        f"[SVG] Config: sparsity={sparsity}, first_layers_fp={first_layers_fp}/{num_layers}, "
        f"L0 T={T_l0} ({num_views}×{H_per_view}×{W} stacked views)\n"
        f"[SVG] Mask patterns: diagonal-band + view-stripe"
    )
    if pattern == "SVG2":
        print(
            f"[SVG] SAP: Q-centroids={num_q_centroids}, K-centroids={num_k_centroids}, "
            f"top_p={top_p_kmeans}"
        )


def disable_svg(unet) -> None:
    """Remove SVG processors and restore default diffusers SDPA."""
    unet.set_default_attn_processor()
    print("[SVG] Disabled — restored default SDPA processor.")
