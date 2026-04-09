"""
Attention mask utilities for SVG applied to Ctrl-World SVD UNet.

SVD UNet spatial self-attention layout:
  - 3 camera views stacked vertically in the latent: H=72, W=40 → T=2880
  - View 0: rows  0-23, View 1: rows 24-47, View 2: rows 48-71 (each 24 rows)

SVG1 online profiling compares 2 patterns per attention call and picks the
one with lower approximation MSE. We define:

  Pattern 0 – "spatial diagonal":
    Each token attends to tokens within a fixed bandwidth in the flat sequence.
    Captures local spatial correlation within a view.

  Pattern 1 – "view stripe":
    Each token attends to all tokens in the same column (W-position) across all
    3 stacked views. Captures cross-view semantic alignment.
"""

from __future__ import annotations

import math
from typing import List

import torch


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def get_svd_attention_masks(
    num_views: int = 3,
    H_per_view: int = 24,
    W: int = 40,
    sparsity: float = 0.25,
    sample_mse_max_row: int = 3000,
    device: str = "cuda",
) -> List[torch.Tensor]:
    """
    Build the two candidate attention masks used by SVG1 online profiling.

    Args:
        num_views:          Number of camera views stacked (default 3).
        H_per_view:         Latent height of a single view (default 24 = 192/8).
        W:                  Latent width (default 40 = 320/8).
        sparsity:           Target fraction of tokens attended (0 < sparsity ≤ 1).
        sample_mse_max_row: Cap on returned mask rows to avoid OOM.
        device:             Where to put the masks ('cuda' or 'cpu').

    Returns:
        List of two boolean tensors, each of shape [min(T, sample_mse_max_row), T].
    """
    H = num_views * H_per_view
    T = H * W

    band_width = max(1, int(T * sparsity))

    mask0 = _build_diagonal_mask(T, band_width)         # spatial locality
    mask1 = _build_view_stripe_mask(num_views, H_per_view, W)  # cross-view

    # Trim rows to avoid building T×T tensors on small GPUs
    max_row = min(T, sample_mse_max_row)
    mask0 = mask0[:max_row].to(device)
    mask1 = mask1[:max_row].to(device)

    return [mask0, mask1]


def sparsity_to_bandwidth(T: int, sparsity: float) -> int:
    """
    Convert a target sparsity fraction into a diagonal bandwidth.
    sparsity = fraction of sequence that IS attended (density).
    """
    return max(1, int(T * sparsity))


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_diagonal_mask(T: int, band_width: int) -> torch.Tensor:
    """
    Boolean mask where token i attends to j iff |i - j| <= band_width // 2.
    Shape: [T, T].
    """
    idx = torch.arange(T)
    diff = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()   # [T, T]
    return diff <= (band_width // 2)


def _build_view_stripe_mask(
    num_views: int,
    H_per_view: int,
    W: int,
) -> torch.Tensor:
    """
    Boolean mask where token i attends to j iff they share the same
    (y % H_per_view, x) position — i.e. same spatial position across views.

    For 3-view stacked latent (H=72, W=40, T=2880):
      - Each token has a row index 0-71 and col index 0-39.
      - Tokens are matched if (row % 24, col) are equal.

    This promotes cross-view semantic coherence during sparse attention.
    Shape: [T, T].
    """
    T = num_views * H_per_view * W
    idx = torch.arange(T)

    row = idx // W                  # overall row (0 to H-1)
    col = idx % W                   # column (0 to W-1)
    row_local = row % H_per_view    # position within a single view

    r_local_i = row_local.unsqueeze(1)   # [T, 1]
    c_i       = col.unsqueeze(1)          # [T, 1]
    r_local_j = row_local.unsqueeze(0)   # [1, T]
    c_j       = col.unsqueeze(0)          # [1, T]

    return (r_local_i == r_local_j) & (c_i == c_j)
