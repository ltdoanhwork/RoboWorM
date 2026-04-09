"""
Standalone benchmark: SVG1 vs SVG2 vs PISA vs SDPA on Ctrl-World attention shapes.

Run without loading any full model checkpoint:
    python methods/sparse_attention/svg/benchmark.py
    python methods/sparse_attention/svg/benchmark.py --pattern SVG2 --sparsity 0.3
"""

from __future__ import annotations

import argparse
import math
import sys
import os
import time
import warnings
from typing import List, Tuple

import torch
import torch.nn.functional as F

# ── Real Ctrl-World attention shapes ────────────────────────────────────────
# (label, B, num_heads, T, head_dim)
SVD_SHAPES: List[Tuple[str, int, int, int, int]] = [
    # Temporal self-attn (T=11) — expect SDPA fallback
    ("temporal (T=11)",               1,  8,   11,  64),
    # Spatial self-attn levels (3-view stacked, 72×40 latent)
    ("spatial L2 (18×10, T=180)",     1,  8,  180,  64),
    ("spatial L1 (36×20, T=720)",     1,  8,  720,  64),
    ("spatial L0 (72×40, T=2880) ←", 1,  8, 2880,  64),   # BOTTLENECK
]

WARMUP = 3
REPEATS = 10


def time_fn(fn, *args, warmup=WARMUP, repeats=REPEATS) -> float:
    """Return mean wall-clock ms over `repeats` runs."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000  # ms


def sdpa_attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def svg1_attn(q, k, v, masks, sparsity=0.25):
    """
    Masked SDPA with a precomputed diagonal-band boolean mask.
    Mirrors SVDSVGAttnProcessor._masked_sparse_attention().
    """
    T = q.shape[2]
    bandwidth = max(1, int(T * sparsity))
    idx = torch.arange(T, device=q.device)
    mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= bandwidth // 2  # [T, T]
    additive = torch.zeros(T, T, device=q.device, dtype=q.dtype).masked_fill(~mask, float("-inf"))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=additive, dropout_p=0.0)


def benchmark_shapes(pattern: str, sparsity: float, device: str):
    print(f"\n{'='*70}")
    print(f"SVG Benchmark  |  pattern={pattern}  |  sparsity={sparsity}  |  device={device}")
    print(f"{'='*70}")
    print(f"{'Shape':<40} {'SDPA (ms)':>10} {'SVG (ms)':>10} {'Speedup':>8}")
    print(f"{'-'*70}")

    for label, B, H, T, D_h in SVD_SHAPES:
        q = torch.randn(B, H, T, D_h, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        t_sdpa = time_fn(sdpa_attn, q, k, v)

        if T < 128:
            # SVG would also fallback here
            t_svg = t_sdpa
            label_suffix = " [fallback→SDPA]"
        elif pattern == "SVG1":
            t_svg = time_fn(svg1_attn, q, k, v, None, sparsity)
            label_suffix = ""
        else:
            # SVG2: approximate with SDPA (no kmeans available without full install)
            warnings.warn("SVG2 benchmark requires full SVG install + flashinfer. Showing SDPA.")
            t_svg = t_sdpa
            label_suffix = " [SVG2 needs flashinfer]"

        speedup = t_sdpa / t_svg if t_svg > 0 else float("inf")
        print(f"{label+label_suffix:<40} {t_sdpa:>10.2f} {t_svg:>10.2f} {speedup:>7.2f}×")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVG benchmark on Ctrl-World attention shapes")
    parser.add_argument("--pattern", type=str, default="SVG1", choices=["SVG1", "SVG2"])
    parser.add_argument("--sparsity", type=float, default=0.25,
                        help="Fraction of tokens attended per row (0,1]")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    benchmark_shapes(args.pattern, args.sparsity, args.device)
    print(f"\n{'='*70}")
    print("Done. Note: SVG1 benchmark uses diagonal-band masked SDPA only (no FlexAttention).")
    print("Install SVG package + flashinfer for full SVG2/SAP benchmark.")
