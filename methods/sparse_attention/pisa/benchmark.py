"""
Standalone benchmark: PISA vs SDPA on SVD attention dimensions.

Run without loading any model checkpoint:
    python methods/sparse_attention/pisa/benchmark.py

Optional args:
    --device cuda:0
    --dtype bfloat16  (or float16, float32)
    --n_iter 100
    --density 0.5
    --block_size 16
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# ---- locate PISA source ------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PISA_SRC = os.path.normpath(os.path.join(_HERE, "..", "piecewise-sparse-attention"))
sys.path.insert(0, _PISA_SRC)

try:
    from piecewise_attn import piecewise_sparse_attention
    PISA_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] PISA not importable: {e}")
    PISA_AVAILABLE = False

# -----------------------------------------------------------------------

SVD_SHAPES = [
    # (label, B, H, T, D_head)   — typical shapes in Ctrl-World SVD UNet
    ("temporal_attn (T=25)",    2, 8,   25, 64),
    ("spatial_attn (T=60)",     2, 8,   60, 64),
    ("spatial_attn (T=240)",    2, 8,  240, 64),   # 2x resolution
    ("spatial_attn (T=960)",    2, 8,  960, 64),   # 4x resolution / mid-block
    ("spatial_attn (T=2048)",   1, 8, 2048, 64),   # hypothetical large
]


def bench_shape(label, B, H, T, D_head, density, block_size, device, dtype, n_iter):
    q = torch.randn(B, H, T, D_head, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    results = {}

    # ---- SDPA baseline ------------------------------------------------
    # warmup
    for _ in range(5):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        _ = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    results["SDPA"] = (time.perf_counter() - t0) / n_iter * 1000  # ms

    # ---- PISA ---------------------------------------------------------
    if PISA_AVAILABLE:
        bs = min(block_size, T)
        try:
            for _ in range(5):  # warmup
                _ = piecewise_sparse_attention(q, k, v, density=density, block_size=bs)
            torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(n_iter):
                _ = piecewise_sparse_attention(q, k, v, density=density, block_size=bs)
            torch.cuda.synchronize()
            results["PISA"] = (time.perf_counter() - t0) / n_iter * 1000
        except Exception as e:
            results["PISA"] = float("nan")
            results["PISA_error"] = str(e)
    else:
        results["PISA"] = float("nan")

    return results


def main():
    parser = argparse.ArgumentParser(description="PISA vs SDPA benchmark")
    parser.add_argument("--device", default="cuda", help="PyTorch device (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--n_iter", type=int, default=100, help="Iterations per shape")
    parser.add_argument("--density", type=float, default=0.5,
                        help="PISA density (fraction of exact blocks)")
    parser.add_argument("--block_size", type=int, default=16, help="PISA block size")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = args.device

    if not torch.cuda.is_available():
        print("[WARNING] CUDA not available — skipping GPU benchmark.")
        return

    gpu_name = torch.cuda.get_device_name(device)
    print(f"\n{'='*70}")
    print(f"  PISA vs SDPA Benchmark")
    print(f"  GPU     : {gpu_name}")
    print(f"  dtype   : {args.dtype}")
    print(f"  density : {args.density}  |  block_size : {args.block_size}")
    print(f"  n_iter  : {args.n_iter}")
    print(f"{'='*70}")
    print(f"{'Shape':<35} {'SDPA (ms)':>10} {'PISA (ms)':>10} {'Speedup':>10}")
    print(f"{'-'*70}")

    for label, B, H, T, D_head in SVD_SHAPES:
        res = bench_shape(
            label, B, H, T, D_head,
            args.density, args.block_size,
            device, dtype, args.n_iter,
        )
        sdpa_ms = res["SDPA"]
        pisa_ms = res.get("PISA", float("nan"))
        if not (pisa_ms != pisa_ms):  # not NaN
            speedup = f"{sdpa_ms / pisa_ms:.2f}x"
        else:
            speedup = "N/A"
            if "PISA_error" in res:
                speedup = f"ERR: {res['PISA_error'][:30]}"
        print(f"{label:<35} {sdpa_ms:>10.3f} {pisa_ms:>10.3f} {speedup:>10}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
