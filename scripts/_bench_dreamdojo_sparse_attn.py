"""Microbenchmark: dense SDPA vs PISA / Radial / SVG kernels at the DreamDojo
humanoid-singleview self-attention shape. Tuning helper only."""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "DreamDojo"))


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--t", type=int, default=4)
    p.add_argument("--h", type=int, default=30)
    p.add_argument("--w", type=int, default=40)
    args = p.parse_args()

    S = args.t * args.h * args.w
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"shape: B={args.batch} H={args.heads} S={S} D={args.head_dim} (T={args.t},H={args.h},W={args.w})")

    q_bhsd = torch.randn(args.batch, args.heads, S, args.head_dim, device=dev, dtype=dtype)
    k_bhsd = torch.randn_like(q_bhsd)
    v_bhsd = torch.randn_like(q_bhsd)
    q_bshd = q_bhsd.transpose(1, 2).contiguous()
    k_bshd = k_bhsd.transpose(1, 2).contiguous()
    v_bshd = v_bhsd.transpose(1, 2).contiguous()

    dense_ms = bench(lambda: F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd))
    print(f"dense SDPA          : {dense_ms:.3f} ms")

    from cosmos_predict2._src.predict2.networks.a2a_cp import PISALocalAttnOp

    for density in (0.5, 0.3, 0.2, 0.1, 0.05):
        op = PISALocalAttnOp(density=density, block_size=64, use_bias=False)
        try:
            ms = bench(lambda o=op: o(q_bshd, k_bshd, v_bshd))
            print(f"PISA-op density={density:<5} : {ms:.3f} ms   ({dense_ms / ms:.2f}x vs dense) [incl. layout]")
        except Exception as exc:  # noqa: BLE001
            print(f"PISA-op density={density}: FAILED {type(exc).__name__}: {exc}")

    from cosmos_predict2._src.predict2.networks.a2a_cp import _load_piecewise_sparse_attention

    pisa = _load_piecewise_sparse_attention()
    for density in (0.5, 0.3, 0.2, 0.1):
        try:
            ms = bench(
                lambda d=density: pisa(
                    q_bhsd, k_bhsd, v_bhsd, density=d, block_size=64, scale=None, use_bias=False
                )
            )
            print(f"PISA density={density:<4}    : {ms:.3f} ms   ({dense_ms / ms:.2f}x vs dense)")
        except Exception as exc:  # noqa: BLE001
            print(f"PISA density={density}: FAILED {type(exc).__name__}: {exc}")

    class VS:
        def __init__(self, T, H, W):
            self.T, self.H, self.W = T, H, W

    vs = VS(args.t, args.h, args.w)

    try:
        from cosmos_predict2._src.predict2.networks.a2a_cp import _load_radial_runtime_class

        for model_type in ("hunyuan", "wan"):
            rt = _load_radial_runtime_class()(block_size=64, decay_factor=1.0, model_type=model_type)
            ms = bench(lambda r=rt: r.forward(q_bshd, k_bshd, v_bshd, video_size=vs))
            print(f"Radial {model_type:<8}     : {ms:.3f} ms   ({dense_ms / ms:.2f}x vs dense)")
    except Exception as exc:  # noqa: BLE001
        print(f"Radial: FAILED {type(exc).__name__}: {exc}")

    try:
        from cosmos_predict2._src.predict2.networks.a2a_cp import _load_svg_runtime_class

        svg_cls = _load_svg_runtime_class()
        for sparsity in (0.25, 0.15, 0.10):
            rt = svg_cls(
                variant="svg1",
                sparsity=sparsity,
                num_sampled_rows=16,
                sample_mse_max_row=1200,
                dense_step_frac=0.0,
            )
            ms = bench(lambda r=rt: r.forward(q_bshd, k_bshd, v_bshd, video_size=vs), iters=10, warmup=3)
            print(f"SVG1 sparsity={sparsity:<4}   : {ms:.3f} ms   ({dense_ms / ms:.2f}x vs dense)")
    except Exception as exc:  # noqa: BLE001
        print(f"SVG1: FAILED {type(exc).__name__}: {exc}")

    try:
        svg_cls = _load_svg_runtime_class()
        for top_p in (0.9, 0.6):
            rt = svg_cls(
                variant="svg2",
                num_q_centroids=32,
                num_k_centroids=64,
                top_p_kmeans=top_p,
                min_kc_ratio=0.15,
                kmeans_iter_init=8,
                kmeans_iter_step=2,
                dense_step_frac=0.0,
            )
            q1 = q_bshd[:1].contiguous()
            k1 = k_bshd[:1].contiguous()
            v1 = v_bshd[:1].contiguous()
            dense1 = bench(lambda: F.scaled_dot_product_attention(q_bhsd[:1], k_bhsd[:1], v_bhsd[:1]))
            ms = bench(lambda r=rt: r.forward(q1, k1, v1, video_size=vs), iters=10, warmup=3)
            print(
                f"SVG2 top_p={top_p:<5}      : {ms:.3f} ms (B=1)  "
                f"({dense1 / ms:.2f}x vs dense B=1 {dense1:.3f} ms)"
            )
    except Exception as exc:  # noqa: BLE001
        print(f"SVG2: FAILED {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
