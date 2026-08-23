"""Diagnose why SVG1/SVG2 are slower than dense on DreamDojo humanoid singleview.

Measures, at the EXACT production shape (B=1, H=16, S=4800, T=4, H=30, W=40):
  - dense flash attention cost (the baseline the sparse path must beat)
  - the SVG1 block-mask density actually produced by svg_sparsity
  - the cost of each stage of the SVG1 runtime (rearrange / MSE profile /
    head placement / flex_attention / output placement)

Run: python scripts/_diag_dreamdojo_svg.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_LOGS", "")

B, H, S, D = 1, 16, 4800, 128
T, HH, WW = 4, 30, 40
FRAME_SIZE = HH * WW
DEV = "cuda"
DTYPE = torch.bfloat16


def _bench(fn, warmup=3, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device=DEV, dtype=DTYPE)
    k = torch.randn(B, S, H, D, device=DEV, dtype=DTYPE)
    v = torch.randn(B, S, H, D, device=DEV, dtype=DTYPE)

    print(f"shape: B={B} H={H} S={S} D={D} (T={T}, H={HH}, W={WW}, frame_size={FRAME_SIZE})")

    # ---- dense baseline (what the model actually uses) ----
    from cosmos_predict2._src.predict2.networks.attention import attention as dense_attn

    dense_ms = _bench(lambda: dense_attn(q, k, v))
    print(f"\ndense (flash, bshd)            : {dense_ms:.3f} ms   <-- must beat this")

    # ---- SVG1 mask geometry ----
    from methods.sparse_attention.svg.dreamgen_runtime import (
        _sparsity_to_width,
        _generate_temporal_head_mask_mod,
    )
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    print("\n--- SVG1 block-mask density vs svg_sparsity ---")
    for sp in (0.05, 0.10, 0.15, 0.25, 0.35):
        width = _sparsity_to_width(sp, 0, T, FRAME_SIZE)
        mask_mod = _generate_temporal_head_mask_mod(0, T, FRAME_SIZE, width)
        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=DEV)
        # fraction of 128x128 blocks kept
        dens = bm.sparsity()
        print(f"  sparsity={sp:<5} -> diag_width={width:6.3f} frames, blocks KEPT={100 - dens:5.1f}%")

    # ---- cost of flex_attention itself at the tuned point ----
    print("\n--- where SVG1's 2.75 ms/call goes (sparsity=0.25) ---")
    width = _sparsity_to_width(0.25, 0, T, FRAME_SIZE)
    mask_mod = _generate_temporal_head_mask_mod(0, T, FRAME_SIZE, width)
    block_mask = create_block_mask(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=DEV)

    compiled_flex = torch.compile(
        lambda a, b, c, m: flex_attention(a, b, c, block_mask=m), backend="inductor", fullgraph=False
    )

    qb = q.transpose(1, 2).contiguous()
    kb = k.transpose(1, 2).contiguous()
    vb = v.transpose(1, 2).contiguous()

    lay_ms = _bench(lambda: (q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()))
    print(f"  3x rearrange->contiguous     : {lay_ms:.3f} ms")

    flex_ms = _bench(lambda: compiled_flex(qb, kb, vb, block_mask))
    print(f"  compiled flex_attention      : {flex_ms:.3f} ms")

    # MSE profile cost
    runtime_cls = None
    from methods.sparse_attention.svg.dreamgen_runtime import DreamGenSVG1Runtime

    rt = DreamGenSVG1Runtime(sparsity=0.25, num_sampled_rows=16, sample_mse_max_row=1200)
    masks = rt._attention_masks(seq_len=S, num_frame=T, frame_size=FRAME_SIZE, device=torch.device(DEV))
    mse_ms = _bench(lambda: rt._sample_mse_profile(qb, kb, vb, attention_masks=masks))
    print(f"  _sample_mse_profile (per call): {mse_ms:.3f} ms")

    # head placement cost
    from methods.sparse_attention.svg.dreamgen_runtime import _load_svg1_modules

    placement_mod = _load_svg1_modules()["placement_mod"]
    best = rt._sample_mse_profile(qb, kb, vb, attention_masks=masks)
    qo, ko, vo = torch.empty_like(qb), torch.empty_like(kb), torch.empty_like(vb)
    oh = torch.empty_like(qb)

    def _place():
        placement_mod.cosmos_sparse_head_placement(qb, kb, vb, qo, ko, vo, best, 0, T, FRAME_SIZE)

    place_ms = _bench(_place)
    print(f"  cosmos_sparse_head_placement : {place_ms:.3f} ms")

    def _unplace():
        placement_mod.cosmos_hidden_states_placement(qb, oh, best, 0, T, FRAME_SIZE)

    unplace_ms = _bench(_unplace)
    print(f"  cosmos_hidden_states_placement: {unplace_ms:.3f} ms")

    total = lay_ms + flex_ms + mse_ms + place_ms + unplace_ms
    print(f"\n  SUM of SVG1 stages           : {total:.3f} ms  vs dense {dense_ms:.3f} ms")
    print(f"  -> flex_attention alone is    : {flex_ms / dense_ms:.2f}x dense")
    print(f"  -> overhead (non-attn stages) : {total - flex_ms:.3f} ms ({(total - flex_ms) / dense_ms:.2f}x dense)")

    # ---- SVG2 stage breakdown ----
    print("\n--- where SVG2's 6.72 ms/call goes (q32/k64, top_p 0.6) ---")
    try:
        from methods.sparse_attention.svg.dreamgen_runtime import (
            DreamGenSVG2Runtime,
            _load_svg2_modules,
        )

        rt2 = DreamGenSVG2Runtime(
            num_q_centroids=32,
            num_k_centroids=64,
            top_p_kmeans=0.6,
            min_kc_ratio=0.15,
            kmeans_iter_init=8,
            kmeans_iter_step=2,
        )
        mods = _load_svg2_modules()
        kmeans_utils = mods["kmeans_utils"]
        permute_mod = mods["permute_mod"]

        bsz, nh, sl, hd = qb.shape
        qf = qb.reshape(bsz * nh, sl, hd)
        kf = kb.reshape(bsz * nh, sl, hd)

        # warm the centroids so we measure the steady-state iter_step cost
        rt2._cluster(qf, kf)
        km_ms = _bench(lambda: rt2._cluster(qf, kf), warmup=2, iters=10)
        print(f"  kmeans _cluster (warm)       : {km_ms:.3f} ms")

        ql, qc, qcs, kl, kc, kcs = rt2._cluster(qf, kf)
        dmap = kmeans_utils.identify_dynamic_map(
            qc.view(bsz, nh, 32, hd), kc.view(bsz, nh, 64, hd),
            qcs.view(bsz, nh, 32), kcs.view(bsz, nh, 64), 0.6, 0.15,
        )
        idm_ms = _bench(
            lambda: kmeans_utils.identify_dynamic_map(
                qc.view(bsz, nh, 32, hd), kc.view(bsz, nh, 64, hd),
                qcs.view(bsz, nh, 32), kcs.view(bsz, nh, 64), 0.6, 0.15,
            ), warmup=2, iters=10,
        )
        print(f"  identify_dynamic_map         : {idm_ms:.3f} ms")

        def _perm():
            permute_mod.permute_tensor_by_labels_triton(qb, ql.view(bsz, nh, sl), dim=2)
            permute_mod.permute_tensor_by_labels_triton(kb, kl.view(bsz, nh, sl), dim=2)

        perm_ms = _bench(_perm, warmup=2, iters=10)
        print(f"  2x permute_by_labels         : {perm_ms:.3f} ms")

        qp, qsi = permute_mod.permute_tensor_by_labels_triton(qb, ql.view(bsz, nh, sl), dim=2)
        kp, ksi = permute_mod.permute_tensor_by_labels_triton(kb, kl.view(bsz, nh, sl), dim=2)
        vp, _ = permute_mod.permute_tensor_by_labels_triton(vb, kl.view(bsz, nh, sl), dim=2, sorted_indices=ksi)

        bs_ms = _bench(
            lambda: kmeans_utils.dynamic_block_sparse_fwd_flashinfer(
                qp, kp, vp, dmap, qcs.view(bsz, nh, 32), kcs.view(bsz, nh, 64), is_cpu=False,
            ), warmup=2, iters=10,
        )
        print(f"  dynamic_block_sparse (attn)  : {bs_ms:.3f} ms")
        kept = float(dmap.float().mean().item()) * 100
        print(f"  -> dynamic_map blocks kept   : {kept:.1f}%")
        tot2 = km_ms + idm_ms + perm_ms + bs_ms + lay_ms
        print(f"\n  SUM of SVG2 stages           : {tot2:.3f} ms  vs dense {dense_ms:.3f} ms")
        print(f"  -> non-attn overhead         : {tot2 - bs_ms:.3f} ms ({(tot2 - bs_ms) / dense_ms:.2f}x dense)")
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"  SVG2 breakdown unavailable: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
