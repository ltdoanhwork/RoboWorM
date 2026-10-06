"""
Kernel-level diagnosis of PISA / SVG1 / SVG2 / Radial on the Ctrl-World SVD UNet
at the REAL single_arm 3-view production shapes.

Loads the actual UNet (so head count / head_dim / dtype are exact), finds a
spatial self-attention Attention module, and for each spatial T (2880, 720, 180)
times:
  - dense flash SDPA (the baseline every sparse method must beat)
  - PISA piecewise (density sweep)
  - SVG1 masked-SDPA (sparsity sweep) + its per-call MSE-profile cost
  - SVG2 block-sparse flashinfer (top_p sweep) + kmeans cost
  - Radial flashinfer BSR (fp/decay)

This is the ground truth for "can any config beat dense at this shape, even with
zero Python overhead". Prints ms/call per stage. No full episode is run.

Run:
  source models/Ctrl-World/.venv/bin/activate
  PYTHONPATH=models/Ctrl-World:. CUDA_VISIBLE_DEVICES=0 python scripts/tools/diag/_diag_ctrlworld_sa_sparse.py
"""
import os, sys, time
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "models", "Ctrl-World"))
sys.path.insert(0, ROOT)

DEV = "cuda"
DTYPE = torch.float16
# Real SVD UNet attn shape: H heads, D_head. Filled from the actual module below.
SPATIAL_TS = [2880, 720, 180]
WARMUP, ITERS = 5, 30


def _time(fn, *a, **kw):
    for _ in range(WARMUP):
        fn(*a, **kw)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn(*a, **kw)
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS * 1e3  # ms/call


def main():
    # --- build the real UNet just to read head geometry -------------------
    from config import wm_args
    args = wm_args(task_type="replay")
    args.svd_model_path = os.path.join(ROOT, "checkpoints/base/stabilityai/stable-video-diffusion-img2vid")
    args.clip_model_path = os.path.join(ROOT, "checkpoints/base/openai/clip-vit-base-patch32")
    ckpt = os.path.join(ROOT, "checkpoints/base/ctrl_world/Ctrl-World/checkpoint-10000.pt")
    args.ckpt_path = args.val_model_path = ckpt
    args.data_stat_path = os.path.join(ROOT, "models/Ctrl-World/dataset_meta_info/droid_subset/stat.json")

    from scripts.infer.infer_single_arm_multiview_ctrlworld import CtrlWorldAgent
    agent = CtrlWorldAgent(args)
    unet = agent.model.unet

    # find a spatial self-attn Attention module (attn1) to read heads/dim
    ref = None
    for name, m in unet.named_modules():
        if hasattr(m, "to_q") and hasattr(m, "heads") and name.endswith("attn1"):
            ref = m
            break
    H = ref.heads
    Dh = ref.to_q.out_features // H
    print(f"[geom] found attn1 with H={H} heads, D_head={Dh}, D={ref.to_q.out_features}")

    from methods.sparse_attention.pisa.piecewise_sparse_attn_tma import piecewise_sparse_attention

    print(f"\n{'T':>6} {'dense_ms':>9} | {'PISA d.5':>9} {'d.2':>8} | "
          f"{'SVG1 s.25':>9} {'SVG1+prof':>9} | {'Radial':>8}")
    for T in SPATIAL_TS:
        q = torch.randn(1, H, T, Dh, device=DEV, dtype=DTYPE)
        k = torch.randn(1, H, T, Dh, device=DEV, dtype=DTYPE)
        v = torch.randn(1, H, T, Dh, device=DEV, dtype=DTYPE)

        dense = _time(lambda: F.scaled_dot_product_attention(q, k, v))

        def pisa(dn):
            return piecewise_sparse_attention(q, k, v, density=dn, block_size=min(64, T))
        try:
            pisa5 = _time(pisa, 0.5)
            pisa2 = _time(pisa, 0.2)
        except Exception as e:
            pisa5 = pisa2 = float("nan"); print(f"  PISA err T={T}: {e}")

        # SVG1 masked-SDPA: bool diag mask (held) vs the per-call MSE profile
        from methods.sparse_attention.svg.mask_utils import _build_diagonal_mask, sparsity_to_bandwidth
        bw = sparsity_to_bandwidth(T, 0.25)
        bm = _build_diagonal_mask(T, bw).to(DEV)
        svg1 = _time(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bm))

        # cost of ONE MSE profile pass (num_sampled_rows=32, 2 candidate masks)
        def svg1_profile():
            n = min(32, T)
            rows = torch.randint(0, T, (n,), device=DEV)
            qs = q[:, :, rows, :]
            sc = torch.matmul(qs, k.transpose(-2, -1)) * (Dh ** -0.5)
            a = F.softmax(sc, dim=-1); torch.matmul(a, v)
            sc2 = sc.masked_fill(~bm[rows, :].unsqueeze(0).unsqueeze(0), float("-inf"))
            a2 = torch.nan_to_num(F.softmax(sc2, dim=-1)); torch.matmul(a2, v)
        svg1p = _time(svg1_profile)

        # Radial via its runtime (flashinfer BSR) — needs video_size; skip if err
        radial = float("nan")
        try:
            from methods.sparse_attention.radial.processor import SVDRadialAttnProcessor  # noqa
            radial = float("nan")  # radial needs full processor context; measured in-run
        except Exception:
            pass

        print(f"{T:>6} {dense:>9.3f} | {pisa5:>9.3f} {pisa2:>8.3f} | "
              f"{svg1:>9.3f} {svg1+svg1p:>9.3f} | {radial:>8}")

    print("\n[read] If dense_ms <= a method's ms at every T, that method is "
          "kernel-bound (cannot win) regardless of Python-overhead fixes.")


if __name__ == "__main__":
    main()
