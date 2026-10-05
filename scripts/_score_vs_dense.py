#!/usr/bin/env python3
"""Score accelerated predictions against the matched dense prediction.

Reference = the dense prediction of the SAME episode produced through the SAME
entrypoint (PhyFid-DR convention in .cursor/rules/phyworld-benchmark.mdc): the
question is "does the cache change what the model would have generated", not
"does the model match ground truth". Metrics are PSNR / SSIM / LPIPS, computed
with the same backends the inference scripts use (skimage + piq.LPIPS).

Usage:
  python scripts/_score_vs_dense.py --dense DIR --test DIR [--out out.json]
Both dirs are expected to hold <episode>.mp4 files with matching names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_video(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    frames = [np.asarray(f) for f in iio.imiter(path)]
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames)


def score_pair(ref: np.ndarray, test: np.ndarray, lpips_fn) -> dict:
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    n = min(len(ref), len(test))
    if n == 0:
        raise RuntimeError("empty video")
    ref, test = ref[:n], test[:n]
    if ref.shape[1:] != test.shape[1:]:
        raise RuntimeError(f"resolution mismatch {ref.shape[1:]} vs {test.shape[1:]}")

    psnrs, ssims = [], []
    for a, b in zip(ref, test):
        psnrs.append(peak_signal_noise_ratio(a, b, data_range=255))
        ssims.append(structural_similarity(a, b, channel_axis=2, data_range=255))

    # Bit-identical frames give PSNR=inf, which is not representable in strict
    # JSON and poisons any mean. Record identity explicitly instead.
    identical = bool(np.array_equal(ref, test))
    finite = [p for p in psnrs if np.isfinite(p)]
    psnr_mean = float(np.mean(finite)) if finite else None

    # LPIPS in batches to bound memory; piq expects float [0,1], NCHW.
    lp = []
    with torch.no_grad():
        for i in range(0, n, 16):
            x = torch.from_numpy(ref[i : i + 16]).permute(0, 3, 1, 2).float().div_(255).cuda()
            y = torch.from_numpy(test[i : i + 16]).permute(0, 3, 1, 2).float().div_(255).cuda()
            lp.append(lpips_fn(x, y).mean().item())

    return {
        "frames": int(n),
        "ref_frames": int(len(ref)),
        "identical": identical,
        "n_identical_frames": int(sum(1 for p in psnrs if not np.isfinite(p))),
        "psnr": psnr_mean,
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lp)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", required=True, help="dir of dense reference mp4s")
    ap.add_argument("--test", required=True, help="dir of accelerated mp4s")
    ap.add_argument("--out", default=None, help="write json here")
    args = ap.parse_args()

    import piq

    lpips_fn = piq.LPIPS()

    dense_dir, test_dir = Path(args.dense), Path(args.test)
    per_ep, missing = {}, []
    for mp4 in sorted(test_dir.glob("*.mp4")):
        ref = dense_dir / mp4.name
        if not ref.is_file():
            missing.append(mp4.name)
            continue
        per_ep[mp4.stem] = score_pair(read_video(ref), read_video(mp4), lpips_fn)
        m = per_ep[mp4.stem]
        psnr_s = "identical" if m["psnr"] is None else f"{m['psnr']:.3f}"
        print(
            f"  {mp4.stem}: PSNR={psnr_s} SSIM={m['ssim']:.4f} "
            f"LPIPS={m['lpips']:.4f} ({m['frames']}f)"
            + (" [BIT-IDENTICAL]" if m["identical"] else "")
        )

    if not per_ep:
        raise SystemExit(f"ERROR: no comparable episodes ({len(missing)} missing refs)")

    psnr_vals = [m["psnr"] for m in per_ep.values() if m["psnr"] is not None]
    out = {
        "dense_dir": str(dense_dir),
        "test_dir": str(test_dir),
        "n_episodes": len(per_ep),
        "n_identical": sum(1 for m in per_ep.values() if m["identical"]),
        "missing_reference": missing,
        # Mean over episodes that actually differ; bit-identical episodes have
        # undefined (infinite) PSNR and are counted in n_identical instead.
        "mean_psnr": float(np.mean(psnr_vals)) if psnr_vals else None,
        "n_psnr_episodes": len(psnr_vals),
        "mean_ssim": float(np.mean([m["ssim"] for m in per_ep.values()])),
        "mean_lpips": float(np.mean([m["lpips"] for m in per_ep.values()])),
        "per_episode": per_ep,
    }
    mp = "n/a (all identical)" if out["mean_psnr"] is None else f"{out['mean_psnr']:.3f}"
    print(
        f"MEAN vs dense: PSNR={mp} SSIM={out['mean_ssim']:.4f} "
        f"LPIPS={out['mean_lpips']:.4f} over {out['n_episodes']} episodes "
        f"({out['n_identical']} bit-identical)"
    )
    if missing:
        print(f"WARNING: {len(missing)} episodes had no dense reference: {missing}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
