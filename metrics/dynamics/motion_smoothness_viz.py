"""Visualize motion_smoothness (VFIMamba) step-by-step for a few sample triples.

For each clip it dumps, at a handful of evenly-spaced interpolation triples, a
side-by-side panel:  [ frame_{2k}  |  VFIMamba interp mid  |  real frame_{2k+1}
|  |interp - real| error heatmap ]  with per-triple SSIM/MSE annotated, plus a
per-triple SSIM & MSE series (json) so we can chart how smoothness varies over time.

Reuses the exact interpolation call of motion_mag_smooth_probe.py (WorldScore
MotionSmoothnessMetric): even frames are the anchors, the odd frame is the target
the model must reconstruct. Pure-PyTorch mamba_ssm shim, no TTA by default.
"""
import os, sys, json, types, argparse
import cv2
import numpy as np
import torch


def decode(video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    i, frames = 0, []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        p = os.path.join(out_dir, f"{i:05d}.jpg")
        cv2.imwrite(p, fr)
        frames.append(p)
        i += 1
    cap.release()
    return frames


def build_vfimamba(device):
    import config as _cfg
    _cfg.MODEL_CONFIG["LOGNAME"] = "VFIMamba"
    _cfg.MODEL_CONFIG["MODEL_ARCH"] = _cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])
    chain = "worldscore.benchmark.metrics.third_party.VFIMamba"
    parts = chain.split(".")
    for k in range(1, len(parts) + 1):
        n = ".".join(parts[:k])
        sys.modules.setdefault(n, types.ModuleType(n))
    sys.modules[chain + ".config"] = _cfg
    from Trainer_finetune import Model
    from benchmark.utils.padder import InputPadder
    m = Model(-1); m.load_model(); m.eval(); m.device()
    return m, InputPadder


def ssim_global(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                 ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))


def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_clip(video, tag, out_dir, model, InputPadder, device, n_panels=4, stride=8, tta=False):
    work = os.path.join(out_dir, f"_{tag}_frames")
    frames = decode(video, work)
    even, odd = frames[::2], frames[1::2]
    pairs = list(zip(even[:-1], even[1:]))
    n = min(len(pairs), len(odd))
    # per-triple series over a strided sweep
    sel_all = list(range(0, n, stride))
    series = []
    # panel indices: evenly spaced across the VISITED (strided) triples
    if sel_all:
        pick = np.linspace(0, len(sel_all) - 1, n_panels).astype(int).tolist()
        panel_idx = set(sel_all[j] for j in pick)
    else:
        panel_idx = set()
    for i in sel_all:
        f0, f2 = pairs[i]
        I0 = cv2.imread(f0); I2 = cv2.imread(f2); mid_gt = cv2.imread(odd[i])
        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        padder = InputPadder(I0_.shape, divisor=32)
        I0p, I2p = padder.pad(I0_, I2_)
        with torch.no_grad():
            mid = model.inference(I0p, I2p, True, TTA=tta, fast_TTA=tta, scale=0.0)
        mid = (padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        s = ssim_global(mid, mid_gt)
        m = float(np.mean((mid.astype(np.float64) - mid_gt.astype(np.float64)) ** 2))
        series.append({"triple": int(i), "pct": round(100.0 * i / max(1, n - 1), 1),
                       "ssim": s, "mse": m})
        if i in panel_idx:
            err = np.abs(mid.astype(np.float32) - mid_gt.astype(np.float32)).mean(axis=2)
            err = np.clip(err / 60.0 * 255.0, 0, 255).astype(np.uint8)  # amplify for visibility
            errc = cv2.applyColorMap(err, cv2.COLORMAP_INFERNO)
            panel = np.hstack([
                label(I0, "frame 2k (anchor)"),
                label(mid, f"VFIMamba interp  SSIM={s:.4f}"),
                label(mid_gt, "real frame 2k+1"),
                label(errc, f"|err| MSE={m:.1f}"),
            ])
            cv2.imwrite(os.path.join(out_dir, f"{tag}_panel_{i:04d}.png"), panel)
    import shutil
    shutil.rmtree(work, ignore_errors=True)
    ssim_mean = float(np.mean([x["ssim"] for x in series]))
    mse_mean = float(np.mean([x["mse"] for x in series]))
    return {"tag": tag, "video": video, "n_frames": len(frames),
            "ssim_mean": ssim_mean, "mse_mean": mse_mean, "series": series}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True,
                    help="tag=path pairs, e.g. gt_dg=/path.mp4 dreamgen=/path.mp4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_panels", type=int, default=4)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, InputPadder = build_vfimamba(device)
    out = {}
    for spec in args.clips:
        tag, path = spec.split("=", 1)
        print(f"=== {tag} :: {path}")
        out[tag] = run_clip(path, tag, args.out, model, InputPadder, device,
                            n_panels=args.n_panels, stride=args.stride, tta=args.tta)
        print(f"    ssim_mean={out[tag]['ssim_mean']:.4f}  mse_mean={out[tag]['mse_mean']:.2f}  "
              f"panels+series={len(out[tag]['series'])}", flush=True)
    with open(os.path.join(args.out, "smoothness_viz.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote", os.path.join(args.out, "smoothness_viz.json"))


if __name__ == "__main__":
    main()
