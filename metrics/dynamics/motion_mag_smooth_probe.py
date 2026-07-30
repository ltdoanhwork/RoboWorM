"""motion_magnitude + motion_smoothness probe on matched-grid clips.

Runs the WorldScore dynamics metrics that were NOT covered by the motion_accuracy
(gdino) probe, on a single (gt_matched, pred) pair that is already on ONE common
(resolution, fps) grid — so the numbers are directly comparable within the pair:

  - motion_magnitude : mean over frame-pairs of median |optical flow|  (SEA-RAFT)
                       (WorldScore OpticalFlowMetric). Reported for BOTH gt and pred
                       so we can take ratio pred/gt (fps/res already matched).
  - motion_smoothness: VFIMamba interpolates the odd frames from the even frames;
                       compare interpolated-vs-real odd frame by MSE / SSIM / LPIPS.
                       Higher SSIM + lower LPIPS/MSE = temporally smoother/consistent.

Imports SEA-RAFT and VFIMamba submodules DIRECTLY from third_party (like the gdino
probe imports SEA-RAFT), shimming the one `worldscore.benchmark.metrics.third_party.
VFIMamba.config` import so we do NOT trigger the worldscore package __init__ (which
pulls pyiqa). Everything runs from cwd = metrics/WorldScore so the VFIMamba .pkl path
`worldscore/benchmark/metrics/checkpoints/VFIMamba.pkl` resolves.
"""
import os
import sys
import json
import types
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- SEA-RAFT (direct, same as gdino probe) ---
from core.raft import RAFT
from core.utils.utils import load_ckpt
from core.parser import parse_args


def decode(video, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    i = 0
    frames = []
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


# --------------------------------------------------------------------------- #
# motion_magnitude (SEA-RAFT median flow) — matches OpticalFlowMetric
# --------------------------------------------------------------------------- #
class Flow:
    def __init__(self, cfg, ckpt, device):
        args = parse_args(argparse.Namespace(cfg=cfg, path=ckpt))
        m = RAFT(args)
        load_ckpt(m, ckpt)
        m.to(device).eval()
        self.m, self.a, self.dev = m, args, device

    def _img(self, f):
        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        return torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None].to(self.dev)

    def median_mag(self, f1, f2):
        with torch.no_grad():
            i1, i2 = self._img(f1), self._img(f2)
            s = self.a.scale
            i1 = F.interpolate(i1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            i2 = F.interpolate(i2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type="cuda"):
                out = self.m(i1, i2, iters=self.a.iters, test_mode=True)
            fl = out["flow"][-1]
            fl = F.interpolate(fl, scale_factor=0.5 ** s, mode="bilinear", align_corners=False) * (0.5 ** s)
            fl = fl.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
            mag = np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)
            return float(np.median(mag))


def motion_magnitude(frames, flow):
    vals = [flow.median_mag(a, b) for a, b in zip(frames[:-1], frames[1:])]
    return float(np.mean(vals)), vals


# --------------------------------------------------------------------------- #
# motion_smoothness (VFIMamba) — matches MotionSmoothnessMetric
# --------------------------------------------------------------------------- #
def build_vfimamba(device):
    """Import VFIMamba directly; shim the worldscore.* config name to dodge pyiqa."""
    import config as _cfg  # from VFIMamba/ on sys.path
    _cfg.MODEL_CONFIG["LOGNAME"] = "VFIMamba"
    _cfg.MODEL_CONFIG["MODEL_ARCH"] = _cfg.init_model_config(F=32, depth=[2, 2, 2, 3, 3])
    # Trainer_finetune does `from worldscore.benchmark.metrics.third_party.VFIMamba.config import *`
    # register a fake package chain that resolves that name to the module we just imported.
    chain = "worldscore.benchmark.metrics.third_party.VFIMamba"
    parts = chain.split(".")
    for k in range(1, len(parts) + 1):
        name = ".".join(parts[:k])
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules[chain + ".config"] = _cfg
    from Trainer_finetune import Model
    from benchmark.utils.padder import InputPadder
    m = Model(-1)
    m.load_model()
    m.eval()
    m.device()
    return m, InputPadder


def _ssim(a, b):
    # simple global SSIM on uint8 BGR (channel-mean), constants per Wang et al.
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return float(((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                 ((mu_a ** 2 + mu_b ** 2 + C1) * (va + vb + C2)))


def motion_smoothness(frames, model, InputPadder, device, TTA=True, stride=1, max_triples=None):
    even = frames[::2]
    odd = frames[1::2]
    ssim_l, mse_l = [], []
    pairs = list(zip(even[:-1], even[1:]))
    # subsample frame-triples for speed (representative average); keep index into odd
    sel = list(range(0, len(pairs), stride))
    if max_triples is not None:
        sel = sel[:max_triples]
    for i in sel:
        f0, f2 = pairs[i]
        if i >= len(odd):
            break
        I0 = cv2.imread(f0); I2 = cv2.imread(f2); mid_gt = cv2.imread(odd[i])
        I0_ = (torch.tensor(I0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        I2_ = (torch.tensor(I2.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        padder = InputPadder(I0_.shape, divisor=32)
        I0p, I2p = padder.pad(I0_, I2_)
        with torch.no_grad():
            mid = model.inference(I0p, I2p, True, TTA=TTA, fast_TTA=TTA, scale=0.0)
        mid = (padder.unpad(mid)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        ssim_l.append(_ssim(mid, mid_gt))
        mse_l.append(float(np.mean((mid.astype(np.float64) - mid_gt.astype(np.float64)) ** 2)))
    return {
        "ssim": float(np.mean(ssim_l)) if ssim_l else float("nan"),
        "mse": float(np.mean(mse_l)) if mse_l else float("nan"),
        "n": len(ssim_l), "tta": TTA, "stride": stride,
    }


def run_one(video, tag, work_root, flow, vfi_model, InputPadder, device, do_smooth,
            tta=False, smooth_stride=1, max_triples=None):
    fdir = os.path.join(work_root, f"{tag}_frames")
    frames = decode(video, fdir)
    mag_mean, _ = motion_magnitude(frames, flow)
    out = {"tag": tag, "video": video, "n_frames": len(frames),
           "motion_magnitude": mag_mean}
    if do_smooth:
        out["motion_smoothness"] = motion_smoothness(
            frames, vfi_model, InputPadder, device,
            TTA=tta, stride=smooth_stride, max_triples=max_triples)
    import shutil
    shutil.rmtree(fdir, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="pair")
    ap.add_argument("--raft_cfg", required=True)
    ap.add_argument("--raft_ckpt", required=True)
    ap.add_argument("--no_smoothness", action="store_true")
    ap.add_argument("--tta", action="store_true", help="VFIMamba test-time aug (2x slower)")
    ap.add_argument("--smooth_stride", type=int, default=8,
                    help="evaluate every Nth interpolation triple (speed)")
    ap.add_argument("--max_triples", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    flow = Flow(args.raft_cfg, args.raft_ckpt, device)
    vfi_model = InputPadder = None
    if not args.no_smoothness:
        vfi_model, InputPadder = build_vfimamba(device)

    work = os.path.join(args.out, "_work")
    res = {}
    for tag, path in [("gt", args.gt), ("pred", args.pred)]:
        res[tag] = run_one(path, tag, work, flow, vfi_model, InputPadder, device,
                           do_smooth=not args.no_smoothness, tta=args.tta,
                           smooth_stride=args.smooth_stride, max_triples=args.max_triples)
        print(f"[{tag}] motion_magnitude={res[tag]['motion_magnitude']:.4f}"
              + ("" if args.no_smoothness else
                 f"  smooth(ssim={res[tag]['motion_smoothness']['ssim']:.4f},"
                 f" mse={res[tag]['motion_smoothness']['mse']:.2f},"
                 f" n={res[tag]['motion_smoothness']['n']})"), flush=True)

    mm_gt = res["gt"]["motion_magnitude"]
    mm_pr = res["pred"]["motion_magnitude"]
    summary = {
        "label": args.label,
        "motion_magnitude": {
            "gt": mm_gt, "pred": mm_pr,
            "ratio_pred_over_gt": (mm_pr / mm_gt) if mm_gt else float("nan"),
            "abs_log_ratio": abs(np.log(mm_pr / mm_gt)) if (mm_gt and mm_pr) else float("nan"),
        },
        "detail": res,
    }
    if not args.no_smoothness:
        summary["motion_smoothness"] = {
            "gt": res["gt"]["motion_smoothness"],
            "pred": res["pred"]["motion_smoothness"],
        }
    with open(os.path.join(args.out, "mag_smooth_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["motion_magnitude"], indent=2))


if __name__ == "__main__":
    main()
