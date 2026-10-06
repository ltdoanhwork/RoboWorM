#!/usr/bin/env python3
"""Can one SHARED real pool per (embodiment x view x regime) actually be built?

The recommended protocol -- one fixed real pool per setting, shared by every model -- is
only sound if all models' real videos land in the SAME distribution. Measured so far:

  mosaic, as shipped   : cross-model real-vs-real FVD 1295-2895  (vs floor ~200-290)
  per-camera, no other : cross-model real-vs-real FVD ~300

so per-camera already removes most of it, but a residual remains. Two candidate causes,
both real and both measurable:

  fps     Ctrl-World GT is 5 fps, DreamGen GT is 16 fps. A 16-frame I3D window therefore
          spans 3.2 s vs 1.0 s of real time. I3D is a motion network, so this alone shifts
          the feature distribution.
  aspect  the source camera is 4:3, but DreamGen's tile is 16:9 (384x216) and Ctrl-World's
          is 5:3 (320x192). Squashing both to 224x224 distorts them by different amounts.

This ablates the two so we know what normalisation the shared pool requires, instead of
assuming. Verdict is the ratio of cross-model real-vs-real to the within-model split-half
floor: ~1x means the two models' real videos are the same distribution and may be pooled.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/shared_pool_feasibility.py --clips 60
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'metrics'))

from export_dr_grid_xlsx import model_label                              # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet, ref_identity            # noqa: E402
from perview_fvd_probe import split_mosaic                               # noqa: E402

I3D_WINDOW, I3D_SIZE = 16, 224
TARGET_FPS = 5.0          # the slowest model; resampling down never invents motion
CANON = (192, 256)        # h,w -- 4:3, the source camera aspect for these cameras
# cameras every bimanual multiview configuration has (Ctrl-World MV lacks cam_low)
COMMON_CAMS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


def probe_fps(path: str) -> float:
    try:
        o = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=r_frame_rate', '-of', 'json', path],
            capture_output=True, text=True, timeout=60).stdout
        rf = json.loads(o)['streams'][0]['r_frame_rate']
        n, d = rf.split('/')
        return float(n) / float(d)
    except Exception:                                                # noqa: BLE001
        return float('nan')


def resample_fps(vid: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Nearest-frame temporal decimation. Never interpolates -- inventing frames would
    invent motion, which is exactly the thing being measured."""
    if not np.isfinite(src_fps) or src_fps <= dst_fps * 1.01:
        return vid
    n_out = max(int(round(len(vid) * dst_fps / src_fps)), I3D_WINDOW)
    idx = np.clip(np.round(np.arange(n_out) * src_fps / dst_fps).astype(int),
                  0, len(vid) - 1)
    return vid[idx]


def ep_key(p: str):
    m = re.search(r'/episode_([a-z0-9_]+?)__(\d+)/', p)
    if m:
        return (m.group(1), m.group(2).lstrip('0') or '0')
    m = re.search(r'/([a-z0-9_]+)/episode_(\d+)/', p)
    if m:
        return (m.group(1), m.group(2).lstrip('0') or '0')
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=60)
    ap.add_argument('--embodiment', default='bimanual')
    ap.add_argument('--split', default='makovian')
    ap.add_argument('--json', default='metrics/reports/shared_pool_feasibility.json')
    args = ap.parse_args()

    import torch
    from compute_dr_metrics import get_models, read_video

    run = REPO / 'metrics/WorldScore/results/dr_full'
    man = json.loads((run / 'manifest.json').read_text())
    rows = man['rows'] if isinstance(man, dict) else man

    pick: dict[tuple[str, str], str] = {}
    for r in rows:
        if r.get('ref_kind') != 'vs_gt' or 'multiview' not in r.get('relative', ''):
            continue
        if r.get('embodiment') != args.embodiment or r.get('split') != args.split:
            continue
        lab = model_label(r.get('relative', ''))
        if lab is not None:
            pick.setdefault((lab, ref_identity(r['ref'])), r['ref'])

    per_model = collections.defaultdict(dict)
    for (lab, _i), p in pick.items():
        k = ep_key(p)
        if k:
            per_model[lab][k] = p

    # use only episodes EVERY model has, so episode identity is never a confound
    common_eps = None
    for lab in per_model:
        s = set(per_model[lab])
        common_eps = s if common_eps is None else (common_eps & s)
    common_eps = sorted(common_eps or [])[:args.clips]
    print(f'{args.embodiment}/{args.split}: {len(per_model)} models, '
          f'{len(common_eps)} shared episodes used')
    fps = {lab: probe_fps(next(iter(per_model[lab].values()))) for lab in per_model}
    for lab in sorted(per_model):
        print(f'  {lab:18s} {len(per_model[lab]):4d} eps, GT fps={fps[lab]:.1f}')

    dev = 'cuda'
    m = get_models(dev)
    VARIANTS = ('as_is', 'fps_only', 'aspect_only', 'fps+aspect')
    feats = {v: collections.defaultdict(list) for v in VARIANTS}

    for lab in sorted(per_model):
        for k in common_eps:
            p = per_model[lab].get(k)
            if p is None:
                continue
            try:
                vid = read_video(Path(p))
            except Exception:                                        # noqa: BLE001
                continue
            views = split_mosaic(vid)
            if not views:
                print(f'  !! unknown mosaic {vid.shape} for {lab}')
                break
            for cam in COMMON_CAMS:
                if cam not in views:
                    continue
                base = views[cam]
                for var in VARIANTS:
                    cv = base
                    if 'fps' in var:
                        cv = resample_fps(cv, fps[lab], TARGET_FPS)
                    with torch.no_grad():
                        for s in range(0, len(cv) - I3D_WINDOW + 1, I3D_WINDOW):
                            w = torch.from_numpy(np.ascontiguousarray(
                                cv[s:s + I3D_WINDOW])).to(dev)
                            w = w.permute(0, 3, 1, 2).float() / 255
                            if 'aspect' in var:
                                # undo each model's own squash: back to the 4:3 the
                                # source camera actually has, THEN to I3D's square
                                w = torch.nn.functional.interpolate(
                                    w, size=CANON, mode='bilinear', align_corners=False)
                            w = torch.nn.functional.interpolate(
                                w, size=(I3D_SIZE, I3D_SIZE), mode='bilinear',
                                align_corners=False)
                            w = (w * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)
                            f = m['i3d'](w, rescale=False, resize=False,
                                         return_features=True)
                            feats[var][(lab, cam)].append(
                                f.squeeze(0).cpu().numpy())
        print(f'  {lab}: extracted')

    def mom(x):
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean(0)
        xc = x - mu
        return mu, xc.T.dot(xc) / max(len(x) - 1, 1)

    rng = np.random.default_rng(0)
    out = {}
    for var in VARIANTS:
        F = feats[var]
        floors = {}
        for key in F:
            x = np.asarray(F[key], dtype=np.float64)
            if len(x) < 128:
                continue
            idx = rng.permutation(len(x))
            h = len(x) // 2
            floors[key] = frechet(*mom(x[idx[:h]]), *mom(x[idx[h:2 * h]]))
        cams = sorted({k[1] for k in F})
        models = sorted({k[0] for k in F})
        print(f'\n=== variant: {var} ===')
        ratios, recs = [], []
        for cam in cams:
            have = [mo for mo in models if (mo, cam) in F and len(F[(mo, cam)]) >= 128]
            for i in range(len(have)):
                for j in range(i + 1, len(have)):
                    d = frechet(*mom(F[(have[i], cam)]), *mom(F[(have[j], cam)]))
                    fl = max(floors.get((have[i], cam), 0), floors.get((have[j], cam), 0))
                    # conservative: also compare against the TIGHTEST floor, since a model
                    # with many windows has a low floor and that is the honest yardstick
                    flmin = min(v for k2, v in floors.items() if k2[1] == cam) \
                        if any(k2[1] == cam for k2 in floors) else fl
                    r_max = d / fl if fl > 0 else float('inf')
                    r_min = d / flmin if flmin > 0 else float('inf')
                    ratios.append(r_min)
                    recs.append(dict(cam=cam, a=have[i], b=have[j], fvd=d,
                                     floor_lenient=fl, floor_strict=flmin,
                                     ratio_lenient=r_max, ratio_strict=r_min))
                    print(f'  {cam:16s} {have[i]:15s} vs {have[j]:15s} '
                          f'FVD={d:7.1f} floor={fl:6.1f}/{flmin:6.1f} '
                          f'ratio={r_max:4.1f}x/{r_min:5.1f}x')
        w = float(np.mean(ratios)) if ratios else float('nan')
        print(f'  -> mean STRICT ratio = {w:.1f}x '
              f'({"POOLABLE" if w <= 3 else "NOT poolable"})')
        out[var] = dict(pairs=recs, mean_strict_ratio=w,
                        floors={f'{k[0]}|{k[1]}': v for k, v in floors.items()})

    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(embodiment=args.embodiment, split=args.split,
                                 target_fps=TARGET_FPS, canon_hw=list(CANON),
                                 gt_fps=fps, variants=out), indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
