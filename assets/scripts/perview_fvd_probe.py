#!/usr/bin/env python3
"""Go/no-go for per-camera FVD: do the models' GTs agree once the mosaic is split?

WHY THIS MUST RUN FIRST
The whole point of per-camera FVD is to restore a SHARED real reference so the FVD column
becomes comparable across configurations. That only works if, after cropping each model's
mosaic into single-camera videos, the GT of DreamGen and the GT of Ctrl-World are the same
distribution. They come from the same source mp4, but they travel through different
resize/crop paths, so "same source" does not prove "same distribution".

The test is real-vs-real FVD per camera: take model A's GT and model B's GT for the SAME
camera and compute FVD between them. If the models agree, that number is near the
within-model split-half noise floor, and pooling them into one reference is sound. If it is
far above the floor -- as the mosaic-level test showed (that is how the layout bug was
found) -- then per-camera does NOT fix the problem and we must not promise it does.

It also times I3D so the migration ETA is measured rather than guessed.

Layouts (verified in the infer scripts):
  DreamGen MV  432x768  2x2 grid  [cam_high, cam_low, cam_left_wrist, cam_right_wrist]
  CW-4view MV  192x1280 h-strip   [cam_high, cam_low, cam_left_wrist, cam_right_wrist]
  Ctrl-World   192x960  h-strip   [cam_high, cam_left_wrist, cam_right_wrist]

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/perview_fvd_probe.py --clips 40
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts' / 'metrics'))

from export_dr_grid_xlsx import model_label                              # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet                          # noqa: E402

I3D_WINDOW, I3D_SIZE = 16, 224

# rows x cols x camera order, keyed by the mosaic shape the DR results recorded
LAYOUTS = {
    (432, 768):  dict(grid=(2, 2), cams=['cam_high', 'cam_low',
                                         'cam_left_wrist', 'cam_right_wrist']),
    (192, 1280): dict(grid=(1, 4), cams=['cam_high', 'cam_low',
                                         'cam_left_wrist', 'cam_right_wrist']),
    (192, 960):  dict(grid=(1, 3), cams=['cam_high', 'cam_left_wrist',
                                         'cam_right_wrist']),
}


def split_mosaic(vid: np.ndarray) -> dict[str, np.ndarray]:
    """Cut a mosaic clip (T,H,W,3) into {camera: (T,h,w,3)}; {} if shape unknown."""
    T, H, W, _ = vid.shape
    lay = LAYOUTS.get((H, W))
    if lay is None:
        return {}
    r, c = lay['grid']
    th, tw = H // r, W // c
    out = {}
    for i, cam in enumerate(lay['cams']):
        y, x = (i // c) * th, (i % c) * tw
        out[cam] = vid[:, y:y + th, x:x + tw]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=40,
                    help='GT clips per model to sample (per split)')
    ap.add_argument('--embodiment', default='bimanual')
    ap.add_argument('--split', default='makovian')
    ap.add_argument('--json', default='metrics/reports/perview_fvd_probe.json')
    args = ap.parse_args()

    import torch
    from compute_dr_metrics import get_models, read_video
    from export_dr_grid_xlsx_vs_real import ref_identity

    run = REPO / 'metrics/WorldScore/results/dr_full'
    man = json.loads((run / 'manifest.json').read_text())
    rows = man['rows'] if isinstance(man, dict) else man

    # one representative GT per (model, content) -- dedupe so a GT shared by 10 methods
    # is read once, exactly like the exporter's reference pooling does
    pick: dict[tuple[str, str], str] = {}
    for r in rows:
        if r.get('ref_kind') != 'vs_gt' or 'multiview' not in r.get('relative', ''):
            continue
        if r.get('embodiment') != args.embodiment or r.get('split') != args.split:
            continue
        lab = model_label(r.get('relative', ''))
        if lab is None:
            continue
        pick.setdefault((lab, ref_identity(r['ref'])), r['ref'])

    per_model = collections.defaultdict(list)
    for (lab, _ident), path in pick.items():
        per_model[lab].append(path)
    print(f'GT pool for {args.embodiment}/{args.split}:')
    for lab in sorted(per_model):
        print(f'  {lab:20s} {len(per_model[lab]):5d} unique GT clips')

    dev = 'cuda'
    m = get_models(dev)
    feats = collections.defaultdict(list)     # (model, cam) -> [feature]
    t_i3d = 0.0
    n_win = n_clip = 0

    for lab in sorted(per_model):
        paths = sorted(per_model[lab])[:args.clips]
        for p in paths:
            try:
                vid = read_video(Path(p))
            except Exception as e:                                    # noqa: BLE001
                print(f'  !! unreadable {p}: {e}')
                continue
            views = split_mosaic(vid)
            if not views:
                print(f'  !! unknown mosaic {vid.shape} for {lab}')
                break
            n_clip += 1
            for cam, cv in views.items():
                with torch.no_grad():
                    for s in range(0, len(cv) - I3D_WINDOW + 1, I3D_WINDOW):
                        w = torch.from_numpy(cv[s:s + I3D_WINDOW]).to(dev)
                        w = w.permute(0, 3, 1, 2).float() / 255
                        w = torch.nn.functional.interpolate(
                            w, size=(I3D_SIZE, I3D_SIZE), mode='bilinear',
                            align_corners=False)
                        w = (w * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)
                        t0 = time.time()
                        f = m['i3d'](w, rescale=False, resize=False,
                                     return_features=True)
                        torch.cuda.synchronize()
                        t_i3d += time.time() - t0
                        feats[(lab, cam)].append(f.squeeze(0).cpu().numpy())
                        n_win += 1
        print(f'  {lab}: done, {n_win} windows so far')

    def mom(x):
        x = np.asarray(x, dtype=np.float64)
        mu = x.mean(0)
        xc = x - mu
        return mu, xc.T.dot(xc) / max(len(x) - 1, 1)

    print('\n=== within-model split-half NOISE FLOOR (per camera) ===')
    floors = {}
    rng = np.random.default_rng(0)
    for key in sorted(feats):
        x = np.asarray(feats[key], dtype=np.float64)
        if len(x) < 128:
            continue
        idx = rng.permutation(len(x))
        h = len(x) // 2
        fl = frechet(*mom(x[idx[:h]]), *mom(x[idx[h:2 * h]]))
        floors[key] = fl
        print(f'  {key[0]:20s} {key[1]:18s} n={len(x):5d}  floor={fl:7.1f}')

    print('\n=== cross-model REAL-vs-REAL per camera (the go/no-go) ===')
    print('  a pooled reference is sound only if cross ~ floor\n')
    cams = sorted({k[1] for k in feats})
    models = sorted({k[0] for k in feats})
    verdicts = []
    for cam in cams:
        have = [mo for mo in models if (mo, cam) in feats
                and len(feats[(mo, cam)]) >= 128]
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = feats[(have[i], cam)], feats[(have[j], cam)]
                d = frechet(*mom(a), *mom(b))
                fl = max(floors.get((have[i], cam), 0),
                         floors.get((have[j], cam), 0))
                ratio = d / fl if fl > 0 else float('inf')
                ok = 'SAME-DIST' if ratio <= 3 else 'DIFFERENT'
                print(f'  {cam:18s} {have[i]:16s} vs {have[j]:16s} '
                      f'FVD={d:8.1f}  floor={fl:6.1f}  ratio={ratio:5.1f}x  {ok}')
                verdicts.append(dict(cam=cam, a=have[i], b=have[j], fvd=d,
                                     floor=fl, ratio=ratio, same=ratio <= 3))

    if n_win:
        print(f'\n=== throughput ===\n  I3D {t_i3d / n_win * 1000:.2f} ms/window, '
              f'{n_win} windows from {n_clip} clips '
              f'({n_win / max(n_clip, 1):.1f} win/clip incl. all cameras)')

    out = REPO / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        embodiment=args.embodiment, split=args.split,
        floors={f'{k[0]}|{k[1]}': v for k, v in floors.items()},
        cross=verdicts,
        ms_per_window=t_i3d / n_win * 1000 if n_win else None,
        windows=n_win, clips=n_clip), indent=2))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
