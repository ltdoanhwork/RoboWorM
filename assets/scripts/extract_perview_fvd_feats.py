#!/usr/bin/env python3
"""Extract per-camera I3D features for every multiview clip, at a common fps.

This is the re-scoring step that makes a SHARED real pool possible. It does NOT regenerate
video and does NOT touch the validated PSNR/SSIM/LPIPS results: the existing DR run stored
I3D features computed on the MOSAIC, which cannot be pooled across models (measured:
cross-model real-vs-real FVD 1295-2895 vs a split-half floor of 200-290). Here every
mosaic is cut into camera tiles and decimated to a common 5 fps, which brings cross-model
real-vs-real to 0.3x the floor -- i.e. one distribution, safe to pool.

Protocol: .cursor/skills/fvd-shared-real-pool/SKILL.md

One output .npz per manifest row, so the job is resumable and shardable across machines
exactly like scripts/compute_dr_metrics.py:

    MACHINES=6 bash scripts/_perview_wave_fg.sh <machine 0..5>

Each .npz holds float16 features (feature values are O(1) logits; fp16 costs ~0.1% of the
FVD value and halves the 1.1M-window footprint) plus the keys needed to pool later.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))
sys.path.insert(0, str(REPO / 'scripts'))

from export_dr_grid_xlsx import METHOD_LABEL, model_label                # noqa: E402
from export_dr_grid_xlsx_vs_real import ref_identity                     # noqa: E402
from fvd_pool_common import (TARGET_FPS, ep_key, i3d_features,           # noqa: E402
                            resample_fps, split_mosaic)


def probe_fps(path: str) -> float:
    import subprocess
    try:
        o = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=r_frame_rate', '-of', 'json', path],
            capture_output=True, text=True, timeout=60).stdout
        n, d = json.loads(o)['streams'][0]['r_frame_rate'].split('/')
        return float(n) / float(d)
    except Exception:                                                # noqa: BLE001
        return float('nan')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='metrics/WorldScore/results/dr_full')
    ap.add_argument('--out', default='metrics/WorldScore/results/fvd_perview')
    ap.add_argument('--camera-map', default='metrics/reports/fvd_camera_map.json')
    ap.add_argument('--machines', type=int, default=1)
    ap.add_argument('--machine', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    from compute_dr_metrics import get_models, read_video

    run = REPO / args.run
    out = REPO / args.out
    (out / 'feats').mkdir(parents=True, exist_ok=True)
    (out / 'errors').mkdir(parents=True, exist_ok=True)

    man = json.loads((run / 'manifest.json').read_text())
    rows = man['rows'] if isinstance(man, dict) else man

    # Only multiview vs_gt rows: single-view is already poolable at mosaic level
    # (real-vs-real 7.6-8.3), so its existing features stay valid and are reused.
    todo = []
    for r in rows:
        if r.get('ref_kind') != 'vs_gt' or 'multiview' not in r.get('relative', ''):
            continue
        if model_label(r.get('relative', '')) is None:
            continue
        if METHOD_LABEL.get(r.get('method')) is None:
            continue
        todo.append(r)
    todo.sort(key=lambda r: r['id'])
    mine = [r for i, r in enumerate(todo) if i % args.machines == args.machine]
    if args.limit:
        mine = mine[:args.limit]
    print(f'[perview] {len(todo)} multiview vs_gt rows; this machine '
          f'({args.machine}/{args.machines}) takes {len(mine)}', flush=True)

    m = get_models('cuda')
    fps_cache: dict[str, float] = {}
    t0 = time.time()
    done = skipped = failed = 0

    for n, r in enumerate(mine, 1):
        dst = out / 'feats' / f"{r['id']}.npz"
        if dst.exists():
            skipped += 1
            continue
        try:
            payload = {}
            for tag, path in (('pred', r['pred']), ('ref', r['ref'])):
                p = str(path)
                if p not in fps_cache:
                    fps_cache[p] = probe_fps(p)
                vid = read_video(Path(p))
                tiles = split_mosaic(vid)
                if tiles is None:
                    raise ValueError(f'unknown mosaic {vid.shape} for {tag} {p}')
                for ti, t in enumerate(tiles):
                    t = resample_fps(t, fps_cache[p], TARGET_FPS)
                    f = i3d_features(t, m['i3d'], 'cuda')
                    payload[f'{tag}_{ti}'] = (np.asarray(f, dtype=np.float16)
                                              if f else np.zeros((0, 400), np.float16))
                del vid, tiles
            ek = ep_key(r['ref'])
            meta = dict(id=r['id'], embodiment=r['embodiment'], split=r['split'],
                        method=r['method'], label=model_label(r.get('relative', '')),
                        ref_identity=ref_identity(r['ref']),
                        episode=json.dumps(ek), target_fps=TARGET_FPS)
            # np.savez_compressed appends '.npz' unless the name already ends in it, so
            # the temp name must carry the suffix or the atomic rename below misses.
            tmp = dst.with_name(dst.name + '.tmp.npz')
            np.savez_compressed(tmp, meta=json.dumps(meta), **payload)
            tmp.rename(dst)
            done += 1
        except Exception as e:                                       # noqa: BLE001
            failed += 1
            (out / 'errors' / f"{r['id']}.txt").write_text(
                f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}')
        if n % 25 == 0 or n == len(mine):
            el = time.time() - t0
            rate = (done + failed) / el if el > 0 else 0
            left = len(mine) - n
            print(f'[perview] {n}/{len(mine)} done={done} skip={skipped} '
                  f'fail={failed} {rate:.2f} clip/s ETA {left / rate / 60:.0f}m'
                  if rate > 0 else f'[perview] {n}/{len(mine)}', flush=True)

    print(f'[perview] FINISHED done={done} skipped={skipped} failed={failed} '
          f'in {(time.time() - t0) / 60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
