#!/usr/bin/env python3
"""Pool per-camera I3D features into FVD against a SHARED real reference, and export.

Reference key is `(embodiment, camera, split)` -- the SAME real pool for every world model
and every acceleration method in that setting. This is the protocol in
metrics/fvd/README.md and it is what makes the FVD column
comparable across configurations, which a per-model reference cannot be.

Camera identity comes from `fvd_camera_match.py` (data-driven assignment), never from the
stored view names: humanoid Ctrl-World annotations have no `view_keys`, so the inference
script fell back to bimanual names and mislabels that cell's tiles.

Reported FVD for a multiview cell = mean over the cameras common to every configuration in
that cell, matching how the repo already averages multiview PSNR.

Also equalises sample size. FVD is biased by the number of windows (measured: 10x fewer
windows inflates FVD ~8.5x), so with `--budget` every cell is scored on the same number of
windows, drawn deterministically, and the budget is reported.

Usage:
  models/dreamgen/.venv/bin/python assets/scripts/pool_perview_fvd.py \
      --feats metrics/WorldScore/results/fvd_perview \
      --json results/reports/fvd_perview_pooled.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')

import numpy as np                                                       # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'assets' / 'scripts'))

from export_dr_grid_xlsx import METHOD_LABEL, METHOD_ORDER, MODEL_ORDER  # noqa: E402
from export_dr_grid_xlsx_vs_real import frechet                          # noqa: E402
from fvd_pool_common import moments                                      # noqa: E402

# Shared-pool protocol constants (tile -> camera map, camera restrictions, budgeted
# subsampling) live in metrics/fvd/pool.py, documented in metrics/fvd/README.md.
from metrics.fvd.pool import (FVD_MIN_WINDOWS, RESTRICT_CAMERAS, TILE_CAMERA,  # noqa: E402
                              budgeted, split_half)


def load(feats_dir: Path):
    """Read every .npz once, returning per-(cell, camera) pred features and the deduped
    per-(embodiment, camera, split) real pool."""
    pred = collections.defaultdict(list)     # (emb,label,split,method,cam) -> [feat]
    real = collections.defaultdict(list)     # (emb,cam,split)               -> [feat]
    real_a, real_b = collections.defaultdict(list), collections.defaultdict(list)
    seen_ref = set()
    audit = collections.Counter()

    files = sorted((feats_dir / 'feats').glob('*.npz'))
    print(f'[pool] reading {len(files)} feature files', flush=True)
    for i, f in enumerate(files, 1):
        try:
            z = np.load(f, allow_pickle=False)
            meta = json.loads(str(z['meta']))
        except Exception:                                            # noqa: BLE001
            audit['unreadable'] += 1
            continue
        label, emb, split = meta['label'], meta['embodiment'], meta['split']
        method = METHOD_LABEL.get(meta['method'])
        cam_of = TILE_CAMERA.get(label)
        if method is None or cam_of is None:
            audit['unmapped'] += 1
            continue
        keep = RESTRICT_CAMERAS.get((emb, split))
        ident = meta['ref_identity']
        for k in z.files:
            if k == 'meta':
                continue
            tag, ti = k.rsplit('_', 1)
            cam = cam_of.get(int(ti))
            if cam is None or (keep is not None and cam not in keep):
                continue
            arr = z[k]
            if not len(arr):
                continue
            if tag == 'pred':
                pred[(emb, label, split, method, cam)].append(arr)
            else:
                # DEDUPE by GT content: one real video is hard-linked into every method
                # tree, so counting it per pair would inflate n ~10x and void the
                # >=64-window guard.
                rk = ((emb, cam, split), ident)
                if rk in seen_ref:
                    continue
                seen_ref.add(rk)
                real[(emb, cam, split)].append(arr)
                # deterministic split-half for the noise floor (md5: PYTHONHASHSEED
                # randomises str hashing, which would move the floor between runs)
                bit = split_half(ident)
                (real_a if bit == 0 else real_b)[(emb, cam, split)].append(arr)
        if i % 2000 == 0:
            print(f'[pool]   {i}/{len(files)}', flush=True)

    cat = lambda d: {k: np.concatenate(v).astype(np.float64)              # noqa: E731
                     for k, v in d.items() if v}
    return cat(pred), cat(real), cat(real_a), cat(real_b), audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='metrics/WorldScore/results/fvd_perview')
    ap.add_argument('--json', default='results/reports/fvd_perview_pooled.json')
    ap.add_argument('--budget', type=int, default=0,
                    help='windows per cell and per reference (0 = use all)')
    args = ap.parse_args()
    budget = args.budget or None

    pred, real, real_a, real_b, audit = load(REPO / args.feats)
    if audit:
        print(f'[pool] audit: {dict(audit)}')

    print('\n=== shared real pools (embodiment, camera, split) ===')
    for k in sorted(real):
        print(f'  {"|".join(k):38s} n_windows={len(real[k]):6d}')

    ref_mom = {}
    for k, v in real.items():
        vb = budgeted(v, budget, 'ref|' + '|'.join(k))
        if len(vb) >= FVD_MIN_WINDOWS:
            ref_mom[k] = (moments(vb), len(vb))

    # noise floor: two disjoint halves of the same shared pool.
    #
    # CAVEAT ON READING THIS NUMBER. Each half holds a DISJOINT set of episodes, so this
    # is an episode-diversity floor, not a same-content sampling floor: with ~75 episodes
    # per half, two halves differ by task content as well as by sampling noise. A method
    # whose predictions cover ALL episodes is compared against the FULL pool and can
    # therefore score below this figure -- 14 of 28 dense cells do. So the floor is an
    # upper bound on the estimator's noise, useful as an order-of-magnitude sanity check
    # ("is 300 large?"), and must NOT be read as a value no method can beat.
    floor = {}
    for k in real:
        a, b = real_a.get(k), real_b.get(k)
        if a is None or b is None:
            continue
        n = min(len(a), len(b), budget or 10 ** 9)
        if n < FVD_MIN_WINDOWS:
            continue
        floor[k] = frechet(*moments(budgeted(a, n, 'fa|' + '|'.join(k))),
                           *moments(budgeted(b, n, 'fb|' + '|'.join(k))))

    per_cam, rows = {}, {}
    keys = sorted(pred)
    for i, key in enumerate(keys, 1):
        emb, label, split, method, cam = key
        rk = (emb, cam, split)
        if rk not in ref_mom:
            continue
        x = budgeted(pred[key], budget, 'p|' + '|'.join(key))
        if len(x) < FVD_MIN_WINDOWS:
            continue
        (mr, cr), nref = ref_mom[rk]
        per_cam[key] = dict(fvd=frechet(*moments(x), mr, cr), n_pred=len(x),
                            n_ref=nref, floor=floor.get(rk))
        if i % 200 == 0 or i == len(keys):
            print(f'[pool] scored {i}/{len(keys)} (cell,camera) pairs', flush=True)

    # average over cameras common to EVERY configuration in the cell
    cams_by_cell = collections.defaultdict(lambda: collections.defaultdict(set))
    for (emb, label, split, method, cam) in per_cam:
        cams_by_cell[(emb, split)][label].add(cam)
    common_cams = {}
    for cs, bylab in cams_by_cell.items():
        inter = set.intersection(*bylab.values()) if bylab else set()
        common_cams[cs] = sorted(inter)

    print('\n=== cameras averaged per cell (intersection of all configurations) ===')
    for cs in sorted(common_cams):
        print(f'  {cs[0]:11s} {cs[1]:13s} -> {common_cams[cs]}')

    agg = collections.defaultdict(list)
    for (emb, label, split, method, cam), v in per_cam.items():
        if cam in common_cams.get((emb, split), ()):
            agg[(emb, label, split, method)].append((cam, v))
    for k, vs in agg.items():
        rows['|'.join(k)] = dict(
            fvd=float(np.mean([v['fvd'] for _c, v in vs])),
            per_camera={c: round(v['fvd'], 2) for c, v in sorted(vs)},
            n_cameras=len(vs),
            n_windows=int(np.mean([v['n_pred'] for _c, v in vs])),
            n_ref_windows=int(np.mean([v['n_ref'] for _c, v in vs])),
            floor=float(np.mean([v['floor'] for _c, v in vs
                                 if v['floor'] is not None]))
            if any(v['floor'] is not None for _c, v in vs) else None)

    print('\n=== pooled FVD (mean over common cameras), shared reference ===')
    hdr = f'{"cell":34s} ' + ' '.join(f'{m:>9s}' for m in METHOD_ORDER[:5])
    print(hdr)
    for emb in sorted({k[0] for k in agg}):
        for label in [m for m in MODEL_ORDER if any(k[1] == m for k in agg)]:
            for split in ('makovian', 'non_makovian'):
                cells = {k[3]: v for k, v in agg.items()
                         if k[0] == emb and k[1] == label and k[2] == split}
                if not cells:
                    continue
                name = f'{emb}|{label}|{split}'
                vals = []
                for mth in METHOD_ORDER[:5]:
                    v = cells.get(mth)
                    vals.append('      ---' if not v else
                                f'{np.mean([x[1]["fvd"] for x in v]):9.1f}')
                print(f'{name:34s} ' + ' '.join(vals))

    out = dict(protocol='shared real pool keyed (embodiment, camera, split); '
                        'per-camera FVD averaged over cameras common to all '
                        'configurations; all clips decimated to 5 fps',
               budget=budget, min_windows=FVD_MIN_WINDOWS,
               common_cameras={'|'.join(k): v for k, v in common_cams.items()},
               real_pool_windows={'|'.join(k): len(v) for k, v in real.items()},
               floor={'|'.join(k): v for k, v in floor.items()},
               cells=rows,
               per_camera={'|'.join(k): v for k, v in per_cam.items()})
    p = REPO / args.json
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f'\nwrote {p}')


if __name__ == '__main__':
    main()
