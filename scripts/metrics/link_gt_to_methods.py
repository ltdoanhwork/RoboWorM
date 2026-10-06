#!/usr/bin/env python3
"""Expose the ground truth rendered under sampling_dataset/dense to every other method.

`gt_for()` looks for the reference BESIDE the prediction, so a GT that exists only in the
dense tree gives the other nine methods no `vs_gt` row. The GT for a clip is the same file
regardless of which acceleration method produced the prediction, so this hard-links rather
than copies: ~1.6k references across 10 methods would otherwise be ~15k identical files.

A link is created only where a prediction actually exists, so a method that never ran a
cell does not acquire an orphan reference that would later look like missing coverage.

Usage:
    python scripts/metrics/link_gt_to_methods.py --dry-run
    python scripts/metrics/link_gt_to_methods.py
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DENSE = REPO / 'sampling_dataset' / 'dense'
METHODS = ['worldcache', 'dicache', 'fastercache', 'sito', 'itm',
           'pisa', 'radial', 'svg1', 'svg2']


def pred_for(gt: Path) -> Path:
    """The prediction this reference belongs to."""
    if gt.name == 'full_gt.mp4':
        return gt.with_name('full_pred.mp4')
    if gt.name == 'gt_all_views.mp4':
        return gt.with_name('pred_all_views.mp4')
    return gt.with_name(f'{gt.stem[:-3]}.mp4')          # <episode>_gt.mp4 -> <episode>.mp4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--method', action='append', default=[], choices=METHODS)
    args = ap.parse_args()
    methods = args.method or METHODS

    gts = sorted(p for p in DENSE.rglob('*_gt.mp4')) + \
          sorted(p for p in DENSE.rglob('full_gt.mp4'))
    print(f'{len(gts)} GT files under sampling_dataset/dense')

    tally: Counter = Counter()
    per_method: Counter = Counter()
    for gt in gts:
        rel = gt.relative_to(DENSE)
        for method in methods:
            target = REPO / 'sampling_dataset' / method / rel
            if target.exists():
                tally['already'] += 1
                continue
            if not target.parent.is_dir():
                tally['no_cell_dir'] += 1
                continue
            if not pred_for(target).exists():
                tally['no_prediction'] += 1
                continue
            if args.dry_run:
                tally['would_link'] += 1
                per_method[method] += 1
                continue
            try:
                os.link(gt, target)
                tally['linked'] += 1
                per_method[method] += 1
            except OSError as exc:
                if exc.errno == 18:          # cross-device: fall back to a copy
                    import shutil
                    shutil.copy2(gt, target)
                    tally['copied'] += 1
                    per_method[method] += 1
                else:
                    tally[f'error_{exc.errno}'] += 1

    print('\nresult:')
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f'  {k:16s} {v:6d}')
    if per_method:
        print('\nper method:')
        for k, v in sorted(per_method.items()):
            print(f'  {k:14s} {v:6d}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
