"""Sanity report for a dense NR smoke run: coverage, distributions, confounds, GT pairs.

Usage: python scripts/analyze_dense_nr_smoke.py --run metrics/WorldScore/results/dense_nr_v3_smoke
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

KEYS = ['visual_integrity', 'text_condition_alignment', 'actor_motion_raw',
        'actor_motion_active_rate', 'active_motion_continuity',
        'motion_continuity_global_ssim', 'actor_flicker_raw']
LOWER_BETTER = {'actor_flicker_raw'}


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5:
        return float('nan')
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def q(values):
    v = np.array([x for x in values if x is not None], float)
    if not len(v):
        return 'n=0'
    return f'n={len(v)} p10={np.percentile(v, 10):.3g} med={np.median(v):.3g} p90={np.percentile(v, 90):.3g}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--extremes', type=int, default=3)
    args = ap.parse_args()
    manifest = json.loads((args.run / 'manifest.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((args.run / 'results').glob('*.json'))]
    errors = [json.loads(p.read_text()) for p in sorted((args.run / 'errors').glob('*.json'))]
    group = lambda r: (r['model'], r['embodiment'], r['camera_mount'])

    print(f'# coverage: {len(rows)}/{len(manifest)} done, {len(errors)} errors')
    for e in errors:
        print('  ERROR', e['id'], e['relative'], e['view'], e['traceback'].strip().splitlines()[-1][:160])
    done = collections.Counter(map(group, rows)); total = collections.Counter(map(group, manifest))
    print('  elapsed/clip', q([r['elapsed_seconds'] for r in rows]))
    print('  validity', dict(collections.Counter(r['actor_validity'] for r in rows)))

    print('\n# per group (model, embodiment, mount)')
    by = collections.defaultdict(list)
    for r in rows:
        by[group(r)].append(r)
    for g in sorted(total):
        rs = by.get(g, [])
        print(f'## {g} {done[g]}/{total[g]}  motion_none={sum(r["actor_motion_raw"] is None for r in rs)}'
              f'  amc_none={sum(r["active_motion_continuity"] is None for r in rs)}'
              f'  valid_pair_frac {q([r["actor_motion_valid_pair_frac"] for r in rs])}')
        for k in KEYS:
            print(f'   {k:32s} {q([r.get(k) for r in rs])}')

    print('\n# confounds (Spearman, robot_detected)')
    ok = [r for r in rows if r['actor_motion_raw'] is not None]
    for k in ('active_motion_continuity', 'motion_continuity_global_ssim', 'actor_flicker_raw', 'visual_integrity'):
        rr = [r for r in ok if r.get(k) is not None]
        per_mount = {m: round(spearman([r[k] for r in rr if r['camera_mount'] == m],
                                       [r['actor_motion_raw'] for r in rr if r['camera_mount'] == m]), 2)
                     for m in ('static', 'head', 'wrist')}
        print(f'  {k:32s} vs actor_motion: all={spearman([r[k] for r in rr], [r["actor_motion_raw"] for r in rr]):.2f} {per_mount}')

    print('\n# GT vs pred on same episode/view (pred - gt)')
    by_key = collections.defaultdict(dict)
    for r in rows:
        by_key[r['pair_key']][r['model']] = r
    pairs = collections.defaultdict(list)
    for d in by_key.values():
        if 'gt' in d:
            for m, r in d.items():
                if m != 'gt':
                    pairs[(m, r['camera_mount'])].append((d['gt'], r))
    for (m, mount), ps in sorted(pairs.items()):
        print(f'## {m} {mount} n_pairs={len(ps)}')
        for k in KEYS:
            vals = [(g[k], p[k]) for g, p in ps if g.get(k) is not None and p.get(k) is not None]
            if not vals:
                continue
            diff = np.array([p - g for g, p in vals])
            better = np.mean([(g < p) if k in LOWER_BETTER else (g > p) for g, p in vals])
            extra = ''
            if k == 'actor_motion_raw':
                ratio = [p / g for g, p in vals if g > 1e-3]
                extra = f' pred/gt ratio med={np.median(ratio):.2f}' if ratio else ''
            print(f'   {k:32s} n={len(vals)} med_diff={np.median(diff):+.3g} gt_better={better:.2f}{extra}')

    print('\n# extremes for visual audit')
    for k in ('active_motion_continuity', 'actor_motion_raw', 'actor_flicker_raw', 'visual_integrity'):
        rr = sorted((r for r in rows if r.get(k) is not None), key=lambda r: r[k])
        for tag, sel in (('low', rr[:args.extremes]), ('high', rr[-args.extremes:])):
            for r in sel:
                print(f'  {k} {tag} {r[k]:.3g} {r["model"]} {r["camera_mount"]} {r["relative"]} {r["view"]}')


if __name__ == '__main__':
    main()
