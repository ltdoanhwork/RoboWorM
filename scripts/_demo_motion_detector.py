"""Does a LARGE Actor Activity / Active Rate flag an anomaly on its own?

Two tests, same data:

  A) ABSOLUTE value as a detector. Bucket every clip by its own activity / active
     rate and look at the quality columns. If the absolute number were a detector,
     the extreme buckets would be the bad ones.
  B) DEVIATION from the clip's own dense counterpart. Same buckets, but on
     |activity(accel) - activity(dense)| for the SAME episode+view.

Usage:
  python scripts/_demo_motion_detector.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402


def quantile_edges(values, k):
    v = sorted(values)
    return [v[int(i * len(v) / k)] for i in range(1, k)]


def which(edges, x):
    i = 0
    while i < len(edges) and x >= edges[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    rows = T.load_records(Path(args.run))

    # -------------------------------------------------- A) absolute as detector
    clips = []
    for r in rows:
        sm = T.subject_metrics(r)
        if sm['actor_motion_raw'] is None:
            continue
        clips.append((sm, r))

    print('=' * 88)
    print('A) Is the ABSOLUTE number a detector?  (all clips, quintiles of Activity)')
    print('=' * 88)
    edges = quantile_edges([sm['actor_motion_raw'] for sm, _ in clips], 5)
    groups = collections.defaultdict(list)
    for sm, r in clips:
        groups[which(edges, sm['actor_motion_raw'])].append((sm, r))
    print(f'{"quintile":<10}{"Activity":>10}{"n":>7}{"VisualInteg":>13}{"AMC":>8}'
          f'{"%GT clips":>11}{"%accel":>9}')
    for q in sorted(groups):
        g = groups[q]
        vi = [v for v in (T.gated(r, 'visual_integrity') for _, r in g) if v is not None]
        amc = [sm['active_motion_continuity'] for sm, _ in g
               if sm['active_motion_continuity'] is not None]
        n_gt = sum(1 for _, r in g if r.get('method') == 'gt')
        n_ac = sum(1 for _, r in g if r.get('method') not in ('gt', 'dense', None))
        print(f'Q{q + 1:<9}{T.median([sm["actor_motion_raw"] for sm, _ in g]):>10.2f}'
              f'{len(g):>7}{T.median(vi) if vi else float("nan"):>13.1f}'
              f'{T.median(amc) if amc else float("nan"):>8.1f}'
              f'{100 * n_gt / len(g):>10.1f}%{100 * n_ac / len(g):>8.1f}%')
    print('  If absolute Activity were a detector, quality would fall monotonically')
    print('  across the quintiles AND the top bucket would not be GT-rich.')

    # ------------------------------------------------- B) deviation as detector
    dense = {}
    accel = collections.defaultdict(list)
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
        elif r.get('method') not in (None, 'gt'):
            accel[(r['pair_key'], r['model'])].append(r)
    pairs = []
    for key, variants in accel.items():
        base = dense.get(key)
        if base is None:
            continue
        a = T.subject_metrics(base)
        if a['actor_motion_raw'] is None:
            continue
        vi_a = T.gated(base, 'visual_integrity')
        for r in variants:
            if not T.comparable_pair(r, base):
                continue
            b = T.subject_metrics(r)
            if b['actor_motion_raw'] is None:
                continue
            vi_b = T.gated(r, 'visual_integrity')
            d_amc = (None if a['active_motion_continuity'] is None
                     or b['active_motion_continuity'] is None
                     else b['active_motion_continuity'] - a['active_motion_continuity'])
            pairs.append({
                'rel': abs(b['actor_motion_raw'] - a['actor_motion_raw'])
                / max(a['actor_motion_raw'], 0.1),
                'd_vi': None if vi_a is None or vi_b is None else vi_b - vi_a,
                'd_amc': d_amc,
                'method': r['method'],
            })

    print('\n' + '=' * 88)
    print('B) Is the DEVIATION from its own dense clip a detector?')
    print('=' * 88)
    edges = quantile_edges([p['rel'] for p in pairs], 5)
    groups = collections.defaultdict(list)
    for p in pairs:
        groups[which(edges, p['rel'])].append(p)
    print(f'{"quintile":<10}{"|dActivity|/dense":>19}{"n":>7}'
          f'{"dVisualInteg":>14}{"dAMC":>9}')
    for q in sorted(groups):
        g = groups[q]
        vi = [p['d_vi'] for p in g if p['d_vi'] is not None]
        amc = [p['d_amc'] for p in g if p['d_amc'] is not None]
        print(f'Q{q + 1:<9}{T.median([p["rel"] for p in g]):>18.1%}{len(g):>7}'
              f'{T.median(vi) if vi else float("nan"):>+14.2f}'
              f'{T.median(amc) if amc else float("nan"):>+9.2f}')
    print('  A monotone fall here means the deviation IS the detector: the further a')
    print('  method drifts from the motion its own dense run produced, the more quality')
    print('  it has lost. That is the anomaly signal -- not the raw magnitude.')


if __name__ == '__main__':
    main()
