"""What pattern does each (Actor Activity, Active Rate) combination actually mean?

Actor Activity = average subject motion, px per source frame.
Active Rate    = fraction of frame pairs where that motion is >= 1 px.

Neither number alone says "good" or "bad". The PAIR of them names the failure mode,
because they answer different questions: "how much does it move" vs "how often".

  FREEZE   rate ~ 0                    -> nothing moves at all
  JITTER   rate high, activity low     -> moves constantly, tiny amplitude = shivering
  LURCH    rate low, activity high     -> still, then one big jump = teleport
  EXPLODE  activity enormous           -> the frame is melting
  normal   everything else

For each bucket this prints how the OTHER metrics behave, which is the point: the
buckets where Visual Integrity / AMC stay high are exactly the ones you cannot see
without these two columns.

Usage:
  python scripts/_demo_motion_patterns.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402

FREEZE_RATE = 0.05
JITTER_RATE = 0.80
JITTER_ACT = 1.5
LURCH_RATE = 0.30
LURCH_ACT = 3.0


def bucket(activity, rate, explode_at):
    if activity >= explode_at:
        return 'EXPLODE'
    if rate < FREEZE_RATE:
        return 'FREEZE'
    if rate >= JITTER_RATE and activity < JITTER_ACT:
        return 'JITTER'
    if rate < LURCH_RATE and activity >= LURCH_ACT:
        return 'LURCH'
    return 'normal'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--examples', type=int, default=2)
    args = ap.parse_args()
    rows = T.load_records(Path(args.run))

    clips = []
    for r in rows:
        sm = T.subject_metrics(r)
        act, rate = sm['actor_motion_raw'], sm['actor_motion_active_rate']
        if act is None or rate is None:
            continue
        clips.append((act, rate, sm, r))
    acts = sorted(c[0] for c in clips)
    explode_at = acts[int(0.99 * len(acts))]
    print(f'{len(clips)} clips with a motion reading; EXPLODE bar = p99 activity = '
          f'{explode_at:.2f} px/frame\n')

    buckets = collections.defaultdict(list)
    for act, rate, sm, r in clips:
        buckets[bucket(act, rate, explode_at)].append((act, rate, sm, r))

    print('=' * 92)
    print('WHAT EACH PATTERN LOOKS LIKE ON THE OTHER METRICS')
    print('=' * 92)
    print(f'{"pattern":<9}{"n":>7}{"share":>8}{"Activity":>10}{"ActiveRate":>12}'
          f'{"VisualInteg":>13}{"AMC":>8}{"AMC unmeasurable":>19}')
    order = ['FREEZE', 'JITTER', 'LURCH', 'EXPLODE', 'normal']
    for b in order:
        g = buckets.get(b) or []
        if not g:
            continue
        vi = [v for v in (T.gated(r, 'visual_integrity') for *_, r in g) if v is not None]
        amc = [sm['active_motion_continuity'] for _, _, sm, _ in g
               if sm['active_motion_continuity'] is not None]
        dead = sum(1 for _, _, sm, _ in g if sm['active_motion_continuity'] is None)
        print(f'{b:<9}{len(g):>7}{100 * len(g) / len(clips):>7.1f}%'
              f'{T.median([a for a, _, _, _ in g]):>10.2f}'
              f'{T.median([x for _, x, _, _ in g]):>12.3f}'
              f'{T.median(vi) if vi else float("nan"):>13.1f}'
              f'{T.median(amc) if amc else float("nan"):>8.1f}'
              f'{100 * dead / len(g):>18.1f}%')

    print('\n' + '=' * 92)
    print('WHICH METHOD PRODUCES WHICH PATTERN  (% of that method\'s clips)')
    print('=' * 92)
    per_method = collections.defaultdict(collections.Counter)
    for b, g in buckets.items():
        for *_, r in g:
            per_method[r.get('method') or 'gt'][b] += 1
    hdr = [b for b in order if b != 'normal']
    print(f'{"method":<14}{"n":>7}' + ''.join(f'{b:>10}' for b in hdr))
    for m in sorted(per_method, key=lambda k: -sum(
            per_method[k][b] for b in hdr) / max(sum(per_method[k].values()), 1)):
        c = per_method[m]
        tot = sum(c.values())
        print(f'{m:<14}{tot:>7}' + ''.join(f'{100 * c[b] / tot:>9.1f}%' for b in hdr))

    print('\n' + '=' * 92)
    print('REAL CLIPS TO GO AND WATCH')
    print('=' * 92)
    for b in order:
        if b == 'normal':
            continue
        g = buckets.get(b) or []
        if not g:
            continue
        # most extreme examples of each pattern
        if b == 'FREEZE':
            g = sorted(g, key=lambda x: x[1])
        elif b == 'JITTER':
            g = sorted(g, key=lambda x: (x[0], -x[1]))
        elif b == 'LURCH':
            g = sorted(g, key=lambda x: (x[1], -x[0]))
        else:
            g = sorted(g, key=lambda x: -x[0])
        print(f'\n--- {b} ---')
        for act, rate, sm, r in g[:args.examples]:
            vi = T.gated(r, 'visual_integrity')
            amc = sm['active_motion_continuity']
            print(f'  Activity {act:6.2f}  ActiveRate {rate:5.3f}  '
                  f'VI {vi if vi is None else round(vi, 1)}  '
                  f'AMC {amc if amc is None else round(amc, 1)}')
            print(f'    {r["method"]:<12} {r["model"]} / {r["embodiment"]} / '
                  f'{r["view"]} ({r["camera_mount"]})')
            print(f'    {r["path"]}')


if __name__ == '__main__':
    main()
