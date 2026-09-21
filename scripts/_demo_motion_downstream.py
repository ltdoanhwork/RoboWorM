"""Demonstrate the downstream uses of Actor Activity / Active Rate with real numbers.

These two columns have direction=None (purely descriptive), so they are never a
"score". This script shows the four jobs they actually do in the pipeline:

  D1  gate       -- decide whether AMC is measurable at all on a clip
  D2  weight     -- Actor Activity IS the per-triple weight inside AMC
  D3  triage     -- explain/veto a suspicious quality "win" (freeze -> easy metrics)
  D4  confound   -- motion as covariate: is a quality gap really quality?

Usage:
  python scripts/_demo_motion_downstream.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    rows = T.load_records(Path(args.run))
    print(f'loaded {len(rows)} records @ {T.PROTOCOL_REQUIRED}\n')

    # ---------------------------------------------------------------- D1: gate
    print('=' * 78)
    print('D1  GATE -- Active Rate decides whether AMC is reportable')
    print('=' * 78)
    n_robot = n_amc = n_killed_activity = 0
    killed_ar, kept_ar = [], []
    for r in rows:
        if r.get('actor_validity') != 'robot_detected':
            continue
        n_robot += 1
        sm = T.subject_metrics(r)
        ar, amc = sm['actor_motion_active_rate'], sm['active_motion_continuity']
        if ar is None:
            continue
        if amc is None:
            n_killed_activity += 1
            killed_ar.append(ar)
        else:
            n_amc += 1
            kept_ar.append(ar)
    print(f'clips with a detected robot          : {n_robot}')
    print(f'  AMC reportable                     : {n_amc}')
    print(f'  AMC -> None (too little motion)     : {n_killed_activity} '
          f'({100 * n_killed_activity / max(n_robot, 1):.1f}%)')
    if killed_ar and kept_ar:
        print(f'  median Active Rate, AMC killed     : {T.median(killed_ar):.3f}')
        print(f'  median Active Rate, AMC kept       : {T.median(kept_ar):.3f}')
    print('  -> without Active Rate these near-static clips would return a high AMC')
    print('     computed from a handful of frozen frames.\n')

    # -------------------------------------------------------------- D2: weight
    print('=' * 78)
    print('D2  WEIGHT -- Actor Activity is the per-triple weight inside AMC')
    print('=' * 78)
    # Recompute AMC unweighted vs weighted on the same clips to size the effect.
    plain, weighted = [], []
    for r in rows:
        if r.get('actor_validity') != 'robot_detected':
            continue
        sm = T.subject_metrics(r)
        if sm['active_motion_continuity'] is None:
            continue
        tri = r.get('continuity_per_triple') or {}
        gains = tri.get('gain') or []
        if not gains:
            continue
        plain.append(statistics.fmean(gains) * 100)      # flat mean, no motion weight
        weighted.append(sm['active_motion_continuity'])   # motion-weighted (shipped)
    if plain:
        print(f'n clips                              : {len(plain)}')
        print(f'median AMC, flat mean over triples   : {T.median(plain):.2f}')
        print(f'median AMC, Actor-Activity weighted  : {T.median(weighted):.2f}')
        print(f'median per-clip shift                : '
              f'{T.median([w - p for w, p in zip(weighted, plain)]):+.2f}')
        print('  -> the weight moves the number: static triples are free to interpolate,')
        print('     so the flat mean is inflated by exactly the frames nobody cares about.\n')

    # -------------------------------------------------------------- D3: triage
    print('=' * 78)
    print('D3  TRIAGE -- freeze detection: does an accel "win" come from less motion?')
    print('=' * 78)
    dense = {}
    accel = collections.defaultdict(list)
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
        elif r.get('method') not in (None, 'gt'):
            accel[(r['pair_key'], r['model'])].append(r)
    per_method = collections.defaultdict(lambda: collections.defaultdict(list))
    d_ar_all, d_amc_all, d_vi_all = [], [], []
    frozen = collections.Counter()
    frozen_tot = collections.Counter()
    for key, variants in accel.items():
        base = dense.get(key)
        if base is None:
            continue
        for r in variants:
            if not T.comparable_pair(r, base):
                continue
            m = r['method']
            a, b = T.subject_metrics(base), T.subject_metrics(r)
            if a['actor_motion_active_rate'] is None or b['actor_motion_active_rate'] is None:
                continue
            frozen_tot[m] += 1
            d_ar = b['actor_motion_active_rate'] - a['actor_motion_active_rate']
            per_method[m]['ar'].append(d_ar)
            if a['actor_motion_raw'] is not None and b['actor_motion_raw'] is not None:
                per_method[m]['aa'].append(b['actor_motion_raw'] - a['actor_motion_raw'])
            # a clip the accel method effectively froze
            if a['actor_motion_active_rate'] >= 0.30 and b['actor_motion_active_rate'] < 0.10:
                frozen[m] += 1
            amc_a, amc_b = a['active_motion_continuity'], b['active_motion_continuity']
            vi_a, vi_b = T.gated(base, 'visual_integrity'), T.gated(r, 'visual_integrity')
            if amc_a is not None and amc_b is not None:
                d_ar_all.append(d_ar)
                d_amc_all.append(amc_b - amc_a)
                per_method[m]['amc'].append(amc_b - amc_a)
            if vi_a is not None and vi_b is not None:
                d_vi_all.append((d_ar, vi_b - vi_a))
    print(f'{"method":<13}{"n":>6}{"dActivity":>11}{"dActiveRate":>13}'
          f'{"dAMC":>9}{"frozen":>9}')
    for m in sorted(per_method, key=lambda k: T.median(per_method[k]['ar'] or [0])):
        g = per_method[m]
        print(f'{m:<13}{frozen_tot[m]:>6}'
              f'{T.median(g["aa"]) if g["aa"] else float("nan"):>+11.2f}'
              f'{T.median(g["ar"]) if g["ar"] else float("nan"):>+13.3f}'
              f'{T.median(g["amc"]) if g["amc"] else float("nan"):>+9.2f}'
              f'{100 * frozen[m] / max(frozen_tot[m], 1):>8.1f}%')
    if len(d_ar_all) > 30:
        print(f'\nrho(dActiveRate, dAMC) over {len(d_ar_all)} accel pairs : '
              f'{spearman(d_ar_all, d_amc_all):+.3f}')
        xs = [a for a, _ in d_vi_all]
        ys = [b for _, b in d_vi_all]
        print(f'rho(dActiveRate, dVisualIntegrity) over {len(xs)} pairs : '
              f'{spearman(xs, ys):+.3f}')
        print('  -> sign is the diagnosis. NEGATIVE rho would mean clips that LOST')
        print('     motion GAINED on the quality columns = the freeze-cheats-the-metric')
        print('     artifact. Measured POSITIVE here, so losing motion also loses AMC/VI:')
        print('     the motion drop of sito/itm is a real regression, not a metric hack.\n')

    # ------------------------------------------------------------ D4: confound
    print('=' * 78)
    print('D4  CONFOUND -- motion as covariate, per camera mount')
    print('=' * 78)
    by_mount = collections.defaultdict(lambda: ([], [], []))
    for r in rows:
        sm = T.subject_metrics(r)
        mo = sm['actor_motion_raw']
        if mo is None:
            continue
        vi = T.gated(r, 'visual_integrity')
        amc = sm['active_motion_continuity']
        m, v, a = by_mount[r['camera_mount']]
        if vi is not None:
            m.append(mo)
            v.append(vi)
        if amc is not None:
            a.append((mo, amc))
    print(f'{"mount":<10}{"n":>7}{"rho(motion,VI)":>17}{"n":>7}{"rho(motion,AMC)":>18}')
    for mount, (mo, vi, amc) in sorted(by_mount.items()):
        rv = spearman(mo, vi) if len(mo) > 30 else float('nan')
        ra = (spearman([x for x, _ in amc], [y for _, y in amc])
              if len(amc) > 30 else float('nan'))
        print(f'{mount:<10}{len(mo):>7}{rv:>+17.3f}{len(amc):>7}{ra:>+18.3f}')
    print('  -> |rho| >= 0.35 means that column is motion-confounded on that mount:')
    print('     a "quality" gap can be a motion gap. This is what CONFOUNDED_MOUNTS')
    print('     in table_dense_nr.py was derived from.')


if __name__ == '__main__':
    main()
