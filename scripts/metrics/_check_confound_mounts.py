"""Check whether CONFOUNDED_MOUNTS is still correct on the full run.

The shipped dict was derived on a ~1.5k-clip head-heavy sample. The full run has
26k records, so re-measure rho(actor_motion_raw, col) per mount with a bootstrap CI,
and separately per method-class (gt / dense / accel) so a pooled-population artifact
cannot masquerade as a within-clip confound.

Usage:
  python scripts/metrics/_check_confound_mounts.py --run metrics/WorldScore/results/dense_nr_full
"""
import argparse
import collections
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'assets' / 'scripts'))
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


def rho_ci(pairs, n_boot=400, seed=0):
    rnd = random.Random(seed)
    point = spearman([a for a, _ in pairs], [b for _, b in pairs])
    boots = []
    n = len(pairs)
    for _ in range(n_boot):
        s = [pairs[rnd.randrange(n)] for _ in range(n)]
        boots.append(spearman([a for a, _ in s], [b for _, b in s]))
    boots.sort()
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def klass(r):
    m = r.get('method')
    if m == 'gt' or str(r.get('model', '')).startswith('gt_'):
        return 'gt'
    return 'dense' if m == 'dense' else 'accel'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--bar', type=float, default=0.35)
    args = ap.parse_args()
    rows = T.load_records(Path(args.run))

    # (col, mount, class) -> [(motion, value)]
    data = collections.defaultdict(list)
    for r in rows:
        sm = T.subject_metrics(r)
        mo = sm['actor_motion_raw']
        if mo is None:
            continue
        k = klass(r)
        for col in ('visual_integrity', 'active_motion_continuity', 'text_condition_alignment'):
            v = T.gated(r, col)
            if v is not None:
                data[(col, r['camera_mount'], k)].append((mo, v))
                data[(col, r['camera_mount'], 'ALL')].append((mo, v))

    print(f'bar = |rho| >= {args.bar} means confounded\n')
    for col in ('visual_integrity', 'active_motion_continuity', 'text_condition_alignment'):
        print('=' * 76)
        print(col)
        print('=' * 76)
        print(f'{"mount":<9}{"class":<8}{"n":>7}{"rho":>9}{"95% CI":>20}   verdict')
        for mount in ('head', 'static', 'wrist'):
            for k in ('ALL', 'gt', 'dense', 'accel'):
                pairs = data.get((col, mount, k)) or []
                if len(pairs) < 50:
                    continue
                rho, lo, hi = rho_ci(pairs)
                bad = abs(rho) >= args.bar
                shipped = T.CONFOUNDED_MOUNTS.get(col, {}).get(mount) is not None
                tag = ''
                if bad and not shipped:
                    tag = '  <<< CONFOUNDED but NOT in CONFOUNDED_MOUNTS'
                elif not bad and shipped and k == 'ALL':
                    tag = '  <<< flagged in code but now below bar'
                print(f'{mount:<9}{k:<8}{len(pairs):>7}{rho:>+9.3f}'
                      f'{f"[{lo:+.3f},{hi:+.3f}]":>20}   '
                      f'{"CONFOUNDED" if bad else "ok":<11}{tag}')
        print()

    print('shipped CONFOUNDED_MOUNTS =', dict(T.CONFOUNDED_MOUNTS))


if __name__ == '__main__':
    main()
