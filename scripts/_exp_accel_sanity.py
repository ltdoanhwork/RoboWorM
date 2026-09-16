"""E7 - Do the NR metrics PENALISE acceleration artifacts, or reward them?

Before scoring 9 acceleration trees we must know the metrics move the right way. So
we synthesise the artifact each accel family produces, from real dense pred clips,
and check the sign of the metric change. Ground truth here is definitional: every
degradation below is strictly worse video than its source, so any metric that goes
UP is broken for the acceleration benchmark.

Degradations (each mimics one accel family's failure mode):
  dup2      every 2nd frame replaced by its predecessor  -> step caching / frame reuse
  smooth3   temporal box filter over 3 frames            -> over-smoothing from cache
  blur      spatial gaussian blur                        -> token pruning / smear
  block     8x8 downsample-upsample                      -> block-sparse attn artifact

Writes a manifest.json in run_dense_nr.py's own row schema, so the real scorer runs
unmodified via `--reuse-manifest` (no second implementation of the metrics).

Usage:
  python scripts/_exp_accel_sanity.py --build --n 8 --out /tmp/accel_sanity
  bash scripts/run_dense_nr.sh --out /tmp/accel_sanity --workers 2 --reuse-manifest
  python scripts/_exp_accel_sanity.py --report --out /tmp/accel_sanity
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parent.parent

VARIANTS = ('dup2', 'smooth3', 'blur', 'block', 'blur_mild', 'blur_real')


def degrade(frames, kind):
    import cv2
    out = frames.copy()
    if kind == 'dup2':
        for i in range(1, len(out), 2):
            out[i] = out[i - 1]
    elif kind == 'smooth3':
        f = frames.astype(np.float32)
        for i in range(1, len(out) - 1):
            out[i] = np.clip((f[i - 1] + f[i] + f[i + 1]) / 3.0, 0, 255).astype(np.uint8)
    elif kind in ('blur', 'blur_mild', 'blur_real'):
        # The sigma=1.6 'blur' destroys sharpness to ~7% of the original, far beyond
        # anything a real method does (measured: SiTo 0.76x, ITM 0.74x, SVG2 0.93x).
        # These two milder points calibrate the AMC bias at the REAL severity, so the
        # bias can be bounded instead of extrapolated from an extreme.
        sigma = {'blur': 1.6, 'blur_mild': 0.8, 'blur_real': 0.45}[kind]
        for i in range(len(out)):
            out[i] = cv2.GaussianBlur(out[i], (0, 0), sigma)
    elif kind == 'block':
        h, w = out.shape[1:3]
        for i in range(len(out)):
            small = cv2.resize(out[i], (w // 8, h // 8), interpolation=cv2.INTER_AREA)
            out[i] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        raise ValueError(kind)
    return out


def build(args):
    import cv2
    src_run = args.source_run
    rows = [json.loads(p.read_text()) for p in sorted((src_run / 'results').glob('*.json'))]
    # Only real predictions that produced a valid AMC: the question is whether AMC
    # *changes in the right direction*, so it must exist on the source clip.
    pool = [r for r in rows if r['method'] == 'dense' and not r['model'].startswith('gt')
            and r.get('active_motion_continuity') is not None
            and r.get('actor_motion_raw') is not None]
    # Spread over model x mount so the verdict is not one family's quirk.
    buckets = {}
    for r in pool:
        buckets.setdefault((r['model'], r['camera_mount']), []).append(r)
    for v in buckets.values():
        v.sort(key=lambda r: r['id'])
    picks, i = [], 0
    while len(picks) < args.n and any(len(v) > i for v in buckets.values()):
        for k in sorted(buckets):
            if len(buckets[k]) > i and len(picks) < args.n:
                picks.append(buckets[k][i])
        i += 1

    vid = args.out / 'videos'
    if vid.exists():
        shutil.rmtree(vid)
    vid.mkdir(parents=True)
    manifest = []
    for r in picks:
        cap = cv2.VideoCapture(r['path'])
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        if len(frames) < 6:
            continue
        arr = np.stack(frames)
        # 'orig' is re-encoded through the same writer as the variants, so the
        # comparison isolates the degradation and not the codec.
        for kind in ('orig',) + VARIANTS:
            out_arr = arr if kind == 'orig' else degrade(arr, kind)
            name = f"{r['id']}__{kind}.mp4"
            dest = vid / name
            w = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                (arr.shape[2], arr.shape[1]))
            for fr in out_arr:
                w.write(fr)
            w.release()
            row = dict(r)
            for k in list(row):
                # keep only manifest fields; drop every scored metric
                if k not in ('model', 'embodiment', 'split', 'view', 'camera_mount',
                             'layout', 'view_index', 'n_views', 'instruction', 'method'):
                    row.pop(k)
            row['path'] = str(dest)
            row['relative'] = name
            row['method'] = 'dense' if kind == 'orig' else kind
            row['id'] = hashlib.sha256(name.encode()).hexdigest()[:20]
            # pair the variant to its own source clip
            row['pair_key'] = f"sanity/{r['id']}/{row['view']}"
            row['sanity_source_id'] = r['id']
            row['sanity_variant'] = kind
            manifest.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'built {len(manifest)} clips ({len(manifest)//(len(VARIANTS)+1)} sources x '
          f'{len(VARIANTS)+1} variants) -> {args.out}/manifest.json')
    print('now run:')
    print(f'  bash scripts/run_dense_nr.sh --out {args.out} --workers 2 --reuse-manifest')


def boot_ci(d, alpha=0.05, resamples=10000):
    """Percentile bootstrap CI of the median, matching table_dense_nr.py."""
    if len(d) < 3:
        return None, None
    a = np.asarray(d, float)
    rng = np.random.default_rng(0)
    stats = np.median(rng.choice(a, (resamples, a.size), replace=True), axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


KEYS = [('active_motion_continuity', 'AMC', 'high'),
        ('actor_flicker_raw', 'ActorFlicker', 'low'),
        ('actor_motion_raw', 'ActorActivity', 'desc'),
        ('visual_integrity', 'VisualIntegrity', 'high'),
        ('motion_continuity_global_ssim', 'globalSSIM(v2)', 'high')]

# E3 candidate flicker statistics live in a nested dict, so they are flattened onto the
# record before the report runs. This is THE decisive test for them, stronger than the
# rho-vs-motion criterion: a flicker statistic that goes DOWN under a definitionally
# worse video cannot be repaired by any normalization, because the artifact suppresses
# the very residual the statistic measures.
FLICKER_VARIANT_PREFIX = 'flickvar_'


def flatten_variants(rows):
    """Hoist `actor_flicker_variants` onto each record and return the extra KEYS."""
    names = set()
    for r in rows:
        for k, v in (r.get('actor_flicker_variants') or {}).items():
            r[FLICKER_VARIANT_PREFIX + k] = v
            names.add(k)
    # `copy`/`flow`/`grad`/`align_pred` are diagnostics, not candidate statistics.
    cands = sorted(names - {'copy', 'flow', 'grad', 'align_pred'})
    return [(FLICKER_VARIANT_PREFIX + k, f'flick:{k}', 'low') for k in cands]


def report(args):
    rows = [json.loads(p.read_text()) for p in sorted((args.out / 'results').glob('*.json'))]
    keys = KEYS + flatten_variants(rows)
    by = {}
    for r in rows:
        by.setdefault(r.get('sanity_source_id'), {})[r.get('sanity_variant')] = r
    print(f'{len(rows)} scored clips, {len(by)} sources\n')
    print('Paired delta (degraded - orig). A DEGRADED clip is definitionally worse,')
    print('so for a "high is better" metric the delta MUST be negative.')
    print('Verdict requires the 95% bootstrap CI to EXCLUDE 0 -- a median with the')
    print('wrong sign but a CI spanning 0 is noise, not evidence of a broken metric.\n')
    print('  %-14s %-16s %5s %10s %20s %s' % (
        'variant', 'metric', 'n', 'median', '95% CI', 'verdict'))
    verdicts = {}
    for kind in VARIANTS:
        for key, label, direction in keys:
            d = []
            for sid, v in by.items():
                a, b = v.get('orig'), v.get(kind)
                if not a or not b:
                    continue
                if a.get(key) is not None and b.get(key) is not None:
                    d.append(b[key] - a[key])
            if len(d) < 3:
                print('  %-14s %-16s %5d %10s %20s (insufficient)' % (kind, label, len(d), '--', '--'))
                continue
            med = float(np.median(d))
            lo, hi = boot_ci(d)
            resolved = lo is not None and (lo > 0 or hi < 0)
            ci = f'[{lo:+.4f}, {hi:+.4f}]' if lo is not None else '--'
            if direction == 'desc':
                verdict = 'descriptive'
            elif not resolved:
                verdict = 'ns (CI spans 0)'
            elif direction == 'high':
                verdict = 'OK penalises' if med < 0 else '** REWARDS ARTIFACT **'
            else:
                verdict = 'OK penalises' if med > 0 else '** REWARDS ARTIFACT **'
            if direction != 'desc':
                verdicts[(kind, label)] = verdict
            print('  %-14s %-16s %5d %+10.4f %20s %s' % (kind, label, len(d), med, ci, verdict))
        print()
    bad = sorted(k for k, v in verdicts.items() if 'REWARDS' in v)
    weak = sorted(k for k, v in verdicts.items() if 'ns (' in v)
    print('== SUMMARY ==')
    if bad:
        print('  BROKEN for the acceleration benchmark (wrong sign, CI excludes 0):')
        for kind, label in bad:
            print(f'    {label} is rewarded by "{kind}" degradation')
    else:
        print('  no directional metric is significantly rewarded by an artifact')
    if weak:
        print('  Blind (no significant response to the artifact at all):')
        for kind, label in weak:
            print(f'    {label} does not react to "{kind}"')
    (args.out / 'sanity_report.json').write_text(json.dumps(
        {f'{k[0]}/{k[1]}': v for k, v in verdicts.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source-run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_v3_div30')
    ap.add_argument('--n', type=int, default=8)
    args = ap.parse_args()
    if args.build:
        build(args)
    if args.report:
        report(args)


if __name__ == '__main__':
    main()
