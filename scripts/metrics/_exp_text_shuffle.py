"""E2 - Does TextAlignment (CLIPScore) measure anything, or is it saturated?

Discriminating test: score every clip twice with the SAME frames -- once with its own
instruction, once with an instruction taken from a DIFFERENT task in the same
embodiment. If CLIPScore works, matched >> shuffled. If it is saturated on robot
imagery, the two arms coincide and the column must be dropped from the paper.

Deliberately does NOT call score_qual: that also runs GroundingDINO per frame (the
expensive part) and, critically, substitutes a generic caption
("a robot performing a task") when the instruction is empty -- which would silently
dilute the matched arm and fake interpretation (b). Here every caption is asserted
non-empty and clips with no task text (all DreamDojo) are excluded up front.

Usage (via the venv/PYTHONPATH that run_dense_nr.sh sets):
    python scripts/metrics/_exp_text_shuffle.py --run metrics/WorldScore/results/dense_nr_v3_div30 \
        --n 80 --out metrics/WorldScore/results/exp_text_shuffle.json
"""
import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_dense_nr import prepare, task_text, task_of  # noqa: E402  same preprocessing

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from metrics.text_alignment import CLIPScorer  # noqa: E402
from metrics.third_party import write_frames  # noqa: E402

B = 10000


def boot_ci(d, alpha=0.05):
    if len(d) < 3:
        return None, None
    a = np.asarray(d, float)
    rng = np.random.default_rng(0)
    stats = np.median(rng.choice(a, (B, a.size), replace=True), axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--n', type=int, default=80)
    ap.add_argument('--stride', type=int, default=4, help='score every Nth analysis frame')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(p.read_text()) for p in sorted((args.run / 'results').glob('*.json'))]
    # Only clips that actually carry a prompt: this is what excludes the
    # generic-caption fallback from contaminating the matched arm.
    usable = []
    for r in rows:
        text = task_text(r.get('instruction'))
        if text and r.get('text_condition_applicable'):
            usable.append((r, text))
    print(f'{len(usable)}/{len(rows)} clips have a real prompt (rest excluded, no fallback used)')

    # A shuffled prompt must come from a DIFFERENT task, same embodiment: same domain
    # vocabulary, wrong content. Anything else confounds domain shift with task match.
    pool = {}
    for r, text in usable:
        pool.setdefault(r['embodiment'], {}).setdefault(task_of(r), set()).add(text)

    rng = random.Random(0)
    # Spread the budget over models so no single model dominates the median.
    by_model = {}
    for item in usable:
        by_model.setdefault(item[0]['model'], []).append(item)
    for v in by_model.values():
        rng.shuffle(v)
    picks, i = [], 0
    while len(picks) < args.n and any(len(v) > i for v in by_model.values()):
        for m in sorted(by_model):
            if len(by_model[m]) > i and len(picks) < args.n:
                picks.append(by_model[m][i])
        i += 1
    print(f'scoring {len(picks)} clips x 2 arms, stride {args.stride}')

    clip = CLIPScorer('cuda')
    out = []
    for k, (r, matched) in enumerate(picks, 1):
        tasks = pool[r['embodiment']]
        others = [t for name, ts in tasks.items() if name != task_of(r) for t in ts]
        if not others:
            continue
        shuffled = rng.choice(sorted(others))
        assert matched and shuffled and matched != shuffled

        tmpdir = Path(tempfile.mkdtemp(prefix='e2_'))
        try:
            frames, _ = prepare(r, tmpdir / 'clip.mp4')
            paths = write_frames(frames, tmpdir, stride=args.stride)
            ms, ss = [], []
            for p in paths:
                ms.append(clip(p, matched))
                ss.append(clip(p, shuffled))
        except Exception as exc:  # a bad clip must not kill an 80-clip sweep
            print(f'  [{k}] SKIP {r["model"]}: {type(exc).__name__}: {exc}')
            continue
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        rec = dict(id=r['id'], model=r['model'], embodiment=r['embodiment'],
                   camera_mount=r['camera_mount'], n_frames=len(paths),
                   matched_prompt=matched, shuffled_prompt=shuffled,
                   matched=float(np.mean(ms)), shuffled=float(np.mean(ss)))
        rec['delta'] = rec['matched'] - rec['shuffled']
        out.append(rec)
        if k % 10 == 0 or k == len(picks):
            d = [o['delta'] for o in out]
            print(f'  [{k}/{len(picks)}] n={len(d)} running median delta = {np.median(d):+.3f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    deltas = [o['delta'] for o in out]
    lo, hi = boot_ci(deltas)
    verdict = ('keep (a): CLIPScore discriminates' if lo is not None and lo > 0 and np.median(deltas) > 2.0
               else 'DROP (b): saturated / not discriminative')
    summary = dict(n=len(out), median_delta=float(np.median(deltas)) if deltas else None,
                   ci_lo=lo, ci_hi=hi,
                   matched_median=float(np.median([o['matched'] for o in out])) if out else None,
                   shuffled_median=float(np.median([o['shuffled'] for o in out])) if out else None,
                   pass_criterion='median delta > 2.0 and CI excludes 0',
                   verdict=verdict)
    args.out.write_text(json.dumps(dict(summary=summary, clips=out), indent=2))

    print('\n== E2 result ==')
    print(f'  n              = {summary["n"]}')
    print(f'  matched  median = {summary["matched_median"]:.3f}')
    print(f'  shuffled median = {summary["shuffled_median"]:.3f}')
    print(f'  paired delta    = {summary["median_delta"]:+.3f}  [{lo:+.3f}, {hi:+.3f}]')
    print(f'  VERDICT: {verdict}')
    # Per-model, to see whether any single model carries the effect.
    print('\n  per model:')
    for m in sorted({o['model'] for o in out}):
        d = [o['delta'] for o in out if o['model'] == m]
        print('    %-24s n=%3d  delta %+.3f' % (m, len(d), np.median(d)))
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
