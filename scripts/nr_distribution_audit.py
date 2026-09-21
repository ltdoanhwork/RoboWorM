"""Audit the manifest (and what is scored) against the AUTHORITATIVE episode distribution.

The spec the paper must cover is stated in EPISODES, per
(embodiment, view_type, model, split) x method. The manifest is in VIEW-CLIPS: one row per
camera view, so a 4-view ctrlworld episode is 4 rows. Comparing the two directly is what
made "86k" look inexplicable. This script collapses view-clips back to episodes via
`pair_key` minus its trailing view segment, then diffs against the target.

It answers the only three questions that matter before launching:
  1. does the manifest CONTAIN the whole target distribution (or is a cell short on disk)?
  2. which cells are N/a in the target but present in the manifest (= must be excluded)?
  3. how many view-clips does full coverage of the target actually cost, and how many of
     those are already scored?

Usage:
  python scripts/nr_distribution_audit.py --run metrics/WorldScore/results/dense_nr_full
  python scripts/nr_distribution_audit.py --run ... --ids-out /tmp/spec_ids.json
"""
import argparse
import collections
import json
from pathlib import Path

# ---------------------------------------------------------------- the target spec
# episodes per (embodiment, view_type, model): (makovian, non_makovian)
# 'N/a' cells are simply absent -- DreamDojo has no multiview, Ctrl-World no singleview.
DENSE_TARGET = {
    ('bimanual', 'MV', 'dreamgen'): (150, 150),
    ('bimanual', 'MV', 'ctrlworld'): (150, 150),
    ('bimanual', 'SV', 'dreamgen'): (150, 150),
    ('bimanual', 'SV', 'dreamdojo'): (150, 150),
    ('humanoid', 'MV', 'dreamgen'): (98, 150),
    ('humanoid', 'MV', 'ctrlworld'): (98, 150),
    ('humanoid', 'SV', 'dreamgen'): (130, 130),
    ('humanoid', 'SV', 'dreamdojo'): (118, 98),
    ('single_arm', 'MV', 'dreamgen'): (128, 129),
    ('single_arm', 'MV', 'ctrlworld'): (128, 129),
    ('single_arm', 'SV', 'dreamgen'): (120, 120),
    ('single_arm', 'SV', 'dreamdojo'): (120, 120),
}
# per-method episode totals from the spec. dense and 6 methods hit the full 3166 grid;
# itm/sito/fastercache are knowingly short (those trees were never fully rendered).
METHOD_TARGET = {
    'dense': 3166, 'worldcache': 3166, 'dicache': 3166, 'fastercache': 2361,
    'svg1': 3166, 'svg2': 3166, 'radial': 3166, 'pisa': 3166,
    'itm': 2410, 'sito': 2410,
}
CLASS = {'dense': 'Dense', 'worldcache': 'Acceleration caching',
         'dicache': 'Acceleration caching', 'fastercache': 'Acceleration caching',
         'svg1': 'Sparse attention', 'svg2': 'Sparse attention',
         'radial': 'Sparse attention', 'pisa': 'Sparse attention',
         'itm': 'Token reduction', 'sito': 'Token reduction'}


def view_type(r):
    """MV / SV from the path, which is the only place the distinction is recorded."""
    rel = r['relative']
    if '/multiview/' in rel:
        return 'MV'
    if '/singleview/' in rel:
        return 'SV'
    return '?'


def base_model(r):
    """Fold ctrlworld_4view_grid and the gt_* anchors onto the spec's model names."""
    m = r['model']
    if m.startswith('gt_'):
        m = m[3:]
    if m.startswith('ctrlworld'):
        return 'ctrlworld'
    return m


def episode_of(r):
    """Episode identity = pair_key without its trailing view segment."""
    return r['pair_key'].rsplit('/', 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ids-out', default=None,
                    help='write the view-clip ids that the spec requires (all methods, '
                         'excluding gt) so a run can be launched with --only-ids')
    ap.add_argument('--with-gt', action='store_true',
                    help='include the gt anchor clips in --ids-out (needed for the paired '
                         'vs-GT columns)')
    args = ap.parse_args()
    run = Path(args.run)
    rows = json.loads((run / 'manifest.json').read_text())
    scored = {p.stem for p in (run / 'results').glob('*.json')}
    print(f'manifest {len(rows)} view-clips, {len(scored)} scored\n')

    # -------------------------------------------------- collapse to episodes
    # (embodiment, view_type, model, split, method) -> {episode -> [rows]}
    cells = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = (r['embodiment'], view_type(r), base_model(r), r['split'], r['method'])
        cells[key][episode_of(r)].append(r)

    # --------------------------------------------- 1. dense grid vs the target
    print('=' * 100)
    print('1. DENSE GRID: target episodes vs what the manifest has')
    print('=' * 100)
    print(f'{"embodiment":<12}{"view":<5}{"model":<11}'
          f'{"mak tgt":>9}{"mak have":>10}{"non tgt":>9}{"non have":>10}'
          f'{"view-clips":>12}   status')
    short = []
    tgt_total = 0
    for (emb, vt, model), (ta, tb) in sorted(DENSE_TARGET.items()):
        ha = len(cells[(emb, vt, model, 'makovian', 'dense')])
        hb = len(cells[(emb, vt, model, 'non_makovian', 'dense')])
        vc = (sum(len(v) for v in cells[(emb, vt, model, 'makovian', 'dense')].values())
              + sum(len(v) for v in
                    cells[(emb, vt, model, 'non_makovian', 'dense')].values()))
        tgt_total += ta + tb
        bad = []
        if ha < ta:
            bad.append(f'makovian SHORT {ta - ha}')
        if hb < tb:
            bad.append(f'non_makovian SHORT {tb - hb}')
        if ha > ta or hb > tb:
            bad.append(f'EXTRA (+{max(ha - ta, 0)}/+{max(hb - tb, 0)})')
        if bad:
            short.append(((emb, vt, model), bad))
        print(f'{emb:<12}{vt:<5}{model:<11}{ta:>9}{ha:>10}{tb:>9}{hb:>10}{vc:>12}   '
              f'{"; ".join(bad) if bad else "ok"}')
    print(f'{"TOTAL":<28}{"":<11}{tgt_total:>9}'
          f'{sum(len(cells[(e, v, m, s, "dense")]) for (e, v, m) in DENSE_TARGET for s in ("makovian", "non_makovian")):>10}')

    # ------------------------------------ 2. cells present but N/a in the spec
    print()
    print('=' * 100)
    print('2. CELLS IN THE MANIFEST THAT THE SPEC MARKS N/a (must be excluded)')
    print('=' * 100)
    stray = collections.defaultdict(int)
    for (emb, vt, model, split, method), eps in cells.items():
        if method == 'gt':
            continue
        if (emb, vt, model) not in DENSE_TARGET:
            stray[(emb, vt, model)] += len(eps)
    if stray:
        for k, n in sorted(stray.items(), key=lambda x: -x[1]):
            print(f'  {k[0]:<12}{k[1]:<5}{k[2]:<24} {n} episode-rows across methods')
    else:
        print('  none -- the manifest contains no (embodiment, view, model) outside the spec')

    # ------------------------------------------- 3. per-method episode coverage
    print()
    print('=' * 100)
    print('3. PER-METHOD EPISODES: spec vs manifest vs fully scored')
    print('=' * 100)
    print(f'{"method":<14}{"class":<22}{"spec":>7}{"manifest":>10}{"delta":>8}'
          f'{"view-clips":>12}{"scored":>9}{"%":>7}')
    spec_ids, need_ids = [], []
    cls_tot = collections.Counter()
    for method, spec_n in sorted(METHOD_TARGET.items(),
                                 key=lambda x: (CLASS[x[0]], x[0])):
        eps, vclips, sc = 0, 0, 0
        for (emb, vt, model) in DENSE_TARGET:
            for split in ('makovian', 'non_makovian'):
                d = cells[(emb, vt, model, split, method)]
                eps += len(d)
                for rs in d.values():
                    for r in rs:
                        vclips += 1
                        spec_ids.append(r['id'])
                        if r['id'] in scored:
                            sc += 1
                        else:
                            need_ids.append(r['id'])
        cls_tot[CLASS[method]] += eps
        print(f'{method:<14}{CLASS[method]:<22}{spec_n:>7}{eps:>10}{eps - spec_n:>+8}'
              f'{vclips:>12}{sc:>9}{100 * sc / max(vclips, 1):>6.1f}%')
    print(f'{"TOTAL":<36}{sum(METHOD_TARGET.values()):>7}{sum(cls_tot.values()):>10}')
    print()
    for c in ('Dense', 'Acceleration caching', 'Sparse attention', 'Token reduction'):
        spec_c = sum(n for m, n in METHOD_TARGET.items() if CLASS[m] == c)
        print(f'  {c:<24}spec {spec_c:>6}   manifest {cls_tot[c]:>6}'
              f'   {cls_tot[c] - spec_c:+}')

    # ------------------------------------------------------------ 4. GT anchors
    print()
    print('=' * 100)
    print('4. GT ANCHORS (needed for every paired vs-GT column)')
    print('=' * 100)
    gt_eps = gt_vc = gt_sc = 0
    for (emb, vt, model) in DENSE_TARGET:
        for split in ('makovian', 'non_makovian'):
            d = cells[(emb, vt, model, split, 'gt')]
            gt_eps += len(d)
            for rs in d.values():
                for r in rs:
                    gt_vc += 1
                    if r['id'] in scored:
                        gt_sc += 1
                    if args.with_gt:
                        spec_ids.append(r['id'])
                        if r['id'] not in scored:
                            need_ids.append(r['id'])
    print(f'  gt episodes {gt_eps}, view-clips {gt_vc}, scored {gt_sc} '
          f'({100 * gt_sc / max(gt_vc, 1):.1f}%)')

    # ----------------------------------------------------------------- 5. the ask
    print()
    print('=' * 100)
    print('5. WHAT FULL COVERAGE OF THE SPEC COSTS')
    print('=' * 100)
    spec_ids = sorted(set(spec_ids))
    need_ids = sorted(set(need_ids))
    print(f'  view-clips the spec requires : {len(spec_ids)}'
          + ('  (incl. GT)' if args.with_gt else '  (excl. GT)'))
    print(f'  already scored               : {len(spec_ids) - len(need_ids)}'
          f' ({100 * (len(spec_ids) - len(need_ids)) / max(len(spec_ids), 1):.1f}%)')
    print(f'  STILL TO SCORE               : {len(need_ids)}')
    for boxes in (1, 4, 5, 6):
        # 2.18 clips/min/box measured at 8 workers on an 80GB A100
        hours = len(need_ids) / (2.18 * 60 * boxes)
        print(f'    {boxes} box{"es" if boxes > 1 else "  "} @2.18 clips/min : '
              f'{hours:6.1f} h  ({hours / 24:.2f} days)')
    if args.ids_out:
        Path(args.ids_out).write_text(json.dumps(spec_ids, indent=0))
        print(f'\n  wrote {len(spec_ids)} spec ids to {args.ids_out}')


if __name__ == '__main__':
    main()
