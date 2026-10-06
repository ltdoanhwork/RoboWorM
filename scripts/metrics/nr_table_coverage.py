"""Coverage of the paper's main-table grid, and what is still missing.

The paper table is one cell per (embodiment, model, split) x method-row, where the
method rows are Real Video (gt), Dense, and the 9 acceleration methods. That grid is
FAR coarser than the 86,743-clip manifest, so "score the whole manifest" is not the
same job as "fill the table". This prints, per cell:

  scored  : records already on disk for that cell
  planned : rows in the manifest for that cell
  need    : how many more to reach --target (0 if already there)

and then how many clips a 5-machine wave has to do to fill the table only.

Usage:
  python scripts/metrics/nr_table_coverage.py --run metrics/WorldScore/results/dense_nr_full \
      --target 30
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'assets' / 'scripts'))
import table_dense_nr as T  # noqa: E402

METHOD_ROWS = ['gt', 'dense', 'itm', 'sito', 'worldcache', 'fastercache', 'dicache',
               'svg2', 'svg1', 'radial', 'pisa']
EMB = ['bimanual', 'humanoid', 'single_arm']
SPLITS = ['makovian', 'non_makovian']


def cell(r):
    """(embodiment, model, split) of a manifest row, with the GT anchor folded onto
    the model it anchors.

    !! DO NOT USE THIS FOR COVERAGE SAMPLING -- measured defect, 2026-09-21 !!

    This key omits VIEW_TYPE (multiview / singleview), but the spec distribution is one
    row per (embodiment, view, model). Merging them makes the MV and SV branches of the
    SAME model compete for one `--target` budget, and since the budget is counted in
    VIEW-CLIPS while MV episodes are 3-4 clips each, MV eats it:

      bimanual/dreamgen/makovian/dense pool = 600 MV + 150 SV clips, cap 300
        -> SV kept 126 of 300 episodes; 174 episodes got NOTHING
        -> bimanual/SV/dreamdojo, which has no MV branch to compete with, kept 300/300

    That is an unfair model-to-model comparison: DreamGen SV lost 58% of its episodes
    purely because DreamGen also has a multiview branch, while DreamDojo does not.

    Worse, because the cap is on clips and `sorted(pool)[:target]` orders by id (a sha256
    hash, i.e. effectively at random), it slices ACROSS the views of one episode:

      bimanual/MV/dreamgen : 251 episodes scored with SOME views missing, only 4 complete
      bimanual/MV/ctrlworld_4view_grid : 256 partial, 24 complete

    A 4-view episode with 2 views scored cannot be aggregated against one with 3 different
    views scored, so per-episode and per-view pooling both break.

    If per-cell sampling is ever needed again, key on (embodiment, view_type, model, split)
    AND cap on EPISODES, taking all views of a chosen episode. For full spec coverage do
    not sample at all -- see docs/dense_nr_full_spec_coverage.md.

    A GT row carries `model='gt_<model>'` so its records never collide with the
    prediction records, but in the paper table "Real Video" is a ROW of the model's
    block, not a separate model. Without this strip, every real model appears to be
    missing its gt column and every `gt_*` pseudo-model appears to be missing all 10
    prediction columns -- 22 phantom holes per method.
    """
    model = r['model']
    if model.startswith('gt_'):
        model = model[3:]
    return (r['embodiment'], model, r['split'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--target', type=int, default=30,
                    help='clips wanted per (cell, method) for the paper table')
    ap.add_argument('--models', nargs='*', default=None,
                    help='restrict to these models (default: all in the manifest)')
    ap.add_argument('--ids-out', default=None,
                    help='where to write the JSON id list (default '
                         '<run>/manifest_table_subset.json)')
    ap.add_argument('--keep-missing', action='store_true',
                    help='keep manifest rows whose video no longer exists on disk. Off '
                         'by default: 267 of 19,979 ids at target=300 pointed at deleted '
                         'fastercache trees (dreamdojo 255, ctrlworld 12). The worker '
                         'handles them (SKIP-GONE) but they can never satisfy the '
                         'per-cell target, so every rebuild would re-queue the same dead '
                         'ids and the printed "need" would never reach 0.')
    args = ap.parse_args()
    run = Path(args.run)

    manifest = json.loads((run / 'manifest.json').read_text())
    scored_ids = {p.stem for p in (run / 'results').glob('*.json')}
    print(f'manifest {len(manifest)} clips, {len(scored_ids)} scored '
          f'({100 * len(scored_ids) / len(manifest):.1f}%)')
    if not args.keep_missing:
        n0 = len(manifest)
        # Deliberately NOT `r['id'] in scored_ids or ...exists()`: the emitted id list
        # must be a pure function of (manifest, target) -- see the id-list comment at the
        # bottom. Folding `scored_ids` in here made the list shrink as the wave advanced.
        manifest = [r for r in manifest if Path(r['path']).exists()]
        if len(manifest) != n0:
            print(f'dropped {n0 - len(manifest)} rows whose video no longer exists '
                  f'(--keep-missing to keep them)')
    print()

    models = args.models or sorted({r['model'] for r in manifest})
    planned = collections.defaultdict(collections.Counter)
    done = collections.defaultdict(collections.Counter)
    pool_ids = collections.defaultdict(list)
    for r in manifest:
        if r['model'] not in models:
            continue
        # gt records carry model 'gt_<model>' in results but the manifest keeps the
        # real model on the gt row too; normalise on `method`.
        m = r.get('method') or 'gt'
        c = cell(r)
        planned[c][m] += 1
        if r['id'] in scored_ids:
            done[c][m] += 1
        pool_ids[(c, m)].append(r['id'])

    cells = sorted(planned, key=lambda c: (EMB.index(c[0]) if c[0] in EMB else 9,
                                           c[1], c[2]))
    print('=' * 118)
    print(f'GRID COVERAGE   scored/planned per method   (target {args.target} per cell)')
    print('=' * 118)
    hdr = ''.join(f'{m[:9]:>10}' for m in METHOD_ROWS)
    print(f'{"embodiment / model / split":<40}{hdr}')
    need_total = 0
    empty_cells = []
    for c in cells:
        line = f'{c[0]}/{c[1]}/{c[2]:<12}'[:40].ljust(40)
        for m in METHOD_ROWS:
            p, d = planned[c][m], done[c][m]
            if not p:
                line += f'{"-":>10}'
                empty_cells.append((c, m))
                continue
            want = min(args.target, p)
            need = max(want - d, 0)
            need_total += need
            line += f'{f"{d}/{p}" + ("*" if need else ""):>10}'
        print(line)
    print(f'\n* = still short of target. clips still needed to fill the table: '
          f'{need_total}')
    print(f'cells with NO clips in the manifest (method never rendered): '
          f'{len(empty_cells)}')

    print('\n' + '=' * 118)
    print('STRUCTURALLY EMPTY (method x cell that was never generated -- scoring '
          'cannot fix these)')
    print('=' * 118)
    by_method = collections.defaultdict(list)
    for c, m in empty_cells:
        by_method[m].append(f'{c[0]}/{c[1]}/{c[2]}')
    for m in METHOD_ROWS:
        if by_method[m]:
            print(f'  {m:<12} missing {len(by_method[m])}: '
                  + ', '.join(sorted(set(by_method[m]))))

    # what a 5-machine wave costs for the table only, vs the whole manifest
    print('\n' + '=' * 118)
    print('COST')
    print('=' * 118)
    remaining_all = len(manifest) - len(scored_ids)
    for label, n in (('whole manifest', remaining_all),
                     (f'table only @ {args.target}/cell', need_total)):
        for boxes, rate in ((3, 2.18), (5, 2.18)):
            mins = n / (boxes * rate)
            print(f'  {label:<28} {n:>7} clips  {boxes} boxes @2.18/min/box '
                  f'-> {mins / 60 / 24:5.2f} days')
    # Emit the id list for the table-only job.
    #
    # This list MUST be a pure function of (manifest, target) -- it must NOT depend on
    # what is already scored. run_dense_nr.py applies `--only-ids` BEFORE the stride
    # partition `rows[offset::stride]`, which is positional, so every box must filter to
    # the exact same list or the slices stop lining up. An earlier version emitted only
    # the UNSCORED ids ("need" per cell), so a list rebuilt at launch shrank as the wave
    # advanced and 5 boxes launched minutes apart each walked a different list. Simulated
    # at 5 boxes x 8 workers: 30% of remaining clips were owned by NO box (lost silently,
    # since every write is idempotent and nothing errors) and 24% were done twice.
    #
    # So: take the first `target` ids of each cell's FULL pool, scored or not. Boxes skip
    # the already-scored ones for free via `dest.exists()`, which is what makes the
    # earlier wave's 31k records count towards the subset.
    ids = []
    for (c, m), pool in sorted(pool_ids.items()):
        ids += sorted(pool)[:args.target]
    outp = Path(args.ids_out) if args.ids_out else run / 'manifest_table_subset.json'
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(sorted(set(ids)), indent=0))
    print(f'\nwrote {len(set(ids))} ids to {outp}')
    print(f'  ({len(set(ids) & scored_ids)} already scored -> skipped for free, '
          f'{len(set(ids) - scored_ids)} left to do)')
    print('  NOTE: this list is deterministic given (manifest, target) -- every box '
          'must use the SAME list, so do not make it depend on progress.')


if __name__ == '__main__':
    main()
