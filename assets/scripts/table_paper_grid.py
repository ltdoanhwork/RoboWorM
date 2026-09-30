"""The paper's main NR grid, in the layout the KIVA sheet uses.

One block per embodiment; inside a block, rows are Real Video (the GT anchor) then
each model's Dense + 9 acceleration methods; columns are Markovian | Non-Markovian,
repeated for each of the four reported metrics.

    BIMANUAL          Visual Integrity      Text Alignment    ...
                      Mark.   Non-Mark.     Mark.  Non-Mark.
    Real Video        27.9    28.1          ...
    Dreamgen  Dense   29.4    29.8
              itm     22.9    23.1
              ...

Why the grid is read DOWN a column only
---------------------------------------
A cell pools every camera mount present in that (embodiment, model, split), because
the KIVA layout has no mount axis. Every method row inside one block therefore shares
the same episode+mount mix, so comparing methods DOWN a column is like-for-like.
Comparing ACROSS blocks is not: Actor Activity is a per-mount statistic and AMC is
barred from cross-embodiment reading (`NOT_CROSS_EMBODIMENT`), and the mount mix
differs by embodiment. The per-block layout is exactly what keeps that honest.

Text Alignment is `PAIRED_ONLY` upstream because CLIPScore carries a per-caption
bias. Inside one block all rows are scored against the SAME caption set, so the
column is readable here; it still may not be compared across blocks. `--delta`
switches every method row to (method - dense of the same clip), which removes the
caption bias entirely and is the defensible form for a claim.

Usage:
  python assets/scripts/table_paper_grid.py --run metrics/WorldScore/results/dense_nr_full
  python assets/scripts/table_paper_grid.py --run <run> --delta      # vs dense
  python assets/scripts/table_paper_grid.py --run <run> --models dreamgen dreamdojo
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_dense_nr as T  # noqa: E402

EMB_ORDER = ['bimanual', 'humanoid', 'single_arm']
SPLITS = [('makovian', 'Markovian'), ('non_makovian', 'Non-Markovian')]
# paper row order, matching the KIVA sheet
METHOD_ORDER = ['dense', 'itm', 'sito', 'worldcache', 'fastercache', 'dicache',
                'svg2', 'svg1', 'radial', 'pisa']
METHOD_LABEL = {'dense': 'Dense', 'itm': 'itm', 'sito': 'sito',
                'worldcache': 'World-cache', 'fastercache': 'Faster-cache',
                'dicache': 'Di-cache', 'svg2': 'SVG2', 'svg1': 'SVG1',
                'radial': 'Radial Attention', 'pisa': 'PISA'}
MODEL_LABEL = {'dreamgen': 'Dreamgen', 'dreamdojo': 'DreamDojo',
               'ctrlworld': 'Ctrl-World', 'ctrlworld_4view_grid': 'CW-4view'}


def model_label(model):
    """Display label, understanding the `<model>|MV` / `|SV` suffix --by-track adds."""
    base, _, tr = model.partition('|')
    lbl = MODEL_LABEL.get(base, base)
    return f'{lbl} {tr}' if tr else lbl


def base_model(model):
    return model[3:] if model.startswith('gt_') else model


def track_of(rec):
    """'multiview' / 'singleview' from the record's path, or None.

    The camera track is NOT in the model name -- DreamGen is the only family rendered on
    both tracks and the only thing separating them in a record is `relative` containing
    `/multiview/` or `/singleview/`. Without this the two are pooled into one DreamGen
    line, which mixes a 4-view grid with a single ego view.
    """
    rel = rec.get('relative', '')
    if '/multiview/' in rel:
        return 'multiview'
    if '/singleview/' in rel:
        return 'singleview'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--models', nargs='*', default=None)
    ap.add_argument('--delta', action='store_true',
                    help='print (method - dense) on the same clip instead of the '
                         'absolute value')
    ap.add_argument('--min-n', type=int, default=T.MIN_PAIRS_FOR_CLAIM,
                    help='cells with fewer clips print as "?"')
    ap.add_argument('--by-track', action='store_true',
                    help='split multi-view from single-view instead of pooling them. '
                         'Only DreamGen has both, so this turns its one line into two '
                         '(Dreamgen MV / Dreamgen SV); Ctrl-World stays multi-view only '
                         'and DreamDojo single-view only. Writes '
                         'table_paper_grid_<tag>_by_track.csv so the pooled CSV is kept.')
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run / 'tables'
    out.mkdir(parents=True, exist_ok=True)

    rows = T.load_records(run)

    # ---- absolute values: (emb, model, split, method, col) -> [values]
    absolute = collections.defaultdict(list)
    # ---- deltas vs dense on the same clip
    dense_by_clip = {}
    for r in rows:
        if r.get('method') == 'dense':
            dense_by_clip[(r['pair_key'], r['model'])] = r
    deltas = collections.defaultdict(list)

    for r in rows:
        method = r.get('method') or 'gt'
        model = base_model(r['model'])
        if args.by_track:
            # Fold the track INTO the model label so every downstream consumer (the
            # printed table, the CSV, the xlsx exporter) gets one row per
            # (embodiment, model+track, split, method) with no further changes.
            tr = track_of(r)
            if tr is None:
                continue
            if model != 'gt':
                model = f'{model}|{"MV" if tr == "multiview" else "SV"}'
        key_base = (r['embodiment'], model, r['split'])
        for col, _, _, _ in T.COLUMNS:
            v = T.gated(r, col)
            if v is not None:
                absolute[key_base + (method, col)].append(v)
        if method in ('gt', 'dense'):
            continue
        base = dense_by_clip.get((r['pair_key'], r['model']))
        if base is None or not T.comparable_pair(r, base):
            continue
        for col, _, _, _ in T.COLUMNS:
            a, b = T.gated(base, col), T.gated(r, col)
            if a is not None and b is not None:
                deltas[key_base + (method, col)].append(b - a)

    table = deltas if args.delta else absolute
    # Derive the model list from the TABLE, not from `rows`: with --by-track the keys carry
    # a `|MV` / `|SV` suffix that base_model() knows nothing about, so re-deriving it from
    # the raw records yields plain 'dreamgen' and matches no key -- every model block then
    # comes out empty (measured: a 693-byte CSV with only the Real Video rows).
    present = {k[1] for k in table} - {'gt', 'Real Video'}
    if args.models:
        models = args.models
    else:
        # Keep the paper's model order (Dreamgen, DreamDojo, Ctrl-World, CW-4view) instead
        # of sorting alphabetically, and put MV before SV within a family.
        rank = {m: i for i, m in enumerate(
            ['dreamgen', 'dreamdojo', 'ctrlworld', 'ctrlworld_4view_grid'])}
        models = sorted(present, key=lambda m: (
            rank.get(m.partition('|')[0], 99),
            {'MV': 0, 'SV': 1}.get(m.partition('|')[2], 0), m))

    csv_rows = []
    for emb in [e for e in EMB_ORDER if any(k[0] == e for k in table)] + \
            sorted({k[0] for k in table} - set(EMB_ORDER)):
        print('\n' + '=' * 118)
        print(f'{emb.upper()}   ' + ('(method - dense, same clip)' if args.delta
                                     else '(absolute value)'))
        print('=' * 118)
        widths = [(col, label, digits) for col, label, _, digits in T.COLUMNS]
        head1 = f'{"":<14}{"":<18}'
        head2 = f'{"Model":<14}{"Method":<18}'
        for _, label, _ in widths:
            head1 += f'{label[:21]:^23}'
            head2 += f'{"Markovian":>11}{"Non-Mark.":>12}'
        print(head1)
        print(head2)

        def emit(model_label, method_label, getter):
            line = f'{model_label[:13]:<14}{method_label:<18}'
            rec = {'embodiment': emb, 'model': model_label, 'method': method_label}
            for col, _, digits in widths:
                for split, sname in SPLITS:
                    vals = getter(split, col)
                    if not vals:
                        cellstr, n, val = '-', 0, None
                    elif len(vals) < args.min_n:
                        cellstr, n, val = f'?({len(vals)})', len(vals), None
                    else:
                        val = T.median(vals)
                        cellstr, n = f'{val:.{digits}f}', len(vals)
                    line += f'{cellstr:>12}' if split != 'makovian' else f'{cellstr:>11}'
                    rec[f'{col}_{split}'] = val
                    rec[f'{col}_{split}_n'] = n
            print(line)
            csv_rows.append(rec)

        if not args.delta:
            emit('Real Video', '', lambda s, c: [
                v for m in models
                for v in absolute.get((emb, m, s, 'gt', c), [])])
        for model in models:
            if not any(k[:2] == (emb, model) for k in table):
                continue
            first = True
            for method in METHOD_ORDER:
                if args.delta and method == 'dense':
                    continue
                if not any((emb, model, s, method, c) in table
                           for s, _ in SPLITS for c, _, _, _ in T.COLUMNS):
                    continue
                emit(model_label(model) if first else '',
                     METHOD_LABEL[method],
                     lambda s, c, _m=model, _me=method: table.get(
                         (emb, _m, s, _me, c), []))
                first = False

    tag = 'delta_vs_dense' if args.delta else 'absolute'
    if args.by_track:
        tag += '_by_track'
    path = out / f'table_paper_grid_{tag}.csv'
    if csv_rows:
        keys = list(csv_rows[0])
        for r in csv_rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with path.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(csv_rows)
        print(f'\nwrote {path}')
    print(f'\n"?" = fewer than {args.min_n} clips; "-" = method never rendered for '
          f'that cell.')
    print('Read DOWN a column (methods within one model+embodiment+split). Do NOT '
          'compare numbers ACROSS embodiment blocks: Actor Activity is per-mount and '
          'AMC is barred cross-embodiment.')


if __name__ == '__main__':
    main()
