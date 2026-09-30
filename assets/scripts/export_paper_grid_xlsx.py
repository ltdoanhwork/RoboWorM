"""Export the NR paper grid to an .xlsx laid out exactly like the RIGIS/KIVA sheets.

The source of the layout is the benchmark workbook's `KIVA-main` / `RIGIS-main` tabs:

    A            B                 C          (E,F,G = same block, Non-Markovian)
  1 BIMANUAL     Method            <METRIC>
  2 Markovian                                 <- merged A2:C2
  3 Real Video                     0.5427     <- merged A3:B3
  4 Dreamgen     Dense             0.7822     <- A4:A13 merged (model spans its methods)
  5              itm               1.1648
 ...
 14 DreamDojo    Dense             ...        <- A14:A23 merged
 ...
 26 HUMANOID ...  (next embodiment block, 25 rows apart)
 51 SINGLE-ARM ...

`RIGIS-main` adds a second pair of blocks to the right under a `NORM -->` label, whose
cells are the same numbers divided by that block's Real Video row. The original holds those
as LITERAL values, not formulas (verified on the source workbook: `L4` is
0.9264723238522365, not `=C4/$C$3`), so this matches that — which also keeps the file free
of any formula needing a LibreOffice recalc pass, unavailable on this box.

One sheet per metric, named for the metric. Values come from
`table_paper_grid.py --run <run>` (its `table_paper_grid_absolute.csv`), i.e. the median
over every clip of that (embodiment, model, split, method) cell.

Usage:
  python assets/scripts/export_paper_grid_xlsx.py \
      --run metrics/WorldScore/results/dense_nr_full \
      --out "sampling_dataset/data_for_web/BENCHMARK_NR_grid.xlsx"
"""
import argparse
import csv
import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

REPO = Path(__file__).resolve().parents[2]

# (csv key, sheet name, sheet title, decimals, higher-is-better)
METRICS = [
    ('visual_integrity', 'VisualIntegrity-main', 'Visual Integrity', 2, True),
    ('text_condition_alignment', 'TextAlign-main', 'Text Alignment', 3, True),
    ('actor_motion_raw', 'ActorActivity-main', 'Actor Activity', 2, None),
    ('active_motion_continuity', 'AMC-main', 'Active Motion Continuity', 2, True),
]

EMB_LABEL = [('bimanual', 'BIMANUAL'), ('humanoid', 'HUMANOID'),
             ('single_arm', 'SINGLE-ARM')]
SPLITS = [('makovian', 'Markovian'), ('non_makovian', 'Non \u2013 Markovian')]
# The sheet's method row order (same as the KIVA tab).
METHOD_ORDER = ['Dense', 'itm', 'sito', 'World-cache', 'Faster-cache', 'Di-cache',
                'SVG2', 'SVG1', 'Radial Attention', 'PISA']
MODEL_ORDER = ['Dreamgen MV', 'Dreamgen SV', 'DreamDojo SV', 'Ctrl-World MV',
               'CW-4view MV',
               # pooled (no --by-track) labels, kept so the exporter still reads an
               # older CSV that did not separate the camera tracks
               'Dreamgen', 'DreamDojo', 'Ctrl-World', 'CW-4view']

# Layout constants, matching the original tabs.
# The original hardcodes a 25-row stride (1 -> 26 -> 51) because its blocks hold only two
# models (2 x 10 methods + 3 header = 23). This run has up to FOUR models per embodiment
# (Dreamgen, DreamDojo, Ctrl-World, CW-4view = 43 rows), so the stride is computed per
# block instead -- a fixed 25 makes block 3 land inside block 2's merged ranges and
# openpyxl raises `MergedCell object attribute 'value' is read-only`.
BLOCK_GAP = 2              # blank rows between embodiment blocks
ROWS_PER_MODEL = len(METHOD_ORDER)
ABS_COLS = (1, 2, 3)       # A,B,C  Markovian
ABS_COLS_NM = (5, 6, 7)    # E,F,G  Non-Markovian
NORM_LABEL_COL = 9         # I
NORM_COLS = (10, 11, 12)   # J,K,L
NORM_COLS_NM = (14, 15, 16)  # N,O,P

ARIAL = 'Arial'
HDR = Font(name=ARIAL, size=12, bold=True)
TXT = Font(name=ARIAL, size=12)
VAL = Font(name=ARIAL, size=11)
MISSING = Font(name=ARIAL, size=11, color='999999')
NOTE = Font(name=ARIAL, size=9, italic=True, color='666666')
CENTER = Alignment(horizontal='center', vertical='center')
LEFT = Alignment(horizontal='left', vertical='center')
RIGHT = Alignment(horizontal='right', vertical='center')
HDR_FILL = PatternFill('solid', fgColor='D9D9D9')
GT_FILL = PatternFill('solid', fgColor='FFF2CC')
THIN = Side(style='thin', color='BFBFBF')
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def load(run, tag='absolute'):
    """(embodiment, model, method) -> row dict, from table_paper_grid.py's CSV.

    The CSV mirrors the printed table, so the `model` column is blank on every row after
    a model's first (that is how the text table shows a model spanning its methods).
    Forward-fill it, or those rows key on model='' and a block appears to have one extra
    phantom model with nine methods.
    """
    path = Path(run) / 'tables' / f'table_paper_grid_{tag}.csv'
    if not path.exists():
        flag = ' --delta' if tag != 'absolute' else ''
        raise SystemExit(
            f'{path} not found -- run first:\n'
            f'  models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py '
            f'--run {run}{flag}')
    out = {}
    last = {}
    for r in csv.DictReader(path.open()):
        emb = r['embodiment']
        model = r['model'] or last.get(emb, '')
        if r['model']:
            last[emb] = r['model']
        out[(emb, model, r['method'])] = r
    return out, path


def num(row, key):
    """Float value, or None when the cell is empty (method never rendered / too few)."""
    if row is None:
        return None
    v = row.get(key)
    if v in (None, '', 'None'):
        return None
    return float(v)


def write_block(ws, top, emb_key, emb_label, data, mkey, title, digits, models):
    """One embodiment block: Markovian at A:C, Non-Markovian at E:G, plus the two
    normalised mirrors at J:L / N:P. Returns the last row used.

    `top` is the header row (1, 26, 51 ...). The block's geometry is identical in all
    four column groups, so the value/normalised variants are driven by the same loop.
    """
    groups = [(ABS_COLS, 'makovian', False), (ABS_COLS_NM, 'non_makovian', False),
              (NORM_COLS, 'makovian', True), (NORM_COLS_NM, 'non_makovian', True)]
    r_gt = top + 2
    for (ca, cb, cc), split, norm in groups:
        ws.cell(top, ca, emb_label).font = HDR
        ws.cell(top, cb, 'Method').font = HDR
        ws.cell(top, cc, title).font = HDR
        for c in (ca, cb, cc):
            ws.cell(top, c).alignment = CENTER
            ws.cell(top, c).fill = HDR_FILL
            ws.cell(top, c).border = BOX
        # split label spans the three columns, like `A2:C2` in the original
        ws.cell(top + 1, ca, dict(SPLITS)[split]).font = TXT
        ws.cell(top + 1, ca).alignment = LEFT
        ws.merge_cells(start_row=top + 1, start_column=ca,
                       end_row=top + 1, end_column=cc)

        # Real Video row (merged across the label+method columns)
        ws.cell(r_gt, ca, 'Real Video').font = TXT
        ws.cell(r_gt, ca).alignment = LEFT
        ws.merge_cells(start_row=r_gt, start_column=ca, end_row=r_gt, end_column=cb)
        gt = num(data.get((emb_key, 'Real Video', '')), f'{mkey}_{split}')
        cell = ws.cell(r_gt, cc)
        if norm:
            # The original RIGIS-main NORM block holds LITERAL values, not formulas
            # (checked: L4 = 0.9264723238522365, not '=C4/$C$3'), so match that. It also
            # removes the only thing that would have needed a LibreOffice recalc pass,
            # which is unavailable on this box.
            cell.value = 1.0 if gt is not None else None
            cell.number_format = '0.0000'
        else:
            cell.value = round(gt, digits) if gt is not None else None
            cell.number_format = f'0.{"0" * digits}'
        cell.font = VAL
        cell.alignment = RIGHT
        cell.fill = GT_FILL
        cell.border = BOX
        for c in (ca, cb):
            ws.cell(r_gt, c).fill = GT_FILL
            ws.cell(r_gt, c).border = BOX

        row = r_gt + 1
        for model in models:
            first = row
            for method in METHOD_ORDER:
                rec = data.get((emb_key, model, method))
                ws.cell(row, cb, method).font = TXT
                ws.cell(row, cb).alignment = LEFT
                ws.cell(row, cb).border = BOX
                ws.cell(row, ca).border = BOX
                cell = ws.cell(row, cc)
                v = num(rec, f'{mkey}_{split}')
                if norm:
                    # method / Real Video of THIS block+split. Literal, like the original.
                    # Guarded: a GT cell can legitimately be absent (metric gated out for
                    # that mount), and 0 is possible for Actor Activity.
                    cell.value = (round(v / gt, 4)
                                  if (v is not None and gt not in (None, 0)) else None)
                    cell.number_format = '0.0000'
                else:
                    cell.value = round(v, digits) if v is not None else None
                    cell.number_format = f'0.{"0" * digits}'
                if cell.value is None:
                    # '---' is how the original marks a method that was never rendered
                    # for that cell (e.g. Faster-cache on Ctrl-World). Keep it visually
                    # distinct from a real 0.
                    cell.value = '---'
                    cell.font = MISSING
                else:
                    cell.font = VAL
                cell.alignment = RIGHT
                cell.border = BOX
                row += 1
            ws.cell(first, ca, model).font = TXT
            ws.cell(first, ca).alignment = CENTER
            ws.merge_cells(start_row=first, start_column=ca,
                           end_row=first + ROWS_PER_MODEL - 1, end_column=ca)
    ws.cell(top, NORM_LABEL_COL, 'NORM -->').font = HDR
    ws.cell(top, NORM_LABEL_COL).alignment = CENTER
    return r_gt + len(models) * ROWS_PER_MODEL - 1


def write_delta_block(ws, top, emb_key, emb_label, data, mkey, title, digits, models):
    """Delta block: (method - dense) on the SAME clip, Markovian at A:C, Non-Mark at E:G.

    Deliberately different from the absolute block:
      * no Real Video row -- a delta is against dense, not against GT;
      * no Dense row -- it is 0 by construction and would read as a real measurement;
      * no NORM mirror -- dividing a difference by a GT level is meaningless.
    This is the form a claim is made on: differencing a method against its own dense clip
    cancels the per-caption CLIPScore bias and the VisualIntegrity ceiling bias.
    """
    methods = [m for m in METHOD_ORDER if m != 'Dense']
    for (ca, cb, cc), split in ((ABS_COLS, 'makovian'), (ABS_COLS_NM, 'non_makovian')):
        ws.cell(top, ca, emb_label).font = HDR
        ws.cell(top, cb, 'Method').font = HDR
        ws.cell(top, cc, f'{title} \u0394').font = HDR
        for c in (ca, cb, cc):
            ws.cell(top, c).alignment = CENTER
            ws.cell(top, c).fill = HDR_FILL
            ws.cell(top, c).border = BOX
        ws.cell(top + 1, ca, dict(SPLITS)[split]).font = TXT
        ws.cell(top + 1, ca).alignment = LEFT
        ws.merge_cells(start_row=top + 1, start_column=ca,
                       end_row=top + 1, end_column=cc)
        row = top + 2
        for model in models:
            first = row
            for method in methods:
                rec = data.get((emb_key, model, method))
                ws.cell(row, cb, method).font = TXT
                ws.cell(row, cb).alignment = LEFT
                ws.cell(row, cb).border = BOX
                ws.cell(row, ca).border = BOX
                cell = ws.cell(row, cc)
                v = num(rec, f'{mkey}_{split}')
                if v is None:
                    cell.value = '---'
                    cell.font = MISSING
                else:
                    cell.value = round(v, digits)
                    cell.number_format = f'+0.{"0" * digits};-0.{"0" * digits};0'
                    cell.font = VAL
                cell.alignment = RIGHT
                cell.border = BOX
                row += 1
            ws.cell(first, ca, model).font = TXT
            ws.cell(first, ca).alignment = CENTER
            ws.merge_cells(start_row=first, start_column=ca,
                           end_row=first + len(methods) - 1, end_column=ca)
    return top + 2 + len(models) * len(methods) - 1


def write_flat(ws, data, delta):
    """One ROW per (embodiment, model, split, method); every metric is a COLUMN.

    The *-main / *-delta sheets copy the original workbook's shape: three column groups
    side by side, one sheet per metric. That is good for eyeballing one metric and useless
    for anything else -- you cannot sort it, filter it, pivot it, or read two metrics of
    the same cell without jumping sheets. This tab is the same numbers in long form, so a
    single row carries a cell's absolute value, its clip count, and its delta vs dense for
    all four metrics at once.

    `n` is kept beside each value on purpose: the wave is unfinished, so cells differ in
    size by an order of magnitude and a bare median hides that. Sort by `n` to find the
    cells that are still too thin to quote.
    """
    head = ['Embodiment', 'Model', 'Method', 'Split']
    for _, _, title, _, _ in METRICS:
        head += [title, f'{title} n', f'{title} \u0394 vs dense']
    for i, h in enumerate(head, start=1):
        c = ws.cell(1, i, h)
        c.font = HDR
        c.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        c.fill = HDR_FILL
        c.border = BOX

    row = 2
    for emb_key, emb_label in EMB_LABEL:
        models = ['Real Video'] + [m for m in MODEL_ORDER
                                   if any(k[:2] == (emb_key, m) for k in data)]
        for model in models:
            # Real Video is the GT anchor: one row, method '' , and no delta (a delta is
            # against dense, and GT has no dense counterpart).
            methods = [''] if model == 'Real Video' else METHOD_ORDER
            for method in methods:
                rec = data.get((emb_key, model, method))
                if rec is None:
                    continue
                drec = delta.get((emb_key, model, method))
                for split, split_label in SPLITS:
                    vals = [(num(rec, f'{k}_{split}'),
                             rec.get(f'{k}_{split}_n'),
                             num(drec, f'{k}_{split}'))
                            for k, _, _, _, _ in METRICS]
                    if all(v is None for v, _, _ in vals):
                        continue
                    for col, value in ((1, emb_label), (2, model),
                                       (3, method or 'Real Video'), (4, split_label)):
                        c = ws.cell(row, col, value)
                        c.font = TXT
                        c.alignment = LEFT
                        c.border = BOX
                    col = 5
                    for (v, n, d), (_, _, _, digits, _) in zip(vals, METRICS):
                        cv = ws.cell(row, col)
                        if v is None:
                            cv.value, cv.font = '---', MISSING
                        else:
                            cv.value = round(v, digits)
                            cv.number_format = f'0.{"0" * digits}'
                            cv.font = VAL
                        cn = ws.cell(row, col + 1, int(n) if n else 0)
                        cn.font = VAL
                        cn.number_format = '#,##0'
                        cd = ws.cell(row, col + 2)
                        if d is None:
                            cd.value, cd.font = '---', MISSING
                        else:
                            cd.value = round(d, digits)
                            cd.number_format = f'+0.{"0" * digits};-0.{"0" * digits};0'
                            cd.font = VAL
                        for c in (cv, cn, cd):
                            c.alignment = RIGHT
                            c.border = BOX
                        col += 3
                    row += 1

    widths = [13, 12, 17, 15] + [11, 8, 13] * len(METRICS)
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
    ws.row_dimensions[1].height = 32
    ws.freeze_panes = 'E2'
    ws.auto_filter.ref = f'A1:{ws.cell(1, len(head)).column_letter}{row - 1}'
    return row - 2


def write_readme(ws, run, src, n_scored, n_manifest):
    """Provenance + the two reading rules the numbers are only valid under.

    These caveats are not decoration: Actor Activity is a per-mount statistic and AMC is
    barred from cross-embodiment comparison (table_dense_nr.NOT_CROSS_EMBODIMENT), so a
    reader who scans ACROSS the three blocks gets a wrong answer. The original sheet has
    no such note, which is exactly why it needs one here.

    The scored count is reported TWICE on purpose. This script is cheap (seconds) while
    table_paper_grid.py takes ~25 min over 30k+ JSONs on the shared filesystem, so the CSV
    it reads is routinely older than the run. Printing only the live count would label
    medians computed from 33k clips as if they came from 39k.
    """
    src_mtime = datetime.datetime.fromtimestamp(Path(src).stat().st_mtime)
    age_h = (datetime.datetime.now() - src_mtime).total_seconds() / 3600
    lines = [
        ('PhyWorld NR grid \u2014 auto-generated', True),
        ('', False),
        (f'run                    : {run}', False),
        (f'source table           : {src}', False),
        (f'  computed at          : {src_mtime:%Y-%m-%d %H:%M} local '
         f'({age_h:.1f} h ago)  <- THE NUMBERS BELOW ARE AS OF THIS TIME', False),
        (f'run progress NOW       : {n_scored:,} of {n_manifest:,} manifest rows '
         f'({100 * n_scored / n_manifest:.1f}%)', False),
        (f'workbook written       : {datetime.datetime.now():%Y-%m-%d %H:%M} local',
         False),
        ('', False),
    ]
    if age_h > 2:
        lines += [
            ('  NOTE: the run has advanced since the source table was computed. '
             'Re-run the two', False),
            ('  table_paper_grid.py commands at the bottom to refresh, then re-run this '
             'exporter.', False),
            ('', False),
        ]
    lines += [
        ('LAYOUT (mirrors the KIVA-main / RIGIS-main tabs)', True),
        ('  All-metrics-flat \u2014 START HERE. One ROW per '
         '(embodiment, model, split, method);', False),
        ('     every metric is a column, with its clip count n and its delta vs dense '
         'beside it.', False),
        ('     Sortable / filterable (autofilter on row 1). Sort by n to spot cells that '
         'are still', False),
        ('     too thin to quote.', False),
        ('  *-main sheets  \u2014 ABSOLUTE medians', False),
        ('     A:C  Markovian        E:G  Non-Markovian', False),
        ('     J:L / N:P  the same numbers divided by that block\'s Real Video row',
         False),
        ('  *-delta sheets \u2014 (method \u2212 its OWN dense clip), the form a claim '
         'is made on:', False),
        ('     differencing cancels the per-caption CLIPScore bias and the '
         'VisualIntegrity ceiling bias.', False),
        ('     No Real Video / Dense row (Dense is 0 by construction) and no NORM '
         'mirror.', False),
        ('  Each cell is the MEDIAN over every scored clip of that '
         '(embodiment, model, split, method).', False),
        ('  "---" = that method was never rendered for that cell (not a zero).', False),
        ('', False),
        ('HOW TO READ IT', True),
        ('  Read DOWN a column: methods inside one model+embodiment+split share the '
         'same episode', False),
        ('  and camera-mount mix, so those comparisons are like-for-like.', False),
        ('  Do NOT compare numbers ACROSS the three embodiment blocks: Actor Activity '
         'is a', False),
        ('  per-mount statistic, and Active Motion Continuity is barred '
         'cross-embodiment (humanoid', False),
        ('  GT AMC ~48 vs ~66 for bimanual/single_arm at matched motion).', False),
        ('  Text Alignment is paired-only (CLIPScore carries a per-caption bias): '
         'readable down a', False),
        ('  column, never across blocks.', False),
        ('', False),
        ('STILL FILLING', True),
        ('  The wave is not finished, so cell sizes differ; a cell with very few clips '
         'is noisy.', False),
        ('  Regenerate after more clips land:', False),
        ('    models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py '
         '--run <run>', False),
        ('    models/dreamgen/.venv/bin/python assets/scripts/table_paper_grid.py '
         '--run <run> --delta', False),
        ('    .venv_edit/bin/python assets/scripts/export_paper_grid_xlsx.py '
         '--run <run> --out <xlsx>', False),
    ]
    for i, (text, bold) in enumerate(lines, start=1):
        c = ws.cell(i, 1, text)
        c.font = Font(name=ARIAL, size=11, bold=bold)
        c.alignment = LEFT
    ws.column_dimensions['A'].width = 105


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--main-only', action='store_true',
                    help='emit only the four *-main sheets (drop the *-delta ones and '
                         'All-metrics-flat)')
    ap.add_argument('--with-reference', nargs='?', const=str(
        REPO / 'sampling_dataset/data_for_web/'
        '[BENCHMARK] Accelerating Inference of Video World Models (1).xlsx'),
        default=None, metavar='XLSX',
        help='copy the RIGIS-main / KIVA-main tabs out of the benchmark workbook so the '
             'NR grid sits beside them in one file (values only, no formatting)')
    ap.add_argument('--ctrlworld-single-layout', action='store_true',
                    help="keep ONE Ctrl-World line per embodiment: the 3-view render for "
                         "single_arm (its only one) and the 4-view grid for bimanual / "
                         "humanoid, whose 3-view clips are excluded from the wave. "
                         "Without this both layouts appear, and the stale 3-view records "
                         "that predate the exclusion show up as a second line.")
    ap.add_argument('--by-track', action='store_true',
                    help='read the *_by_track CSVs (multi-view and single-view kept '
                         'apart) instead of the pooled ones. Build them first with '
                         'table_paper_grid.py --by-track [--delta].')
    args = ap.parse_args()

    suffix = '_by_track' if args.by_track else ''
    data, src = load(args.run, 'absolute' + suffix)
    delta, delta_src = load(args.run, 'delta_vs_dense' + suffix)

    if args.ctrlworld_single_layout:
        # Ctrl-World exists in two multi-view renders and only ONE is wanted per
        # embodiment. single_arm has 3-view only ('Ctrl-World'); bimanual and humanoid are
        # 4-grid ('CW-4view') because their 3-view clips are excluded from scoring
        # (excluded_row() in scripts/run_dense_nr.py). Their 'Ctrl-World' rows in the CSV
        # come from records scored BEFORE that exclusion landed, so they are stale and
        # would otherwise print a second, thinner Ctrl-World line.
        cw3 = 'Ctrl-World MV' if args.by_track else 'Ctrl-World'
        cw4 = 'CW-4view MV' if args.by_track else 'CW-4view'
        drop = {('bimanual', cw3), ('humanoid', cw3)}
        for d in (data, delta):
            for k in [k for k in d if k[:2] in drop]:
                del d[k]
        # Relabel the surviving 4-grid rows to plain Ctrl-World: with one layout per
        # embodiment the distinction no longer carries information in the sheet.
        for d in (data, delta):
            for k in [k for k in d if k[1] == cw4]:
                d[(k[0], cw3) + k[2:]] = d.pop(k)

    run = Path(args.run)
    n_scored = len(list((run / 'results').glob('*.json')))
    import json
    n_manifest = len(json.loads((run / 'manifest.json').read_text()))

    wb = Workbook()
    write_readme(wb.active, args.run, src, n_scored, n_manifest)
    wb.active.title = 'README'

    # Flat long-form tab first: it is the one you can sort/filter/pivot, and it is the
    # only place where all four metrics of one cell sit on a single row.
    n_flat = 0
    if not args.main_only:
        n_flat = write_flat(wb.create_sheet('All-metrics-flat'), data, delta)

    for mkey, sheet, title, digits, _higher in METRICS:
        ws = wb.create_sheet(sheet)
        top = 1
        for emb_key, emb_label in EMB_LABEL:
            # Only the models that actually have a row in this embodiment, in paper order.
            models = [m for m in MODEL_ORDER
                      if any(k[:2] == (emb_key, m) for k in data)]
            if not models:
                continue
            last = write_block(ws, top, emb_key, emb_label,
                               data, mkey, title, digits, models)
            top = last + 1 + BLOCK_GAP
        for col, width in (('A', 17), ('B', 18), ('C', 14), ('D', 3),
                           ('E', 17), ('F', 18), ('G', 14), ('H', 3), ('I', 10),
                           ('J', 17), ('K', 18), ('L', 14), ('M', 3),
                           ('N', 17), ('O', 18), ('P', 14)):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = 'A2'

    # The defensible form for a claim: each method minus its OWN dense clip.
    for mkey, sheet, title, digits, _higher in ([] if args.main_only else METRICS):
        ws = wb.create_sheet(sheet.replace('-main', '-delta'))
        top = 1
        for emb_key, emb_label in EMB_LABEL:
            models = [m for m in MODEL_ORDER
                      if any(k[:2] == (emb_key, m) for k in delta)]
            if not models:
                continue
            last = write_delta_block(ws, top, emb_key, emb_label,
                                     delta, mkey, title, digits, models)
            top = last + 1 + BLOCK_GAP
        for col, width in (('A', 17), ('B', 18), ('C', 14), ('D', 3),
                           ('E', 17), ('F', 18), ('G', 14)):
            ws.column_dimensions[col].width = width
        ws.freeze_panes = 'A2'

    ref_tabs = []
    if args.with_reference:
        # Copy the two reference tabs in VALUES-only form. A cross-workbook openpyxl copy
        # cannot carry formatting or formulas, and a formula would break anyway: it would
        # point at rows that do not exist here. Reading with data_only=True gives the
        # cached results, which is what a reader of a comparison sheet wants.
        ref_path = Path(args.with_reference)
        if not ref_path.exists():
            raise SystemExit(f'--with-reference: {ref_path} not found')
        ref = load_workbook(ref_path, data_only=True)
        for name in ('RIGIS-main', 'KIVA-main'):
            if name not in ref.sheetnames:
                print(f'  ! reference workbook has no {name!r} -- skipped')
                continue
            src_ws = ref[name]
            dst = wb.create_sheet(name)
            for row in src_ws.iter_rows():
                for c in row:
                    if c.value is not None:
                        dst.cell(c.row, c.column, c.value).font = VAL
            for col, width in (('A', 17), ('B', 18), ('C', 14), ('D', 3),
                               ('E', 17), ('F', 18), ('G', 14), ('H', 3), ('I', 10),
                               ('J', 17), ('K', 18), ('L', 14)):
                dst.column_dimensions[col].width = width
            dst.freeze_panes = 'A2'
            ref_tabs.append(name)
        ref.close()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f'wrote {args.out}')
    print(f'  sheets: README'
          + (f', All-metrics-flat ({n_flat} rows)' if n_flat else ''))
    print(f'          + {", ".join(s for _, s, _, _, _ in METRICS)}')
    if not args.main_only:
        print('          + ' + ', '.join(
            s.replace('-main', '-delta') for _, s, _, _, _ in METRICS))
    if ref_tabs:
        print(f'          + {", ".join(ref_tabs)}  (copied from the benchmark workbook, '
              f'values only)')
    if args.ctrlworld_single_layout:
        print('  Ctrl-World: 3-view for single_arm, 4-view grid for bimanual/humanoid '
              '(one line each)')
    print(f'  based on {n_scored:,}/{n_manifest:,} scored clips')


if __name__ == '__main__':
    main()
