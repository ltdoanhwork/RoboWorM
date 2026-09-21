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
cells are the same numbers divided by that block's Real Video row. Those are written as
real formulas (`=C4/$C$3`) so the sheet recalculates, which is also how the original does
it.

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

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

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
MODEL_ORDER = ['Dreamgen', 'DreamDojo', 'Ctrl-World', 'CW-4view']

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


def write_readme(ws, run, src, n_scored, n_manifest):
    """Provenance + the two reading rules the numbers are only valid under.

    These caveats are not decoration: Actor Activity is a per-mount statistic and AMC is
    barred from cross-embodiment comparison (table_dense_nr.NOT_CROSS_EMBODIMENT), so a
    reader who scans ACROSS the three blocks gets a wrong answer. The original sheet has
    no such note, which is exactly why it needs one here.
    """
    lines = [
        ('PhyWorld NR grid \u2014 auto-generated', True),
        ('', False),
        (f'run              : {run}', False),
        (f'source table     : {src}', False),
        (f'clips scored     : {n_scored:,} of {n_manifest:,} manifest rows '
         f'({100 * n_scored / n_manifest:.1f}%)', False),
        (f'generated        : {datetime.datetime.now():%Y-%m-%d %H:%M} local', False),
        ('', False),
        ('LAYOUT (mirrors the KIVA-main / RIGIS-main tabs)', True),
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
    args = ap.parse_args()

    data, src = load(args.run)
    delta, delta_src = load(args.run, 'delta_vs_dense')
    run = Path(args.run)
    n_scored = len(list((run / 'results').glob('*.json')))
    import json
    n_manifest = len(json.loads((run / 'manifest.json').read_text()))

    wb = Workbook()
    write_readme(wb.active, args.run, src, n_scored, n_manifest)
    wb.active.title = 'README'

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
    for mkey, sheet, title, digits, _higher in METRICS:
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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(args.out)
    print(f'wrote {args.out}')
    print(f'  sheets: README + {", ".join(s for _, s, _, _, _ in METRICS)}')
    print(f'          + {", ".join(s.replace("-main", "-delta") for _, s, _, _, _ in METRICS)}')
    print(f'  based on {n_scored:,}/{n_manifest:,} scored clips')


if __name__ == '__main__':
    main()
