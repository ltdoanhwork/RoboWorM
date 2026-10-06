#!/usr/bin/env python3
"""Circular ("round") views of the NR workbook, with SPEEDUP folded in.

Exploratory answer to "plot the workbook as a circle, and get speedup into it".
Five different circular encodings of the SAME two sources, because there is no
single honest circle: speedup and NR quality have different units, different
polarities, and different coverage.

  c1  radar / spider     one panel per (embodiment x track); spokes = the 4 NR
                         metrics as ratio-to-dense + a 5th SPEEDUP spoke.
  c2  circular barplot   one bar per (track x method), bar LENGTH = speedup,
                         bar COLOUR = NR quality retention. Dense = ring at 1.0.
  c3  donut grid         a TRUE pie: dense compute split into "time still spent"
                         (1/speedup) vs "time saved" (1 - 1/speedup). Methods
                         slower than dense get an explicit overhead wedge.
  c4  polar scatter      angle = method, radius = speedup, one dot per cell,
                         colour = NR retention -> shows the spread a mean hides.
  c5  sunburst           inner ring = method family, outer = method; arc width
                         proportional to #cells measured, radius = mean speedup.

WHY A PIE IS ONLY LEGAL IN c3
-----------------------------
A pie encodes parts of a whole. NR metrics are levels, not shares, so slicing
them by method would invent a whole that does not exist. The one real
parts-of-a-whole quantity here is the dense TIME BUDGET: a method with speedup
s spends 1/s of the dense time and saves 1 - 1/s of it. That is what c3 draws.
c1/c2/c4/c5 are circular but not pies, on purpose.

POLARITY / STATUS (see .cursor/rules/phyworld-benchmark.mdc, nr_viz_common.py)
------------------------------------------------------------------------------
Visual Integrity          higher is better  (CEILING-failed: read deltas, not levels)
Text Alignment            higher is better
Actor Activity            DESCRIPTIVE -- no polarity. Ratio-to-dense is drawn as
                          CLOSENESS, min(r, 1/r), so overshoot is not rewarded.
Active Motion Continuity  higher is better

SOURCES -- the workbook is self-contained, nothing else is read by default
--------------------------------------------------------------------------
  --xlsx  assets/BENCHMARK_NR_main (3).xlsx
          `Speedup` tab             -> every speedup (12 cells x 10 methods)
          4 NR tabs                 -> every quality number
          Ctrl-World MV rows are the 4-view grid for bimanual/humanoid and the
          3-view strip for single_arm, per assets/scripts/benchmark_cells.py.

  --baseline real_video (default) reproduces the workbook's OWN `NORM -->` block
          (cols J:P): every 0.xx is value / Real Video. `--baseline dense` divides
          by the Dense row instead -- useful for "what did acceleration cost?",
          but that block is NOT in the sheet, it is derived from the same cells.

  --speedup-source csv reads results/reports/speedup_table_full.csv instead of the
          `Speedup` tab. NOTE the two disagree on 40 of 92 shared cells (the tab
          has no value below 1.00x; the CSV has 24), so they must never be mixed
          in one figure. Default is the tab, i.e. the numbers you supplied.

Usage (needs matplotlib + openpyxl -- the dreamgen venv has both):
  models/dreamgen/.venv/bin/python assets/scripts/make_nr_circular_figures.py
  ... --only c1 c3            # a subset, by id
  ... --split non_makovian    # default: makovian
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]

# (sheet name, short label, polarity) -- polarity None == DESCRIPTIVE, no "better".
METRICS = [
    ('VisualIntegrity-main', 'Visual\nIntegrity', True),
    ('TextAlign-main', 'Text\nAlignment', True),
    ('ActorActivity-main', 'Actor\nActivity', None),
    ('AMC-main', 'Motion\nContinuity', True),
]
METRIC_KEY = {m[0]: m[1].replace('\n', ' ') for m in METRICS}

# --------------------------------------------------------------- physics audits
#
# RIGIS and KIVA are the workbook's two PHYSICAL audits, and they are NOT two more
# columns of the same kind as the four NR metrics above:
#
#   * their tabs use a DIFFERENT layout (3 blocks at rows 1/26/51 -- stride 25 --
#     with only TWO model groups per block, versus the NR tabs' four), and the
#     model column holds 'Dreamgen'/'Ctrlworld'/'DreamDojo' rather than the
#     'Dreamgen MV'-style track names, so `read_workbook`'s `a in TRACK_ORDER`
#     branch never fires on them;
#
#   * their track coverage is DISJOINT. RIGIS measures cross-view coherence, so it
#     only exists where there are multiple views; KIVA lifts body keypoints to 3D,
#     so it only exists on single-view body tracks. No (embodiment, track) panel
#     ever has both. That is why they get their own figure (c6) instead of two more
#     radar spokes -- as spokes, one of the two would be blank on every panel.
#
#   * their polarities OPPOSE. RIGIS is 0..1 higher-better. KIVA is a PERCENTAGE of
#     invalid 16-frame segments, so LOWER is better, and Real Video is a non-zero
#     reference LEVEL (5.05 / 4.98 / 10.12 %) rather than a ceiling. Putting KIVA on
#     a "further from centre = better" radial axis would require inverting it.
#
# Track sets mirror RIGIS_TRACKS / KIVA_TRACKS in make_rq1_dense_table.py.
AUDITS = [
    # (sheet, key, short label, unit, higher_is_better, tracks it applies to)
    ('RIGIS-main', 'RIGIS', 'RIGIS', '', True,
     ('Dreamgen MV', 'Ctrl-World MV')),
    ('KIVA-main', 'KIVA', 'KIVA', '%', False,
     ('Dreamgen SV', 'DreamDojo SV')),
]
AUDIT_KEYS = [a[1] for a in AUDITS]
AUDIT_BY_KEY = {a[1]: a for a in AUDITS}
# Audit-tab model label -> the track name used everywhere else in this script.
# Mirrors AUDIT_ROW in make_rq1_dense_table.py.
AUDIT_ALIAS = {
    'RIGIS-main': {'Dreamgen': 'Dreamgen MV', 'Ctrlworld': 'Ctrl-World MV'},
    'KIVA-main': {'Dreamgen': 'Dreamgen SV', 'DreamDojo': 'DreamDojo SV'},
}

# Workbook method label -> speedup-CSV method name. 'Dense' is the reference.
METHOD_TO_SPEEDUP = {
    'itm': 'ITM', 'sito': 'SiTo',
    'World-cache': 'WorldCache', 'Faster-cache': 'FasterCache', 'Di-cache': 'DiCache',
    'SVG2': 'SVG2', 'SVG1': 'SVG1', 'Radial Attention': 'Radial', 'PISA': 'PISA',
}
METHOD_ORDER = ['World-cache', 'Di-cache', 'Faster-cache', 'sito', 'itm',
                'PISA', 'Radial Attention', 'SVG1', 'SVG2']
METHOD_SHORT = {'World-cache': 'WorldCache', 'Di-cache': 'DiCache',
                'Faster-cache': 'FasterCache', 'sito': 'SiTo', 'itm': 'ITM',
                'PISA': 'PISA', 'Radial Attention': 'Radial',
                'SVG1': 'SVG1', 'SVG2': 'SVG2'}
FAMILY = {'World-cache': 'caching', 'Di-cache': 'caching', 'Faster-cache': 'caching',
          'sito': 'pruning', 'itm': 'pruning',
          'PISA': 'sparse-attn', 'Radial Attention': 'sparse-attn',
          'SVG1': 'sparse-attn', 'SVG2': 'sparse-attn'}
FAMILY_ORDER = ['caching', 'pruning', 'sparse-attn']
FAMILY_COLOR = {'caching': '#2f7d4f', 'pruning': '#c4622d', 'sparse-attn': '#3b5fa8'}

# Audit method label -> the method names used elsewhere. The audit tabs happen to
# use the same labels as the NR tabs ('itm', 'sito', 'World-cache', ...), so this
# is identity today; it exists so a future relabel is a one-line change here
# rather than a silent 'method not found' that drops rows from a figure.
# (Defined here, below METHOD_ORDER, rather than beside the other AUDIT_* maps.)
AUDIT_METHOD = {k: k for k in ['Dense'] + METHOD_ORDER}

EMB_XL = ['BIMANUAL', 'HUMANOID', 'SINGLE-ARM']
EMB_CSV = {'BIMANUAL': 'bimanual', 'HUMANOID': 'humanoid', 'SINGLE-ARM': 'single_arm'}
EMB_LABEL = {'BIMANUAL': 'Bimanual (ALOHA)', 'HUMANOID': 'Humanoid (GR1)',
             'SINGLE-ARM': 'Single-arm (DROID)'}
TRACK_ORDER = ['Dreamgen MV', 'Dreamgen SV', 'DreamDojo SV', 'Ctrl-World MV']
TRACK_LABEL = {'Dreamgen MV': 'DreamGen MV', 'Dreamgen SV': 'DreamGen SV',
               'DreamDojo SV': 'DreamDojo SV', 'Ctrl-World MV': 'Ctrl-World MV'}

# (embodiment, workbook track) -> speedup-CSV `cell`.
# Ctrl-World: bimanual/humanoid report the 4-view grid, single_arm the 3-view
# strip (benchmark_cells.py). The workbook labels both rows 'Ctrl-World MV', so
# the layout MUST be resolved here or the grid rows would silently inherit
# strip timings -- the exact bug tmp/20261005_refactor/docs/speedup_coverage_gaps.md 3.1 records.
CELL_OF = {}
for _e in ('bimanual', 'humanoid', 'single_arm'):
    CELL_OF[(_e, 'Dreamgen MV')] = 'dreamgen_%s_mv' % _e
    CELL_OF[(_e, 'Dreamgen SV')] = 'dreamgen_%s_sv' % _e
    CELL_OF[(_e, 'DreamDojo SV')] = 'dreamdojo_%s_sv' % _e
CELL_OF[('bimanual', 'Ctrl-World MV')] = 'ctrlworld4_bimanual_mv'
CELL_OF[('humanoid', 'Ctrl-World MV')] = 'ctrlworld4_humanoid_mv'
CELL_OF[('single_arm', 'Ctrl-World MV')] = 'ctrlworld_single_arm_mv'

SPLITS = {'makovian': 'Markovian', 'non_makovian': 'Non-Markovian'}
# Column of the absolute-value block for each split, in the workbook layout
# (A:C = Markovian, E:G = non-Markovian; J:P repeat them normalised by Real Video).
SPLIT_COL = {'makovian': (1, 2, 3), 'non_makovian': (5, 6, 7)}
MISSING = {'---', '--', '', None}


def is_missing(v) -> bool:
    """True for an empty / 'not applicable' cell, for ANY dash spelling.

    The workbook is hand-maintained and genuinely contains four different dash
    tokens: '--' (RIGIS x1), '---' (most tabs), and '----' (RIGIS NORM x4). A
    literal set kept missing one of them -- `float('----')` is how that surfaced.
    So the test is structural (all characters are dashes) rather than a fixed list,
    and it accepts en/em dashes too since those are one autocorrect away.

    Deliberately NOT treated as missing: 0 and 0.0. KIVA has a genuine 0.0 (0% of
    segments invalid = the best possible score), so a truthiness test here would
    delete the single best cell in that tab.
    """
    if v is None:
        return True
    if isinstance(v, (int, float)):
        return False                 # keeps a real 0.0
    s = str(v).strip()
    return not s or set(s) <= set('-\u2013\u2014')



# --------------------------------------------------------------------------- IO

def read_workbook(xlsx: Path) -> dict:
    """{(emb, track, method, metric): value} + {(emb, 'Real Video', metric): value}.

    The four NR tabs share one layout: 3 embodiment blocks, each = header row,
    split row, 'Real Video' row, then one 10-row group per track.
    """
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, data_only=True)
    out, real = {}, {}
    for sheet, _label, _pol in METRICS:
        if sheet not in wb.sheetnames:
            raise SystemExit('missing tab %r in %s' % (sheet, xlsx))
        ws = wb[sheet]
        key = METRIC_KEY[sheet]
        for split, (ca, cb, cc) in SPLIT_COL.items():
            emb = track = None
            for r in range(1, ws.max_row + 1):
                a = ws.cell(r, ca).value
                a = a.strip() if isinstance(a, str) else a
                if a in EMB_XL:                      # new embodiment block
                    emb, track = a, None
                    continue
                if emb is None:
                    continue
                if a == 'Real Video':
                    v = ws.cell(r, cc).value
                    if not is_missing(v):
                        real[(emb, split, key)] = float(v)
                    continue
                if isinstance(a, str) and a in TRACK_ORDER:
                    track = a
                if track is None:
                    continue
                meth = ws.cell(r, cb).value
                meth = meth.strip() if isinstance(meth, str) else meth
                if is_missing(meth) or meth == 'Method':
                    continue
                v = ws.cell(r, cc).value
                if is_missing(v):
                    continue
                out[(emb, track, meth, split, key)] = float(v)
    return {'val': out, 'real': real}


def read_audit_tabs(xlsx: Path) -> dict:
    """RIGIS + KIVA from their own tabs -> absolute values, real-video levels, NORM.

    Returns
        {'val':  {(emb, track, method, split, audit_key): value},
         'real': {(emb, split, audit_key): real_video_level},
         'norm': {(emb, track, method, split, audit_key): workbook's own ratio}}

    `norm` is only populated for RIGIS (its tab ships a `NORM -->` mirror in cols
    J:L / N:P). It is read but NEVER plotted -- it exists so `selftest` can prove
    the recomputed ratio matches the workbook's own, which is what catches a tab
    whose absolute and normalised halves have drifted apart.

    Two data hazards this handles explicitly, because both look like "no data" to a
    naive reader and are not:

      * KIVA `DreamDojo / bimanual / non-Markovian / Dense == 0.0` is a GENUINE
        zero (0% invalid segments = a perfect score on a lower-is-better metric).
        A truthiness test (`if v:`) would silently discard the single best cell in
        the tab, so the check here is `is None` / MISSING-token only. It is also
        why KIVA is never used as a ratio DENOMINATOR anywhere.
      * `itm` and `sito` are `---` on every DreamDojo group (6 rows). Those methods
        were never run there, so the cells are ABSENT, not zero -- they must not
        become 0.0 and drag a mean down.
    """
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, data_only=True)
    val, real, norm = {}, {}, {}
    for sheet, key, _label, _unit, _hib, tracks in AUDITS:
        if sheet not in wb.sheetnames:
            raise SystemExit('missing audit tab %r in %s (tabs: %s)'
                             % (sheet, xlsx, ', '.join(wb.sheetnames)))
        ws = wb[sheet]
        alias = AUDIT_ALIAS[sheet]
        # (value cols, norm cols) per split. A:C / E:G hold the absolute blocks;
        # J:L / N:P the RIGIS NORM mirror (absent on KIVA, guarded by max_column).
        blocks = {'makovian': ((1, 2, 3), (10, 11, 12)),
                  'non_makovian': ((5, 6, 7), (14, 15, 16))}
        for split, ((ca, cb, cc), (na, nb, nc)) in blocks.items():
            emb = track = None
            for r in range(1, ws.max_row + 1):
                a = ws.cell(r, ca).value
                a = a.strip() if isinstance(a, str) else a
                if a in EMB_XL:
                    emb, track = a, None
                    continue
                if emb is None:
                    continue
                if a == 'Real Video':
                    v = ws.cell(r, cc).value
                    if not is_missing(v):
                        real[(emb, split, key)] = float(v)
                    track = None
                    continue
                if isinstance(a, str) and a in alias:
                    track = alias[a]
                if track is None or track not in tracks:
                    continue
                meth = ws.cell(r, cb).value
                meth = meth.strip() if isinstance(meth, str) else meth
                if is_missing(meth) or meth == 'Method':
                    continue
                meth = AUDIT_METHOD.get(meth, meth)
                v = ws.cell(r, cc).value
                if not is_missing(v):           # is_missing() keeps a real 0.0
                    val[(emb, track, meth, split, key)] = float(v)
                if nc <= ws.max_column:
                    nv = ws.cell(r, nc).value
                    # The NORM mirror repeats the method label in `nb`; only trust
                    # the row when it lines up, so a shifted block is skipped
                    # rather than mis-attributed to the wrong method.
                    nm = ws.cell(r, nb).value
                    nm = nm.strip() if isinstance(nm, str) else nm
                    if not is_missing(nv) and AUDIT_METHOD.get(nm, nm) == meth:
                        norm[(emb, track, meth, split, key)] = float(nv)
    return {'val': val, 'real': real, 'norm': norm}


def read_speedup(path: Path) -> dict:
    """{(cell, speedup-method): (speedup, provenance)} from the standalone CSV."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                s = float(row['speedup'])
            except (TypeError, ValueError):
                continue
            out[(row['cell'], row['method'])] = (s, row.get('source', ''))
    return out


# The `Speedup` tab's 12 data columns, in sheet order. Row 1 gives the mount
# (Humanoid MV / SV, Bimanual MV / SV, Single-arm MV / SV) and row 2 the model
# (DG = DreamGen, CW = Ctrl-World, DD = DreamDojo); together they name exactly the
# 12 reported cells, so this maps 1:1 onto CELL_OF's values.
SPEEDUP_SHEET = 'Speedup'
SPEEDUP_SHEET_COLS = [
    (2, 'dreamgen_humanoid_mv'), (3, 'ctrlworld4_humanoid_mv'),
    (4, 'dreamgen_humanoid_sv'), (5, 'dreamdojo_humanoid_sv'),
    (6, 'dreamgen_bimanual_mv'), (7, 'ctrlworld4_bimanual_mv'),
    (8, 'dreamgen_bimanual_sv'), (9, 'dreamdojo_bimanual_sv'),
    (10, 'dreamgen_single_arm_mv'), (11, 'ctrlworld_single_arm_mv'),
    (12, 'dreamgen_single_arm_sv'), (13, 'dreamdojo_single_arm_sv'),
]
# Sheet method labels -> the names used everywhere else in this script.
SPEEDUP_SHEET_METHOD = {
    'Dense': 'Dense', 'SiTo': 'SiTo', 'ITM': 'ITM', 'WorldCache': 'WorldCache',
    'DiCache': 'DiCache', 'FasterCache': 'FasterCache', 'PISA': 'PISA',
    'Radial': 'Radial', 'SVG1': 'SVG1', 'SVG2': 'SVG2',
}


def parse_speedup_cell(raw):
    """'1.44x' -> 1.44. Returns (value, note) where note flags a repaired cell.

    The tab is hand-maintained, so it carries hand-typing artefacts: a trailing
    'x', and at least one comma decimal separator ('1,18x'). Both are repaired
    rather than dropped -- but the repair is REPORTED, because a silently coerced
    cell is how a typo becomes a published number.
    """
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        return float(raw), None
    s = str(raw).strip()
    if is_missing(s):
        return None, None
    body = s.rstrip('xX').strip()
    note = None
    if ',' in body:
        body = body.replace(',', '.')
        note = 'comma decimal separator repaired: %r -> %s' % (s, body)
    try:
        return float(body), note
    except ValueError:
        return None, 'unparseable: %r' % s


def read_speedup_xlsx(xlsx: Path) -> tuple[dict, list]:
    """Read the workbook's own `Speedup` tab. Returns (lookup, repair notes)."""
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, data_only=True)
    if SPEEDUP_SHEET not in wb.sheetnames:
        raise SystemExit(
            'no %r tab in %s (tabs: %s).\nUse --speedup-source csv to read '
            'results/reports/speedup_table_full.csv instead.'
            % (SPEEDUP_SHEET, xlsx.name, ', '.join(wb.sheetnames)))
    ws = wb[SPEEDUP_SHEET]
    out, notes = {}, []
    for r in range(1, ws.max_row + 1):
        label = ws.cell(r, 1).value
        label = label.strip() if isinstance(label, str) else label
        meth = SPEEDUP_SHEET_METHOD.get(label)
        if meth is None:
            continue
        for col, cell in SPEEDUP_SHEET_COLS:
            v, note = parse_speedup_cell(ws.cell(r, col).value)
            if note:
                notes.append('%s / %s: %s' % (meth, cell, note))
            if v is None:
                continue
            out[(cell, meth)] = (v, 'workbook:%s!%s%d' % (
                SPEEDUP_SHEET, ws.cell(r, col).column_letter, r))
    return out, notes


def closeness(ratio: float) -> float:
    """Descriptive-metric ratio folded so overshoot is penalised like undershoot."""
    if ratio <= 0:
        return 0.0
    return min(ratio, 1.0 / ratio)


def pooled_speedup(speedups) -> float:
    """HARMONIC mean -- the only aggregate a time budget may use.

    Speedup is a ratio of times, so the quantity that adds across cells is TIME,
    not speed. Pooling with an arithmetic mean and then inverting it (1/mean(s))
    understates the remaining time and so overstates the saving: for 1.0x and
    2.0x, arithmetic gives 1.50x ("33% still spent") while the two runs actually
    take 1.0 + 0.5 = 1.5 dense-units for 2 dense-units of work = 1.33x (75%
    still spent). c3 draws a parts-of-a-whole pie of exactly that time, so it
    must pool the time fractions.
    """
    s = [x for x in speedups if x and x > 0]
    if not s:
        raise ValueError('no positive speedups to pool')
    return len(s) / sum(1.0 / x for x in s)


def build_records(book: dict, speed: dict, split: str,
                  baseline: str = 'real_video', audit: dict | None = None) -> list[dict]:
    """One record per (embodiment, track, method) with NR ratios + speedup.

    `baseline` picks the denominator of every 0.xx in the figures:

      'real_video'  value / Real Video  -- reproduces the workbook's OWN `NORM -->`
                    block (cols J:P) cell for cell, so every number on a figure can
                    be found in the sheet you supplied. Dense becomes a data row,
                    not a reference, because dense is not 1.0 on this baseline.
      'dense'       value / Dense       -- what an acceleration paper usually wants
                    (how much a method lost relative to the dense run it replaces).
                    NOT stored in the workbook; derived here from the same cells.

    Neither invents a number: both are ratios of two cells of the supplied sheet.
    """
    if baseline not in ('real_video', 'dense'):
        raise ValueError('baseline must be real_video or dense')
    val, real, recs = book['val'], book['real'], []
    aval = (audit or {}).get('val', {})
    areal = (audit or {}).get('real', {})
    methods = (['Dense'] + METHOD_ORDER) if baseline == 'real_video' else METHOD_ORDER
    for emb in EMB_XL:
        for track in TRACK_ORDER:
            for meth in methods:
                ratios, raw, denoms = {}, {}, {}
                for sheet, _lab, pol in METRICS:
                    key = METRIC_KEY[sheet]
                    if baseline == 'dense':
                        d = val.get((emb, track, 'Dense', split, key))
                    else:
                        d = real.get((emb, split, key))
                    m = val.get((emb, track, meth, split, key))
                    if d in (None, 0) or m is None:
                        continue
                    r = m / d
                    ratios[key] = r if pol is not None else closeness(r)
                    raw[key], denoms[key] = m, d
                if not ratios:
                    continue
                cell = CELL_OF[(EMB_CSV[emb], track)]
                sp = speed.get((cell, METHOD_TO_SPEEDUP.get(meth, meth)))
                if sp is None and meth == 'Dense':
                    sp = (1.0, 'reference')      # dense IS the 1.00x reference
                # The polarity-bearing metrics only; Actor Activity is DESCRIPTIVE
                # and must not be averaged into a "quality" scalar.
                scored_keys = [METRIC_KEY[s] for s, _l, p in METRICS
                               if p and METRIC_KEY[s] in ratios]
                scored = [ratios[k] for k in scored_keys]
                # Which polarity-bearing metrics this cell is MISSING. DreamDojo has
                # no Text Alignment at all (absent upstream too, not dropped here),
                # so its `retention` is a 2-metric mean while every other track's is
                # a 3-metric mean. Carried per record and flagged in every figure,
                # because a scalar whose definition changes per row is the one thing
                # a single colour scale cannot express on its own.
                missing = [METRIC_KEY[s] for s, _l, p in METRICS
                           if p and METRIC_KEY[s] not in ratios]
                # ---- physics audits (RIGIS / KIVA) ----------------------------
                # Carried as ABSOLUTE values plus their real-video level and
                # `applies` flag, NOT folded into `retention`. Three reasons:
                # KIVA is lower-better (it would cancel the others out), KIVA is a
                # percentage on a totally different scale (it would dominate any
                # mean), and each audit covers only half the tracks (so folding it
                # in would make `retention` mean different things on different
                # rows -- the exact trap `missing_metrics` above exists to flag).
                audits, audit_real, applies = {}, {}, {}
                for _sheet, akey, _al, _unit, _hib, atracks in AUDITS:
                    applies[akey] = track in atracks
                    if not applies[akey]:
                        continue
                    av = aval.get((emb, track, meth, split, akey))
                    if av is not None:
                        audits[akey] = av
                    ar = areal.get((emb, split, akey))
                    if ar is not None:
                        audit_real[akey] = ar
                recs.append({
                    'emb': emb, 'emb_csv': EMB_CSV[emb], 'track': track,
                    'method': meth, 'family': FAMILY.get(meth, 'dense'), 'cell': cell,
                    'split': split, 'baseline': baseline,
                    'ratios': ratios, 'raw': raw, 'denominators': denoms,
                    'retention': mean(scored) if scored else None,
                    'n_scored': len(scored), 'missing_metrics': missing,
                    'partial': bool(missing),
                    'speedup': sp[0] if sp else None,
                    'provenance': sp[1] if sp else 'unavailable',
                    'audits': audits, 'audit_real': audit_real,
                    'audit_applies': applies,
                })
    return recs


# ---------------------------------------------------------------- plot plumbing

class Ctx:
    """Matplotlib context + the shared radial mappings."""

    # Metric ratios are linear around 1.0; speedup is LOG. The two mappings below
    # are deliberately different and the difference is the whole reason this got a
    # comment:
    #
    #   r_ratio / r_speed_radar  put 1.0 at radius 0.5, because on c1 the ratio
    #       spokes and the speedup spoke SHARE one radial axis -- the dense/baseline
    #       reference has to land at the same radius on every spoke or the reference
    #       "circle" becomes a kite. Speedup therefore only gets the outer half.
    #
    #   r_speed                  puts SPEED_LO at radius 0 and SPEED_HI at 1, i.e.
    #       the FULL dial, for every figure where speedup is the only radial
    #       quantity (c2/c4/c5/c9/c10). Those panels have no ratio spoke to align
    #       with, so spending half the radius on an unused 0.4x..1.0x band is pure
    #       waste -- that waste is exactly why the first draft looked flat.
    RATIO_HALFSPAN = 0.40        # ratio 0.60 -> r=0, 1.40 -> r=1
    # The dial window. Fixed (not auto-fit) so that all 2 splits x 2 baselines
    # stay mutually comparable; measured data is 1.00x..2.28x, so 1.00..2.30
    # spends ~100% of the radius on real values. Override with --speed-range.
    SPEED_LO, SPEED_HI = 1.00, 2.30

    def __init__(self, out, formats=('png', 'pdf'), baseline='real_video',
                 source_note='', speed_range=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        self.plt, self.np, self.mpl = plt, np, matplotlib
        self.out, self.formats = Path(out), formats
        self.baseline = baseline
        self.source_note = source_note
        if speed_range:
            self.speed_lo, self.speed_hi = float(speed_range[0]), float(speed_range[1])
            if not 0 < self.speed_lo < self.speed_hi:
                raise SystemExit('--speed-range needs 0 < LO < HI, got %r' % (speed_range,))
        else:
            self.speed_lo, self.speed_hi = self.SPEED_LO, self.SPEED_HI
        # Any speedup pushed outside the dial is CLAMPED, and every clamp is
        # counted and reported. The repo's own tuning rules record genuine
        # sub-1.0x results (DreamDojo bimanual SVG2 = 0.69x, SVG1 = 0.97x), so a
        # future data source could legitimately carry a slowdown -- and a silently
        # clipped 0.69x would render identically to a 1.00x tie, turning a
        # regression into an apparent no-op. Hence: clamp, but never quietly.
        self.speed_clamped = []
        # Every 0.xx on a figure is a ratio against THIS row of the workbook, so
        # the denominator is named on the panel rather than left to a caption.
        self.base_label = ('Real Video' if baseline == 'real_video' else 'Dense')
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'data').mkdir(exist_ok=True)
        plt.rcParams.update({
            'font.family': 'DejaVu Sans', 'font.size': 7.5,
            'pdf.fonttype': 42, 'ps.fonttype': 42,
            'axes.titlesize': 8.5, 'axes.titleweight': 'bold',
            'figure.dpi': 120, 'savefig.bbox': 'tight',
            'legend.frameon': False, 'grid.alpha': 0.25,
        })

    def r_ratio(self, v):
        # Against Real Video the spread is much wider (DreamGen SV humanoid hits
        # ~1.6x Real Video on Visual Integrity), so the half-span widens with the
        # baseline; otherwise every such panel would clip at the rim.
        half = 0.40 if self.baseline == 'dense' else 0.70
        return max(0.0, min(1.06, 0.5 + (v - 1.0) / (2 * half)))

    def _speed_frac(self, v, who):
        """Speedup -> 0..1 within the dial window, log-spaced.

        `who` identifies the DATA point, and only named points are recorded as
        clamps. Reference geometry (the dense ring, axis ticks) calls in without a
        name: those are drawn AT the dial edge by design, so counting them would
        report a clamp on every panel and bury real out-of-range data.
        """
        lo, hi = self.speed_lo, self.speed_hi
        if v is None or v <= 0:
            return None
        if who and (v < lo - 1e-9 or v > hi + 1e-9):
            self.speed_clamped.append((who, float(v)))
        c = max(lo, min(hi, float(v)))
        return self.np.log(c / lo) / self.np.log(hi / lo)

    def r_speed(self, v, who=''):
        """FULL dial: speed_lo -> r=0, speed_hi -> r=1. Speedup-only figures."""
        f = self._speed_frac(v, who)
        return None if f is None else f

    def r_speed_radar(self, v, who=''):
        """OUTER HALF: speed_lo -> r=0.5, speed_hi -> r=1.0.

        Used by c1 only, where the radius is shared with `r_ratio`; keeping
        speed_lo (= 1.00x = dense) at 0.5 is what makes the dense reference a
        true circle across all five spokes.
        """
        f = self._speed_frac(v, who)
        return None if f is None else 0.5 + 0.5 * f

    def speed_ticks(self, n=5):
        """Evenly log-spaced dial ticks, e.g. 1.00 1.22 1.49 1.82 2.30."""
        lo, hi = self.speed_lo, self.speed_hi
        return [float(self.np.exp(self.np.log(lo) + i * (self.np.log(hi) - self.np.log(lo))
                                  / (n - 1))) for i in range(n)]

    def clamp_note(self):
        """One-line report of every value the dial had to clamp, or ''."""
        if not self.speed_clamped:
            return ''
        uniq = sorted(set(self.speed_clamped), key=lambda t: t[1])
        head = ', '.join('%s=%.2fx' % (w, v) for w, v in uniq[:4])
        more = '' if len(uniq) <= 4 else ' +%d more' % (len(uniq) - 4)
        return ('WARNING: %d value(s) clamped into the [%.2f, %.2f] dial: %s%s'
                % (len(uniq), self.speed_lo, self.speed_hi, head, more))

    def ret_color(self, ret):
        """Quality ratio -> colour, centred on the chosen baseline's own level."""
        if ret is None:
            return '#bbbbbb'
        lo, mid, hi = ((0.80, 0.97, 1.04) if self.baseline == 'dense'
                       else (0.60, 0.90, 1.25))
        norm = self.mpl.colors.TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)
        return self.plt.get_cmap('RdYlGn')(norm(max(lo, min(hi, ret))))

    def ret_norm(self):
        lo, mid, hi = ((0.80, 0.97, 1.04) if self.baseline == 'dense'
                       else (0.60, 0.90, 1.25))
        return self.mpl.colors.TwoSlopeNorm(vmin=lo, vcenter=mid, vmax=hi)

    def ret_label(self):
        return ('NR quality ratio vs %s  (mean of the %d polarity-bearing metrics; '
                '1.00 = %s)' % (self.base_label, 3, self.base_label))

    def stamp(self, fig, y=-0.005):
        """Name both data sources on the figure itself, and any dial clamp."""
        note = self.source_note
        clamp = self.clamp_note()
        if clamp:
            note = (note + '\n' + clamp) if note else clamp
        if note:
            fig.text(0.5, y, note, ha='center', va='top',
                     fontsize=6.2, color=('#a11' if clamp else '#666666'))

    def save(self, fig, name):
        for ext in self.formats:
            fig.savefig(self.out / ('%s.%s' % (name, ext)), bbox_inches='tight')
        self.plt.close(fig)
        print('  wrote %s.{%s}' % (self.out / name, ','.join(self.formats)))


FIGURES = {}


def figure(fid, title):
    def deco(fn):
        FIGURES[fid] = (title, fn)
        return fn
    return deco


# ------------------------------------------------------------------- c1  radar

@figure('c1', 'radar: 4 NR metrics as ratio-to-dense + a speedup spoke')
def c1_radar(ctx, recs, split):
    np, plt = ctx.np, ctx.plt
    # The speedup spoke names its own window, because it is the ONE spoke whose
    # radius is not a ratio -- leaving it labelled "log" made 1.00x look like it
    # shared the ratio-1.0 meaning of the other four. Kept SHORT and two-line: the
    # spoke sits on the left rim, and a one-line version ran off the panel edge.
    spokes = [m[1] for m in METRICS] + [
        'Speedup\n%.2f-%.2fx' % (ctx.speed_lo, ctx.speed_hi)]
    keys = [METRIC_KEY[m[0]] for m in METRICS]
    ang = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False).tolist()
    ang += ang[:1]

    by = {(r['emb'], r['track']): [] for r in recs}
    for r in recs:
        by[(r['emb'], r['track'])].append(r)

    fig, axs = plt.subplots(len(EMB_XL), len(TRACK_ORDER),
                            figsize=(4.5 * len(TRACK_ORDER), 4.5 * len(EMB_XL)),
                            subplot_kw=dict(polar=True), constrained_layout=True)
    cmap = plt.get_cmap('tab10').colors
    handles = {}
    for i, emb in enumerate(EMB_XL):
        for j, track in enumerate(TRACK_ORDER):
            ax = axs[i][j]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(ang[:-1])
            ax.set_xticklabels(spokes, fontsize=6.5)
            ax.set_yticks([0.5])
            ax.set_yticklabels([])
            ax.set_ylim(0, 1.06)
            ax.grid(alpha=0.2)
            # dense reference circle
            ax.plot(ang, [0.5] * len(ang), color='k', lw=1.6, ls='--', zorder=5)
            group = by.get((emb, track), [])
            n = 0
            # On the real_video baseline Dense is DATA (it is not 1.0), so it gets
            # its own bold black polygon; on the dense baseline it is the dashed
            # reference circle and must not be drawn twice.
            here = (['Dense'] + METHOD_ORDER if ctx.baseline == 'real_video'
                    else METHOD_ORDER)
            for k, meth in enumerate(here):
                rec = next((r for r in group if r['method'] == meth), None)
                if rec is None:
                    continue
                vals = [ctx.r_ratio(rec['ratios'][q]) if q in rec['ratios'] else np.nan
                        for q in keys]
                # r_speed_radar (outer half), NOT r_speed: this spoke shares its
                # radial axis with the r_ratio spokes above, so 1.00x must sit at
                # the same radius (0.5) as ratio 1.0 or the reference stops being
                # a circle.
                vals.append(ctx.r_speed_radar(rec['speedup'],
                                              '%s/%s' % (rec['track'], meth))
                            if rec['speedup'] else np.nan)

                vals.append(vals[0])
                v = np.array(vals, dtype=float)
                if np.all(np.isnan(v)):
                    continue
                if meth == 'Dense':
                    col, lw, ms, z = '#000000', 2.2, 3.0, 8
                else:
                    col, lw, ms, z = cmap[(k - 1) % len(cmap)], 1.3, 2.2, 4
                label = METHOD_SHORT.get(meth, meth)
                ax.plot(ang, v, color=col, lw=lw, marker='o', ms=ms, zorder=z,
                        label=label)
                handles[label] = col
                n += 1
            # A spoke no method in this panel has was never MEASURED for this track
            # (DreamDojo has no Text Alignment). Say so on the panel: an open polygon
            # looks like a collapsed score, which is the opposite of "no data".
            gone = [s for s, q in zip(spokes, keys)
                    if not any(q in r['ratios'] for r in group)]
            for lab, txt in zip(spokes, ax.get_xticklabels()):
                if lab in gone:
                    txt.set_color('#bbbbbb')
            if gone:
                # Inside the axes and raised to y=0.10: at y=0.03 it cleared the
                # next row's titles but then sat on the "Motion Continuity" /
                # "Actor Activity" spoke labels, which live at the bottom rim. The
                # radar polygons never reach the bottom-centre, so this is the one
                # spot in the dial that is reliably empty.
                ax.text(0.5, 0.10, 'not measured on this track: %s'
                        % ', '.join(g.replace('\n', ' ') for g in gone),
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=6, color='#b06000', style='italic', zorder=9,
                        bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                  ec='#e0c9a6', lw=0.5, alpha=0.92))
            ax.set_title('%s\n%s  (n=%d methods)' % (EMB_LABEL[emb],
                                                     TRACK_LABEL[track], n), pad=12)

    # suptitle WITHOUT an explicit y: under constrained_layout an out-of-range y
    # (the old 1.005) is not reserved space, so the title landed on top of the
    # first row's panel titles. Letting constrained_layout place it keeps the gap.
    fig.suptitle('c1  NR quality (ratio to %s) + speedup, %s split\n'
                 'dashed circle = the %s reference (1.00 on every spoke).  Actor '
                 'Activity is DESCRIPTIVE: plotted as closeness min(r,1/r), not '
                 '"higher is better".'
                 % (ctx.base_label, SPLITS[split], ctx.base_label),
                 fontsize=10)
    fig.legend([plt.Line2D([], [], color=c, lw=2.2 if k == 'Dense' else 1.6)
                for k, c in handles.items()],
               list(handles), loc='lower center', ncol=10, fontsize=8,
               bbox_to_anchor=(0.5, -0.022))
    ctx.stamp(fig, y=-0.032)
    # No tight_layout(): the figure is constrained_layout, and calling both makes
    # matplotlib warn and then discard the constrained solve.
    ctx.save(fig, 'c1_radar_%s' % split)


# -------------------------------------------------------- c2  circular barplot

@figure('c2', 'circular barplot: bar length = speedup, bar colour = NR retention')
def c2_polar_bars(ctx, recs, split):
    np, plt = ctx.np, ctx.plt
    # c2-local radial map, now built ON TOP of the shared dial rather than
    # re-deriving its own window. `ctx.r_speed` already spends the full 0..1 on
    # speed_lo..speed_hi; here that band is shifted onto 0.22..1.00 so every bar
    # keeps a visible stub at 1.00x (a zero-length bar reads as missing data, not
    # as "no speedup"). The dense ring therefore sits at a constant r=0.22.
    R0 = 0.22

    def r_sp(v, who=''):
        f = ctx.r_speed(v, who)
        return None if f is None else R0 + (1.0 - R0) * f

    # Layout: one row of polar panels, then two reserved strips (colorbar, legend).
    # They are GridSpec rows rather than fig.legend/colorbar with a low bbox
    # anchor, because constrained_layout does not reserve space for those and the
    # previous version drew the legend straight on top of the colorbar label.
    # Height 6.0, not 7.0: each panel is 4.9in wide and the dial is a circle, so
    # anything taller than the panel width is slack the circle cannot use -- it
    # just shrank the dial and left a band of whitespace above and below it.
    fig = plt.figure(figsize=(4.9 * len(EMB_XL), 6.0), constrained_layout=True)
    gs = fig.add_gridspec(3, len(EMB_XL), height_ratios=[1, 0.040, 0.050])
    axs = [fig.add_subplot(gs[0, j], polar=True) for j in range(len(EMB_XL))]
    cb_ax = fig.add_subplot(gs[1, :])
    leg_ax = fig.add_subplot(gs[2, :])
    for ax, emb in zip(axs, EMB_XL):
        group = [r for r in recs if r['emb'] == emb and r['speedup']]
        # bars ordered track-major so each track occupies one contiguous arc
        bars = []
        for track in TRACK_ORDER:
            for meth in METHOD_ORDER:
                rec = next((r for r in group
                            if r['track'] == track and r['method'] == meth), None)
                if rec:
                    bars.append(rec)
            # TWO empty slots between tracks, not one: with a single gap the last
            # method of one track and the first of the next were ~9 degrees apart
            # and their rotated labels overlapped at the rim.
            bars.extend([None, None])
        while bars and bars[-1] is None:
            bars.pop()
        n = len(bars)
        width = 2 * np.pi / max(n, 1)
        ang = np.arange(n) * width

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_ylim(0, 1.42)
        # Ticks come from the dial itself, so they can never fall outside it (the
        # old hardcoded 0.6x/0.8x ticks would now sit below r=0 and clamp).
        ticks = ctx.speed_ticks(5)
        ax.set_yticks([r_sp(v) for v in ticks])
        ax.set_yticklabels(['%.2fx' % v for v in ticks], fontsize=6)

        ax.set_rlabel_position(0)
        ax.grid(alpha=0.25)
        ax.spines['polar'].set_visible(False)
        # dense ring
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(th, [r_sp(1.0)] * len(th), color='k', lw=1.4, ls='--', zorder=6)

        for a, rec in zip(ang, bars):
            if rec is None:
                continue
            r = r_sp(rec['speedup'], '%s/%s' % (rec['track'], rec['method']))
            base = r_sp(1.0)
            ax.bar(a, r - base, width=width * 0.88, bottom=base,
                   color=ctx.ret_color(rec['retention']),
                   edgecolor='#b06000' if rec['partial'] else FAMILY_COLOR[rec['family']],
                   hatch='///' if rec['partial'] else None,
                   linewidth=0.8 if rec['partial'] else 0.7, zorder=3)
            # Rotate to the SCREEN radial direction. theta_offset=pi/2 with
            # theta_direction=-1 puts data angle `a` at screen angle (90 - deg);
            # the old -deg-90 is that line 180 degrees out of phase, so labels
            # read inward across the dial. Flip left-pointing ones upright.
            rot = 90.0 - np.degrees(a)
            flip = 90 < (rot % 360) < 270
            # Method label anchored just past the BAR TIP, not at a fixed radius.
            # At a fixed radius all 36 labels sat on one ring and collided; riding
            # the tip spreads them radially, and the tip is also where the eye
            # already is. Short bars therefore keep their label close in.
            ax.text(a, min(r + 0.05, 1.18), METHOD_SHORT[rec['method']],
                    fontsize=5.8,
                    rotation=rot + (180 if flip else 0), rotation_mode='anchor',
                    ha='right' if flip else 'left', va='center',
                    color='#222222', zorder=7)
        # track arcs, outside the labels
        i = 0
        for track in TRACK_ORDER:
            k = sum(1 for r in bars if r and r['track'] == track)
            if not k:
                continue
            a0, a1 = ang[i] - width / 2, ang[i] + (k - 0.5) * width
            ax.plot(np.linspace(a0, a1, 40), [1.30] * 40, lw=2.2,
                    color='#444444', solid_capstyle='butt')
            ax.text((a0 + a1) / 2, 1.38, TRACK_LABEL[track], fontsize=6.5,
                    ha='center', va='center', fontweight='bold')
            i += k + 2          # +2 = the two gap slots inserted above
        ax.set_title('%s\n(%d timed method-cells)' % (EMB_LABEL[emb],
                     sum(1 for r in bars if r)), pad=26)

    # The colorbar and the family legend are stacked in a reserved bottom strip.
    # Under constrained_layout, a fig.legend() with a negative/low bbox anchor is
    # NOT reserved space, so the previous version drew it straight on top of the
    # colorbar label. Reserving the strip explicitly is what keeps them apart.
    handles = [plt.Line2D([], [], color=FAMILY_COLOR[f], lw=2, label=f)
               for f in FAMILY_ORDER]
    handles.append(ctx.mpl.patches.Patch(facecolor='white', edgecolor='#b06000',
                                         hatch='///',
                                         label='ratio from fewer metrics '
                                               '(no Text Alignment)'))
    leg_ax.axis('off')
    leg_ax.legend(handles=handles, loc='center', ncol=4, fontsize=8,
                  title='bar outline = method family', title_fontsize=8)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=ctx.ret_norm())
    cb = fig.colorbar(sm, cax=cb_ax, orientation='horizontal')
    cb.set_label(ctx.ret_label(), fontsize=8)
    fig.suptitle('c2  Circular barplot -- bar LENGTH = measured speedup (log dial '
                 '%.2fx-%.2fx, dashed ring = dense 1.00x), bar FILL = NR quality '
                 'retention.  %s split'
                 % (ctx.speed_lo, ctx.speed_hi, SPLITS[split]), fontsize=10)
    ctx.stamp(fig)
    ctx.save(fig, 'c2_circular_bars_%s' % split)


# ---------------------------------------------------------------- c3  donut pie

@figure('c3', 'donut grid: the dense TIME BUDGET split into spent vs saved')
def c3_donuts(ctx, recs, split):
    np, plt = ctx.np, ctx.plt
    per = defaultdict(list)
    for r in recs:
        if r['speedup']:
            per[r['method']].append(r)

    cols, rows = 3, 3
    # 3.7 in/row left a wide empty band under every donut (the donut is square and
    # `aspect='equal'`, so the extra height is unusable). 3.0 matches the drawn
    # content plus its outside callouts.
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows),
                            constrained_layout=True)
    for k, meth in enumerate(METHOD_ORDER):
        ax = axs[k // cols][k % cols]
        ax.set_aspect('equal')
        ax.axis('off')
        group = per.get(meth, [])
        if not group:
            ax.text(0.5, 0.5, '%s\nnever timed' % METHOD_SHORT[meth], ha='center',
                    va='center', transform=ax.transAxes, fontsize=9, color='#999')
            continue
        s = pooled_speedup(r['speedup'] for r in group)
        rets = [r['retention'] for r in group if r['retention'] is not None]
        ret = mean(rets) if rets else None
        fam = FAMILY_COLOR[FAMILY[meth]]
        t_method = 1.0 / s                     # method time, dense time == 1.0
        if s >= 1.0:
            sizes = [t_method, 1.0 - t_method]
            colors = [fam, '#e9e9e9']
            labels = ['time still spent\n%.0f%% of dense' % (100 * t_method),
                      'time SAVED\n%.0f%% of dense' % (100 * (1 - t_method))]
        else:
            # whole circle = the method's OWN runtime (> dense), so the red wedge
            # is labelled as a share of that runtime -- the same quantity the
            # wedge angle encodes.
            sizes = [1.0, t_method - 1.0]
            colors = [fam, '#c0392b']
            labels = ['work dense would\nhave done',
                      'pure OVERHEAD\n%.0f%% of its runtime'
                      % (100 * (t_method - 1.0) / t_method)]
        ax.pie(sizes, colors=colors, labels=labels, startangle=90,
               counterclock=False, labeldistance=1.16,
               textprops=dict(fontsize=6.4),
               wedgeprops=dict(width=0.34, edgecolor='white', linewidth=1.2))
        ax.text(0, 0.10, METHOD_SHORT[meth], ha='center', va='center',
                fontsize=10, fontweight='bold')
        ax.text(0, -0.06, '%.2fx' % s, ha='center', va='center', fontsize=14,
                color=fam if s >= 1 else '#c0392b', fontweight='bold')
        ax.text(0, -0.26, 'NR vs %s: %s\n%d cells' % (
            ctx.base_label, ('%.3f' % ret) if ret is not None else 'n/a',
            len(group)),
            ha='center', va='center', fontsize=6.4, color='#555')
    for k in range(len(METHOD_ORDER), rows * cols):
        axs[k // cols][k % cols].axis('off')

    fig.suptitle('c3  The only honest PIE here: the dense TIME BUDGET.\n'
                 'A method with speedup s spends 1/s of the dense time (coloured) and '
                 'saves the rest (grey); s<1 shows the overhead it ADDS in red.  '
                 'Speedup = HARMONIC mean over that method\'s timed cells '
                 '(time adds, speed does not), %s split.'
                 % SPLITS[split], fontsize=10)
    # No tight_layout(): c3 is constrained_layout now, and calling both makes
    # matplotlib warn and discard the constrained solve.
    ctx.stamp(fig)
    ctx.save(fig, 'c3_time_budget_donuts_%s' % split)


# ------------------------------------------------------------ c4  polar scatter

@figure('c4', 'polar scatter: the per-cell spread a mean speedup hides')
def c4_polar_scatter(ctx, recs, split):
    np, plt = ctx.np, ctx.plt
    fig = plt.figure(figsize=(9.2, 9.2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.24)
    n = len(METHOD_ORDER)
    sector = 2 * np.pi / n
    # Method names go on as text at a fixed radius rather than as theta ticks:
    # matplotlib places polar tick labels at a constant offset, so at 9 spokes they
    # land on top of the per-method mean annotation.
    ax.set_xticks([i * sector for i in range(n)])
    ax.set_xticklabels([])
    ticks = ctx.speed_ticks(5)
    ax.set_yticks([ctx.r_speed(v) for v in ticks])
    ax.set_yticklabels(['%.2fx' % v for v in ticks], fontsize=6.5)
    ax.grid(alpha=0.25)
    ax.spines['polar'].set_visible(False)
    th = np.linspace(0, 2 * np.pi, 300)
    # With the dial starting AT 1.00x, the dense reference is now the innermost
    # ring (r=0) rather than a mid-radius circle, so there is no "slower than
    # dense" annulus left to shade -- the old red fill would have covered nothing.
    ax.plot(th, [ctx.r_speed(1.0)] * len(th), color='k', lw=1.5, ls='--', zorder=6)

    # sector separators, so each method's dots read as one group
    for i in range(len(METHOD_ORDER)):
        ax.plot([i * sector - sector / 2] * 2, [0, 1.06], color='#dddddd', lw=0.7,
                zorder=1)

    marker = {'Dreamgen MV': 'o', 'Dreamgen SV': 's', 'DreamDojo SV': '^',
              'Ctrl-World MV': 'D'}
    for i, meth in enumerate(METHOD_ORDER):
        group = [r for r in recs if r['method'] == meth and r['speedup']]
        if not group:
            continue
        # spread the cells across the method's sector, deterministic order
        group.sort(key=lambda r: (r['track'], r['emb']))
        for j, rec in enumerate(group):
            off = (j + 1) / (len(group) + 1) - 0.5
            a = i * sector + off * sector * 0.78
            # hollow marker = retention computed from FEWER metrics (DreamDojo has
            # no Text Alignment), so its colour is not the same average as the rest.
            ax.scatter(a, ctx.r_speed(rec['speedup'],
                                      '%s/%s' % (rec['track'], meth)), s=46,
                       marker=marker[rec['track']],
                       color=ctx.ret_color(rec['retention']),
                       edgecolor='#b06000' if rec['partial'] else '#333',
                       linewidth=1.3 if rec['partial'] else 0.5, zorder=5)
        m = pooled_speedup(r['speedup'] for r in group)
        a0, a1 = i * sector - sector * 0.44, i * sector + sector * 0.44
        ax.plot(np.linspace(a0, a1, 30), [ctx.r_speed(m)] * 30, color='#111',
                lw=2.0, zorder=7)
        ax.text(i * sector, 1.15, '%s\n%.2fx  (n=%d)'
                % (METHOD_SHORT[meth], m, len(group)),
                ha='center', va='center', fontsize=7.5, fontweight='bold',
                color=FAMILY_COLOR[FAMILY[meth]])

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=ctx.ret_norm())
    cb = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.035,
                      pad=0.02, shrink=0.7)
    cb.set_label(ctx.ret_label(), fontsize=8)
    handles = [plt.Line2D([], [], marker=marker[t], ls='', color='#555',
                          label=TRACK_LABEL[t]) for t in TRACK_ORDER]
    handles.append(plt.Line2D([], [], marker='o', ls='', mfc='white',
                              mec='#b06000', mew=1.4,
                              label='ratio from fewer metrics'))
    fig.legend(handles=handles, loc='upper left', fontsize=7.5,
               bbox_to_anchor=(0.02, 0.94))
    fig.suptitle('c4  Every timed (model x embodiment) cell, one dot.  Radius = speedup '
                 '(log %.2fx-%.2fx,\ncentre = dense 1.00x), black arc = method pooled '
                 '(harmonic) mean.  %s split'
                 % (ctx.speed_lo, ctx.speed_hi, SPLITS[split]),
                 fontsize=9.5, y=1.0)
    ctx.stamp(fig)
    ctx.save(fig, 'c4_polar_scatter_%s' % split)


# ---------------------------------------------------------------- c5  sunburst

@figure('c5', 'sunburst: family -> method, arc width = #cells, radius = speedup')
def c5_sunburst(ctx, recs, split):
    np, plt = ctx.np, ctx.plt
    timed = [r for r in recs if r['speedup']]
    # Dense is the 1.00x reference, not an acceleration family: it belongs in the
    # hub text, not as a wedge competing with caching/pruning/sparse-attn.
    per_meth = defaultdict(list)
    for r in timed:
        if r['method'] != 'Dense':
            per_meth[r['method']].append(r)
    per_fam = defaultdict(list)
    for m, g in per_meth.items():
        per_fam[FAMILY[m]].extend(g)
    total = sum(len(g) for g in per_meth.values())

    # Square canvas + an explicitly reserved colorbar strip. The dial is a circle,
    # so its drawn size is limited by the SHORTER axes dimension; the old 9.0x9.4
    # with a 3-line title and an `ax=ax` colorbar left ~a third of the height as
    # dead space under the circle. GridSpec gives the colorbar its own row instead
    # of carving it out of the dial's.
    # 9.0 x 7.2. The dial is a circle whose drawn size is min(box width, box
    # height); at 8.4 tall the box was ~1024x750px while the circle only needed
    # ~610px, so ~130px under the dial was dead. Trimming the height moves that
    # slack into the circle instead of leaving it blank.
    fig = plt.figure(figsize=(9.0, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.035])
    ax = fig.add_subplot(gs[0, 0], polar=True)
    cax = fig.add_subplot(gs[1, 0])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.axis('off')
    # 2.15 left a third of the canvas empty: the outermost drawn radius is the
    # wedge top (1.12 + 0.95) plus its label at +0.10, so ~2.17 is the real extent
    # and anything beyond that is pure whitespace that shrinks the whole plot.
    ax.set_ylim(0, 2.18)

    a = 0.0
    for fam in FAMILY_ORDER:
        fam_cells = per_fam.get(fam, [])
        if not fam_cells:
            continue
        span = 2 * np.pi * len(fam_cells) / total
        fam_mean = pooled_speedup(r['speedup'] for r in fam_cells)
        # inner ring: the family, height still encodes its mean speedup
        ax.bar(a + span / 2, 0.55, width=span * 0.985, bottom=0.55,
               color=FAMILY_COLOR[fam], alpha=0.85, edgecolor='white', linewidth=1.4)
        ax.text(a + span / 2, 0.82, '%s\n%.2fx' % (fam, fam_mean), ha='center',
                va='center', fontsize=8, color='white', fontweight='bold')
        # outer ring: methods inside the family, angular width == its cell count
        b = a
        for meth in METHOD_ORDER:
            if FAMILY[meth] != fam or meth not in per_meth:
                continue
            g = per_meth[meth]
            w = 2 * np.pi * len(g) / total
            m = pooled_speedup(r['speedup'] for r in g)
            rets = [r['retention'] for r in g if r['retention'] is not None]
            # Radial height of the outer wedge = speedup on the shared dial. The
            # normaliser is the dial TOP (r_speed(speed_hi) == 1.0 by definition),
            # not a hardcoded 2.5x -- with the old constant the wedge heights
            # silently rescaled whenever the dial window moved.
            h = 0.95 * ctx.r_speed(m, 'family:%s/%s' % (fam, meth))

            ax.bar(b + w / 2, h, width=w * 0.96, bottom=1.12,
                   color=ctx.ret_color(mean(rets) if rets else None),
                   edgecolor=FAMILY_COLOR[fam], linewidth=1.0)
            ax.text(b + w / 2, 1.12 + h + 0.10,
                    '%s\n%.2fx  (n=%d)' % (METHOD_SHORT[meth], m, len(g)),
                    ha='center', va='center', fontsize=6.6, rotation=0)
            b += w
        a += span

    # dense reference ring on the outer scale. r_speed(1.00) == 0 now that the dial
    # starts at 1.00x, so this is the wedge BASELINE ring.
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(th, [1.12 + 0.95 * ctx.r_speed(1.0)] * len(th),
            color='k', lw=1.3, ls='--')
    ax.text(0, 0.18, 'dense\n%d timed\nmethod-cells' % total, ha='center',
            va='center', fontsize=8, fontweight='bold')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=ctx.ret_norm())
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(ctx.ret_label(), fontsize=8)
    fig.suptitle('c5  Sunburst -- inner ring = method family, outer = method.  '
                 'Arc WIDTH = number of timed cells (so a\nmethod measured on fewer '
                 'cells cannot look as solid as one measured everywhere), HEIGHT = pooled\n'
                 '(harmonic) mean speedup (dashed ring = dense 1.00x), FILL = NR retention.  %s split'
                 % SPLITS[split], fontsize=9.5)
    ctx.stamp(fig)
    ctx.save(fig, 'c5_sunburst_%s' % split)


# ----------------------------------------------------- c6  RIGIS / KIVA audits

@figure('c6', 'physics audits: RIGIS on multi-view, KIVA on single-view')
def c6_audits(ctx, recs, split):
    """The two physical audits, each on its own track set and its own scale.

    NOT a radar, and not spokes on c1, for three reasons that all come from the
    data rather than from taste:

      1. COVERAGE IS DISJOINT. RIGIS needs several views (cross-view geometric
         consistency); KIVA lifts body keypoints to 3D from one view. So RIGIS
         exists only on {DreamGen MV, Ctrl-World MV} and KIVA only on
         {DreamGen SV, DreamDojo SV}. On a shared radar, whichever audit did not
         apply would be a blank spoke on every single panel.
      2. POLARITY OPPOSES. RIGIS higher-better, KIVA lower-better. A radar says
         "further out = better", so KIVA would have to be silently inverted --
         and an inverted axis is exactly how a reader misreads a table.
      3. SCALES DIFFER BY 2 ORDERS. RIGIS is 0.68..0.91; KIVA is 0.5%..86%. One
         shared radial axis would flatten RIGIS into a single ring.

    So: horizontal bars, one row per method, grouped by embodiment, RIGIS left and
    KIVA right (log x), each panel carrying its own dashed real-video level.
    """
    np, plt = ctx.np, ctx.plt

    # A log x-axis cannot place 0, and KIVA has exactly one GENUINE 0.0
    # (bimanual / DreamDojo SV / non-Markovian / Dense = 0% invalid segments). That
    # cell is the BEST score in the whole tab, so dropping it would delete the one
    # result the figure most needs to show -- and leaving it in collapsed the whole
    # constrained_layout solve ("axes sizes collapsed to zero").
    # Fix: clamp such a bar to a floor one decade below the smallest positive
    # value, draw it, and mark it "0.0" in its label so the reader is told the bar
    # is at the axis floor rather than at its true value.
    kiva_pos = [r['audits']['KIVA'] for r in recs
                if r['audits'].get('KIVA') is not None and r['audits']['KIVA'] > 0]
    kiva_floor = (min(kiva_pos) / 3.0) if kiva_pos else 0.1

    # Order rows by the DENSE value of the panel's own audit, best first, so the
    # reading order is "who is closest to the ceiling" rather than alphabetical.
    fig, axs = plt.subplots(len(EMB_XL), len(AUDITS),
                            figsize=(7.2 * len(AUDITS), 3.5 * len(EMB_XL)),
                            constrained_layout=True)
    axs = np.atleast_2d(axs)

    for i, emb in enumerate(EMB_XL):
        for j, (_sheet, akey, alabel, unit, hib, atracks) in enumerate(AUDITS):
            ax = axs[i][j]
            tracks = [t for t in TRACK_ORDER if t in atracks]
            rows = ['Dense'] + METHOD_ORDER
            n_t = len(tracks)
            h = 0.80 / max(n_t, 1)
            got_any = False
            for ti, track in enumerate(tracks):
                ys, xs, cols, edges, raws = [], [], [], [], []
                for k, meth in enumerate(rows):
                    rec = next((r for r in recs if r['emb'] == emb
                                and r['track'] == track and r['method'] == meth), None)
                    v = rec['audits'].get(akey) if rec else None
                    if v is None:
                        continue
                    got_any = True
                    ys.append(k + (ti - (n_t - 1) / 2) * h)
                    raws.append(v)
                    # Only the log panel needs the floor; the linear RIGIS panel
                    # plots 0 perfectly well.
                    xs.append(kiva_floor if (unit and v <= 0) else v)
                    cols.append('#333333' if meth == 'Dense'
                                else FAMILY_COLOR[FAMILY[meth]])
                    edges.append('#000000' if meth == 'Dense' else 'none')
                if not xs:
                    continue
                ax.barh(ys, xs, height=h * 0.92, color=cols, edgecolor=edges,
                        linewidth=1.0, alpha=0.55 if ti else 0.95,
                        hatch='' if ti == 0 else '///', zorder=3,
                        label=TRACK_LABEL[track])
                # Value labels go just OUTSIDE the bar end -- multiplicative offset
                # on the log axis, additive on the linear one -- so they never land
                # on top of the neighbouring group's bar. xlim is padded below to
                # leave room, which is also why they are not drawn inside the bar
                # (the 2nd track is alpha-0.55 and white-on-pale is unreadable).
                for y, x, raw in zip(ys, xs, raws):
                    if unit and raw <= 0:
                        # Say so explicitly: the bar is at the axis floor, and the
                        # true value (a perfect 0%) cannot exist on a log scale.
                        txt = '0.0%  (perfect; log axis floor)'
                    else:
                        txt = ('%.1f%%' % raw) if unit else ('%.3f' % raw)
                    ax.text(x * 1.04 if unit else x + 0.012, y, txt,
                            va='center', ha='left', fontsize=5.2,
                            color='#0a6d3a' if (unit and raw <= 0) else '#333',
                            fontweight='bold' if (unit and raw <= 0) else 'normal',
                            zorder=6)

            # real-video level: a CEILING for RIGIS, a reference LEVEL for KIVA.
            rv = next((r['audit_real'].get(akey) for r in recs
                       if r['emb'] == emb and r['audit_real'].get(akey) is not None),
                      None)
            if rv is not None:
                ax.axvline(rv, color='#c0392b', lw=1.5, ls='--', zorder=5)
                ax.text(rv, -1.18, ' real video %s' % (
                    '%.3f' % rv if not unit else '%.1f%%' % rv),
                    color='#c0392b', fontsize=6.2, va='bottom', ha='left')

            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([METHOD_SHORT.get(x, x) for x in rows], fontsize=7)
            # y limits are set explicitly below (reversed, so Dense is on top) --
            # invert_yaxis() would fight that and re-flip it.
            ax.grid(axis='x', alpha=0.25)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            if unit:
                # log x: KIVA spans 0.5%..86%, so a linear axis would squash the
                # whole low-percent end (where the good methods live) into the
                # first few pixels. Left bound is pinned to the 0-clamp floor (an
                # autoscaled log axis with a floored bar is what collapsed the
                # layout); right pad is multiplicative to fit the labels.
                ax.set_xscale('log')
                hi_x = max([x for x in ax.get_xlim()] + [1.0])
                ax.set_xlim(kiva_floor * 0.75, hi_x * 1.75)
                ax.set_xlabel('%s %s  (LOWER is better %s)  [log]'
                              % (alabel, unit, u'\u2193'), fontsize=7.5)
            else:
                # 1.15 not 1.0: RIGIS tops out near 0.91, and the extra headroom is
                # where the value labels live.
                ax.set_xlim(0, 1.15)
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xlabel('%s  (HIGHER is better %s;  1.0 = perfect)'
                              % (alabel, u'\u2191'), fontsize=7.5)
            ax.set_ylim(len(rows) - 0.4, -1.25)     # top room for the rv caption
            ax.set_title('%s  --  %s' % (EMB_LABEL[emb], alabel), fontsize=8.5)
            if not got_any:
                ax.text(0.5, 0.5, '%s does not apply\nto %s tracks'
                        % (alabel, 'multi-view' if unit else 'single-view'),
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8, color='#999999')
            if i == 0:
                # Legend ABOVE the plot area, not 'lower right' -- inside the axes
                # it sat on top of the SVG1/SVG2 bars.
                ax.legend(fontsize=6.4, loc='lower left', ncol=2, framealpha=0.9,
                          frameon=True, bbox_to_anchor=(0.0, 1.06),
                          title='track (hatched = 2nd)', title_fontsize=6.4)

    fig.suptitle('c6  Physics audits, absolute values -- RIGIS (cross-view geometric '
                 'consistency) applies ONLY to multi-view tracks,\nKIVA (%% of invalid '
                 '16-frame keypoint segments) ONLY to single-view. Disjoint by '
                 'construction, opposite polarities,\nso they are NOT radar spokes and '
                 'are NOT folded into NR retention.  %s split' % SPLITS[split],
                 fontsize=10)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c6_audits_%s' % split)


# --------------------------------------------------- c7  speedup/quality Pareto

@figure('c7', 'Pareto scatter: speedup vs NR retention, frontier drawn')
def c7_pareto(ctx, recs, split):
    """The trade-off question directly: is the speedup worth the quality?

    Cartesian on purpose. Every circular figure above has to encode speedup as a
    RADIUS, which means the reader compares arc lengths -- fine for ranking, poor
    for "how much quality did 2.28x cost". Here both axes are linear positions, so
    the trade-off is a direction on the page, and the Pareto frontier is a line you
    can point at.

    The frontier is computed on (speedup up, retention up) and it is per PANEL
    (per embodiment), because retention is only comparable within an embodiment:
    the NR metrics are reference-free absolute scores, so a bimanual 0.78 and a
    humanoid 0.86 are not the same "amount" of quality.
    """
    np, plt = ctx.np, ctx.plt
    fig = plt.figure(figsize=(5.4 * len(EMB_XL), 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, len(EMB_XL), height_ratios=[1, 0.10])
    axs = [fig.add_subplot(gs[0, j]) for j in range(len(EMB_XL))]
    leg_ax = fig.add_subplot(gs[1, :])
    marker = {'Dreamgen MV': 'o', 'Dreamgen SV': 's', 'DreamDojo SV': '^',
              'Ctrl-World MV': 'D'}
    for ax, emb in zip(axs, EMB_XL):
        pts = [r for r in recs if r['emb'] == emb and r['speedup']
               and r['retention'] is not None]
        dense = [r for r in pts if r['method'] == 'Dense']
        meths = [r for r in pts if r['method'] != 'Dense']

        # Dense reference lines: its own speedup is 1.00x by definition, and its
        # retention is the per-track quality the accelerations are trying to keep.
        for r in dense:
            ax.scatter(r['speedup'], r['retention'], marker=marker[r['track']],
                       s=120, facecolor='none', edgecolor='#000000', linewidth=1.8,
                       zorder=6, label='_')
            ax.axhline(r['retention'], color='#000000', lw=0.6, ls=':', alpha=0.45)
        for r in meths:
            ax.scatter(r['speedup'], r['retention'], marker=marker[r['track']],
                       s=52, color=FAMILY_COLOR[r['family']],
                       edgecolor='#b06000' if r['partial'] else '#333',
                       linewidth=1.2 if r['partial'] else 0.4, zorder=5)

        # Pareto frontier over the acceleration methods: a point is dominated if
        # some other point is BOTH faster and better. Ties broken by keeping the
        # first, so a duplicate does not create a fake frontier step.
        front = []
        for r in sorted(meths, key=lambda z: (-z['speedup'], -z['retention'])):
            if not front or r['retention'] > front[-1]['retention'] + 1e-12:
                front.append(r)
        front = list(reversed(front))
        if len(front) > 1:
            ax.plot([r['speedup'] for r in front], [r['retention'] for r in front],
                    color='#c0392b', lw=1.4, ls='-', alpha=0.8, zorder=4,
                    label='Pareto frontier')
        # Frontier labels alternate above/below the point. The frontier's top-left
        # end is where several near-tied points bunch up, and a single fixed offset
        # stacked their labels on top of each other.
        for li, r in enumerate(front):
            dy = 7 if li % 2 == 0 else -11
            ax.annotate('%s\n%s %.2fx' % (METHOD_SHORT[r['method']],
                                          TRACK_LABEL[r['track']], r['speedup']),
                        (r['speedup'], r['retention']),
                        textcoords='offset points', xytext=(6, dy), fontsize=5.4,
                        color='#7b241c', zorder=7)

        ax.axvline(1.0, color='#666', lw=0.8, ls='--')
        ax.set_xlabel('speedup vs dense  (1.00x = dense)', fontsize=8)
        ax.set_ylabel('NR quality retention vs %s' % ctx.base_label, fontsize=8)
        ax.set_title('%s  (%d accelerations)' % (EMB_LABEL[emb], len(meths)),
                     fontsize=9)
        ax.grid(alpha=0.22)
        # Headroom so the topmost frontier label is not clipped by the axes.
        ys = [r['retention'] for r in pts]
        if ys:
            pad = 0.10 * (max(ys) - min(ys) or 0.1)
            ax.set_ylim(min(ys) - pad, max(ys) + 2.2 * pad)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)

    handles = [plt.Line2D([], [], marker=marker[t], ls='', color='#555',
                          label=TRACK_LABEL[t]) for t in TRACK_ORDER]
    handles += [plt.Line2D([], [], marker='s', ls='', color=FAMILY_COLOR[f],
                           label=f) for f in FAMILY_ORDER]
    handles += [plt.Line2D([], [], marker='o', ls='', mfc='none', mec='k',
                           label='Dense (1.00x reference)'),
                plt.Line2D([], [], color='#c0392b', lw=1.4,
                           label='Pareto frontier (per embodiment)')]
    # Two reserved legend rows in a GridSpec strip, not a low-anchored fig.legend:
    # at ncol=5 a wrapped fig.legend row landed on the source stamp, and at ncol=9
    # the 9 entries were squeezed into one unreadable line.
    leg_ax.axis('off')
    leg_ax.legend(handles=handles, loc='center', ncol=5, fontsize=7.5,
                  frameon=False)
    fig.suptitle('c7  Is the speedup worth it?  Up-and-right dominates; the red line '
                 'is the Pareto frontier.\nHollow marker + dotted line = that track\'s '
                 'Dense reference.  %s split' % SPLITS[split], fontsize=10)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c7_pareto_%s' % split)


# ------------------------------------------------------------ c8  polar heatmap

@figure('c8', 'polar heatmap: method sectors x metric rings, colour = value')
def c8_polar_heatmap(ctx, recs, split):
    """Every metric for every method in one overlap-free circular grid.

    This is the circular figure that CANNOT collide: each cell is a fixed
    (sector, ring) tile, so adding methods or metrics shrinks the tiles instead of
    pushing labels into each other. It is the answer to "more chart types" that
    still keeps the circular look the radar gives.

    Each ring is normalised INDEPENDENTLY (per-ring min-max over the plotted
    cells) and every ring is oriented so that BRIGHTER (yellow) == BETTER:
      * higher-better metrics and speedup map straight through;
      * KIVA (lower-better) is inverted;
      * Actor Activity has no polarity, so it enters as `closeness` -- already
        folded to "1.0 = matches the reference" by build_records.
    Per-ring normalisation is what makes a 0..1 RIGIS and a 0..86 KIVA legible
    side by side; the cost is that colour is only comparable WITHIN a ring, which
    is why the numeric value is printed in every tile.

    One sector is left EMPTY as a label gutter: the ring names have to sit at each
    ring's own radius (they are concentric), so without a reserved wedge they land
    on top of whatever tile happens to be at that angle.
    """
    np, plt = ctx.np, ctx.plt
    # ring spec: (label, getter, higher_is_better)
    rings = [(METRIC_KEY[s].replace(' ', '\n'),
              (lambda q: (lambda r: r['ratios'].get(q)))(METRIC_KEY[s]), True)
             for s, _l, _p in METRICS]
    rings += [(a[2], (lambda k: (lambda r: r['audits'].get(k)))(a[1]), a[4])
              for a in AUDITS]
    rings += [('Speedup', lambda r: r['speedup'], True)]

    fig, axs = plt.subplots(1, len(EMB_XL),
                            figsize=(5.6 * len(EMB_XL), 6.6),
                            subplot_kw=dict(polar=True), constrained_layout=True)
    axs = np.atleast_1d(axs)
    methods = ['Dense'] + METHOD_ORDER
    # The gutter is 2.6 method-sectors WIDE, not 1. At one sector the wedge's arc
    # length at the innermost ring (r=1.5 of 9.6) was ~35px while "Integrity" needs
    # ~27px, so the inner ring names spilled onto the Dense / SVG2 tiles either
    # side. 2.6 sectors + keeping the labels on two lines clears both neighbours.
    GUT = 2.6
    n_m = len(methods)
    sector = 2 * np.pi / (n_m + GUT)
    gut_half = GUT * sector / 2
    cmap = plt.get_cmap('viridis')

    for ax, emb in zip(axs, EMB_XL):
        # Offset by +gut_half so the FIRST method sector starts exactly at 12
        # o'clock and the methods run clockwise; the gutter is then the wedge
        # immediately counter-clockwise of 12 o'clock (upper-left).
        ax.set_theta_offset(np.pi / 2 + gut_half)
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        # Extra outer room for the radial method labels; the gutter sector is empty
        # so the visual centre is offset, and this headroom is what keeps the rim
        # labels (ITM / PISA at the bottom) inside the axes.
        ax.set_ylim(0, len(rings) + 3.0)
        # One track per panel column is impossible in a single dial, so the value
        # shown is the MEAN over the tracks where that metric exists. That is safe
        # for RIGIS/KIVA precisely because they are disjoint (the mean is over the
        # 2 tracks the audit applies to, never a mix of measured and absent).
        for ri, (rlab, get, hib) in enumerate(rings):
            vals = []
            for meth in methods:
                got = [get(r) for r in recs
                       if r['emb'] == emb and r['method'] == meth
                       and get(r) is not None]
                vals.append(mean(got) if got else None)
            have = [v for v in vals if v is not None]
            lo, hi = (min(have), max(have)) if have else (0.0, 1.0)
            for mi, (meth, v) in enumerate(zip(methods, vals)):
                a = gut_half + (mi + 0.5) * sector   # gutter occupies data ~0
                if v is None:
                    ax.bar(a, 1.0, width=sector * 0.97, bottom=ri + 1.0,
                           color='#f2f2f2', edgecolor='white', linewidth=0.8)
                    continue
                f = 0.5 if hi - lo < 1e-12 else (v - lo) / (hi - lo)
                if not hib:
                    f = 1.0 - f            # lower-better ring: invert so bright=good
                ax.bar(a, 1.0, width=sector * 0.97, bottom=ri + 1.0,
                       color=cmap(0.10 + 0.85 * f), edgecolor='white',
                       linewidth=0.8)
                txt = ('%.2f' % v) if v < 10 else ('%.0f' % v)
                ax.text(a, ri + 1.5, txt, ha='center', va='center', fontsize=5.0,
                        color='#111' if f > 0.55 else 'white', zorder=5)
            # Ring name in the reserved gutter, at this ring's own radius. The
            # newline is KEPT (not flattened to a space): two short lines are half
            # as wide, which is what lets the innermost rings fit the wedge.
            ax.text(0.0, ri + 1.5, rlab, ha='center', va='center',
                    fontsize=5.2, color='#222', zorder=6, linespacing=0.9,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white', ec='none',
                              alpha=0.85))
        for mi, meth in enumerate(methods):
            a = gut_half + (mi + 0.5) * sector
            deg = np.degrees(a) % 360
            # HORIZONTAL rim labels. Both the tangential and the radial rotated
            # versions stayed hard to read at 10 sectors; upright text with the
            # alignment chosen from the angle (so the label grows away from the
            # dial) is legible with no rotation at all.
            ha = 'center' if deg < 12 or deg > 348 or abs(deg - 180) < 12 else (
                'left' if deg < 180 else 'right')
            ax.text(a, len(rings) + 1.15, METHOD_SHORT.get(meth, meth),
                    fontsize=6.6,
                    fontweight='bold' if meth == 'Dense' else 'normal',
                    rotation=0, ha=ha, va='center', color='#111')
        ax.set_title(EMB_LABEL[emb], fontsize=9, pad=22)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.04,
                      pad=0.02, aspect=50)
    cb.set_ticks([0.05, 0.95])
    cb.set_ticklabels(['worst in ring', 'best in ring'])
    cb.set_label('each RING is min-max normalised on its own and oriented so '
                 'BRIGHTER = BETTER (KIVA inverted); numbers are the raw values',
                 fontsize=7.5)
    fig.suptitle('c8  Polar heatmap -- sector = method, ring = metric. Collision-free '
                 'by construction (fixed tiles).\nGrey tile = not measured for that '
                 'method on any track of this embodiment.  %s split\n' % SPLITS[split],
                 fontsize=10)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c8_polar_heatmap_%s' % split)


# ----------------------------------------------------------- c9  lollipop/bullet

@figure('c9', 'lollipop: 1.00x -> speedup stem per method, with a quality marker')
def c9_lollipop(ctx, recs, split):
    """Speedup as a LENGTH on a common linear axis -- the most readable form.

    c2 already shows speedup as a length, but on a polar axis, where equal
    speedups at different angles have different arc footprints and the eye has to
    correct for it. Here every stem starts at the same x (1.00x = dense) on a
    shared axis, so the ranking and the magnitudes are both read directly.

    The dot colour carries quality retention, so a long stem in red is exactly the
    "fast but lossy" case the repo's tuning rules keep warning about.
    """
    np, plt = ctx.np, ctx.plt
    fig, axs = plt.subplots(1, len(EMB_XL), figsize=(5.0 * len(EMB_XL), 6.2),
                            constrained_layout=True, sharex=True)
    axs = np.atleast_1d(axs)
    for ax, emb in zip(axs, EMB_XL):
        rows = []
        for track in TRACK_ORDER:
            for meth in METHOD_ORDER:
                r = next((z for z in recs if z['emb'] == emb and z['track'] == track
                          and z['method'] == meth and z['speedup']), None)
                if r:
                    rows.append(r)
            rows.append(None)                       # blank separator row
        while rows and rows[-1] is None:
            rows.pop()

        labels = []
        for y, r in enumerate(rows):
            if r is None:
                labels.append('')
                continue
            labels.append(METHOD_SHORT[r['method']])
            ax.hlines(y, 1.0, r['speedup'], color=FAMILY_COLOR[r['family']],
                      lw=2.0, alpha=0.75, zorder=3)
            ax.scatter(r['speedup'], y, s=64, color=ctx.ret_color(r['retention']),
                       edgecolor='#b06000' if r['partial'] else '#333',
                       linewidth=1.2 if r['partial'] else 0.5, zorder=5)
            ax.text(r['speedup'] + 0.035, y, '%.2fx' % r['speedup'], fontsize=5.6,
                    va='center', ha='left', color='#333')
        # track banners in the left margin, so the track name is never a row label
        i = 0
        for track in TRACK_ORDER:
            k = sum(1 for r in rows if r and r['track'] == track)
            if not k:
                continue
            ax.annotate(TRACK_LABEL[track], xy=(0, 0), xytext=(-0.30, i + (k - 1) / 2),
                        textcoords=('axes fraction', 'data'), fontsize=6.4,
                        fontweight='bold', va='center', ha='left', annotation_clip=False)
            i += k + 1
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=6.2)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.axvline(1.0, color='#000', lw=1.0, ls='--')
        ax.set_xlim(0.97, max(ctx.speed_hi, 2.32) + 0.10)
        ax.set_xlabel('speedup vs dense', fontsize=8)
        ax.set_title(EMB_LABEL[emb], fontsize=9)
        ax.grid(axis='x', alpha=0.22)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=ctx.ret_norm())
    cb = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.035,
                      pad=0.02, aspect=50)
    cb.set_label(ctx.ret_label(), fontsize=7.5)
    fig.suptitle('c9  Lollipop -- stem = 1.00x to the measured speedup on a SHARED '
                 'linear axis (no arc-length\ncorrection needed), stem colour = method '
                 'family, dot colour = NR quality retention.  %s split' % SPLITS[split],
                 fontsize=10)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c9_lollipop_%s' % split)


# ---------------------------------------------------- c10  nightingale / rose

@figure('c10', 'nightingale rose: wedge AREA proportional to time saved')
def c10_nightingale(ctx, recs, split):
    """Coxcomb where AREA, not radius, carries the quantity.

    A polar bar's visual weight grows with r^2, so the plain radius encoding used
    by c2 makes a 2.28x look more than twice as strong as a 1.14x. The classic
    Nightingale fix is to set r = sqrt(quantity), which makes the wedge AREA
    linear in the quantity. This panel is the honest-area counterpart to c2.

    The quantity is TIME SAVED (1 - 1/speedup), not speedup itself, because that
    is the thing an area can legitimately represent: it is a FRACTION of the dense
    time budget, it is bounded in [0, 1), and it adds across cells -- whereas
    speedup is a ratio and areas of ratios mean nothing. So a 2.00x wedge has
    exactly twice the area of a 1.33x one (0.50 vs 0.25 of dense time).
    """
    np, plt = ctx.np, ctx.plt
    fig, axs = plt.subplots(1, len(EMB_XL), figsize=(5.2 * len(EMB_XL), 6.0),
                            subplot_kw=dict(polar=True), constrained_layout=True)
    axs = np.atleast_1d(axs)
    # A common radial scale across panels, so the three embodiments are comparable.
    saved_max = max([1.0 - 1.0 / r['speedup'] for r in recs if r['speedup']] or [0.6])
    r_max = np.sqrt(saved_max)

    for ax, emb in zip(axs, EMB_XL):
        wedges = []
        for track in TRACK_ORDER:
            for meth in METHOD_ORDER:
                r = next((z for z in recs if z['emb'] == emb and z['track'] == track
                          and z['method'] == meth and z['speedup']), None)
                if r:
                    wedges.append(r)
            wedges.append(None)
        while wedges and wedges[-1] is None:
            wedges.pop()
        n = len(wedges)
        width = 2 * np.pi / max(n, 1)
        ang = np.arange(n) * width

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks([])
        ax.set_ylim(0, r_max * 1.52)
        # Ticks are placed at sqrt(saved) but LABELLED with the speedup that
        # produces that saving, so the reader never has to think in square roots.
        # Only TWO ticks: sqrt(1 - 1/s) saturates hard (1.60x -> r=0.61,
        # 2.10x -> r=0.73, dial top 2.28x -> r=0.75), so every tick above ~1.75x
        # lands within a few pixels of the rim and the labels ran together.
        tick_sp = [1.25, 1.75]
        ax.set_yticks([np.sqrt(1.0 - 1.0 / s) for s in tick_sp])
        ax.set_yticklabels(['%.2fx' % s for s in tick_sp], fontsize=5.6,
                           color='#555')
        # Put the radial tick labels in one of the blank separator slots between
        # tracks instead of at a fixed 4 degrees, where they landed on the first
        # track's wedges (which are also the tallest, so the clash was guaranteed).
        gaps = [i for i, w in enumerate(wedges) if w is None]
        ax.set_rlabel_position(np.degrees(ang[gaps[0]]) if gaps else 4)
        ax.grid(alpha=0.28)
        ax.spines['polar'].set_visible(False)

        # Label only the EXTREMES (per track: the largest and smallest wedge).
        # Labelling all 33 wedges made them illegible, and this figure's job is the
        # area comparison -- c9 is where every number is read off. The per-track
        # max/min is what an area chart is actually good at showing.
        extremes = set()
        for track in TRACK_ORDER:
            g = [r for r in wedges if r and r['track'] == track]
            if g:
                extremes.add(id(max(g, key=lambda z: z['speedup'])))
                extremes.add(id(min(g, key=lambda z: z['speedup'])))
        for a, r in zip(ang, wedges):
            if r is None:
                continue
            saved = 1.0 - 1.0 / r['speedup']
            ax.bar(a, np.sqrt(saved), width=width * 0.90, bottom=0.0,
                   color=ctx.ret_color(r['retention']),
                   edgecolor='#b06000' if r['partial'] else FAMILY_COLOR[r['family']],
                   hatch='///' if r['partial'] else None,
                   linewidth=0.8, zorder=3)
            if id(r) not in extremes:
                continue
            deg = np.degrees(a)
            # Rotate to the wedge's OUTWARD radial direction. With
            # theta_offset=pi/2 and theta_direction=-1 a data angle `a` sits at
            # SCREEN angle (90 - deg), so that -- not -deg-90, which is the same
            # line 180 degrees out of phase and made every label read inward
            # across the rose -- is the rotation. Flip the ones pointing left so
            # no label is upside down.
            scr = 90.0 - deg
            flip = 90 < (scr % 360) < 270
            ax.text(a, np.sqrt(saved) + r_max * 0.04,
                    '%s %.2fx' % (METHOD_SHORT[r['method']], r['speedup']),
                    fontsize=5.8, fontweight='bold',
                    rotation=scr + (180 if flip else 0),
                    rotation_mode='anchor', ha='right' if flip else 'left',
                    va='center', color='#111', zorder=6)
        i = 0
        for track in TRACK_ORDER:
            k = sum(1 for r in wedges if r and r['track'] == track)
            if not k:
                continue
            a0, a1 = ang[i] - width / 2, ang[i] + (k - 0.5) * width
            ax.plot(np.linspace(a0, a1, 40), [r_max * 1.38] * 40, lw=2.0,
                    color='#444444', solid_capstyle='butt')
            ax.text((a0 + a1) / 2, r_max * 1.47, TRACK_LABEL[track], fontsize=6.0,
                    ha='center', va='center', fontweight='bold')
            i += k + 1
        ax.set_title('%s\n(%d timed method-cells)'
                     % (EMB_LABEL[emb], sum(1 for r in wedges if r)),
                     fontsize=8.5, pad=20)

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=ctx.ret_norm())
    cb = fig.colorbar(sm, ax=axs, orientation='horizontal', fraction=0.04,
                      pad=0.02, aspect=50)
    cb.set_label(ctx.ret_label(), fontsize=7.5)
    fig.suptitle('c10  Nightingale rose -- wedge AREA is proportional to the fraction '
                 'of dense time SAVED (1 - 1/speedup),\nso r = sqrt(saved): a 2.00x '
                 'wedge is exactly twice the area of a 1.33x one. Ring labels give the '
                 'speedup.\nFILL = NR quality retention; only the fastest and slowest '
                 'wedge per track is labelled (see c9 for every value).  %s split'
                 % SPLITS[split], fontsize=9.5)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c10_nightingale_%s' % split)


# ------------------------------------------------------- c11  slope/bump chart

@figure('c11', 'slope chart: rank by speedup vs rank by quality (crossings = trade-off)')
def c11_slope(ctx, recs, split):
    """Two rankings side by side; a crossing line IS the trade-off.

    Every other figure asks the reader to compare two visual channels (length vs
    colour, radius vs fill) to see whether a fast method is also a good one. This
    one converts both to the same channel -- vertical POSITION -- so "fast but
    lossy" becomes a steeply falling line, which is much harder to miss.

    Ranks are computed per embodiment (retention is only comparable within an
    embodiment, as in c7) and per track-pooled method: a method's speedup is the
    HARMONIC mean over its timed cells (`pooled_speedup`, the only correct
    aggregate for a time ratio) and its quality is the arithmetic mean retention
    (a mean of levels, where the arithmetic mean IS correct).
    """
    np, plt = ctx.np, ctx.plt
    fig = plt.figure(figsize=(4.6 * len(EMB_XL), 6.8), constrained_layout=True)
    gs = fig.add_gridspec(2, len(EMB_XL), height_ratios=[1, 0.10])
    axs = [fig.add_subplot(gs[0, j]) for j in range(len(EMB_XL))]
    leg_ax = fig.add_subplot(gs[1, :])
    for ax, emb in zip(axs, EMB_XL):
        per = {}
        for meth in METHOD_ORDER:
            g = [r for r in recs if r['emb'] == emb and r['method'] == meth
                 and r['speedup']]
            rets = [r['retention'] for r in g if r['retention'] is not None]
            if not g or not rets:
                continue
            per[meth] = (pooled_speedup(r['speedup'] for r in g), mean(rets), len(g))
        if not per:
            continue
        # rank 1 = best. Speed: higher is better. Quality: higher retention better.
        by_sp = sorted(per, key=lambda m: -per[m][0])
        by_q = sorted(per, key=lambda m: -per[m][1])
        rank_sp = {m: i + 1 for i, m in enumerate(by_sp)}
        rank_q = {m: i + 1 for i, m in enumerate(by_q)}

        for meth in per:
            y0, y1 = rank_sp[meth], rank_q[meth]
            sp, ret, nn = per[meth]
            drop = y1 - y0          # >0 means it ranks WORSE on quality than speed
            # Colour the LINE by the direction of the trade-off, not by family:
            # that is the question this figure exists to answer.
            col = ('#c0392b' if drop >= 2 else
                   '#2f7d4f' if drop <= -2 else '#999999')
            ax.plot([0, 1], [y0, y1], color=col, lw=2.0, alpha=0.85, zorder=3)
            ax.scatter([0, 1], [y0, y1], s=34, color=FAMILY_COLOR[FAMILY[meth]],
                       zorder=5, edgecolor='white', linewidth=0.8)
            ax.text(-0.045, y0, '%s  %.2fx' % (METHOD_SHORT[meth], sp),
                    ha='right', va='center', fontsize=6.4)
            ax.text(1.045, y1, '%s  %.3f' % (METHOD_SHORT[meth], ret),
                    ha='left', va='center', fontsize=6.4)

        ax.set_xlim(-0.60, 1.60)
        ax.set_ylim(len(per) + 0.6, 0.4)          # rank 1 on top
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['rank by\nSPEEDUP', 'rank by\nNR QUALITY'], fontsize=7.5)
        ax.set_yticks(range(1, len(per) + 1))
        ax.set_yticklabels(['#%d' % i for i in range(1, len(per) + 1)], fontsize=6.4)
        ax.set_title('%s  (%d methods)' % (EMB_LABEL[emb], len(per)), fontsize=9)
        for s in ('top', 'right', 'bottom'):
            ax.spines[s].set_visible(False)
        ax.grid(axis='y', alpha=0.18)

    handles = [plt.Line2D([], [], color='#c0392b', lw=2,
                          label='drops >=2 ranks (fast but lossy)'),
               plt.Line2D([], [], color='#2f7d4f', lw=2,
                          label='gains >=2 ranks (better than its speed suggests)'),
               plt.Line2D([], [], color='#999999', lw=2, label='within 1 rank')]
    handles += [plt.Line2D([], [], marker='o', ls='', color=FAMILY_COLOR[f],
                           label=f) for f in FAMILY_ORDER]
    # Reserved legend row (GridSpec strip, not fig.legend): a 6-entry single row
    # ran wider than the figure, and wrapping it to 2 rows put the second row on
    # top of the source stamp. A real axes takes the space out of the layout.
    leg_ax.axis('off')
    leg_ax.legend(handles=handles, loc='center', ncol=4, fontsize=7.0,
                  frameon=False)
    fig.suptitle('c11  Slope chart -- the SAME method ranked by speedup (left) and by '
                 'NR quality (right).\nA steeply falling line is a method that buys its '
                 'speed with quality; a rising one is the opposite.\nSpeedup pooled as '
                 'the HARMONIC mean over that method\'s timed cells.  %s split'
                 % SPLITS[split], fontsize=9.5)
    ctx.stamp(fig, y=-0.02)
    ctx.save(fig, 'c11_slope_%s' % split)


# ------------------------------------------- c12  speedup bars + audit/NR labels

# `retention` on a record is the mean of the 3 POLARITY-BEARING metrics; c12 was
# asked for the mean of all FOUR. The difference is Actor Activity, which METRICS
# marks `None` = descriptive (more motion is not "better", it is just more), so
# build_records deliberately keeps it out of the quality scalar.
#
# Folding it in is still defensible HERE because build_records already passed it
# through `closeness()` -- ratio 1.30 and ratio 0.77 both become 0.77, i.e. "how
# close to the reference's amount of motion". That is monotone ("1.0 = matches
# reference"), so it shares an orientation with the other three and the mean is
# not adding a quantity that points the other way.
#
# It is computed here instead of changing `retention` so the other ten figures
# keep the 3-metric definition their colour scales and captions were tuned for.
def nr_mean4(rec):
    """Mean of all FOUR NR ratios (Actor Activity via its closeness fold).

    Returns (value, n_used, is_partial). n_used is carried because DreamDojo has
    no Text Alignment upstream, so its mean is over 3 metrics, not 4 -- a scalar
    whose definition changes per row must never be shown without saying so.
    """
    vals = [rec['ratios'][METRIC_KEY[s]] for s, _l, _p in METRICS
            if METRIC_KEY[s] in rec['ratios']]
    if not vals:
        return None, 0, True
    return mean(vals), len(vals), len(vals) < len(METRICS)


def band_ylim(lo, hi, f_lo, f_hi):
    """ylim that maps the data range [lo, hi] onto the panel band [f_lo, f_hi].

    This is what lets three series with unrelated units share one panel without
    overlapping: each gets its own axis, and each axis is padded so its data lands
    in its own horizontal stripe. Solving it instead of hand-tuning pads matters --
    hand-tuned pads kept breaking on the tallest bar in a row, whose value label
    ran into the series above it.

    Returns (y_lo, y_hi). Both axes are TRUNCATED by construction (they do not
    reach 0), so only spacing is readable, never ratios; ticks are therefore
    restricted to the band by `band_ticks`.
    """
    rng = (hi - lo) or (abs(hi) * 0.15 + 0.02)
    total = rng / max(f_hi - f_lo, 1e-6)
    return lo - f_lo * total, hi + (1.0 - f_hi) * total


def band_ticks(np, lo, hi, band=None, fig_h=10.0, n=5):
    """Nice ticks covering ONLY [lo, hi], plus a format string matching the range.

    Restricting ticks to the band matters because the axes here are padded far
    beyond their data: without it the empty region still carries labels and reads
    as "these values are attainable" -- e.g. a speedup axis ticking to 4.4x on a row
    whose fastest method is 2.28x.

    `band` is the (f_lo, f_hi) panel fraction the data occupies. The tick BUDGET is
    derived from that band's height in inches, because a count that is fine on a
    half-panel band silently overlaps on a 13%-tall one: the metric band had four
    labels stacked inside ~0.7in and they printed on top of each other.

    The format is derived from the SPAN, not hardcoded: single-arm's metric means sit
    in 0.89..0.93, where '%.1f' collapses every tick to '0.9'.
    """
    from matplotlib.ticker import MaxNLocator
    if band is not None:
        # ~0.32in per tick label at the font sizes used here.
        n = max(2, min(n, int((band[1] - band[0]) * fig_h / 0.32)))
    rng = hi - lo
    fmt = '%.0f' if rng >= 20 else ('%.1f' if rng >= 1 else
                                    ('%.2f' if rng >= 0.1 else '%.3f'))
    t = [v for v in MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi)
         if lo - 1e-9 <= v <= hi + 1e-9]
    # Drop ticks that collide once rounded to `fmt`. MaxNLocator picks a step from the
    # UNrounded range, so on a narrow band (single-arm RIGIS spans 0.72..0.91) two
    # adjacent ticks can both print '0.80' -- a duplicated label reads as a bug.
    seen, keep = set(), []
    for v in t:
        s = fmt % v
        if s not in seen:
            seen.add(s)
            keep.append(v)
    return keep, fmt


@figure('c12', 'three y axes: speedup bars (left), RIGIS/KIVA (middle), adapted-metric mean dots (right)')
def c12_speedup_bars(ctx, recs, split):
    """Three y axes, each owning one horizontal band of the panel.

      * LEFT   axis, bottom band -> speedup, as BARS (value printed inside)
      * MIDDLE axis, middle band -> the physics audit (RIGIS or KIVA), as a dash
                                    marker with its value printed above it
      * RIGHT  axis, top band    -> mean of the adapted NR metrics, as a DOT

    Three axes rather than two because all three quantities have unrelated units
    and ranges (a speedup ratio, a percentage or a 0..1 score, and a metric ratio);
    on a shared axis two of the three collapse. The right spine is pushed outward
    so the middle and right scales are both readable.

    Every marker for a given method sits at the SAME x as its bar, so the three
    bands read as one column per method and need no per-method legend entry.

    Rows are split by VIEW TYPE, which the data forces rather than a preference:
      * top row    = single-view tracks (DreamGen SV, DreamDojo SV) -> KIVA
      * bottom row = multi-view tracks  (DreamGen MV, Ctrl-World MV) -> RIGIS
    RIGIS and KIVA cover disjoint track sets (see the AUDITS comment), so a row
    carrying both audits would leave one of the two blank on every bar.

    No suptitle and no source stamp: the caption goes in the paper, per request.
    """
    np, plt = ctx.np, ctx.plt
    C_AUD, C_MET = '#1f6f8b', '#7b241c'
    # Band layout, as fractions of panel height. Bars own the bottom, the audit the
    # middle, the metric the top; the gaps are what keep the value labels of one
    # band clear of the markers of the next. Bars get the largest share (0.58): they
    # are the only band whose height carries meaning, while the other two are thin
    # marker series, so giving them equal thirds wasted the panel and made the
    # in-bar speedup labels hard to read.
    B_BAR = (0.00, 0.50)
    B_AUD = (0.58, 0.76)
    B_MET = (0.84, 0.96)
    rows = [(('Dreamgen SV', 'DreamDojo SV'), 'KIVA'),
            (('Dreamgen MV', 'Ctrl-World MV'), 'RIGIS')]
    # Derive the method list from the records, do NOT hardcode ['Dense'] + ORDER:
    # on the `dense` baseline build_records drops Dense entirely (it would be a
    # constant 1.0 row), so a hardcoded list leaves an empty leading slot -- and
    # when the legend label was attached at that slot, ax.legend() raised
    # "No artists with labels".
    methods = [m for m in (['Dense'] + METHOD_ORDER)
               if any(r['method'] == m for r in recs)]

    for emb in EMB_XL:
        # MANUAL layout, not constrained_layout: the third axis lives on a spine
        # pushed outward, which constrained_layout cannot account for -- it reserved
        # room for the spine AND for its own padding, squeezing the plot area to
        # ~60% of the width until the method names overlapped. Fixed margins give
        # the outward spine exactly the room it needs and nothing more.
        FIG_H = 10.0
        fig, axs = plt.subplots(2, 1, figsize=(17.0, FIG_H))
        fig.subplots_adjust(left=0.052, right=0.895, top=0.925, bottom=0.055,
                            hspace=0.30)
        # Height of ONE panel in inches -- band_ticks needs it to budget tick labels
        # per band (a 13%-tall band cannot hold 4 labels).
        PANEL_H = FIG_H * (0.925 - 0.055) / (2 + 0.30)

        for ax, (tracks, akey) in zip(axs, rows):
            _a = AUDIT_BY_KEY[akey]
            aunit, ahib = _a[3], _a[4]
            present = [t for t in tracks
                       if any(r['emb'] == emb and r['track'] == t for r in recs)]
            n_t = max(len(present), 1)
            bw = 0.84 / n_t
            x = np.arange(len(methods), dtype=float)
            ax_a = ax.twinx()                                    # middle axis
            ax_m = ax.twinx()                                    # right axis
            ax_m.spines['right'].set_position(('outward', 58))

            sp_max = ctx.speed_lo
            cells = []
            for ti, track in enumerate(present):
                for mi, meth in enumerate(methods):
                    rec = next((r for r in recs if r['emb'] == emb
                                and r['track'] == track and r['method'] == meth), None)
                    if rec is None or rec['speedup'] is None:
                        continue
                    cells.append((x[mi] + (ti - (n_t - 1) / 2) * bw,
                                  rec['speedup'], ti, meth, track, rec))
                    sp_max = max(sp_max, rec['speedup'])

            # ---- LEFT axis (bars): baseline at ZERO, bars strictly proportional ----
            # Earlier revisions floored the axis 0.14 then 0.45 spans BELOW 1.00x to
            # stop a 1.00x bar having height zero. That is no longer needed and was
            # strictly worse: because the bars only get the bottom B_BAR of the panel
            # now, a 1.00x bar from zero is already ~22-30% of the panel (taller than
            # any floored version), AND the axis stops being truncated, so bar height
            # becomes a real ratio again. Floor arithmetic for reference: with floor
            # f, the 2.28x:1.00x height ratio was (2.28-f)/(1-f) = 3.2 at f=0.42,
            # i.e. a 41% exaggeration; at f=0 it is exactly 2.28.
            y_floor = 0.0
            labelled = set()
            for xc, sp, ti, meth, track, rec in cells:
                col = ('#555555' if meth == 'Dense'
                       else FAMILY_COLOR[FAMILY[meth]])
                ax.bar(xc, sp - y_floor, width=bw * 0.90, bottom=y_floor,
                       color=col, alpha=1.0 if ti == 0 else 0.55,
                       hatch='' if ti == 0 else '///',
                       edgecolor='#000000' if meth == 'Dense' else 'white',
                       linewidth=1.0, zorder=3,
                       label=(None if track in labelled else TRACK_LABEL[track]))
                labelled.add(track)
                # Speedup inside the bar, anchored in POINTS off the axes bottom (not
                # at a data y) so it lands the same distance from the baseline on both
                # rows, whose spans differ.
                ax.annotate('%.2fx' % sp, (xc, 0.0),
                            xycoords=ax.get_xaxis_transform(),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', va='bottom', rotation=90,
                            fontsize=13, color='white', fontweight='bold',
                            zorder=6)
            ax.set_ylim(*band_ylim(y_floor, sp_max, *B_BAR))
            tk, fm = band_ticks(np, y_floor, sp_max, B_BAR, PANEL_H)
            ax.set_yticks(tk)
            # 'x' suffix on every tick: this axis is a MULTIPLE, not a duration, and
            # bare '1.5' next to a bar labelled '1.35x' reads as a different quantity.
            ax.set_yticklabels([(fm % v) + 'x' for v in tk])
            ax.spines['left'].set_bounds(min(tk), max(tk))
            # Shade everything at or below dense. The bars all start at 0, so without
            # this the eye has no reference for "how much of this bar is the GAIN";
            # with it, the part of each bar sticking out of the grey band IS the
            # speedup over dense, which is the quantity the figure is about.
            ax.axhspan(0.0, ctx.speed_lo, color='#000000', alpha=0.055, zorder=0)
            ax.axhline(ctx.speed_lo, color='#000000', lw=1.5, ls='--', zorder=4)
            ax.annotate('1.00x = dense', (0.0, ctx.speed_lo),
                        xycoords=('axes fraction', 'data'),
                        textcoords='offset points', xytext=(4, 3),
                        ha='left', va='bottom', fontsize=12, color='#333333',
                        fontweight='bold', zorder=10)

            # ---- MIDDLE axis: the audit, as a dash marker + its value -------------
            # A dash (not a dot) so it cannot be confused with the metric series on
            # the right axis, and because a horizontal tick reads as "a level".
            av_all = [rec['audits'][akey] for _x, _s, _t, _m, _tr, rec in cells
                      if rec['audits'].get(akey) is not None]
            # Within a method group the two tracks can be near-equal (single-arm's
            # Dense is 0.907 vs 0.909), which puts their labels at the same height
            # where '0.907' and '0.909' are wider than the two bars combined. Stagger
            # the second track's label upward, but only in the groups where that
            # actually happens, so the rest stay tight to their marker.
            a_span = (max(av_all) - min(av_all)) if av_all else 1.0
            a_tip = {}
            for _x, _s, ti, meth, _tr, rec in cells:
                v = rec['audits'].get(akey)
                if v is not None:
                    a_tip.setdefault(meth, {})[ti] = v
            a_stag = {m: (len(d) > 1
                          and abs(d.get(0, 0.0) - d.get(1, 0.0)) < 0.10 * a_span)
                      for m, d in a_tip.items()}
            for ti, track in enumerate(present):
                pts = [(xc, rec['audits'][akey], meth)
                       for xc, _s, _t2, meth, tr, rec in cells
                       if tr == track and rec['audits'].get(akey) is not None]
                if not pts:
                    continue
                pts.sort()
                px = np.asarray([p for p, _v, _m in pts])
                pv = np.asarray([v for _p, v, _m in pts])
                ax_a.plot(px, pv, ls=':', lw=1.4, color=C_AUD, alpha=0.55, zorder=7)
                ax_a.scatter(px, pv, marker='_', s=340 if ti == 0 else 240,
                             color=C_AUD, linewidths=3.4 if ti == 0 else 2.4,
                             zorder=8)
                for xx, vv, mm in pts:
                    ax_a.annotate(('%.1f' % vv) if aunit else ('%.3f' % vv),
                                  (xx, vv), textcoords='offset points',
                                  xytext=(0, 7 + (14 if (ti and a_stag[mm]) else 0)),
                                  ha='center', va='bottom',
                                  fontsize=12.5, fontweight='bold', color='#0f3d4d',
                                  zorder=9)
            rv = next((r['audit_real'].get(akey) for r in recs
                       if r['emb'] == emb and r['track'] in present
                       and r['audit_real'].get(akey) is not None), None)
            if av_all:
                # Include the real-video level in the band range: it is drawn on this
                # axis below, and if it falls outside ylim it is silently clipped --
                # a reference line that is invisible is worse than none.
                a_lo = min(min(av_all), rv) if rv is not None else min(av_all)
                a_hi = max(max(av_all), rv) if rv is not None else max(av_all)
                ax_a.set_ylim(*band_ylim(a_lo, a_hi, *B_AUD))
                tk, fm = band_ticks(np, a_lo, a_hi, B_AUD, PANEL_H)
                ax_a.set_yticks(tk)
                ax_a.set_yticklabels([fm % v for v in tk])
                # Spine drawn ONLY across the band its data occupies. A full-height
                # spine with three ticks clustered near the top is what made the right
                # side look wrong: it advertises a range that carries no data. Bounding
                # the spine makes each axis visibly own one stripe.
                ax_a.spines['right'].set_bounds(min(tk), max(tk))
            # ylabel positioned at the BAND's centre, not the panel's. Matplotlib
            # centres it over the full axes height by default, which put both right
            # labels side by side in the middle of a panel whose data sits at the top.
            ax_a.set_ylabel('%s%s (%s)' % (akey, ' %' if aunit else '',
                                           u'\u2191 better' if ahib
                                           else u'\u2193 better'),
                            fontsize=15, color=C_AUD,
                            y=min((B_AUD[0] + B_AUD[1]) / 2, 0.58),
                            ha='center')
            ax_a.tick_params(axis='y', labelsize=13, colors=C_AUD)
            ax_a.spines['right'].set_color(C_AUD)

            # ---- RIGHT axis: mean adapted metric, as a DOT ------------------------
            # Hollow dot = mean over 3 metrics, not 4 (DreamDojo has no Text
            # Alignment upstream). The dots carry no value labels by request, so a
            # scalar whose definition changes per row is flagged by marker fill.
            mv_all = []
            for ti, track in enumerate(present):
                pts = []
                for xc, _s, _t, _m, tr, rec in cells:
                    if tr != track:
                        continue
                    nr, _n, nr_part = nr_mean4(rec)
                    if nr is not None:
                        pts.append((xc, nr, nr_part))
                if not pts:
                    continue
                pts.sort()
                px = np.asarray([p for p, _v, _q in pts])
                pv = np.asarray([v for _p, v, _q in pts])
                mv_all.extend(pv.tolist())
                ax_m.plot(px, pv, ls='--', lw=1.6, color=C_MET, alpha=0.6, zorder=7)
                for xx, vv, pp in zip(px, pv, [q for _p, _v, q in pts]):
                    ax_m.scatter([xx], [vv], s=200 if ti == 0 else 150,
                                 marker='o' if ti == 0 else 'D',
                                 facecolor=('white' if pp else
                                            (C_MET if ti == 0 else '#e8776a')),
                                 edgecolor='#4a120c',
                                 linewidth=2.0 if pp else 1.5, zorder=9)
            if mv_all:
                ax_m.set_ylim(*band_ylim(min(mv_all), max(mv_all), *B_MET))
                tk, fm = band_ticks(np, min(mv_all), max(mv_all), B_MET, PANEL_H)
                ax_m.set_yticks(tk)
                ax_m.set_yticklabels([fm % v for v in tk])
                ax_m.spines['right'].set_bounds(min(tk), max(tk))
            ax_m.set_ylabel('Mean adapted metric', fontsize=15, color=C_MET,
                            y=min((B_MET[0] + B_MET[1]) / 2, 0.72),
                            ha='center')
            ax_m.tick_params(axis='y', labelsize=13, colors=C_MET)
            ax_m.spines['right'].set_color(C_MET)

            ax.set_xticks(x)
            ax.set_xticklabels([METHOD_SHORT.get(m, m) for m in methods],
                               fontsize=16)
            ax.set_ylabel('Speedup', fontsize=15,
                          y=(B_BAR[0] + B_BAR[1]) / 2)
            ax.tick_params(axis='y', labelsize=13)
            ax.tick_params(axis='x', labelsize=16, length=0, pad=2)
            # Grid drawn as explicit short segments, not ax.grid(): a full-width grid
            # line at a SPEEDUP value crossing the audit and metric bands reads as a
            # gridline for those series too, which is wrong (they are different axes).
            for v in tk:
                ax.plot([0, 1], [v, v], transform=ax.get_yaxis_transform(),
                        color='#000000', alpha=0.10, lw=0.8, zorder=0,
                        clip_on=True)
            ax.set_axisbelow(True)
            ax.margins(x=0.012)
            for a in (ax, ax_a, ax_m):
                a.spines['top'].set_visible(False)

            # Legend outside, in the thin strip above each panel: all three bands are
            # occupied by data, so an in-axes legend lands on a series.
            handles, _ = ax.get_legend_handles_labels()
            handles = list(handles) + [
                plt.Line2D([], [], marker='_', ls=':', color=C_AUD, ms=13, mew=3,
                           label='%s%s (middle axis)'
                                 % (akey, ' %' if aunit else '')),
                plt.Line2D([], [], marker='o', ls='--', color=C_MET, mfc=C_MET,
                           mec='#4a120c', ms=11,
                           label='mean adapted metric (right axis)')]
            if any(nr_mean4(r)[2] for r in recs
                   if r['emb'] == emb and r['track'] in present):
                handles.append(
                    plt.Line2D([], [], marker='o', ls='', mfc='white',
                               mec='#4a120c', mew=2, ms=11,
                               label='hollow = mean over 3 of 4 metrics'))
            if rv is not None:
                # Drawn as a real LEVEL on the audit axis, not a legend string. As a
                # legend entry it sat at the right end of the strip, where it ran into
                # the 'Mean adapted metric' ylabel; and a reference level belongs on
                # the axis it refers to, where it can actually be compared to the
                # markers. Bounded to the audit band so it cannot be mistaken for a
                # speedup gridline.
                ax_a.plot([0.01, 0.99], [rv, rv],
                          transform=ax_a.get_yaxis_transform(),
                          color=C_AUD, lw=1.6, ls=(0, (6, 3)), alpha=0.85, zorder=6)
                # Label BELOW the line at the left edge. At the right edge it covered
                # the last method's audit value (single-arm SVG2 = 18.0); below-left is
                # empty because the audit markers there sit above their labels.
                ax_a.annotate('real video %s' % (('%.1f%%' % rv) if aunit
                                                 else '%.3f' % rv),
                              (0.012, rv), xycoords=ax_a.get_yaxis_transform(),
                              textcoords='offset points', xytext=(0, -4),
                              ha='left', va='top', fontsize=11.5,
                              fontweight='bold', color=C_AUD, zorder=11,
                              bbox=dict(boxstyle='round,pad=0.18', fc='white',
                                        ec=C_AUD, lw=0.7, alpha=0.92))
            ax.legend(handles=handles, fontsize=13.5, loc='lower left',
                      bbox_to_anchor=(0.0, 1.0), ncol=len(handles),
                      frameon=False, handlelength=1.6, columnspacing=1.3,
                      borderaxespad=0.15)

        ctx.save(fig, 'c12_speedup_%s_%s' % (EMB_CSV[emb], split))


# -------------------------------------------------------------------- CLI

def dump_data(ctx, recs, split, src_desc='', repairs=()):
    """Every plotted number as CSV + JSON, so a figure can be audited without rerun."""
    cols = ['emb_csv', 'track', 'method', 'family', 'cell', 'split', 'baseline',
            'speedup', 'speedup_source', 'nr_ratio_mean',
            'nr_mean4', 'nr_mean4_n', 'nr_mean4_partial'] \
        + [METRIC_KEY[m[0]] for m in METRICS] \
        + ['rigis', 'rigis_real_video', 'rigis_ratio_real', 'rigis_applies',
           'kiva_pct', 'kiva_real_level', 'kiva_applies']
    path = ctx.out / 'data' / ('circular_%s_%s.csv' % (split, ctx.baseline))
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            rg = r['audits'].get('RIGIS')
            rg_rv = r['audit_real'].get('RIGIS')
            kv = r['audits'].get('KIVA')
            kv_rv = r['audit_real'].get('KIVA')
            _m4, _m4n, _m4p = nr_mean4(r)
            # RIGIS gets a ratio-to-real-video (it is a bounded higher-better score
            # whose real-video level is a ceiling). KIVA deliberately does NOT: it
            # is lower-better AND has a genuine 0.0 cell, so a ratio would be both
            # inverted in meaning and undefined at the best data point.
            w.writerow([r['emb_csv'], r['track'], r['method'], r['family'], r['cell'],
                        r['split'], r['baseline'],
                        '' if r['speedup'] is None else '%.4f' % r['speedup'],
                        r['provenance'],
                        '' if r['retention'] is None else '%.4f' % r['retention'],
                        # nr_mean4 is what c12 prints; it is a DIFFERENT scalar from
                        # nr_ratio_mean (it includes Actor Activity's closeness), so
                        # both are exported and the `_n` column says how many
                        # metrics each row's mean actually covers.
                        '' if _m4 is None else '%.4f' % _m4, _m4n, int(_m4p)]
                       + ['' if METRIC_KEY[m[0]] not in r['ratios'] else
                          '%.4f' % r['ratios'][METRIC_KEY[m[0]]] for m in METRICS]
                       + ['' if rg is None else '%.4f' % rg,
                          '' if rg_rv is None else '%.4f' % rg_rv,
                          '' if (rg is None or not rg_rv) else '%.4f' % (rg / rg_rv),
                          int(r['audit_applies'].get('RIGIS', False)),
                          '' if kv is None else '%.4f' % kv,
                          '' if kv_rv is None else '%.4f' % kv_rv,
                          int(r['audit_applies'].get('KIVA', False))])
    timed = [r for r in recs if r['speedup']]
    meta = {
        'split': split,
        'baseline': ctx.baseline,
        'baseline_meaning': ('every NR number is value / %s of the same '
                             '(embodiment, track, metric, split)' % ctx.base_label),
        'speedup_source': src_desc,
        'cell_repairs': list(repairs),
        'n_method_cells': len(recs),
        'n_timed': len(timed),
        'n_untimed': len(recs) - len(timed),
        'per_method_pooled_speedup_harmonic': {
            METHOD_SHORT.get(m, m): round(pooled_speedup(
                r['speedup'] for r in timed if r['method'] == m), 4)
            for m in (['Dense'] + METHOD_ORDER)
            if any(r['method'] == m for r in timed)},
        'metric_polarity': {METRIC_KEY[s]: ('higher is better' if p
                                            else 'DESCRIPTIVE -- no polarity')
                            for s, _l, p in METRICS},
        # The speedup dial is recorded so a figure can be re-read years later
        # without guessing what radius meant what. `clamped` should be 0; a
        # non-zero value means some measured speedup fell OUTSIDE the window and
        # was drawn at the edge.
        'speedup_dial': {
            'lo': ctx.speed_lo, 'hi': ctx.speed_hi, 'spacing': 'log',
            'meaning': ('r_speed: %.2fx -> r=0, %.2fx -> r=1 (speedup-only '
                        'figures). r_speed_radar: %.2fx -> r=0.5, %.2fx -> r=1.0 '
                        '(c1 only, where the radius is shared with r_ratio).'
                        % (ctx.speed_lo, ctx.speed_hi, ctx.speed_lo, ctx.speed_hi)),
            'n_values_outside': sum(
                1 for r in timed
                if not (ctx.speed_lo <= r['speedup'] <= ctx.speed_hi)),
            'values_outside': sorted(
                '%s/%s/%s=%.2fx' % (r['emb_csv'], r['track'],
                                    METHOD_SHORT.get(r['method'], r['method']),
                                    r['speedup'])
                for r in timed
                if not (ctx.speed_lo <= r['speedup'] <= ctx.speed_hi)),
        },
        'audits': {
            key: {
                'label': label,
                'unit': unit or 'dimensionless 0..1',
                'polarity': 'higher is better' if hib else 'LOWER is better',
                'applies_to_tracks': list(tracks),
                'n_rows_with_value': sum(1 for r in recs if key in r['audits']),
                'real_video_level_per_embodiment': {
                    r['emb_csv']: r['audit_real'][key] for r in recs
                    if key in r['audit_real']},
                'folded_into_nr_ratio_mean': False,
            }
            for _sheet, key, label, unit, hib, tracks in AUDITS},
        'notes': [
            'nr_ratio_mean = mean ratio of the polarity-bearing metrics; Actor '
            'Activity is excluded from it because it is DESCRIPTIVE.',
            'nr_mean4 = mean of ALL FOUR NR ratios (what figure c12 prints above '
            'each bar). It is a different scalar from nr_ratio_mean: it includes '
            'Actor Activity, which is safe only because that metric is already '
            'folded to closeness (1.0 = matches the reference amount of motion), '
            'so all four share an orientation. nr_mean4_n is 3 on DreamDojo rows '
            '(no Text Alignment upstream) and those are flagged nr_mean4_partial.',
            'Ctrl-World MV = 4-view grid for bimanual/humanoid, 3-view strip for '
            'single_arm (assets/scripts/benchmark_cells.py).',
            'RIGIS and KIVA are NOT in nr_ratio_mean. Their track coverage is '
            'disjoint (RIGIS multi-view only, KIVA single-view only), their '
            'polarities oppose, and their scales differ by two orders of '
            'magnitude -- averaging them in would make nr_ratio_mean mean a '
            'different thing on different rows.',
            'kiva_pct has no ratio-to-real-video column on purpose: it is '
            'lower-better, and DreamDojo/bimanual/non_makovian/Dense is a genuine '
            '0.0 (0%% invalid segments), which cannot be a ratio denominator.',
        ],
    }
    out_json = ctx.out / 'data' / ('circular_%s_%s.json' % (split, ctx.baseline))
    with open(out_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print('  wrote %s and its .json sidecar' % path)
    return meta


def selftest(xlsx: Path, speedup_csv: Path) -> int:
    """Guard everything that would silently produce a WRONG figure.

    Grouped by what it protects:
      1-2  the two aggregations (harmonic pooling, descriptive folding)
      3-5  cell/method name routing (a typo must fail, not read as "unmeasured")
      6    the partial-metric flag (DreamDojo has no Text Alignment)
      7    the Speedup sheet is clean and inside the dial
      8    the speedup dial's endpoints, monotonicity and clamp accounting
      9    the KIVA tab is the NEW detector's numbers, not the superseded scale
      10   the RIGIS NORM mirror agrees with the recomputed ratio
      11   RIGIS/KIVA track coverage is disjoint (the premise of c6 and c8)
      12   the two data hazards: a genuine 0.0, and '---' != 0
    """
    bad = []

    def check(name, got, want):
        ok = got == want
        if not ok:
            bad.append(name)
        print('  %s %-58s got %s want %s' % ('ok  ' if ok else 'FAIL', name,
                                             got, want))

    # 1. Pooling must be harmonic. Two cells at 1.0x and 2.0x take 1.5 dense-units
    # to do 2 dense-units of work -> 1.333x, NOT the arithmetic 1.5x.
    check('pooled_speedup(1.0, 2.0)', round(pooled_speedup([1.0, 2.0]), 4), 1.3333)
    check('pooled_speedup(2.0)', round(pooled_speedup([2.0]), 4), 2.0)
    check('pooled <= arithmetic', pooled_speedup([1.0, 2.0]) < mean([1.0, 2.0]), True)

    # 2. Descriptive folding must punish overshoot exactly like undershoot, or a
    # method that doubles Actor Activity would score as if it matched dense.
    check('closeness(2.0) == closeness(0.5)',
          round(closeness(2.0), 6) == round(closeness(0.5), 6), True)
    check('closeness(1.0)', closeness(1.0), 1.0)

    # 3. Ctrl-World layout routing. bimanual/humanoid MUST hit the 4-view grid
    # cells and single_arm the 3-view strip (assets/scripts/benchmark_cells.py).
    # Getting this wrong is how the grid rows silently inherited strip timings
    # (tmp/20261005_refactor/docs/speedup_coverage_gaps.md 3.1).
    check('CW cell bimanual', CELL_OF[('bimanual', 'Ctrl-World MV')],
          'ctrlworld4_bimanual_mv')
    check('CW cell humanoid', CELL_OF[('humanoid', 'Ctrl-World MV')],
          'ctrlworld4_humanoid_mv')
    check('CW cell single_arm', CELL_OF[('single_arm', 'Ctrl-World MV')],
          'ctrlworld_single_arm_mv')

    # 4. Every cell id we ask for must exist in the speedup CSV, otherwise a
    # typo would read as "never measured" instead of failing.
    speed, _ = read_speedup_xlsx(xlsx)
    known = {c for c, _m in speed}
    missing = sorted({c for c in CELL_OF.values()} - known)
    check('all 12 cell ids present in the Speedup tab', missing, [])

    # 5. Every method must map to a name the CSV actually uses.
    csv_methods = {m for _c, m in speed}
    unknown = sorted(set(METHOD_TO_SPEEDUP.values()) - csv_methods)
    check('all method names present in the Speedup tab', unknown, [])

    if xlsx.exists():
        book = read_workbook(xlsx)
        audit = read_audit_tabs(xlsx)
        recs = build_records(book, speed, 'makovian', baseline='real_video',
                             audit=audit)
        # 6. DreamDojo genuinely has no Text Alignment upstream; assert the
        # partial flag catches it rather than the mean quietly changing meaning.
        dd = [r for r in recs if r['track'] == 'DreamDojo SV']
        check('DreamDojo rows all flagged partial',
              bool(dd) and all(r['partial'] for r in dd), True)
        check('DreamDojo missing metric is Text Alignment',
              sorted({m for r in dd for m in r['missing_metrics']}),
              ['Text Alignment'])
        others = [r for r in recs if r['track'] != 'DreamDojo SV']
        check('non-DreamDojo rows score all 3 metrics',
              all(r['n_scored'] == 3 for r in others), True)
        # Baseline fidelity: on real_video the figures must reproduce the
        # workbook's OWN NORM block. Checked to 3dp, not exactly: the NORM block
        # was computed from the unrounded medians (it implies Real Video =
        # 31.0174) while col C displays them rounded to 2dp (31.02), so
        # recomputing from the visible cells lands 1e-4 away by construction.
        vi = [r for r in recs if r['emb'] == 'BIMANUAL'
              and r['track'] == 'Dreamgen MV' and r['method'] == 'itm']
        check('itm/RealVideo reproduces workbook NORM col L (0.6851, 3dp)',
              round(vi[0]['ratios']['Visual Integrity'], 3), 0.685)
        dn = [r for r in recs if r['method'] == 'Dense']
        check('Dense is a plotted row on the real_video baseline', bool(dn), True)
        check('Dense speedup is exactly 1.00x',
              sorted({r['speedup'] for r in dn}), [1.0])

        # ---- 7. The speedup sheet must be clean and inside the dial -----------
        # Both are guaranteed by assets/scripts/fix_speedup_sheet.py; asserted
        # here so a future hand-edit to the workbook fails the self-test instead
        # of quietly producing clamped bars or a bogus "unparseable" warning.
        _sp2, repairs = read_speedup_xlsx(xlsx)
        check('Speedup tab emits 0 repair notes', list(repairs), [])
        sps = [r['speedup'] for r in recs if r['speedup']]
        check('all speedups inside the default dial [%.2f, %.2f]'
              % (Ctx.SPEED_LO, Ctx.SPEED_HI),
              [s for s in sps if not (Ctx.SPEED_LO <= s <= Ctx.SPEED_HI)], [])
        check('the 3 patched cells read back correctly',
              [round(next(r['speedup'] for r in recs if r['emb'] == e
                          and r['track'] == t and r['method'] == m), 2)
               for e, t, m in [('SINGLE-ARM', 'Ctrl-World MV', 'World-cache'),
                               ('SINGLE-ARM', 'Ctrl-World MV', 'Di-cache'),
                               ('BIMANUAL', 'Dreamgen MV', 'Faster-cache')]],
              [1.81, 1.74, 1.29])

        # ---- 8. The speedup dial itself --------------------------------------
        # These four properties are what every speedup radius on every figure
        # depends on; if the endpoints drift, every figure silently rescales.
        c = Ctx('/tmp/_nr_selftest_dial')
        check('r_speed(lo) == 0.0', round(c.r_speed(c.speed_lo), 12), 0.0)
        check('r_speed(hi) == 1.0', round(c.r_speed(c.speed_hi), 12), 1.0)
        check('r_speed_radar(lo) == 0.5',
              round(c.r_speed_radar(c.speed_lo), 12), 0.5)
        check('r_speed_radar(hi) == 1.0',
              round(c.r_speed_radar(c.speed_hi), 12), 1.0)
        probe = [c.r_speed(v) for v in (1.0, 1.2, 1.5, 1.9, 2.3)]
        check('r_speed strictly monotone',
              all(b > a for a, b in zip(probe, probe[1:])), True)
        check('clamping a sub-1.0x value is RECORDED, not silent',
              (c.r_speed(0.69, 'probe/slowdown') == 0.0
               and len(c.speed_clamped) == 1), True)
        check('reference geometry (no `who`) is NOT counted as a clamp',
              (c.r_speed(0.5), len(c.speed_clamped))[1], 1)

        # ---- 9. The KIVA tab must be the NEW detector's numbers --------------
        # make_rq1_dense_table.py documents the risk: an older workbook carries a
        # SUPERSEDED KIVA scale. Pinning the tab to results/reports/kiva_audit.csv
        # is what makes swapping such a workbook in fail loudly here instead of
        # silently republishing the wrong numbers.
        kiva_csv = REPO / 'results/reports/kiva_audit.csv'
        if kiva_csv.exists():
            n_cmp, mism = 0, []
            for row in csv.DictReader(kiva_csv.open()):
                if not row['track']:
                    continue                    # real_video reference rows
                emb = {v: k for k, v in EMB_CSV.items()}.get(row['embodiment'])
                for sp_name, col in (('makovian', 'makovian'),
                                     ('non_makovian', 'non_makovian')):
                    want = row[col]
                    if is_missing(want):
                        continue
                    got = audit['val'].get((emb, row['track'], row['method'],
                                            sp_name, 'KIVA'))
                    n_cmp += 1
                    if got is None or abs(got - float(want)) > 1e-9:
                        mism.append('%s/%s/%s/%s: csv=%s tab=%s'
                                    % (row['embodiment'], row['track'],
                                       row['method'], sp_name, want, got))
            check('KIVA tab == kiva_audit.csv (all %d entries)' % n_cmp, mism, [])
            # 108 = 60 track rows x 2 splits - 12 '---' cells (itm and sito were
            # never run on DreamDojo: 2 methods x 3 embodiments x 2 splits). The
            # count is asserted so a shrunk CSV cannot make the comparison above
            # vacuously pass by simply having fewer rows to disagree about.
            check('KIVA cross-check covered every comparable entry', n_cmp, 108)
            check('KIVA absent cells are the itm/sito DreamDojo rows',
                  120 - n_cmp, 12)
        else:
            print('  skip KIVA cross-check (%s not found)' % kiva_csv)

        # ---- 10. The RIGIS NORM mirror must agree with the recomputed ratio ---
        # The tab ships its own value/RealVideo block. We never plot it, so this
        # is the check that the tab's absolute and normalised halves have not
        # drifted apart (which would mean one of them is stale).
        nbad = []
        for k, nv in audit['norm'].items():
            emb, _tr, _me, sp_name, key = k
            rv = audit['real'].get((emb, sp_name, key))
            av = audit['val'].get(k)
            if rv and av is not None and round(av / rv, 3) != round(nv, 3):
                nbad.append('%s: computed %.4f vs tab %.4f' % ('/'.join(k),
                                                               av / rv, nv))
        check('RIGIS NORM block == computed ratio (3dp, %d rows)'
              % len(audit['norm']), nbad, [])
        check('RIGIS ships a NORM block', len(audit['norm']) > 0, True)
        check('KIVA ships NO NORM block (it is lower-better)',
              [k for k in audit['norm'] if k[4] == 'KIVA'], [])

        # ---- 11. Audit track coverage is disjoint ----------------------------
        # This is the whole reason RIGIS/KIVA get their own figure (c6) instead of
        # two more radar spokes. If it ever stops holding, c6's two-panel split
        # and c8's per-ring means both become wrong.
        rt = {k[1] for k in audit['val'] if k[4] == 'RIGIS'}
        kt = {k[1] for k in audit['val'] if k[4] == 'KIVA'}
        check('RIGIS is multi-view only', sorted(rt),
              ['Ctrl-World MV', 'Dreamgen MV'])
        check('KIVA is single-view only', sorted(kt),
              ['DreamDojo SV', 'Dreamgen SV'])
        check('RIGIS and KIVA track sets are DISJOINT', sorted(rt & kt), [])
        check('no record carries both audits',
              [1 for r in recs if 'RIGIS' in r['audits'] and 'KIVA' in r['audits']],
              [])
        check('audit_applies matches the declared track sets',
              all(r['audit_applies'][k] == (r['track'] in tr)
                  for r in recs for _s, k, _l, _u, _h, tr in AUDITS), True)

        # ---- 12. The two data hazards in the audit tabs -----------------------
        check('the genuine KIVA 0.0 survived the reader',
              audit['val'].get(('BIMANUAL', 'DreamDojo SV', 'Dense',
                                'non_makovian', 'KIVA')), 0.0)
        check('is_missing() does NOT swallow a real zero',
              (is_missing(0), is_missing(0.0)), (False, False))
        check('is_missing() catches every dash spelling in this workbook',
              all(is_missing(t) for t in ('--', '---', '----', '\u2013', '\u2014')),
              True)
        check('itm/sito are ABSENT on DreamDojo (not 0.0)',
              [k for k in audit['val']
               if k[1] == 'DreamDojo SV' and k[2] in ('itm', 'sito')], [])

        # ---- c12: mean-of-4 vs the 3-metric `retention` -----------------------
        # c12 shows a FOUR-metric mean while every other figure shows the
        # 3-metric `retention`. These checks pin that the extra metric really is
        # Actor Activity, that it is the closeness fold (so all four point the
        # same way), and that the two scalars stay distinguishable -- if a future
        # edit made nr_mean4 == retention the figure would silently be mislabelled.
        r4 = [r for r in recs if len(r['ratios']) == 4]
        check('some cells carry all 4 NR metrics', bool(r4), True)
        if r4:
            s = r4[0]
            man = mean([s['ratios'][METRIC_KEY[m[0]]] for m in METRICS])
            check('nr_mean4 == plain mean of the 4 ratios',
                  round(nr_mean4(s)[0], 9), round(man, 9))
            check('nr_mean4 counts 4 metrics and is NOT partial',
                  nr_mean4(s)[1:], (4, False))
            check('nr_mean4 differs from the 3-metric retention',
                  abs(nr_mean4(s)[0] - s['retention']) > 1e-9, True)
            # Actor Activity is the one metric retention drops; it must already be
            # folded to closeness (<=1.0) or averaging it in would be wrong.
            check('Actor Activity enters as closeness (<= 1.0)',
                  all(r['ratios']['Actor Activity'] <= 1.0 + 1e-9
                      for r in recs if 'Actor Activity' in r['ratios']), True)
        dd4 = [r for r in recs if r['track'] == 'DreamDojo SV']
        check('DreamDojo nr_mean4 is flagged partial (3 metrics, no TextAlign)',
              sorted({nr_mean4(r)[1:] for r in dd4}), [(3, True)])
        # c12's rows must partition the tracks exactly the way the audits do.
        check('c12 row tracks == the AUDIT track sets',
              (('Dreamgen SV', 'DreamDojo SV'), ('Dreamgen MV', 'Ctrl-World MV')),
              (AUDIT_BY_KEY['KIVA'][5], AUDIT_BY_KEY['RIGIS'][5]))
        # c12 derives its method list from the records. On the `dense` baseline
        # Dense is absent, and c12 crashed on exactly that case (it labelled its
        # legend at method index 0, which was then an empty slot). Both baselines
        # must yield a non-empty list whose first entry really has bars.
        for _bl in ('real_video', 'dense'):
            _r = build_records(book, speed, 'makovian', baseline=_bl, audit=audit)
            _m = [m for m in (['Dense'] + METHOD_ORDER)
                  if any(z['method'] == m for z in _r)]
            check('c12 method list is non-empty on baseline=%s' % _bl,
                  bool(_m) and _m[0] in {z['method'] for z in _r}, True)
            check('Dense present on baseline=%s only when expected' % _bl,
                  'Dense' in _m, _bl == 'real_video')
    else:
        print('  skip workbook checks (%s not found)' % xlsx)

    print()
    return '%d self-test failure(s)' % len(bad) if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xlsx', type=Path,
                    default=REPO / 'assets/BENCHMARK_NR_main (3).xlsx')
    ap.add_argument('--speedup-source', default='xlsx', choices=('xlsx', 'csv'),
                    help="where speedup comes from: 'xlsx' = the workbook's own "
                         "`Speedup` tab (default), 'csv' = "
                         "results/reports/speedup_table_full.csv")
    ap.add_argument('--speedup', type=Path,
                    default=REPO / 'results/reports/speedup_table_full.csv',
                    help='only used with --speedup-source csv')
    ap.add_argument('--baseline', default='real_video',
                    choices=('real_video', 'dense'),
                    help="denominator of every NR ratio: 'real_video' (default) "
                         "reproduces the workbook's own NORM block; 'dense' shows "
                         'what each method lost vs the dense run it replaces')
    ap.add_argument('--out', type=Path,
                    default=REPO / 'assets/paper_figures/nr_circular')
    ap.add_argument('--split', default='makovian', choices=sorted(SPLITS))
    ap.add_argument('--speed-range', nargs=2, type=float, metavar=('LO', 'HI'),
                    default=None,
                    help='speedup dial window, default %.2f %.2f. Fixed rather '
                         'than auto-fit so every split/baseline stays mutually '
                         'comparable; values outside it are clamped AND reported.'
                         % (Ctx.SPEED_LO, Ctx.SPEED_HI))
    ap.add_argument('--only', nargs='*', metavar='ID',
                    help='subset of %s' % ', '.join(sorted(FIGURES)))
    ap.add_argument('--selftest', action='store_true',
                    help='check the aggregation + cell mapping, then exit')
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest(args.xlsx, args.speedup))

    if not args.xlsx.exists():
        raise SystemExit('workbook not found: %s' % args.xlsx)
    book = read_workbook(args.xlsx)
    if args.speedup_source == 'xlsx':
        speed, repairs = read_speedup_xlsx(args.xlsx)
        src_desc = '%s!%s tab' % (args.xlsx.name, SPEEDUP_SHEET)
        for r in repairs:
            print('  NOTE  %s' % r)
    else:
        speed, repairs = read_speedup(args.speedup), []
        src_desc = str(args.speedup.relative_to(REPO))
    recs = build_records(book, speed, args.split, baseline=args.baseline,
                         audit=read_audit_tabs(args.xlsx))
    if not recs:
        raise SystemExit('no records built -- check the workbook layout')

    note = ('NR metrics + speedup: %s   |   NR ratios are value / %s  |  '
            'speedup source: %s' % (
                args.xlsx.name,
                'Real Video' if args.baseline == 'real_video' else 'Dense',
                src_desc))
    ctx = Ctx(args.out, baseline=args.baseline, source_note=note,
              speed_range=args.speed_range)
    note += '  |  speedup dial %.2fx..%.2fx' % (ctx.speed_lo, ctx.speed_hi)
    ctx.source_note = note
    meta = dump_data(ctx, recs, args.split, src_desc, repairs)
    print('%d method-cells, %d with speedup, %d without'
          % (meta['n_method_cells'], meta['n_timed'], meta['n_untimed']))
    sps = [r['speedup'] for r in recs if r['speedup']]
    if sps:
        out = [s for s in sps if not (ctx.speed_lo <= s <= ctx.speed_hi)]
        print('speedup %.2fx..%.2fx; dial %.2fx..%.2fx; %d value(s) outside'
              % (min(sps), max(sps), ctx.speed_lo, ctx.speed_hi, len(out)))
        if out:
            print('  NOTE  those values are CLAMPED on the figures -- widen '
                  '--speed-range if they are real')

    want = args.only or sorted(FIGURES)
    for fid in want:
        if fid not in FIGURES:
            raise SystemExit('unknown figure %r (have %s)'
                             % (fid, ', '.join(sorted(FIGURES))))
        title, fn = FIGURES[fid]
        print('%s  %s' % (fid, title))
        ctx.speed_clamped = []          # per-figure, so each stamp is accurate
        fn(ctx, recs, args.split)
        if ctx.speed_clamped:
            print('      %s' % ctx.clamp_note())


if __name__ == '__main__':
    main()
