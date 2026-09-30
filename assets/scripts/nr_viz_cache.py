"""Reduce an NR run's `results/*.json` to a small cache the figure/dashboard scripts read.

Why a cache exists at all
-------------------------
The run dir holds ~50k records and each one carries `actor_per_frame`
(~160 pairs x 22 fields) plus `continuity_per_triple`, so a full read is minutes of
PFSS I/O -- measured >6 min for a single pass. Every figure and the dashboard would
pay that again. This script pays it ONCE and writes:

  figures/cache/clips.jsonl.gz   one slim row per clip: strata + the 6 metric values
                                 ALREADY passed through `table_dense_nr.gated()`, the
                                 reason each None was gated out, and light diagnostics
  figures/cache/agg.json         every aggregate table, medians and bootstrap CIs
  figures/cache/perframe/*.json  `actor_per_frame` / `continuity_per_triple` for a
                                 small exemplar set (the per-frame figures)
  figures/cache/meta.json        coverage, protocol, gates, partial flag

Nothing here re-derives a statistic. The aggregates are produced by calling
`table_dense_nr`'s own `stratum_table` / `paired_table` / `accel_table` /
`accel_pooled_table` / `common_support_ranking` on the slim rows, with `T.gated`
swapped for a lookup of the value this script already computed with the real
`T.gated`. That swap is the only trick, and it is what lets the slim rows keep none
of the per-frame payload while every grouping, pairing, comparability gate, bootstrap
and `resolved_flag` stays byte-identical to the paper tables.

Usage:
  python assets/scripts/nr_viz_cache.py --run metrics/WorldScore/results/dense_nr_full
  python assets/scripts/nr_viz_cache.py --run <run> --incremental   # only new clips
  python assets/scripts/nr_viz_cache.py --run <run> --limit 2000    # smoke test
"""
import argparse
import collections
import concurrent.futures as cf
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_dense_nr as T  # noqa: E402

# The six metrics this suite visualises, in reading order. `status` is NOT cosmetic:
# it is the single place a figure or the dashboard learns which comparisons a column
# is allowed to carry, and every entry is derived from the policy dicts in
# table_dense_nr so a policy change here is impossible to forget.
#   VALID        -- may be read as a level and as a delta
#   CEILING      -- CEILING_FAILED: descriptive only, no model ranking; the accel
#                   delta IS claimable (CEILING_FAILURE_CANCELS_VS_DENSE)
#   PAIRED_ONLY  -- absolute medians are meaningless, only paired deltas
#   DIAGNOSTIC   -- NOT_REPORTED: valid but never a result
#   DROPPED      -- invalid metric, appendix autopsy only
VIZ_COLUMNS = [
    ('visual_integrity', 'Visual Integrity', 'CLIP-IQA+ p10 x100', True, 2),
    ('text_condition_alignment', 'Text Alignment', 'CLIPScore', True, 3),
    ('actor_motion_raw', 'Actor Activity', 'px / source frame', None, 2),
    ('actor_motion_active_rate', 'Active Rate', 'fraction of pairs >= 1 px', None, 3),
    ('active_motion_continuity', 'Active Motion Continuity', 'VFI gain x100', True, 2),
    ('actor_flicker_raw', 'Actor Flicker', 'RGB residual (0-1)', None, 4),
]
VIZ_KEYS = [k for k, _, _, _, _ in VIZ_COLUMNS]
# Kept OUT of `COLUMNS` upstream; this suite reports them with a badge instead of
# hiding them, because "why is this not a column" is itself a result worth a figure.
EXTRA_KEYS = [k for k in VIZ_KEYS if k not in {c[0] for c in T.COLUMNS}]


def status_of(key):
    if key in T.DROPPED:
        return 'DROPPED'
    if key in T.NOT_REPORTED:
        return 'DIAGNOSTIC'
    if key in T.PAIRED_ONLY:
        return 'PAIRED_ONLY'
    if key in T.CEILING_FAILED:
        return 'CEILING'
    return 'VALID'


# Light per-clip diagnostics copied into the slim row. Cheap, and they are what the
# validity / confound / anatomy figures need without the per-frame payload.
SCALARS = ['look_score', 'motion_continuity_global_ssim', 'detection_rate_raw',
           'clip_iqa+_raw', 'clip_iqa+_p10_raw', 'clip_score_raw', 'scene_integrity',
           'actor_motion_px_per_second', 'actor_tracked_pair_frac',
           'actor_motion_valid_pair_frac', 'actor_n_active_pairs', 'actor_n_arm_dets',
           'actor_seed_frame', 'qual_n_frames', 'phy_qual',
           'active_motion_continuity_weight_sum']
STRATA = ['id', 'model', 'embodiment', 'split', 'view', 'camera_mount', 'method',
          'pair_key', 'source_fps', 'effective_fps', 'source_interval', 'n_views',
          'layout', 'view_index', 'instruction', 'relative', 'actor_validity',
          'protocol_version', 'elapsed_seconds']


def gate_reason(row, key):
    """Why `T.gated(row, key)` returned None. Diagnostic only -- mirrors the checks in
    `T.gated` in the same order, so a new gate upstream shows up here as 'other'
    rather than as a wrong label."""
    if key in T.SUBJECT_KEYS:
        if row.get('actor_validity') != 'robot_detected':
            return f"actor_validity={row.get('actor_validity')}"
        possible = max(len(row.get('sampled_source_frames') or []) - 1, 1)
        tracked = row.get('actor_tracked_pair_frac')
        if tracked is None:
            tracked = len(row.get('actor_per_frame') or []) / possible
        if tracked < T.MIN_TRACKED_FRAC:
            return f'tracked_frac {tracked:.2f} < {T.MIN_TRACKED_FRAC}'
        if key == 'actor_flicker_raw':
            return 'stored value missing'
        sm = T.subject_metrics(row)
        if sm.get(key) is None:
            if sm.get('actor_motion_raw') is None:
                return 'too few valid motion pairs (arm dets / bg coverage)'
            if key == 'active_motion_continuity':
                return (f'weight sum < {T.MIN_WEIGHT_SUM} or active pairs < '
                        f'max({T.MIN_ACTIVE_PAIRS}, {T.MIN_ACTIVE_FRAC}*n)')
            return 'recompute returned None'
        if key == 'active_motion_continuity' and (
                row.get('active_motion_continuity_weight_sum') or 0.0) < T.MIN_WEIGHT_SUM:
            return f'stored weight sum < {T.MIN_WEIGHT_SUM}'
        return 'other'
    return 'not computed for this clip'


def reduce_record(path):
    """Full record on disk -> slim row. Runs in a worker process."""
    try:
        row = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if row.get('protocol_version') != T.PROTOCOL_REQUIRED:
        return {'_skipped_protocol': row.get('protocol_version')}
    out = {k: row.get(k) for k in STRATA}
    out['n_sampled'] = len(row.get('sampled_source_frames') or [])
    vals, reasons = {}, {}
    for key in VIZ_KEYS:
        v = T.gated(row, key)
        vals[key] = v
        if v is None:
            reasons[key] = gate_reason(row, key)
    out['gated'] = vals
    out['gate_reason'] = reasons
    out['raw'] = {k: row.get(k) for k in SCALARS}
    # 13 flicker variants: the E7 autopsy figure needs every one of them, and they are
    # the reason the raw column was kept in the record at all (auditable drop).
    out['flicker_variants'] = row.get('actor_flicker_variants') or {}
    pf = row.get('actor_per_frame') or []
    tri = row.get('continuity_per_triple') or {}
    out['n_pairs'] = len(pf)
    out['n_triples'] = len(tri.get('gain') or [])
    if pf:
        out['bg_cov_pct_med'] = T.median([p.get('bg_cov_pct') or 0.0 for p in pf])
        out['fg_cov_pct_med'] = T.median([p.get('fg_cov_pct') or 0.0 for p in pf])
        out['obj_cov_pct_med'] = T.median([p.get('obj_cov_pct') or 0.0 for p in pf])
    gains, copy_mse, vfi_mse = (tri.get('gain') or [], tri.get('copy_mse') or [],
                                tri.get('vfi_mse') or [])
    if gains:
        out['gain_med'] = T.median(gains)
        out['copy_mse_med'] = T.median(copy_mse) if copy_mse else None
        out['vfi_mse_med'] = T.median(vfi_mse) if vfi_mse else None
        # How much of the clip AMC actually scored: triples under the copy-error floor
        # carry weight 0, which is the usual reason a busy-looking clip has AMC=None.
        out['n_triples_below_floor'] = sum(1 for c in copy_mse if c < T.COPY_ERR_FLOOR)
    return out


def scan(run, limit=None, workers=None, known=frozenset(), progress_every=2000):
    paths = [p for p in sorted((run / 'results').glob('*.json'))
             if p.stem not in known]
    if limit:
        paths = paths[:limit]
    rows, skipped, unreadable = [], collections.Counter(), 0
    workers = workers or min(32, (os.cpu_count() or 8))
    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for i, out in enumerate(ex.map(reduce_record, [str(p) for p in paths],
                                       chunksize=32), 1):
            if out is None:
                unreadable += 1
            elif '_skipped_protocol' in out:
                skipped[out['_skipped_protocol']] += 1
            else:
                rows.append(out)
            if i % progress_every == 0:
                rate = i / max(time.time() - t0, 1e-9)
                print(f'  scanned {i}/{len(paths)}  ({rate:.0f}/s)', flush=True)
    return rows, skipped, unreadable, len(paths)


def rehydrate(slim):
    """Slim row -> a row the `table_dense_nr` aggregators accept.

    They touch exactly three things this row must answer for: the strata keys they
    group on, `sampled_source_frames` (LENGTH only, in `comparable_pair`), and
    `gated(row, key)`. `range(n)` answers the length in O(1) memory -- storing the real
    list for 50k clips would be ~300 MB of integers nobody reads.
    """
    row = dict(slim)
    row['sampled_source_frames'] = range(slim.get('n_sampled') or 0)
    return row


def patched_gated(row, key):
    return (row.get('gated') or {}).get(key)


def spearman(xs, ys):
    """Rank correlation, self-contained (the cache must run in any venv)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 8:
        return None, len(pairs)

    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return None, n
    return num / (dx * dy), n


def base_model(model):
    return model[3:] if model.startswith('gt_') else model


def grid_tables(rows, with_ci=False):
    """The paper-grid cell (embodiment, model, split, method) for ALL SIX metrics.

    `table_paper_grid.py` builds the same cells but only over `T.COLUMNS` (4 metrics).
    The two diagnostic columns are included here so the figures can show them greyed
    out with their badge instead of leaving a reader wondering where they went.
    Absolute AND delta-vs-dense, same clip, same `comparable_pair` gate.

    `with_ci` is off by default because this table feeds a heatmap of medians: at full
    coverage it is ~2000 cells, and a B=10000 percentile bootstrap on each is ~10
    minutes for intervals no figure draws. The forest figures (T3/T3b), which DO draw
    intervals, always get the real bootstrap.
    """
    absolute = collections.defaultdict(list)
    deltas = collections.defaultdict(list)
    dense_by_clip = {}
    for r in rows:
        if r.get('method') == 'dense':
            dense_by_clip[(r['pair_key'], r['model'])] = r
    for r in rows:
        method = r.get('method') or 'gt'
        cell = (r['embodiment'], base_model(r['model']), r['split'])
        for key in VIZ_KEYS:
            v = patched_gated(r, key)
            if v is not None:
                absolute[cell + (method, key)].append(v)
        if method in ('gt', 'dense'):
            continue
        base = dense_by_clip.get((r['pair_key'], r['model']))
        if base is None or not T.comparable_pair(r, base):
            continue
        for key in VIZ_KEYS:
            a, b = patched_gated(base, key), patched_gated(r, key)
            if a is not None and b is not None:
                deltas[cell + (method, key)].append(b - a)

    def emit(table):
        out = []
        for (emb, model, split, method, key), vals in sorted(table.items()):
            lo, hi = T.boot_ci(vals) if with_ci else (None, None)
            out.append({'embodiment': emb, 'model': model, 'split': split,
                        'method': method, 'metric': key, 'n': len(vals),
                        'median': T.median(vals), 'ci_lo': lo, 'ci_hi': hi,
                        'claim_ok': len(vals) >= T.MIN_PAIRS_FOR_CLAIM})
        return out

    return emit(absolute), emit(deltas)


def coverage_waterfall(rows, n_results, n_manifest, skipped, unreadable):
    """Per metric: how many clips survive each stage. This is the answer to "why does
    every column have a different n", which is otherwise invisible in a table."""
    stages = {'manifest_clips': n_manifest, 'result_files': n_results,
              'wrong_protocol': sum(skipped.values()), 'unreadable': unreadable,
              'in_protocol': len(rows)}
    per_metric = {}
    for key in VIZ_KEYS:
        reasons = collections.Counter()
        kept = 0
        for r in rows:
            if patched_gated(r, key) is not None:
                kept += 1
            else:
                reasons[(r.get('gate_reason') or {}).get(key) or 'missing'] += 1
        per_metric[key] = {'in_protocol': len(rows), 'gate_passed': kept,
                           'gated_out': len(rows) - kept,
                           'reasons': dict(reasons.most_common(12)),
                           'status': status_of(key)}
    return {'stages': stages, 'per_metric': per_metric}


def redundancy(rows):
    """Spearman between every pair of the 6 metrics, pooled and per mount.

    This is the figure that carries the documented reason Active Rate is NOT_REPORTED
    (rho ~ +0.875 pooled, +0.946 on wrist vs Actor Activity) and the reason Actor
    Flicker is motion-confounded -- both re-measured here on the current wave rather
    than quoted.
    """
    def matrix(subset):
        m = {}
        for a in VIZ_KEYS:
            for b in VIZ_KEYS:
                if a >= b:
                    continue
                rho, n = spearman([patched_gated(r, a) for r in subset],
                                  [patched_gated(r, b) for r in subset])
                m[f'{a}|{b}'] = {'rho': rho, 'n': n}
        return m

    out = {'pooled': matrix(rows)}
    by_mount = collections.defaultdict(list)
    for r in rows:
        by_mount[r.get('camera_mount')].append(r)
    out['by_mount'] = {mount: matrix(sub) for mount, sub in sorted(by_mount.items())
                       if mount and len(sub) >= 20}
    # Flicker variants vs motion: the E7 pass criterion was |rho| < 0.25 on EVERY
    # mount, for whichever variant might have rescued the column. Measured per variant.
    variants = sorted({v for r in rows for v in (r.get('flicker_variants') or {})})
    fv = {}
    for mount, sub in sorted(by_mount.items()):
        if not mount or len(sub) < 20:
            continue
        motion = [patched_gated(r, 'actor_motion_raw') for r in sub]
        fv[mount] = {}
        for var in variants:
            vals = [(r.get('flicker_variants') or {}).get(var) for r in sub]
            # A variant is only readable on clips whose motion column itself is valid.
            rho, n = spearman(motion, vals)
            fv[mount][var] = {'rho': rho, 'n': n}
    out['flicker_variants_vs_motion'] = {'bar': 0.25, 'by_mount': fv,
                                         'variants': variants}
    return out


# Methods worth an anatomy trace, in preference order: two blur-like (itm, sito), two
# that produced the documented over-smoothing AMC flag (fastercache, pisa), one cache.
EXEMPLAR_METHODS = ['itm', 'sito', 'fastercache', 'pisa', 'dicache']


def pick_exemplars(rows, per_cell=2, max_files=140):
    """Choose a few (dense + accel) clip groups to keep per-frame data for.

    Selected AFTER the scan, from the slim rows, so the expensive pass stays a single
    streaming reduction: the ~100 chosen records are then re-read individually, which
    costs seconds. Deterministic (sorted by id) so a re-run reproduces the figures.
    """
    dense = {}
    for r in rows:
        if r.get('method') == 'dense':
            dense[(r['pair_key'], r['model'])] = r
    accel = collections.defaultdict(list)
    for r in rows:
        if r.get('method') in (None, 'gt', 'dense'):
            continue
        accel[(r['pair_key'], r['model'])].append(r)

    by_cell = collections.defaultdict(list)
    for key, base in sorted(dense.items(), key=lambda kv: kv[1]['id']):
        if not base.get('n_pairs') or not base.get('n_triples'):
            continue
        if patched_gated(base, 'actor_motion_raw') is None:
            continue
        mates = [r for r in accel.get(key, [])
                 if r['method'] in EXEMPLAR_METHODS and T.comparable_pair(r, base)
                 and r.get('n_pairs')]
        if not mates:
            continue
        mates.sort(key=lambda r: EXEMPLAR_METHODS.index(r['method']))
        by_cell[(base['embodiment'], base['camera_mount'])].append((base, mates[:2]))

    chosen, groups = [], []
    for cell, cands in sorted(by_cell.items()):
        for base, mates in cands[:per_cell]:
            ids = [base['id']] + [m['id'] for m in mates]
            if len(chosen) + len(ids) > max_files:
                break
            chosen += ids
            groups.append({'cell': list(cell), 'pair_key': base['pair_key'],
                           'model': base['model'], 'split': base['split'],
                           'view': base['view'], 'instruction': base.get('instruction'),
                           'dense_id': base['id'],
                           'accel': [{'id': m['id'], 'method': m['method']}
                                     for m in mates]})
    return groups, chosen


PERFRAME_FIELDS = ['pair', 'fg_flow_med', 'fg_flow_p95', 'fg_flow_ego_p95',
                   'fg_flow_ego_med', 'arm_flow_med', 'obj_flow_med', 'bg_flow_med',
                   'frame_flow_med', 'fg_cov_pct', 'obj_cov_pct', 'arm_cov_pct',
                   'bg_cov_pct', 'bg_ego_dx', 'bg_ego_dy', 'actor_flicker_raw']


def write_perframe(run, ids, out_dir):
    """Re-read the chosen records and keep only the per-frame series the figures plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cid in ids:
        src = run / 'results' / f'{cid}.json'
        if not src.exists():
            continue
        try:
            row = json.loads(src.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tri = row.get('continuity_per_triple') or {}
        payload = {
            'id': cid, 'method': row.get('method'), 'model': row.get('model'),
            'embodiment': row.get('embodiment'), 'camera_mount': row.get('camera_mount'),
            'split': row.get('split'), 'view': row.get('view'),
            'source_interval': row.get('source_interval'),
            'instruction': row.get('instruction'), 'relative': row.get('relative'),
            'per_frame': [{k: p.get(k) for k in PERFRAME_FIELDS}
                          for p in (row.get('actor_per_frame') or [])],
            'continuity': {k: tri.get(k) for k in ('gain', 'copy_mse', 'vfi_mse')},
            'gated': {k: T.gated(row, k) for k in VIZ_KEYS},
        }
        (out_dir / f'{cid}.json').write_text(json.dumps(payload))
        written += 1
    return written


def load_existing(path):
    if not path.exists():
        return []
    with gzip.open(path, 'rt') as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None,
                    help='cache dir (default <run>/figures/cache)')
    ap.add_argument('--limit', type=int, default=None, help='smoke test on N records')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--incremental', action='store_true',
                    help='keep clips already in the cache and only read new result files')
    ap.add_argument('--skip-perframe', action='store_true')
    ap.add_argument('--boot', type=int, default=None,
                    help='bootstrap resamples (default %d, the paper value); lower only '
                         'for iteration' % T.B)
    args = ap.parse_args()
    if args.boot:
        T.B = args.boot
    out = args.out or args.run / 'figures' / 'cache'
    out.mkdir(parents=True, exist_ok=True)
    clips_path = out / 'clips.jsonl.gz'

    old = load_existing(clips_path) if args.incremental else []
    known = {r['id'] for r in old}
    if old:
        print(f'incremental: {len(old)} clips already cached')
    print(f'scanning {args.run / "results"} ...', flush=True)
    t0 = time.time()
    new, skipped, unreadable, n_files = scan(args.run, args.limit, args.workers, known)
    rows = old + new
    print(f'scanned {n_files} new files in {time.time() - t0:.0f}s -> '
          f'{len(rows)} clips ({len(new)} new, {sum(skipped.values())} other-protocol, '
          f'{unreadable} unreadable)', flush=True)
    if not rows:
        raise SystemExit('no records in this protocol; nothing to cache')

    with gzip.open(clips_path, 'wt') as fh:
        for r in sorted(rows, key=lambda r: r['id']):
            fh.write(json.dumps(r) + '\n')
    print(f'wrote {clips_path}')

    hydrated = [rehydrate(r) for r in rows]
    # The one swap that lets the real aggregators run on slim rows (see module docstring).
    real_gated, real_columns = T.gated, T.COLUMNS
    T.gated = patched_gated
    T.COLUMNS = [(k, label, higher, digits)
                 for k, label, _unit, higher, digits in VIZ_COLUMNS]
    try:
        print('aggregating (bootstrap B=%d) ...' % T.B, flush=True)
        t1 = time.time()
        t1_desc = T.stratum_table(hydrated)
        print(f'  T1 descriptive: {len(t1_desc)} cells  ({time.time() - t1:.0f}s)',
              flush=True)
        t2_paired = T.paired_table(hydrated)
        print(f'  T2 pred-vs-GT: {len(t2_paired)} cells', flush=True)
        t3_cell, t3_unpaired, t3_incomp = T.accel_table(hydrated)
        print(f'  T3 accel per-cell: {len(t3_cell)} cells', flush=True)
        t3b_pooled, _, _ = T.accel_pooled_table(hydrated)
        print(f'  T3b accel pooled: {len(t3b_pooled)} methods', flush=True)
        common = T.common_support_ranking(hydrated)
        grid_abs, grid_delta = grid_tables(hydrated)
        print(f'  grid: {len(grid_abs)} absolute / {len(grid_delta)} delta cells',
              flush=True)
        cov = coverage_waterfall(hydrated, n_files + len(known), 0, skipped, unreadable)
        red = redundancy(hydrated)
        groups, exemplar_ids = pick_exemplars(hydrated)
    finally:
        T.gated, T.COLUMNS = real_gated, real_columns

    manifest_n = 0
    mpath = args.run / 'manifest.json'
    if mpath.exists():
        try:
            manifest_n = len(json.loads(mpath.read_text()))
        except (json.JSONDecodeError, OSError):
            manifest_n = 0
    cov['stages']['manifest_clips'] = manifest_n

    agg = {
        'columns': [{'key': k, 'label': label, 'unit': unit, 'higher': higher,
                     'digits': digits, 'status': status_of(k)}
                    for k, label, unit, higher, digits in VIZ_COLUMNS],
        'policy': {
            'ceiling_failed': T.CEILING_FAILED,
            'dropped': T.DROPPED,
            'not_reported': T.NOT_REPORTED,
            'paired_only': T.PAIRED_ONLY,
            'not_cross_embodiment': T.NOT_CROSS_EMBODIMENT,
            'accel_caveats': T.ACCEL_CAVEATS,
            'no_improvement_vs_dense': T.NO_IMPROVEMENT_VS_DENSE,
            'confounded_mounts': T.CONFOUNDED_MOUNTS,
            'ceiling_failure_cancels_vs_dense': sorted(T.CEILING_FAILURE_CANCELS_VS_DENSE),
            'min_pairs_for_claim': T.MIN_PAIRS_FOR_CLAIM,
            'alpha': T.ALPHA,
            'bootstrap_resamples': T.B,
            'gates': {'min_tracked_frac': T.MIN_TRACKED_FRAC,
                      'min_weight_sum': T.MIN_WEIGHT_SUM,
                      'min_active_pairs': T.MIN_ACTIVE_PAIRS,
                      'min_active_frac': T.MIN_ACTIVE_FRAC,
                      'min_bg_cov_pct': T.MIN_BG_COV_PCT,
                      'active_px': T.ACTIVE_PX,
                      'copy_err_floor': T.COPY_ERR_FLOOR,
                      'max_length_mismatch': T.MAX_LENGTH_MISMATCH},
        },
        't1_descriptive': t1_desc,
        't2_paired_gt': t2_paired,
        't3_accel_cell': t3_cell,
        't3b_accel_pooled': t3b_pooled,
        't3_unpaired': dict(t3_unpaired),
        't3_incomparable': dict(t3_incomp),
        'common_support': {col: {'cells': v['cells'], 'methods': v['methods'],
                                 'ranked': [[m, d, n] for m, d, n in v['ranked']],
                                 'moved': v['moved'],
                                 'pooled_order': v['pooled_order']}
                           for col, v in common.items()},
        'grid_absolute': grid_abs,
        'grid_delta_vs_dense': grid_delta,
        'coverage': cov,
        'redundancy': red,
        'exemplars': groups,
    }
    (out / 'agg.json').write_text(json.dumps(agg, indent=1))
    print(f'wrote {out / "agg.json"}')

    n_perframe = 0
    if not args.skip_perframe and exemplar_ids:
        n_perframe = write_perframe(args.run, exemplar_ids, out / 'perframe')
        print(f'wrote {n_perframe} per-frame files for {len(groups)} exemplar groups')

    errors = len(list((args.run / 'errors').glob('*.json'))) \
        if (args.run / 'errors').exists() else 0
    meta = {
        'run': str(args.run),
        'generated_by': 'assets/scripts/nr_viz_cache.py',
        'generated_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'protocol': T.PROTOCOL_REQUIRED,
        'manifest_clips': manifest_n,
        'scored_clips': len(rows),
        'errors': errors,
        'other_protocol': dict(skipped),
        'unreadable': unreadable,
        'partial': bool(manifest_n and len(rows) < manifest_n),
        'coverage_frac': round(len(rows) / manifest_n, 4) if manifest_n else None,
        'n_exemplar_groups': len(groups),
        'n_perframe_files': n_perframe,
        'methods': sorted({r.get('method') for r in rows if r.get('method')}),
        'models': sorted({r['model'] for r in rows}),
        'embodiments': sorted({r['embodiment'] for r in rows}),
        'mounts': sorted({r['camera_mount'] for r in rows if r.get('camera_mount')}),
        'limit_applied': args.limit,
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=1))
    print(f'wrote {out / "meta.json"}')
    print(f"\ncoverage {meta['scored_clips']}/{meta['manifest_clips']} "
          f"({'PARTIAL' if meta['partial'] else 'COMPLETE'})")
    for key in VIZ_KEYS:
        pm = cov['per_metric'][key]
        print(f"  {key:28} {pm['gate_passed']:6d} valid  [{pm['status']}]")


if __name__ == '__main__':
    main()
