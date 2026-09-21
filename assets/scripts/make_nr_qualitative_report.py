"""Qualitative HTML report for the PhyWorld-NR columns.

For every DIRECTIONAL column (Visual Integrity, Text Alignment, AMC) it shows the
worst and best real clips side by side; for the DESCRIPTIVE column (Actor Activity)
it shows a low->high ladder per camera mount, because "high" there is not "good".

The clip shown is the **analysis clip** -- re-prepared with the exact same
`prepare()` the scorer used (same view crop, same 5 fps sampling, same 256x192 grid,
conditioning frame dropped). Showing the raw mp4 instead would show a different
video from the one that produced the number (wrong view on multiview, wrong fps).

Values are read through `table_dense_nr.gated()`, so they are the same numbers the
paper tables use, under the current gates.

Usage:
    models/dreamgen/.venv/bin/python assets/scripts/make_nr_qualitative_report.py \
        --run metrics/WorldScore/results/dense_nr_full --n 3
"""
import argparse
import collections
import importlib.util
import json
import random
import shutil
import statistics as st
import subprocess
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load('table_dense_nr', 'assets/scripts/table_dense_nr.py')
NR = _load('run_dense_nr', 'scripts/run_dense_nr.py')

# Directional columns get a poor/good contrast. Actor Activity deliberately does NOT:
# it is descriptive, so the report shows a magnitude ladder and says so.

# Smallest number of pred/real pairs a configuration needs before its median is read as
# a result. Matches the threshold used to read the wave-level audit; below it a single
# odd episode moves the median.
MIN_CELL_PAIRS = 8

DIRECTIONAL = [
    ('visual_integrity', 'Visual Integrity',
     'Is the picture clean? Blur, blocking and noise push it down. It says nothing '
     'about whether the robot does the right thing.',
     'CLIP-IQA+ p10 x100'),
    ('text_condition_alignment', 'Text Alignment',
     'Does the video show the task that was asked for? Compare each clip against '
     'its own prompt, printed under the video.',
     'CLIPScore [0,100]'),
    ('active_motion_continuity', 'Active Motion Continuity',
     'While the robot is moving, does the motion continue sensibly from frame to '
     'frame, or does something jump / pop / teleport? Clips with no motion are not '
     'scored at all (N/A) instead of scoring perfectly.',
     'activity-weighted VFI gain x100'),
]
DESCRIPTIVE = ('actor_motion_raw', 'Actor Activity',
               'How much the robot and the objects move, in pixels per source frame. '
               'This is NOT a score: more motion is not better, because the real '
               'video defines the right amount. It is reported so the other columns '
               'can be read next to it.',
               'px / source frame')
ALL_KEYS = [k for k, *_ in DIRECTIONAL] + [DESCRIPTIVE[0]]
# The analysis grid is 5 fps. Played at native rate a clip lasts a few seconds and
# nothing is judgeable; 3 fps is slow enough to see a pop or a smear.
PLAY_FPS = 3


def read_records(run, limit, seed):
    """Load scored records once and split into predictions and their GT anchors.

    Returns `(pred_rows, gt_rows)`. Both come out of the SAME pass and the SAME sample,
    which matters: the reality check pairs a prediction with its own GT, so sampling the
    two independently would silently drop pairs and move the audit numbers.
    """
    files = sorted((run / 'results').glob('*.json'))
    if limit and limit < len(files):
        random.Random(seed).shuffle(files)
        files = files[:limit]
    pred, gt = [], []
    for p in files:
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        # Records from an older protocol are not recomputable under the current gates
        # (v3 stored no `frame_flow_med`), exactly as `table_dense_nr.load_records`
        # filters them -- keep the same rule so the report cannot show a clip whose
        # number came from a different protocol than the tables quote.
        if r.get('protocol_version') != T.PROTOCOL_REQUIRED:
            continue
        if not Path(r.get('path', '')).exists():
            continue
        vals = {}
        for k in ALL_KEYS:
            try:
                vals[k] = T.gated(r, k)
            except Exception:
                vals[k] = None
        r['_gated'] = vals
        # A GT clip is the plausibility anchor, not a model output -- a gallery of
        # "what a good clip looks like" filled with real video teaches nothing about
        # the models, so galleries use predictions only and the reality check uses GT.
        (gt if str(r.get('model', '')).startswith('gt') else pred).append(r)
    return pred, gt


def pick(rows, key, n, lowest):
    """n clips at one end of `key`, at most one per episode and per config."""
    have = [r for r in rows if r['_gated'].get(key) is not None]
    have.sort(key=lambda r: r['_gated'][key], reverse=not lowest)
    out, seen_ep, seen_cfg = [], set(), set()
    for r in have:
        ep = r.get('episode_key') or r.get('relative')
        cfg = (r.get('model'), r.get('embodiment'), r.get('camera_mount'))
        if ep in seen_ep:
            continue
        # Spread across configs so the gallery is not five clips of one checkpoint.
        if cfg in seen_cfg and len(out) < n:
            continue
        out.append(r)
        seen_ep.add(ep)
        seen_cfg.add(cfg)
        if len(out) == n:
            break
    return out


def ladder(rows, key, mount, n):
    """Evenly spaced quantiles of `key` within one camera mount: a magnitude ladder."""
    have = [r for r in rows
            if r['camera_mount'] == mount and r['_gated'].get(key) is not None]
    have.sort(key=lambda r: r['_gated'][key])
    if len(have) < n:
        return have
    picks, seen_ep = [], set()
    for q in [i / (n - 1) for i in range(n)]:
        idx = min(int(q * (len(have) - 1)), len(have) - 1)
        for off in range(len(have)):
            for j in (idx + off, idx - off):
                if 0 <= j < len(have):
                    ep = have[j].get('episode_key') or have[j].get('relative')
                    if ep not in seen_ep:
                        picks.append(have[j])
                        seen_ep.add(ep)
                        break
            else:
                continue
            break
    return picks


def make_clip(row, dest_dir, tag):
    """Re-create the exact analysis clip that was scored, as browser-playable H.264.

    `prepare()` writes `mp4v` (MPEG-4 Part 2), which Chrome and Safari refuse to
    decode -- the page would show black boxes with working controls. So the analysis
    clip is re-encoded to H.264/yuv420p. The transcode is lossless in content: same
    frames, same order, same count; only the codec changes.

    Playback is slowed to `PLAY_FPS` because the analysis grid is 5 fps: at native
    rate a 46-frame clip is over in 9 seconds and the motion is impossible to judge.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = f'{tag}_{row["id"][:10]}.mp4'
    out = dest_dir / name
    if out.exists():
        return f'clips/{name}'
    raw = dest_dir / f'.raw_{name}'
    arr, _ = NR.prepare(row, raw)
    n_scored = len(arr)
    # `-r` must be an INPUT option: as an output option it RESAMPLES to that rate and
    # silently DROPS frames (measured: a 46-frame analysis clip came out with 30), so
    # the video would no longer be the thing that was scored. As an input option it
    # only rewrites timestamps -- every scored frame is kept, just shown slower.
    cmd = ['ffmpeg', '-y', '-loglevel', 'error',
           '-r', str(PLAY_FPS), '-i', str(raw),
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
           '-pix_fmt', 'yuv420p', '-an', '-vsync', 'passthrough',
           '-movflags', '+faststart', str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    raw.unlink(missing_ok=True)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f'ffmpeg failed for {row["id"]}: {res.stderr.strip()[:300]}')
    n_out = count_frames(out)
    if n_out != n_scored:
        raise RuntimeError(
            f'{row["id"]}: transcode changed the clip ({n_scored} scored frames -> '
            f'{n_out} shown). The report must show exactly what was scored.')
    return f'clips/{name}'


def count_frames(path):
    res = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(path)],
        capture_output=True, text=True)
    try:
        return int(res.stdout.strip().rstrip(','))
    except ValueError:
        return -1


def fmt(v, digits=2):
    return '&mdash;' if v is None else f'{v:.{digits}f}'


def card(row, rel, key, badge, badge_cls):
    g = row['_gated']
    prompt = row.get('text_condition_prompt') or ''
    others = ''.join(
        f'<tr><td>{escape(lab)}</td><td class="num">{fmt(g.get(k), 3 if k=="text_condition_alignment" else 2)}</td></tr>'
        for k, lab in [('visual_integrity', 'Visual Integrity'),
                       ('text_condition_alignment', 'Text Alignment'),
                       ('actor_motion_raw', 'Actor Activity (px/frame)'),
                       ('active_motion_continuity', 'AMC')]
        if k != key)
    val = fmt(g.get(key), 3 if key == 'text_condition_alignment' else 2)
    return f'''
    <figure class="card">
      <span class="badge {badge_cls}">{escape(badge)} &nbsp;{val}</span>
      <video src="{rel}" controls loop muted playsinline preload="metadata"></video>
      <figcaption>
        <div class="meta"><b>{escape(str(row.get("model")))}</b> &middot;
          {escape(str(row.get("embodiment")))} &middot;
          mount <b>{escape(str(row.get("camera_mount")))}</b> &middot;
          view {escape(str(row.get("view")))}</div>
        <div class="path">{escape(str(row.get("relative")))}</div>
        {f'<div class="prompt">prompt: &ldquo;{escape(prompt)}&rdquo;</div>' if prompt else ''}
        <table class="others">{others}</table>
      </figcaption>
    </figure>'''


def paired_gt(pred_rows, gt_rows, key, n, gt_wins):
    """Pred clips paired with their own co-located GT, for the GT-is-ceiling check.

    Pairing rule copied from `table_dense_nr.paired_table`: only `method == 'dense'`
    predictions, same `pair_key`, and the anchor is `gt_<model>` (or plain `gt` for the
    ctrlworld family). `comparable_pair` is enforced, so a pair is only shown when both
    clips are the same measurement (same source/effective fps, frame counts within 10%)
    -- otherwise the delta would be a render difference, not a metric result.

    The `dense` restriction is what makes this page agree with the paper: an
    acceleration clip carries the SAME `model` as its dense counterpart, so admitting
    every method would enter one episode a dozen times and weight each configuration by
    how many acceleration variants it happens to have been run with (7991 pairs instead
    of 1187, and Visual Integrity flipping to a spurious 88% "real wins").
    """
    by_pk = {}
    for r in gt_rows:
        by_pk.setdefault(r['pair_key'], {})[r['model']] = r
    # Same dedup as `paired_table`, which keys records by (pair_key, method, model): if a
    # rescore left two records for one episode+view, counting both would enter the same
    # measurement twice. Without this the pair counts drift off the paper's by ~1%.
    uniq = {}
    for pred in pred_rows:
        if pred.get('method') == 'dense':
            uniq[(pred['pair_key'], pred['model'])] = pred
    out = []
    for pred in uniq.values():
        if pred['_gated'].get(key) is None:
            continue
        anchors = by_pk.get(pred['pair_key']) or {}
        model = pred['model']
        gt = anchors.get(f'gt_{model}') or (
            anchors.get('gt') if model.startswith('ctrlworld') else None)
        if gt is None or gt['_gated'].get(key) is None:
            continue
        if not T.comparable_pair(pred, gt):
            continue
        out.append((gt['_gated'][key] - pred['_gated'][key], pred, gt))
    # `gt_wins=True` -> biggest GT advantage first (the metric behaving correctly);
    # False -> the violations, i.e. where prediction outscores real video.
    out.sort(key=lambda x: -x[0] if gt_wins else x[0])
    picked, seen_ep = [], set()
    for d, pred, gt in out:
        ep = pred.get('episode_key') or pred.get('relative')
        if ep in seen_ep:
            continue
        picked.append((d, pred, gt))
        seen_ep.add(ep)
        if len(picked) == n:
            break
    return picked, out


def pair_card(delta, pred, gt, rel_pred, rel_gt, key):
    digits = 3 if key == 'text_condition_alignment' else 2
    sign = 'good' if delta > 0 else 'bad'
    verdict = 'GT higher' if delta > 0 else 'PRED higher'
    return f'''
    <figure class="card pair">
      <span class="badge {sign}">{verdict} &nbsp;{delta:+.{digits}f}</span>
      <div class="twin">
        <div><video src="{rel_gt}" controls loop muted playsinline preload="metadata"></video>
          <span class="tag real">REAL &nbsp;{fmt(gt["_gated"][key], digits)}</span></div>
        <div><video src="{rel_pred}" controls loop muted playsinline preload="metadata"></video>
          <span class="tag gen">GENERATED &nbsp;{fmt(pred["_gated"][key], digits)}</span></div>
      </div>
      <figcaption>
        <div class="meta"><b>{escape(str(pred.get("model")))}</b> &middot;
          {escape(str(pred.get("embodiment")))} &middot;
          mount <b>{escape(str(pred.get("camera_mount")))}</b> &middot;
          view {escape(str(pred.get("view")))}</div>
        <div class="path">{escape(str(pred.get("episode_key") or pred.get("relative")))}</div>
      </figcaption>
    </figure>'''


CSS = '''
:root{--bg:#0f1115;--card:#181b22;--line:#272b35;--txt:#e6e8ee;--dim:#98a0b0;
      --bad:#e5484d;--good:#30a46c;--neutral:#5b6cff}
*{box-sizing:border-box}
body{margin:0;padding:32px 28px 80px;background:var(--bg);color:var(--txt);
     font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
h1{font-size:26px;margin:0 0 6px} h2{font-size:20px;margin:44px 0 4px}
h3{font-size:14px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);
   margin:26px 0 10px;font-weight:600}
.sub{color:var(--dim);max-width:70ch;margin:0 0 10px}
.unit{display:inline-block;font:12px ui-monospace,Menlo,monospace;color:var(--dim);
      border:1px solid var(--line);border-radius:4px;padding:1px 6px}
.note{background:#1b1f28;border-left:3px solid var(--neutral);padding:10px 14px;
      border-radius:0 6px 6px 0;color:#c3c9d6;max-width:80ch;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}
.card{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;
      overflow:hidden;position:relative}
.card video{width:100%;display:block;background:#000;aspect-ratio:4/3}
.badge{position:absolute;top:8px;left:8px;z-index:2;font:600 11px ui-monospace,Menlo,monospace;
       padding:3px 8px;border-radius:20px;color:#fff;background:#333}
.badge.bad{background:var(--bad)} .badge.good{background:var(--good)}
.badge.mid{background:var(--neutral)}
figcaption{padding:10px 12px 12px;font-size:12px}
.meta{color:var(--txt)} .path{color:var(--dim);font:11px ui-monospace,Menlo,monospace;
      word-break:break-all;margin:3px 0 0}
.prompt{color:#b9c0cf;margin:6px 0 0;font-style:italic}
table.others{width:100%;margin:8px 0 0;border-collapse:collapse;color:var(--dim)}
table.others td{padding:2px 0;border-top:1px solid var(--line);font-size:11px}
table.others td.num{text-align:right;font-family:ui-monospace,Menlo,monospace;color:var(--txt)}
footer{margin-top:60px;color:var(--dim);font-size:12px;border-top:1px solid var(--line);
       padding-top:14px}
code{font-family:ui-monospace,Menlo,monospace;background:#1b1f28;padding:1px 5px;border-radius:4px}
.card.pair{grid-column:span 1}
.twin{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#000}
.twin>div{position:relative}
.twin video{width:100%;display:block;background:#000;aspect-ratio:4/3}
.tag{position:absolute;bottom:4px;left:4px;font:600 9px ui-monospace,Menlo,monospace;
     padding:2px 5px;border-radius:3px;color:#fff;background:#000a}
.tag.real{background:#1f6f4aee} .tag.gen{background:#4a4f8fee}
table.audit{border-collapse:collapse;margin:14px 0;font-size:13px}
table.audit th,table.audit td{border:1px solid var(--line);padding:6px 11px;text-align:left}
table.audit th{background:#1b1f28;color:var(--dim);font-weight:600}
table.audit td.num{text-align:right;font-family:ui-monospace,Menlo,monospace}
.ok{color:var(--good);font-weight:600} .fail{color:var(--bad);font-weight:600}
.warn{color:#e0a83a;font-weight:600}
.note.bad-note{border-left-color:var(--bad)}
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path,
                    default=ROOT / 'metrics/WorldScore/results/dense_nr_full')
    ap.add_argument('--out', type=Path, default=ROOT / 'metrics/reports/nr_qualitative')
    ap.add_argument('--n', type=int, default=3, help='clips per end of each gallery')
    ap.add_argument('--limit', type=int, default=0,
                    help='sample this many records for speed (0 = all, the default). '
                         'The reality-check audit is only comparable to the paper '
                         'tables at 0, since sampling drops pred/GT pairs.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rows, gt_rows = read_records(args.run, args.limit, args.seed)
    print(f'usable prediction clips: {len(rows)}   GT anchor clips: {len(gt_rows)}')
    if not rows:
        raise SystemExit('no usable records')
    clips = args.out / 'clips'
    body = []

    for key, title, why, unit in DIRECTIONAL:
        lo = pick(rows, key, args.n, lowest=True)
        hi = pick(rows, key, args.n, lowest=False)
        vals = [r['_gated'][key] for r in rows if r['_gated'].get(key) is not None]
        body.append(f'<h2>{escape(title)} <span class="unit">{escape(unit)}</span></h2>')
        body.append(f'<p class="sub">{escape(why)}</p>')
        body.append(f'<p class="sub">scored on {len(vals)} clips &middot; '
                    f'median {st.median(vals):.2f} &middot; '
                    f'range shown: {min(r["_gated"][key] for r in lo):.2f} '
                    f'to {max(r["_gated"][key] for r in hi):.2f}</p>')
        if key in T.CEILING_FAILED:
            body.append('<div class="note"><b>Caveat.</b> This column failed the '
                        'GT-is-ceiling audit, so it may not rank models. Use these '
                        'clips to see what the metric reacts to, not to conclude which '
                        'model is better.</div>')
        if key in T.PAIRED_ONLY:
            body.append('<div class="note"><b>Caveat.</b> Paired only: an absolute '
                        'CLIPScore carries a per-caption bias of ~7 points, so compare '
                        'a clip against its own reference, never across prompts. The '
                        'prompts differ between the clips below.</div>')
        if key == 'active_motion_continuity':
            body.append('<div class="note"><b>You have to play these.</b> This column '
                        'is about what happens <i>between</i> frames, so a still frame '
                        'cannot show it &mdash; a low score means something jumps, pops '
                        'or slides wrongly from one frame to the next, not that the '
                        'picture is ugly. The clips are slowed to 3&nbsp;fps and loop.'
                        '</div>')
        for label, group, cls in (('POOR', lo, 'bad'), ('GOOD', hi, 'good')):
            body.append(f'<h3>{label} &mdash; lowest {escape(title)}</h3>'
                        if cls == 'bad' else
                        f'<h3>{label} &mdash; highest {escape(title)}</h3>')
            body.append('<div class="grid">' + ''.join(
                card(r, make_clip(r, clips, f'{key}_{cls}'), key, label, cls)
                for r in group) + '</div>')

    key, title, why, unit = DESCRIPTIVE
    body.append(f'<h2>{escape(title)} <span class="unit">{escape(unit)}</span></h2>')
    body.append(f'<p class="sub">{escape(why)}</p>')
    body.append('<div class="note"><b>No poor/good here.</b> There is no "good" value '
                'to show: the co-located real video defines how much the robot should '
                'move, so this column is descriptive. Below is a magnitude ladder per '
                'camera mount &mdash; the statistic itself differs by mount (static: '
                'subject p95; head: ego-compensated subject p95; wrist: whole-frame '
                'median), because on a wrist camera the "background" moves more than '
                'the subject does.</div>')
    for mount in ('static', 'head', 'wrist'):
        group = ladder(rows, key, mount, max(4, args.n + 1))
        if not group:
            continue
        body.append(f'<h3>{escape(mount)} mount &mdash; still &rarr; fast</h3>')
        body.append('<div class="grid">' + ''.join(
            card(r, make_clip(r, clips, f'{key}_{mount}'), key,
                 'STILL' if i == 0 else ('FAST' if i == len(group) - 1 else 'MID'),
                 'bad' if i == 0 else ('good' if i == len(group) - 1 else 'mid'))
            for i, r in enumerate(group)) + '</div>')

    # ---- reality check: does real video score higher than generated video? ----
    body.append('<h2 id="ceiling">Reality check &mdash; does real video score higher?</h2>')
    body.append('<p class="sub">A metric that claims to measure quality must rank the '
                '<b>real</b> video above the generated one. Each pair below is the same '
                'episode and the same view: real on the left, generated on the right. '
                'Only pairs that are the same measurement are used (identical source '
                'and effective fps, frame counts within 10%), so a difference cannot be '
                'a rendering artifact.</p>')
    audit_rows, gallery = [], []
    for key, title, _why, _unit in DIRECTIONAL:
        wins, allp = paired_gt(rows, gt_rows, key, args.n, gt_wins=True)
        viol, _ = paired_gt(rows, gt_rows, key, args.n, gt_wins=False)
        if not allp:
            continue
        # Aggregate per CONFIGURATION, the way `table_dense_nr.paired_table` does, NOT
        # by pooling every pair. Pooling weights a configuration by how many clips it
        # happens to have, and here that inverts the conclusion: pooled by clip, Visual
        # Integrity looks like it "prefers reality" 68% of the time, because the few
        # dreamgen cells that favour GT (-1.4 to -3.5) are the large ones. Per
        # configuration, prediction scores ABOVE real video in 8 of 17 -- which is the
        # ceiling failure the paper reports. Equal weight per configuration is the
        # honest view for "does this metric know real from generated".
        cells = collections.defaultdict(list)
        for d, pred, _gt in allp:
            cells[(pred['model'], pred['embodiment'], pred['camera_mount'])].append(d)
        med = {c: st.median(v) for c, v in cells.items() if len(v) >= MIN_CELL_PAIRS}
        if not med:
            continue
        gt_cells = sum(1 for v in med.values() if v > 0)
        # Count only the pairs that actually feed the verdict, i.e. those inside the
        # configurations kept above. Summing every pair (including the sub-threshold
        # cells that are then discarded) would print a total the paper table cannot
        # reproduce.
        n_pairs = sum(len(cells[c]) for c in med)
        audit_rows.append((key, title, n_pairs, len(med), gt_cells, st.median(
            list(med.values())), sorted(med.items(), key=lambda x: x[1])))
        gallery.append((key, title, wins, viol))
    body.append('<table class="audit"><tr><th>Metric</th><th>pairs</th>'
                '<th>configs (n&ge;%d)</th><th>configs where real wins</th>'
                '<th>median of config medians</th><th>verdict</th></tr>'
                % MIN_CELL_PAIRS)
    for key, title, n_pairs, n_cells, gt_cells, med, _per in audit_rows:
        # Three states, all taken from `table_dense_nr` rather than re-derived here:
        # a failed ceiling audit (may not rank models at all), a column that passes but
        # carries a documented caveat -- AMC's humanoid offset is exactly the "do not
        # read a small positive humanoid GT-delta as beating reality" case, which is why
        # its raw config count looks weak (47%) while the column is still sound -- and a
        # clean pass.
        failed = key in T.CEILING_FAILED
        caveat = T.NOT_CROSS_EMBODIMENT.get(key)
        if failed:
            cls, verdict = 'fail', 'CEILING FAILED &mdash; may not rank models'
        elif caveat:
            cls, verdict = 'warn', 'prefers reality &mdash; within an embodiment'
        else:
            cls, verdict = 'ok', 'prefers reality'
        body.append(f'<tr><td>{escape(title)}</td><td class="num">{n_pairs}</td>'
                    f'<td class="num">{n_cells}</td>'
                    f'<td class="num">{gt_cells}/{n_cells}'
                    f' ({100*gt_cells/n_cells:.0f}%)</td>'
                    f'<td class="num">{med:+.3f}</td>'
                    f'<td class="{cls}">{verdict}</td></tr>')
    body.append('</table>')
    for key, title, *_ in audit_rows:
        if key in T.CEILING_FAILED:
            body.append(f'<div class="note bad-note"><b>{escape(title)}:</b> '
                        f'{escape(T.CEILING_FAILED[key])}</div>')
        if key in T.NOT_CROSS_EMBODIMENT:
            body.append(f'<div class="note"><b>{escape(title)} &mdash; why the config '
                        f'count above looks weak:</b> '
                        f'{escape(T.NOT_CROSS_EMBODIMENT[key])}</div>')
    body.append('<div class="note"><b>The verdict column is the paper\'s, not this '
                'page\'s.</b> It reports membership of <code>CEILING_FAILED</code> in '
                '<code>assets/scripts/table_dense_nr.py</code>, which was decided by the '
                'pre-registered audit over the whole wave &mdash; not by the sample shown '
                'here. Counting per configuration (not per clip) is deliberate: pooled '
                'over clips, a metric can look like it prefers reality simply because the '
                'configurations that happen to have the most clips are the ones where it '
                'does.</div>')
    body.append('<div class="note"><b>How to read a failure.</b> "Ceiling failed" does '
                'not mean the metric is useless &mdash; it means its <i>absolute</i> '
                'value may not rank models, because it is insensitive to the gap between '
                'real and generated video. Such a column is still valid for comparing a '
                'method against its own dense baseline, where that insensitivity cancels '
                'on both sides of the difference.</div>')
    for key, title, n_pairs, n_cells, gt_cells, med, per in audit_rows:
        body.append(f'<p class="sub"><b>{escape(title)}</b> per configuration '
                    f'(pred &minus; real; negative = real video wins):<br>'
                    + ' &nbsp;&middot;&nbsp; '.join(
                        f'<span class="{"ok" if v < 0 else "fail"}">'
                        f'{escape("/".join(c))} {-v:+.2f}</span>' for c, v in per)
                    + '</p>')
    for key, title, wins, viol in gallery:
        body.append(f'<h3>{escape(title)} &mdash; real video wins here '
                    f'(as it should)</h3>')
        body.append('<div class="grid">' + ''.join(
            pair_card(d, pred, gt,
                      make_clip(pred, clips, f'pair_{key}_pred'),
                      make_clip(gt, clips, f'pair_{key}_gt'), key)
            for d, pred, gt in wins) + '</div>')
        if viol:
            body.append(f'<h3>{escape(title)} &mdash; generated beats real '
                        f'(the metric is wrong here)</h3>')
            body.append('<div class="grid">' + ''.join(
                pair_card(d, pred, gt,
                          make_clip(pred, clips, f'viol_{key}_pred'),
                          make_clip(gt, clips, f'viol_{key}_gt'), key)
                for d, pred, gt in viol) + '</div>')

    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PhyWorld-NR &mdash; qualitative examples</title><style>{CSS}</style></head><body>
<h1>PhyWorld-NR &mdash; what each metric actually reacts to</h1>
<p class="sub">Every clip below is the <b>analysis clip</b>: re-created with the same
<code>prepare()</code> the scorer used, so it is the same view crop, the same 5&nbsp;fps
sampling and the same 256&times;192 grid that produced the number. The conditioning
frame is dropped, exactly as in scoring. Numbers come from
<code>table_dense_nr.gated()</code>, i.e. the current gates.</p>
<p class="sub">The galleries show model predictions only; the
<a href="#ceiling">reality check</a> at the end puts each prediction next to its own
real video. Each card also lists the other columns, so you can see that the metrics
move independently &mdash; and a clip may legitimately appear in two galleries if it is
weak on two of them.</p>
{''.join(body)}
<footer>run <code>{escape(str(args.run))}</code> &middot;
{len(rows)} prediction and {len(gt_rows)} real clips considered &middot;
protocol <code>{escape(T.PROTOCOL_REQUIRED)}</code> &middot;
regenerate with <code>assets/scripts/make_nr_qualitative_report.py</code>.
Active Rate is not shown: it is computed but no longer reported
(&rho;&nbsp;+0.875 with Actor Activity).</footer>
</body></html>'''

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'index.html').write_text(html)
    n_clips = len(list(clips.glob('*.mp4')))
    print(f'wrote {args.out / "index.html"}  ({n_clips} clips in {clips})')


if __name__ == '__main__':
    main()
