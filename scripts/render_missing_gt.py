#!/usr/bin/env python3
"""Render the ground truth that four DreamGen grid cells never got, and prove it lines up.

Those cells have predictions but no GT, so `gt_for()` finds nothing and they contribute
no `vs_gt` rows -- which is why half the grid is empty. GT here is the LeRobot source
episode re-expressed in the prediction's framing: same resolution, same frame count,
same frame-to-frame correspondence.

The correspondence is solved per clip rather than assumed (see solve_convention in
gt_cells.py) because these predictions are not all from one prepare configuration.

    --validate  scans the temporal rate and a frame offset around the solved values and
                reports whether the solved mapping is actually the best one. Run this
                before rendering: a GT that is off by a few frames still produces
                plausible-looking PSNR, so "it ran" is not evidence of correctness.
    --render    writes <pred_stem>_gt.mp4 into the dense tree, then hard-links it beside
                every method's prediction, which is where gt_for() looks. Hard links
                because the same GT serves all ten methods and copying would cost ~10x
                the disk for identical bytes.

Usage:
    python scripts/render_missing_gt.py --all --validate --clips 8
    python scripts/render_missing_gt.py --all --render --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from gt_cells import (CELLS, _dims, _read_all, list_clips,  # noqa: E402
                      solve_convention, spatial_transform, src_index_map)

METHODS = ['dense', 'worldcache', 'dicache', 'fastercache', 'sito', 'itm',
           'pisa', 'radial', 'svg1', 'svg2']


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    m = float((d * d).mean())
    return 99.0 if m <= 1e-9 else 10.0 * float(np.log10(255.0 * 255.0 / m))


def build_gt(clip: dict, cell: dict) -> tuple[np.ndarray, dict]:
    """GT frames in the prediction's exact framing, plus the mapping that produced them."""
    n_pred, w_pred, h_pred, fps_pred = _dims(clip['pred'])
    src = cell['read_source'](clip)
    sol = solve_convention(len(src), n_pred, cell['src_fps'], cell['model_fps'])
    if sol is None:
        raise RuntimeError(f'cannot map n_src={len(src)} onto n_pred={n_pred}')
    idx = src_index_map(sol['n_gt'], len(src), sol['rate'])
    gt = spatial_transform(src[idx], w_pred, h_pred, cell['spatial'])
    return gt, {**sol, 'spatial': cell['spatial'],
                'pred_wh': [w_pred, h_pred], 'pred_fps': fps_pred}


def validate_clip(clip: dict, cell: dict) -> dict:
    """Check the two things about this GT that can actually be checked.

    1. FRAME 0. A DreamGen prediction's first frame is the conditioning frame, which is
       source frame 0 pushed through the model's resize and a video encode. It should
       therefore match GT frame 0 far better than any later frame matches. This is the
       same signal that exposed 33 mislabelled hallucinate clips (frame 0 agreed, frame
       46 did not), and it is the only part of an autoregressive rollout where the
       prediction is required to equal reality.

    2. TEMPORAL RATE, but only where the data can settle it. Scanning a rate multiplier
       measures alignment plus generation drift together, and past the first chunk drift
       dominates. So the scan is restricted to the first chunk and reported with the
       source's own motion: on a near-static clip every candidate scores alike and the
       honest verdict is 'uninformative', not 'pass'. Aggregate evidence for the rate
       came from running this across 24 bimanual clips at once (see solve_convention);
       per-clip it is a sanity check, not the proof.
    """
    n_pred, w_pred, h_pred, _ = _dims(clip['pred'])
    src = cell['read_source'](clip)
    sol = solve_convention(len(src), n_pred, cell['src_fps'], cell['model_fps'])
    if sol is None:
        return {'clip': clip['key'], 'status': 'unsolved',
                'n_src': len(src), 'n_pred': n_pred}

    pred = _read_all(clip['pred'])
    grey = lambda fs: np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 54),
                                           interpolation=cv2.INTER_AREA) for f in fs])
    idx_all = src_index_map(sol['n_gt'], len(src), sol['rate'])
    gt = spatial_transform(src[idx_all], w_pred, h_pred, cell['spatial'])
    pred_s, gt_s = grey(pred), grey(gt)

    # (1) frame-0 control, at full resolution -- this is the strict test
    psnr0 = psnr(pred[0], gt[0])
    # what a typical LATER frame scores, for contrast
    mid = min(n_pred // 2, sol['n_gt'] - 1)
    psnr_mid = psnr(pred[mid], gt[mid])

    # (2) first-chunk rate scan, done by re-indexing the aligned GT
    lim = min(93, len(gt_s))
    idxs = [i for i in range(4, lim, 4) if 2.0 * i < len(gt_s) - 1]
    scan = {}
    for mult in (0.5, 0.8, 1.0, 1.25, 2.0):
        if not idxs:
            break
        j = np.clip(np.rint(np.array(idxs) * mult).astype(int), 0, len(gt_s) - 1)
        scan[mult] = float(np.mean([psnr(pred_s[i], gt_s[jj]) for i, jj in zip(idxs, j)]))
    motion = float(np.mean(np.abs(np.diff(gt_s[:min(60, len(gt_s))].astype(np.float32),
                                          axis=0))))

    best_mult = max(scan, key=scan.get) if scan else None
    spread = (max(scan.values()) - min(scan.values())) if scan else 0.0

    if psnr0 < 20.0:
        status = 'fail_frame0'
    elif psnr0 < psnr_mid + 3.0:
        status = 'fail_no_frame0_advantage'
    elif not scan or spread < 1.0 or motion < 1.0:
        status = 'pass_frame0_rate_uninformative'
    elif best_mult == 1.0:
        status = 'pass'
    else:
        status = 'pass_frame0_rate_ambiguous'

    return {'clip': clip['key'], 'status': status, 'rate': round(sol['rate'], 4),
            'spatial': cell['spatial'],
            'n_src': len(src), 'n_pred': n_pred, 'n_gt': sol['n_gt'],
            'coverage': sol['coverage'],
            'psnr_frame0': round(psnr0, 2), 'psnr_mid': round(psnr_mid, 2),
            'frame0_advantage': round(psnr0 - psnr_mid, 2),
            'rate_scan': {str(k): round(v, 2) for k, v in scan.items()},
            'best_mult': best_mult, 'scan_spread': round(spread, 2),
            'src_motion': round(motion, 3)}



def write_video(frames: np.ndarray, out: Path, fps: float) -> None:
    """Encode GT with ffmpeg at near-transparent quality.

    crf 12 rather than the default 23: this file is a metric REFERENCE, so its own
    compression error should sit well below the differences being measured. Written to a
    temporary name and renamed, so an interrupted run cannot leave a half-written mp4
    that a later resumable pass would accept as complete.
    """
    import subprocess

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.mp4')
    h, w = frames.shape[1:3]
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{w}x{h}', '-r', f'{fps:g}', '-i', 'pipe:0',
           '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '12',
           '-pix_fmt', 'yuv420p', str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = proc.communicate(frames.tobytes())
    if proc.returncode != 0 or not tmp.exists():
        raise RuntimeError(f'ffmpeg failed: {err.decode()[:400]}')
    tmp.replace(out)


def link_to_methods(canonical: Path, clip: dict, cell: dict) -> int:
    """Expose one GT file beside every method's prediction, where gt_for() looks."""
    rel = canonical.relative_to(REPO / 'sampling_dataset' / 'dense')
    linked = 0
    for method in METHODS:
        if method == 'dense':
            continue
        target = REPO / 'sampling_dataset' / method / rel
        if not target.parent.is_dir():
            continue                       # this method never ran this cell
        if not (target.parent / clip['pred'].name).exists():
            continue                       # no prediction here -> no reference needed
        if target.exists():
            linked += 1
            continue
        try:
            os.link(canonical, target)
        except OSError:
            import shutil
            shutil.copy2(canonical, target)
        linked += 1
    return linked


def render_one(args: tuple) -> dict:
    cell_name, split, pred_str, force = args
    cell = CELLS[cell_name]
    from gt_cells import parse_pred
    clip = parse_pred(Path(pred_str), split)
    out = clip['pred'].with_name(f'{clip["pred"].stem}_gt.mp4')
    try:
        if out.exists() and not force:
            n_links = link_to_methods(out, clip, cell)
            return {'clip': clip['key'], 'status': 'exists', 'links': n_links}
        gt, meta = build_gt(clip, cell)
        write_video(gt, out, meta['pred_fps'])
        n_links = link_to_methods(out, clip, cell)
        return {'clip': clip['key'], 'status': 'rendered', 'links': n_links, **meta}
    except Exception as exc:                                        # noqa: BLE001
        return {'clip': clip['key'], 'status': 'error',
                'error': f'{type(exc).__name__}: {exc}',
                'trace': traceback.format_exc(limit=3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', action='append', default=[], choices=sorted(CELLS))
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--split', action='append', default=[])
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--clips', type=int, default=8, help='clips per cell when validating')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='cap clips per split (debug)')
    ap.add_argument('--report', default='metrics/reports/gt_render.json')
    args = ap.parse_args()

    names = sorted(CELLS) if args.all else args.cell
    if not names:
        ap.error('pass --cell NAME (repeatable) or --all')
    if not (args.validate or args.render):
        ap.error('pass --validate and/or --render')
    splits = args.split or ['makovian', 'non_makovian']

    report: dict = {'validate': {}, 'render': {}}

    if args.validate:
        for name in names:
            cell = CELLS[name]
            clips = [c for s in splits for c in list_clips(cell, s, 'dense')]
            if not clips:
                print(f'=== {name}: no clips'); continue
            step = max(1, len(clips) // max(1, args.clips))
            chosen = clips[::step][:args.clips]
            print(f'\n=== VALIDATE {name}  ({len(clips)} clips, checking {len(chosen)})',
                  flush=True)
            rows = []
            for clip in chosen:
                try:
                    r = validate_clip(clip, cell)
                except Exception as exc:                            # noqa: BLE001
                    r = {'clip': clip['key'], 'status': 'error',
                         'error': f'{type(exc).__name__}: {exc}'}
                rows.append(r)
                if r['status'] in ('error', 'unsolved'):
                    print(f'  {r["status"].upper():14s} {r["clip"][:44]:44s} '
                          f'{r.get("error", "")}', flush=True)
                else:
                    print(f'  {r["status"]:31s} {r["clip"][:40]:40s} '
                          f'rate={r["rate"]} n_gt={r["n_gt"]}/{r["n_pred"]} '
                          f'cov={r["coverage"]} '
                          f'psnr0={r["psnr_frame0"]:6.2f} mid={r["psnr_mid"]:6.2f} '
                          f'adv={r["frame0_advantage"]:+6.2f} '
                          f'scan={r["rate_scan"]} motion={r["src_motion"]:.2f}',
                          flush=True)
            from collections import Counter
            tally = Counter(r['status'] for r in rows)
            print(f'  -> {dict(tally)}', flush=True)
            report['validate'][name] = {'tally': dict(tally), 'rows': rows}

    if args.render:
        for name in names:
            cell = CELLS[name]
            jobs = []
            for split in splits:
                clips = list_clips(cell, split, 'dense')
                if args.limit:
                    clips = clips[:args.limit]
                jobs += [(name, split, str(c['pred']), args.force) for c in clips]
            print(f'\n=== RENDER {name}: {len(jobs)} clips on {args.workers} workers',
                  flush=True)
            rows = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(render_one, j): j for j in jobs}
                done = 0
                for fut in as_completed(futs):
                    r = fut.result()
                    rows.append(r)
                    done += 1
                    if r['status'] == 'error':
                        print(f'  ERROR {r["clip"][:48]:48s} {r["error"]}', flush=True)
                    elif done % 25 == 0 or done == len(jobs):
                        print(f'  {done}/{len(jobs)}', flush=True)
            from collections import Counter
            tally = Counter(r['status'] for r in rows)
            rules = Counter(r.get('rule') for r in rows if r['status'] == 'rendered')
            print(f'  -> {dict(tally)}  rules={dict(rules)}', flush=True)
            report['render'][name] = {'tally': dict(tally), 'rules': dict(rules),
                                      'rows': rows}

    out = REPO / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if out.exists():
        try:
            prev = json.loads(out.read_text())
        except Exception:                                           # noqa: BLE001
            prev = {}
    for k, v in report.items():
        prev.setdefault(k, {}).update(v)
    out.write_text(json.dumps(prev, indent=2))
    print(f'\nwrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
