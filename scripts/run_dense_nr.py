"""Resumable dense NR inventory and persistent GPU workers; no dense reference.

Run through run_dense_nr.sh for vendored dependency paths.
"""
import argparse
import collections
import errno
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Kept at v3.2: the two gate changes from the E1/E4 audit (MIN_BG_COV_PCT 10 -> 5,
# MIN_ACTIVE_FRAC) are applied at TABLE time by `table_dense_nr.py`, which recomputes
# the motion columns from the stored per-pair records (validated exact: max abs
# deviation 7e-15 vs the stored values). Records therefore stay comparable and the
# 1,572 already-scored clips did NOT need re-scoring. Bump this only when a change
# alters something that is NOT reconstructible from the record (e.g. the sampling grid,
# the probe itself, or which per-pair fields are stored).
PROTOCOL = 'dense-nr-v3.2-closed-world'
KEYS = ['visual_integrity', 'text_condition_alignment', 'actor_motion_raw',
        'actor_motion_active_rate', 'active_motion_continuity', 'actor_flicker_raw']
TEXT_CONDITIONED_MODELS = {'dreamgen', 'ctrlworld', 'ctrlworld_4view_grid', 'gt'}


def text_conditioned(model):
    """Whether a text instruction is a legitimate condition for this clip.

    Must strip the `gt_` prefix: the per-model co-located anchors (`gt_dreamgen`,
    `gt_ctrlworld`, ...) are real videos of the same instructed task, so they are
    exactly as text-conditioned as the prediction they anchor. Comparing them by
    name alone marked every anchor N/A and left the paired TextAlignment column
    permanently empty.
    """
    return (model[3:] if model.startswith('gt_') else model) in TEXT_CONDITIONED_MODELS
# GT per-view inputs. View order verified against pred strips: GT view_i == per_view[i].
# bimanual/humanoid ctrlworld GT is a view subset of the 4-view GT (same episodes).
GT_SOURCES = {
    'bimanual': ('ctrlworld_4view_grid', ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']),
    'humanoid': ('ctrlworld_4view_grid', ['cam_head_rgb', 'cam_front_rgb', 'cam_left_wrist_rgb', 'cam_right_wrist_rgb']),
    'single_arm': ('ctrlworld', ['exterior_1_left', 'exterior_2_left', 'wrist_left']),
}
ACTIVE_PX = 1.0           # px/source-frame, on the per-pair motion statistic below
# Head ego estimate needs enough background pixels. Was 10; lowered to 5 after E4
# (2026-09-16), which is answerable post-hoc because `actor_per_frame` stores per-pair
# `bg_cov_pct` and `fg_flow_ego_p95` and this gate is applied after the probe. On all
# 241 head clips of the completed v3.2 run, 10 -> 5 changes the head `actor_motion_raw`
# median by 0.00% on a fixed clip set (the discarded pairs sit at bg_cov < 5; the band
# [5,10) is only 5.66% of head pairs) while recovering 15 head clips (+6.2%) that gate
# 10 discarded entirely. 20 is where values actually shift (per-clip max +34%).
MIN_BG_COV_PCT = 5.0
MIN_VALID_PAIR_FRAC = 0.5
MIN_WEIGHT_SUM = 3.0      # AMC needs >=3 activity-equivalent triples, else N/A
MIN_ACTIVE_PAIRS = 3      # ...AND >=3 pairs actually at/above ACTIVE_PX, else N/A
# ...AND that count must be >=5% of the clip's valid pairs. The absolute floor alone
# does not scale with clip length, which is how three clips still reported AMC 71-82 at
# active_rate 0.02-0.04 in the completed v3.2 run: at 79 / 177 / 245 valid pairs, 3-7
# active pairs clear `MIN_ACTIVE_PAIRS` while essentially nothing in the clip moved.
# 0.05 is calibrated on those 1,451 AMC-valid clips: it drops exactly the 15 clips whose
# ActorMotion is <= 0.67 px/source-frame (all genuinely static) and zero clip at
# >= 1 px, so it removes all three artifacts with no false positive. Raising it to 0.15
# starts discarding real motion (4 clips >= 1 px).
MIN_ACTIVE_FRAC = 0.05
COPY_ERR_FLOOR = 2.0      # uint8 MSE; below this frame change is codec noise


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    # The scratch name must be unique per writer. It used to be `<id>.tmp`, which is
    # shared: when two processes score the same clip at once (unavoidable where a
    # forward and a `--reverse` queue meet, and across machines on the shared
    # filesystem) both wrote the same file and the interleaved bytes were then
    # atomically renamed into place as a corrupt result. pid alone is not enough
    # across hosts.
    tmp = path.with_suffix(f'.{os.uname().nodename}.{os.getpid()}.tmp')
    try:
        tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
        tmp.replace(path)
    except BaseException:
        # A failed write leaves a 0-byte scratch file behind. Measured 2026-09-19: an
        # EDQUOT squeeze on the shared fileset left 28 of them in results/, where they
        # were counted as scored clips by `ls results | wc -l` and broke any reader that
        # globs the directory. Remove the stub so results/ only ever holds complete files.
        tmp.unlink(missing_ok=True)
        raise


TASK_CLAUSE = re.compile(r'(?:performing(?: the task)?:|shows that a robot)\s*(.+?)(?:\s+The video\b|$)', re.S)


def task_text(instruction):
    """Task clause only. DreamGen prompts append 2x2-grid layout text that a cropped
    single view cannot match and that pushes the task past CLIP's 77-token window."""
    if not instruction:
        return ''
    m = TASK_CLAUSE.search(instruction)
    return (m.group(1) if m else instruction).strip().rstrip('.').strip()


def camera_mount(embodiment, view):
    """static cameras need no ego compensation; head/wrist cameras move."""
    v = view.lower()
    if 'wrist' in v:
        return 'wrist'
    if embodiment == 'humanoid' and v in ('cam_high', 'cam_high_rgb', 'cam_head_rgb', 'singleview', 'grid_top_left'):
        return 'head'
    return 'static'


def episode_key(rel):
    """Unique episode token `<task>/<episode_id>` from a result path.

    Must be unique per real episode or pairing silently merges clips:
    - DreamGen stores `.../<task>/episode_000004/full_pred.mp4`, so the bare
      `episode_000004` collides across tasks (48 bimanual tasks reuse ep numbers).
    - DreamDojo stores `.../<task>__episode_000039/full_pred.mp4`, where no path
      part startswith 'episode_', so the old code fell back to the file stem and
      EVERY DreamDojo clip collapsed onto the key `full_pred`.
    """
    parts = list(rel.parts)
    for i in range(len(parts) - 1, -1, -1):
        p = parts[i]
        if '__episode_' in p:                      # DreamDojo: <task>__episode_<n>
            task, ep = p.rsplit('__', 1)
            return f'{task}/{ep}'
        if p.startswith('episode_'):
            name = p[len('episode_'):]
            if '__' in name:                       # Ctrl-World: episode_<task>__<n>
                task, ep = name.rsplit('__', 1)
                return f'{task}/episode_{ep}'
            parent = parts[i - 1] if i > 0 else ''  # DreamGen: <task>/episode_<n>
            if parent and not parent.startswith('iter_') and parent not in ('makovian', 'non_makovian'):
                return f'{parent}/{p}'
            return p
    stem = rel.stem
    if '__episode_' in stem:                       # flat singleview: <task>__episode_<n>.mp4
        task, ep = stem.rsplit('__', 1)
        return f'{task}/{ep}'
    return stem


def row_for(p, rel, model, embodiment, split, view, layout, vi, n_views, instruction, method):
    # An accel tree mirrors the dense tree's relative paths exactly, so `rel + view`
    # alone collides across methods and an accel result would overwrite the dense one
    # (same `id` -> same result filename). Namespace by method. `dense`/`gt` keep the
    # bare identity so the 1572 already-scored v3.2 results stay valid and resumable.
    identity = str(rel) + ':' + view
    if method not in ('dense', 'gt'):
        identity = f'{method}:{identity}'
    return dict(id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                path=str(p), relative=str(rel), model=model,
                embodiment=embodiment, split=split, view=view,
                camera_mount=camera_mount(embodiment, view),
                layout=layout, view_index=vi, n_views=n_views,
                instruction=instruction, method=method,
                # Deliberately method-FREE: this is what pairs an accel clip to its
                # dense counterpart (and to GT) for the paired delta.
                pair_key=f'{embodiment}/{split}/{episode_key(rel)}/{view}')


def inventory(out, methods=('dense',)):
    from phyworld_index import _instruction_from_dir, _instruction_from_txt
    dense = ROOT / 'sampling_dataset/dense'
    rows = []
    # GT reference rows first: same analysis protocol, used as plausibility anchor.
    # GT is only ever taken from the dense tree: the accel trees carry byte-identical
    # copies of the same GT, so emitting them again would just duplicate GPU work and
    # collide on pair_key.
    for emb, (src, default_views) in GT_SOURCES.items():
        for ep in sorted((dense / emb / 'input/multiview' / src).glob('*/episode_*')):
            files = sorted(ep.glob('view_*.mp4'))
            if not files:
                continue
            # View names vary per episode (humanoid: cam_head_rgb vs cam_high_rgb);
            # take them from the matching prediction so GT/pred pair_keys line up.
            pred_meta = dense / emb / 'output/multiview' / src / ep.parent.name / ep.name / 'metrics.json'
            views = (list(json.loads(pred_meta.read_text()).get('per_view', {}))
                     if pred_meta.exists() else []) or default_views
            if len(files) != len(views):
                raise ValueError(f'GT view count mismatch: {ep}')
            instruction = _instruction_from_dir(ep)
            for vi, view in enumerate(views):
                p = ep / f'view_{vi}.mp4'
                rows.append(row_for(p, p.relative_to(dense), 'gt', emb, ep.parent.name, view,
                                    'single', 0, 1, instruction, 'gt'))
    for method in methods:
        tree = ROOT / 'sampling_dataset' / method
        if not tree.exists():
            raise ValueError(f'No such method tree: {tree}')
        rows.extend(_pred_rows(tree, method, _instruction_from_dir, _instruction_from_txt))
    dump(out / 'manifest.json', rows)
    print(f'INVENTORY {len(rows)} view-clips / {len(set(r["path"] for r in rows))} videos '
          f'({sum(r["method"] == "gt" for r in rows)} GT view-clips) '
          f'methods={list(methods)}', flush=True)
    return rows


def _pred_rows(tree, method, _instruction_from_dir, _instruction_from_txt):
    """Prediction view-clips of one `sampling_dataset/<method>` tree.

    Identical walk for every method, so an accel tree is inventoried by exactly the
    same rules as dense and the two are comparable by construction. The co-located
    GT sibling is only emitted for the dense tree (`method == 'dense'`): the accel
    trees ship byte-identical GT copies, and scoring them again would duplicate work
    and collide on `pair_key`.
    """
    rows = []
    for p in sorted(tree.rglob('*.mp4')):
        rel = p.relative_to(tree)
        parts = rel.parts
        if len(parts) < 6 or parts[1] != 'output' or parts[4] not in ('makovian', 'non_makovian'):
            continue
        if p.name == 'full_pred.mp4':
            if (p.parent / 'pred_all_views.mp4').exists():
                continue
        elif p.name != 'pred_all_views.mp4' and not ('episode_' in p.stem and p.name.endswith('.mp4')):
            continue
        if '_phyworld' in str(p) or 'seg_' in p.name:
            continue
        # Ctrl-World full_pred is a comparison panel. Only use its prediction strip.
        if parts[3].startswith('ctrlworld') and p.name == 'full_pred.mp4':
            continue
        meta = {}
        if (p.parent / 'metrics.json').exists():
            meta = json.loads((p.parent / 'metrics.json').read_text())
        elif method != 'dense':
            # A few accel episodes were written without metrics.json (interrupted run),
            # so the view layout is unknown. Take it from the dense counterpart at the
            # same relative path: the layout MUST match dense anyway for the clips to
            # be comparable, so this is the authoritative source rather than a guess.
            dense_meta = ROOT / 'sampling_dataset/dense' / rel.parent / 'metrics.json'
            if dense_meta.exists():
                meta = json.loads(dense_meta.read_text())
        instruction = _instruction_from_dir(p.parent) if p.name in ('full_pred.mp4', 'pred_all_views.mp4') else _instruction_from_txt(p.with_suffix('.txt'))
        views = list(meta.get('per_view', {}))
        if parts[2] == 'singleview':
            views, layout = ['singleview'], 'single'
        elif p.name == 'pred_all_views.mp4':
            if not views:
                raise ValueError(f'Missing view layout: {p}')
            layout = 'strip'
        elif parts[3] == 'dreamgen':
            if parts[0] == 'single_arm':
                views, layout = ['exterior_1_left', 'exterior_2_left', 'wrist_left'], 'grid'
            else:
                views, layout = ['grid_top_left', 'grid_top_right', 'left_wrist', 'right_wrist'], 'grid'
        else:
            raise ValueError(f'Unknown layout: {p}')
        for vi, view in enumerate(views):
            rows.append(row_for(p, rel, parts[3], parts[0], parts[4], view, layout, vi,
                                len(views), instruction, method))
        if method != 'dense':
            continue
        # Co-located GT: `full_gt.mp4` / `gt_all_views.mp4` written next to the
        # prediction is byte-for-byte the same geometry (verified: same width,
        # height, fps AND frame count), so `prepare` samples the identical frames
        # and the GT-vs-pred delta carries no fps/crop/resolution caveat. The
        # shared 5 fps Ctrl-World input tree above cannot anchor DreamGen or
        # DreamDojo at all (0 shared episodes), and comparing across 5 vs 16 fps
        # would mix units. One anchor per model so two models' GT for the same
        # episode never collide on `pair_key`.
        gt_sib = p.parent / ('gt_all_views.mp4' if p.name == 'pred_all_views.mp4' else 'full_gt.mp4')
        if gt_sib.exists():
            for vi, view in enumerate(views):
                rows.append(row_for(gt_sib, gt_sib.relative_to(tree), f'gt_{parts[3]}',
                                    parts[0], parts[4], view, layout, vi, len(views),
                                    instruction, 'gt'))
    return rows


def task_of(row):
    """Task template of a clip, from the shared `episode_key` parser."""
    key = episode_key(Path(row['relative']))
    if '/' in key:
        return key.rsplit('/', 1)[0]
    # DROID / singleview episodes carry no task in the path; instruction is the task.
    return (row.get('instruction') or 'unknown').strip().lower()


def spread_sample(items, k, rng, key):
    """Round-robin over strata (task, split) so a sample is not dominated by one task."""
    strata = collections.defaultdict(list)
    for it in items:
        strata[key(it)].append(it)
    for v in strata.values():
        rng.shuffle(v)
    order = sorted(strata, key=lambda s: rng.random())
    out = []
    while len(out) < k and any(strata.values()):
        for s in order:
            if strata[s] and len(out) < k:
                out.append(strata[s].pop())
    return out


def smoke_manifest(rows, paired_episodes, per_group, seed):
    """Validation cohort: GT/pred pairs on same episode-views plus every model/mount group.

    Pairs are drawn per PREDICTION MODEL against its own co-located GT (`gt_<model>`),
    because the paired GT-delta is the only test with enough power here: at n=30 the
    unpaired GT vs pred medians have overlapping 95% CIs for both VisualIntegrity
    (32.5 [31.0,33.6] vs 31.7 [29.7,33.1]) and ActiveMotionContinuity, while the
    paired delta resolves (+1.00 [+0.56,+1.60] and +1.78 [+0.01,+3.05]). Selecting
    only whole pairs guarantees the paper's headline column is computable.

    Both parts spread over task templates and splits for debugging diversity.
    """
    rng = random.Random(seed)
    by_key = collections.defaultdict(dict)
    for r in rows:
        by_key[r['pair_key']][r['model']] = r
    chosen = {}
    pred_models = sorted({r['model'] for r in rows if r['method'] == 'dense'})
    for model in pred_models:
        anchor = f'gt_{model}'
        keys = sorted(k for k, v in by_key.items() if model in v and anchor in v)
        if not keys:
            continue
        per_episode = collections.defaultdict(list)
        for k in keys:
            per_episode[k.rsplit('/', 1)[0]].append(k)
        episodes = sorted(per_episode)
        first = {ep: by_key[per_episode[ep][0]][model] for ep in episodes}
        order = spread_sample(episodes, len(episodes), rng,
                              key=lambda ep: (task_of(first[ep]), first[ep]['split']))
        # Statistical power comes from PAIRED CLIP count, not episode count, and
        # views-per-episode ranges from 1 (DreamDojo singleview) to 4 (grid/strip).
        # Taking a fixed episode count gave DreamDojo 14 paired clips vs Ctrl-World
        # 65. Add episodes until the clip target is met so every model is comparably
        # powered.
        target = paired_episodes * 3
        n_clips = 0
        for ep in order:
            if n_clips >= target:
                break
            for k in per_episode[ep]:
                for m in (anchor, model):
                    chosen[by_key[k][m]['id']] = by_key[k][m]
            n_clips += len(per_episode[ep])
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['model'], r['embodiment'], r['camera_mount'])].append(r)
    for g in groups.values():
        for r in spread_sample(g, per_group, rng, key=lambda r: (task_of(r), r['split'])):
            chosen[r['id']] = r
    return list(chosen.values())


def prepare(row, dest):
    import cv2
    cap = cv2.VideoCapture(row['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError('Invalid FPS')
    interval = max(1, round(fps / 5))
    frames, ids = [], []
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        # Conservatively exclude first conditioning frame for every video.
        if i > 0 and (i - 1) % interval == 0:
            h, w = fr.shape[:2]
            vi = row['view_index']
            if row['layout'] == 'strip':
                fr = fr[:, vi*w//row['n_views']:(vi+1)*w//row['n_views']]
            elif row['layout'] == 'grid':
                fr = fr[(vi//2)*h//2:(vi//2+1)*h//2, (vi%2)*w//2:(vi%2+1)*w//2]
            # Fixed analysis grid. Report aspect-ratio resampling explicitly.
            fr = cv2.resize(fr, (256, 192), interpolation=cv2.INTER_AREA)
            frames.append(fr[:, :, ::-1].copy()); ids.append(i)
        i += 1
    cap.release()
    if len(frames) < 3:
        raise ValueError('Fewer than three post-conditioning frames')
    import numpy as np
    arr = np.stack(frames)
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), fps/interval, (256,192))
    if not writer.isOpened():
        raise RuntimeError('Cannot create analysis video')
    for fr in frames:
        writer.write(fr[:, :, ::-1])
    writer.release()
    return arr, dict(source_fps=fps, effective_fps=fps/interval, source_interval=interval,
                     analysis_width=256, analysis_height=192, sampled_source_frames=ids)


def actor_pair_motion(per_frame, mount, interval):
    """Per-pair actor motion in px/source-frame, keyed by pair index; plus median diagnostic.

    Static: p95 of uncorrected subject flow. Subject median is ~0 because static robot
    body and objects dominate the mask (v3 smoke GT median 0.05 px); ego subtraction
    there removes deformable-object motion. Head: p95 of ego-compensated subject flow
    where background >= MIN_BG_COV_PCT. Wrist: whole-frame median flow, since the
    camera rides the end effector and masks often fill the frame.
    """
    motion, median = {}, {}
    for p in per_frame:
        k = p['pair']
        if mount == 'static':
            motion[k], median[k] = p['fg_flow_p95'], p['fg_flow_med']
        elif mount == 'wrist':
            motion[k] = median[k] = p['frame_flow_med']
        elif p['bg_cov_pct'] >= MIN_BG_COV_PCT:
            motion[k], median[k] = p['fg_flow_ego_p95'], p['fg_flow_ego_med']
    return ({k: v / interval for k, v in motion.items()},
            {k: v / interval for k, v in median.items()})


def continuity(paths, smooth, motion, interval_ok, n_active):
    """Motion-normalized VFI gain on activity-weighted triples.

    gain_j = 1 - MSE(VFI(f0,f2), f1) / MSE(copy, f1), copy error = mean of f0/f2 vs f1.
    Copy error grows with motion like VFI error, so the ratio is not a motion-amount
    proxy (global SSIM was: rho=-0.68 vs actor motion in v2). Coherent motion that
    interpolation explains scores near 100; jerky or popping motion scores near 0.

    Two independent floors are required, because `weight_sum` alone cannot tell
    "few strongly-active triples" from "many barely-moving ones":

    - `weight_sum >= MIN_WEIGHT_SUM` -- the weighted mean is scale-free, so one
      marginal triple otherwise scores like a hundred (v3.1: a wrist clip with
      weight_sum 0.35 over 81 triples reported AMC 83).
    - `n_active >= MIN_ACTIVE_PAIRS` -- weights are `min(motion/ACTIVE_PX, 1)`, so a
      clip at 0.113 px/frame over 88 pairs still sums to weight 4.95 and passed the
      first floor while NO pair was ever active, reporting AMC 78.3 at
      active_rate 0.000. Meanwhile a genuinely active clip (17.6 px, 11/11 pairs
      active) also sums to only 5.0 -- so raising MIN_WEIGHT_SUM would kill the real
      signal and keep the artifact. Counting actually-active pairs separates them:
      it drops the two rate=0 clips and keeps the 11-active-pair one (1% of data).
    - `n_active >= MIN_ACTIVE_FRAC * len(motion)` -- the absolute count above is not
      length-aware, so on long clips it is nearly free: the completed v3.2 run still
      had three clips at AMC 71-82 with active_rate 0.02-0.04 (5 active pairs out of
      245). Requiring 5% of the clip's own valid pairs removes those without touching
      any clip that actually moves (calibrated: every clip this drops is <= 0.67
      px/source-frame).
    """
    import cv2
    import numpy as np
    gains, weights, copy_errs = [], [], []
    for j, vfi_err in enumerate(smooth.get('mse_per_triple', [])):
        f0, f1, f2 = (cv2.imread(paths[2*j+k]).astype(np.float64) for k in range(3))
        copy_err = 0.5 * (np.mean((f0 - f1) ** 2) + np.mean((f2 - f1) ** 2))
        values = [motion[k] for k in (2*j, 2*j+1) if k in motion]
        w = min(float(np.mean(values)) / ACTIVE_PX, 1.0) if values else 0.0
        if copy_err < COPY_ERR_FLOOR:
            w = 0.0
        gains.append(float(np.clip(1 - vfi_err / max(copy_err, 1e-6), 0.0, 1.0)))
        weights.append(w); copy_errs.append(float(copy_err))
    denom = float(np.sum(weights))
    # `motion` is the clip's valid-pair set, i.e. the same denominator `active_rate` is
    # averaged over, so the fractional floor is expressed against it.
    min_active = max(MIN_ACTIVE_PAIRS, MIN_ACTIVE_FRAC * len(motion))
    value = (float(np.dot(weights, gains) / denom * 100)
             if interval_ok and denom >= MIN_WEIGHT_SUM and n_active >= min_active
             else None)
    return value, denom, gains, copy_errs


def excluded_row(row):
    """True for manifest rows the benchmark deliberately does NOT score.

    Ctrl-World multiview exists in two renders for bimanual and humanoid: a 3-view one
    (`model=ctrlworld`: cam_high + 2 wrists) and a 4-view grid (`model=ctrlworld_4view_grid`:
    adds cam_low / a third exterior). Only the 4-view render is used in the paper, so the
    3-view rows are 4,365 clips of pure waste (~5.3 h of wall time across the wave).

    Two things this must NOT do:

    * Do not touch `single_arm`. Its Ctrl-World render only EXISTS as `model=ctrlworld`
      (3 views: exterior_1_left, exterior_2_left, wrist_left) -- there is no 4-view
      variant, so excluding `model == 'ctrlworld'` globally would silently delete the
      whole single_arm/MV/ctrlworld cell from the paper grid.
    * Do not filter this out of manifest.json instead. The shard is `rows[offset::stride]`
      BY POSITION, so dropping rows renumbers every machine's slice; boxes that had not
      restarted would partition a different list and some live rows would be owned by
      NOBODY. Skipping here keeps every index stable -- same reason SKIP-GONE works this
      way.

    The GT anchor (`gt_ctrlworld`) goes with it: it only exists to pair against the 3-view
    predictions, and the 4-view render carries its own `gt_ctrlworld_4view_grid`.
    """
    return (row.get('embodiment') in ('bimanual', 'humanoid')
            and row.get('model') in ('ctrlworld', 'gt_ctrlworld'))


def record_error(args, row):
    """Persist a failure without letting the failure handler itself kill the worker.

    `dump()` can raise EDQUOT, which is precisely the condition that gets us here -- and
    an exception thrown from inside the `except` block escapes the retry loop and takes
    the whole worker process down (measured: a machine silently dropping from 8 workers to
    3 keeps going at a third of the rate with nothing in the log to say so).
    """
    try:
        dump(args.out/'errors'/f'{row["id"]}.json',
             {**row, 'traceback': traceback.format_exc()})
    except OSError:
        print(f'ERRLOG-FAILED {row["id"]} (cannot write errors/, continuing)', flush=True)


def worker(args):
    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2); cv2.setNumThreads(1)
    from compute_phyworld import score_qual, write_frames
    from metrics.quality.quality_control_probe import IQA, Detector
    from metrics.dynamics.motion_mag_smooth_probe import Flow, build_vfimamba, motion_smoothness
    from metrics.dynamics.motion_accuracy_gdino import Grounder, probe
    from metrics.phyworld.schema import EMBODIMENT_PHRASES
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    ck = ROOT / 'checkpoints/metrics/worldscore'
    tp = 'worldscore/benchmark/metrics/third_party'
    cfg = f'{tp}/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    gd = str(ck / 'groundingdino_swint_ogc.pth')
    iqa = IQA('cuda'); det = Detector(cfg, gd, 'cuda')
    flow = Flow(f'{tp}/SEA-RAFT/config/eval/spring-M.json', str(ck/'Tartan-C-T-TSKH-spring540x960-M.pth'), 'cuda')
    flow.a.scale = 0
    vfi, padder = build_vfimamba('cuda')
    GlobalHydra.instance().clear(); hydra.initialize_config_module('sam2', version_base='1.2')
    sc = 'configs/sam2.1/sam2.1_hiera_b+.yaml'
    img = SAM2ImagePredictor(build_sam2(sc, str(ck/'sam2.1_hiera_base_plus.pt'), device='cuda', apply_postprocessing=False))
    vp = build_sam2_video_predictor(sc, str(ck/'sam2.1_hiera_base_plus.pt'), device='cuda', apply_postprocessing=False)
    ground = Grounder(cfg, gd, 'cuda')
    rows = json.loads((args.out/'manifest.json').read_text())
    if args.only_ids:
        # Restrict to a precomputed id subset (scripts/nr_table_coverage.py writes one).
        # The paper grid is a (embodiment, model, split) x method table, which needs only
        # `--target` clips per cell -- measured 20,035 ids at 300/cell versus 55,383 for
        # the whole manifest, i.e. the table is reachable ~2.7x sooner. Filtering here
        # rather than writing a second manifest.json keeps ONE inventory for the run dir:
        # every id, pair_key and shard offset stays identical, so a later full wave simply
        # resumes and the already-scored subset counts towards it.
        keep = set(json.loads(Path(args.only_ids).read_text()))
        before = len(rows)
        rows = [r for r in rows if r['id'] in keep]
        if not rows:
            raise SystemExit(f'--only-ids {args.only_ids} matched 0 of {before} manifest rows')
        print(f'ONLY-IDS {len(rows)} of {before} manifest rows', flush=True)
    # Walk in sha256-id order, NOT manifest order. The manifest is grouped (all GT +
    # Ctrl-World bimanual first, DreamGen at 539, DreamDojo at 599), so in manifest
    # order an untested model family is only reached hours in and partial results are
    # single-embodiment. Sorting by id is a deterministic, stateless shuffle: it covers
    # all 5 models / 3 mounts within 56 clips and all 28 groups within 140, so the run
    # fails fast and is representative if stopped early.
    rows.sort(key=lambda r: r['id'], reverse=args.reverse)

    def score(row, started):
        with tempfile.TemporaryDirectory(prefix='dense_nr_') as td:
            tmp = Path(td)
            arr, protocol = prepare(row, tmp/'analysis.mp4')
            interval = protocol['source_interval']
            result = {**row, **protocol, 'protocol_version': PROTOCOL,
                      'actor_flicker_definition': 'experimental FB-filtered photometric residual, not validated paper reproduction'}
            prompt = task_text(row['instruction'])
            result['text_condition_prompt'] = prompt
            result.update(score_qual(arr, prompt, row['embodiment'], iqa, det, 5))
            # Text alignment and text-derived entity coverage are condition
            # metrics, not universal NR metrics. Do not score image-only
            # methods against a text instruction they never received.
            result['text_condition_applicable'] = text_conditioned(row['model'])
            if not result['text_condition_applicable'] or not row['instruction']:
                result['task_lock'] = None
                result['entity_presence'] = None
            result['text_condition_alignment'] = result['task_lock']
            # Phrase-derived entity detection remains a diagnostic proxy until
            # a task-object ontology declares visibility expectations.
            result['condition_entity_coverage_proxy'] = result['entity_presence']
            result['entity_presence'] = None
            paths = write_frames(arr, tmp/'frames')
            smooth = motion_smoothness(paths, vfi, padder, 'cuda', TTA=False, stride=1)
            result['motion_continuity_global_ssim'] = float(smooth['ssim']*100)
            actor = probe(str(tmp/'analysis.mp4'), 'pred', str(tmp), ground, img, vp, flow,
                          prompt or 'object .',
                          ' . '.join(EMBODIMENT_PHRASES[row['embodiment']])+' .',
                          'cuda', stride=1, compute_flicker=True)
            per_frame = actor.get('per_frame', [])
            robot = bool(actor['n_arm_dets'])
            motion, median = actor_pair_motion(per_frame, row['camera_mount'], interval)
            # `per_frame` is ALREADY filtered by the probe (it skips pairs with empty
            # fg/bg or a failed ego fit), so len(motion)/len(per_frame) can never
            # expose tracking loss. Measure against the analysis pairs that exist.
            n_pairs_possible = max(len(protocol['sampled_source_frames']) - 1, 1)
            tracked_frac = len(per_frame) / n_pairs_possible
            motion_ok = (robot and per_frame
                         and len(motion) >= MIN_VALID_PAIR_FRAC * len(per_frame)
                         and tracked_frac >= MIN_VALID_PAIR_FRAC)
            values = np.array(list(motion.values()), dtype=float)
            result['actor_motion_raw'] = float(values.mean()) if motion_ok else None
            result['actor_motion_median_raw'] = float(np.mean(list(median.values()))) if motion_ok else None
            result['actor_motion_active_rate'] = float(np.mean(values >= ACTIVE_PX)) if motion_ok else None
            result['actor_motion_px_per_second'] = (float(values.mean() * protocol['source_fps'])
                                                    if motion_ok else None)
            result['actor_motion_statistic'] = {'static': 'subject_p95_uncorrected',
                                                'head': 'subject_p95_ego_translation',
                                                'wrist': 'frame_median'}[row['camera_mount']]
            result['actor_seed_frame'] = actor.get('seed_frame')
            result['actor_motion_valid_pair_frac'] = (len(motion) / len(per_frame)) if per_frame else 0.0
            result['actor_tracked_pair_frac'] = tracked_frac
            # Stored so the AMC activity gate is reproducible post-hoc from the record
            # (per-triple weights are not serialized).
            n_active = int((values >= ACTIVE_PX).sum()) if motion_ok else 0
            result['actor_n_active_pairs'] = n_active if motion_ok else None
            amc, weight_sum, gains, copy_errs = continuity(paths, smooth, motion,
                                                           bool(motion_ok), n_active)
            result['active_motion_continuity'] = amc
            result['active_motion_continuity_weight_sum'] = weight_sum
            result['actor_flicker_raw'] = actor.get('actor_flicker_raw') if robot else None
            # E3 candidate statistics, all computed in the same probe pass. Stored so
            # the choice of flicker definition is a post-hoc, zero-GPU decision (the
            # E4 lesson). `actor_flicker_raw` above stays the v1 statistic so this run
            # reproduces the old column exactly and the two are directly comparable.
            result['actor_flicker_variants'] = (actor.get('actor_flicker_variants')
                                                if robot else None)
            result['actor_n_arm_dets'] = actor['n_arm_dets']
            result['actor_validity'] = 'robot_detected' if robot else ('object_only' if actor['n_object_dets'] else 'invalid')
            result['paper_scale'] = {
                'visual_integrity': 'CLIP-IQA+ p10 x100; higher is better',
                'text_condition_alignment': 'CLIPScore [0,100]; higher is better; text-conditioned only',
                'actor_motion_raw': f'px/source-frame; static: subject p95, head: ego-compensated subject p95 (bg>={MIN_BG_COV_PCT:g}%), wrist: frame median; descriptive',
                'actor_motion_active_rate': f'fraction of valid actor pairs at >={ACTIVE_PX} px/source-frame',
                'active_motion_continuity': 'activity-weighted VFI gain over frame copy x100, [0,100]; higher is better',
                'actor_flicker_raw': 'flow-aligned RGB residual [0,1]; lower is better; experimental',
            }
            result['actor_per_frame'] = per_frame
            result['continuity_per_triple'] = {'vfi_mse': smooth.get('mse_per_triple', []),
                                               'copy_mse': copy_errs, 'gain': gains}
            result['elapsed_seconds'] = time.time()-started
            # Strict JSON and independent atomic files avoid summary races.
            def clean(v):
                if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
                if isinstance(v, list): return [clean(x) for x in v]
                if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
                return v
            return clean(result)

    # Global stride partition. `--machines M --machine k` makes machine k own shard
    # indices [k*W, (k+1)*W) of a global M*W stride, so M machines partition the manifest
    # EXACTLY once with no overlap and no coordination. This supersedes --reverse for
    # anything past 2 machines: --reverse relies on two queues meeting in the middle and
    # re-doing whatever they overlap on, which is fine for 2 but has no 3-way analogue.
    stride = args.workers * args.machines
    offset = args.machine * args.workers + args.shard
    for row in rows[offset::stride]:
        dest = args.out/'results'/f'{row["id"]}.json'
        if dest.exists():
            continue
        # Rows whose source video no longer exists are DEAD, not failures. 567 rows of
        # this manifest point into four stale trees that were deleted on 2026-09-19
        # before the HF upload (svg1/pisa/radial `.../dreamgen/*/iter_000090000`, whose
        # canonical tag is `_static16fps`, and `fastercache/single_arm/.../dreamdojo/
        # */iter_000100000`, which held bimanual renders misfiled under single_arm).
        # The manifest was inventoried BEFORE that cleanup, so every one of those rows
        # reaches `prepare()`, gets fps<=0 from cv2 and lands in errors/ as a
        # "permanent failure" -- 445 fake entries that bury the handful of genuinely
        # corrupt source videos and make `failed (perm)` useless as a signal.
        #
        # Skipping here rather than filtering manifest.json is deliberate: the shard is
        # `rows[offset::stride]` by POSITION, so deleting rows renumbers every machine's
        # slice. Machines load the manifest once at startup and a 9-day wave restarts
        # constantly, so a shortened manifest would have boxes partitioning two
        # different lists at the same time -- live rows owned by NOBODY. Same outcome,
        # no resharding.
        if not os.path.exists(row['path']):
            print(f'SKIP-GONE {row["relative"]} {row["view"]}', flush=True)
            continue
        if excluded_row(row):
            print(f'SKIP-EXCLUDED {row["relative"]} {row["view"]}', flush=True)
            continue
        started = time.time()
        print(f'START {row["relative"]} {row["view"]}', flush=True)
        for attempt in range(4):
            try:
                result = score(row, started)
                dump(dest, result)
                (args.out/'errors'/f'{row["id"]}.json').unlink(missing_ok=True)
                print('DONE '+row['id']+' '+str({k:result.get(k) for k in KEYS}), flush=True)
                break
            except (torch.OutOfMemoryError, OSError) as exc:
                # Two DIFFERENT transient faults, both of which must be retried rather
                # than recorded, because a manifest row is owned by exactly ONE worker
                # shard: whatever this worker gives up on is never revisited by the other
                # forward machines, so "record and move on" loses the clip outright.
                #
                #   OOM            several workers peaking together.
                #   EDQUOT/ENOSPC  the fileset is SHARED with other jobs, so the wall
                #                  opens again as soon as they free space. Measured
                #                  2026-09-19: a squeeze from 15:50 failed 6.0% of clips
                #                  (30 of 504 in 76 min) while `df` still showed 383T
                #                  free -- projected ~4.1k clips lost over the rest of
                #                  the wave. Writes succeeded again minutes later, which
                #                  is exactly the case retrying fixes.
                quota = isinstance(exc, OSError) and exc.errno in (errno.EDQUOT, errno.ENOSPC)
                if isinstance(exc, OSError) and not quota:
                    # Any other OSError (bad path, unreadable video) is permanent; fall
                    # through to the same handling as a non-transient exception.
                    record_error(args, row)
                    traceback.print_exc()
                    break
                torch.cuda.empty_cache()
                if attempt == 3:
                    record_error(args, row)
                    traceback.print_exc()
                else:
                    # Quota needs ANOTHER job to release space, which takes longer than a
                    # GPU peak to clear, so it gets the slower backoff (5/10/15 min vs
                    # 1/2/3 min).
                    wait = (300 if quota else 60) * (attempt + 1)
                    print(f'RETRY {"QUOTA" if quota else "OOM"} {attempt+1}/3 '
                          f'in {wait}s {row["id"]}', flush=True)
                    time.sleep(wait)
            except Exception:
                record_error(args, row)
                traceback.print_exc()
                torch.cuda.empty_cache()
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'results/dense_nr_v1')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--shard', type=int, default=-1)
    ap.add_argument('--inventory-only', action='store_true')
    ap.add_argument('--smoke-paired-episodes', type=int, default=0,
                    help='build a small GT/pred validation manifest instead of the full inventory')
    ap.add_argument('--smoke-per-group', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--machines', type=int, default=1,
                    help='TOTAL number of machines/GPUs sharing this run dir. With '
                         '--machine k, machine k processes rows[k*W + s :: W*machines] '
                         'for each of its W workers, so M machines partition the '
                         'manifest exactly once -- no overlap, no coordination, and '
                         'every machine still walks in id order so each one stays '
                         'representative if stopped early. Use this instead of '
                         '--reverse for 3+ machines (--reverse only has a 2-way form).')
    ap.add_argument('--machine', type=int, default=0,
                    help='0-based index of THIS machine, must be < --machines.')
    ap.add_argument('--reverse', action='store_true',
                    help='walk the id-sorted manifest backwards. Lets a SECOND machine '
                         'join an in-flight run without restarting the first: the two '
                         'converge in the middle and the `dest.exists()` skip plus '
                         'atomic per-id writes make the overlap harmless. Never '
                         'reshard a running queue instead -- `rows[shard::workers]` '
                         'already covers the whole manifest, so any added shard index '
                         'would duplicate an existing one.')
    ap.add_argument('--methods', nargs='+', default=['dense'],
                    help='sampling_dataset/<name> trees to inventory, e.g. '
                         '--methods dense dicache pisa. Co-located GT is only taken '
                         'from the dense tree (accel trees hold identical copies). '
                         'Accel rows keep a method-free pair_key so they pair with '
                         'their dense counterpart.')
    ap.add_argument('--reuse-manifest', action='store_true',
                    help='fail instead of building a manifest if one is absent. Guards '
                         'the second machine against silently writing a different '
                         'manifest.json (inventory is filesystem-dependent) into a '
                         'shared run directory.')
    ap.add_argument('--only-ids', default=None,
                    help='JSON list of manifest ids to restrict this run to. Use it to '
                         'score only what the paper grid needs (build the list with '
                         'scripts/nr_table_coverage.py --target N). Applied AFTER the '
                         'manifest is loaded and BEFORE sharding, so every machine must '
                         'pass the SAME file or the stride partition no longer covers '
                         'the subset exactly.')
    args = ap.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.machine < args.machines:
        raise SystemExit(f'--machine {args.machine} must be in [0, {args.machines})')
    if args.machines > 1 and args.reverse:
        # Both mechanisms partition the same manifest; combining them would make machine
        # k walk a stride it does not own from the other end and duplicate work.
        raise SystemExit('--reverse is the 2-machine mechanism; with --machines use '
                         '--machine k on every machine and drop --reverse')
    if args.shard >= 0:
        worker(args); return
    if args.reuse_manifest and not (args.out/'manifest.json').exists():
        raise SystemExit(f'--reuse-manifest: no manifest at {args.out}/manifest.json')
    if args.inventory_only or (not args.reuse_manifest and not (args.out/'manifest.json').exists()):
        rows = inventory(args.out, tuple(args.methods))
        if args.smoke_paired_episodes:
            rows = smoke_manifest(rows, args.smoke_paired_episodes, args.smoke_per_group, args.seed)
            dump(args.out/'manifest.json', rows)
            print(f'SMOKE {len(rows)} view-clips', flush=True)
    if args.inventory_only: return
    processes = []
    tag = f'rev' if args.reverse else (f'm{args.machine}' if args.machines > 1 else 'fwd')
    for i in range(args.workers):
        # Log name carries machine/direction so machines sharing the run dir cannot
        # append into each other's logs.
        log = (args.out/f'worker_{tag}_{i}.log').open('a')
        cmd = [sys.executable, '-u', __file__, '--out', str(args.out),
               '--workers', str(args.workers), '--shard', str(i),
               '--machines', str(args.machines), '--machine', str(args.machine)]
        if args.only_ids:
            cmd += ['--only-ids', str(args.only_ids)]
        if args.reverse:
            cmd.append('--reverse')
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        processes.append(p); log.close()
    codes = [p.wait() for p in processes]
    if args.reverse or args.machine > 0:
        # Only the owning queue (forward / machine 0) runs the plotting hook, so machines
        # cannot race on figures/ and coverage.json.
        dump(args.out/f'queue_finished_{tag}.json',
             {'worker_exit_codes': codes, 'time': time.time()})
        if any(codes): raise SystemExit(1)
        return
    subprocess.run([sys.executable, str(ROOT/'scripts/plot_dense_nr.py'), '--run', str(args.out)], check=True)
    dump(args.out/'queue_finished.json', {'worker_exit_codes':codes, 'time':time.time()})
    if any(codes): raise SystemExit(1)


if __name__ == '__main__': main()
