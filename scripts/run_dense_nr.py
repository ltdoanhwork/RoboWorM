"""Resumable dense NR inventory and persistent GPU workers; no dense reference.

Run through run_dense_nr.sh for vendored dependency paths.
"""
import argparse
import collections
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
PROTOCOL = 'dense-nr-v3.1-closed-world'
KEYS = ['visual_integrity', 'text_condition_alignment', 'actor_motion_raw',
        'actor_motion_active_rate', 'active_motion_continuity', 'actor_flicker_raw']
TEXT_CONDITIONED_MODELS = {'dreamgen', 'ctrlworld', 'ctrlworld_4view_grid', 'gt'}
# GT per-view inputs. View order verified against pred strips: GT view_i == per_view[i].
# bimanual/humanoid ctrlworld GT is a view subset of the 4-view GT (same episodes).
GT_SOURCES = {
    'bimanual': ('ctrlworld_4view_grid', ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']),
    'humanoid': ('ctrlworld_4view_grid', ['cam_head_rgb', 'cam_front_rgb', 'cam_left_wrist_rgb', 'cam_right_wrist_rgb']),
    'single_arm': ('ctrlworld', ['exterior_1_left', 'exterior_2_left', 'wrist_left']),
}
ACTIVE_PX = 1.0           # px/source-frame, on the per-pair motion statistic below
MIN_BG_COV_PCT = 10.0     # head ego estimate needs enough background pixels
MIN_VALID_PAIR_FRAC = 0.5
MIN_WEIGHT_SUM = 3.0      # AMC needs >=3 activity-equivalent triples, else N/A
COPY_ERR_FLOOR = 2.0      # uint8 MSE; below this frame change is codec noise


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


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


def row_for(p, rel, model, embodiment, split, view, layout, vi, n_views, instruction, method):
    identity = str(rel) + ':' + view
    episode = next((x for x in reversed(rel.parts) if x.startswith('episode_')), rel.stem)
    return dict(id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                path=str(p), relative=str(rel), model=model,
                embodiment=embodiment, split=split, view=view,
                camera_mount=camera_mount(embodiment, view),
                layout=layout, view_index=vi, n_views=n_views,
                instruction=instruction, method=method,
                pair_key=f'{embodiment}/{split}/{episode}/{view}')


def inventory(out):
    from phyworld_index import _instruction_from_dir, _instruction_from_txt
    dense = ROOT / 'sampling_dataset/dense'
    rows = []
    # GT reference rows first: same analysis protocol, used as plausibility anchor.
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
    for p in sorted(dense.rglob('*.mp4')):
        rel = p.relative_to(dense)
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
                                len(views), instruction, 'dense'))
    dump(out / 'manifest.json', rows)
    print(f'INVENTORY {len(rows)} view-clips / {len(set(r["path"] for r in rows))} videos '
          f'({sum(r["method"] == "gt" for r in rows)} GT view-clips)', flush=True)
    return rows


def task_of(row):
    """Task template: episode dir `episode_<task>__<n>` or DreamGen `<task>/episode_<n>`."""
    parts = Path(row['relative']).parts
    ep = next((i for i in range(len(parts) - 1, -1, -1) if parts[i].startswith('episode_')), None)
    if ep is not None:
        name = parts[ep][len('episode_'):]
        if '__' in name:
            return name.rsplit('__', 1)[0]
        parent = parts[ep - 1] if ep > 0 else ''
        if not (parent.startswith('iter_') or parent in ('makovian', 'non_makovian')):
            return parent
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

    Both parts spread over task templates and splits for debugging diversity.
    """
    rng = random.Random(seed)
    by_key = collections.defaultdict(dict)
    for r in rows:
        by_key[r['pair_key']][r['model']] = r
    chosen = {}
    for emb in GT_SOURCES:
        src = GT_SOURCES[emb][0]
        keys = sorted(k for k, v in by_key.items() if k.startswith(emb + '/') and 'gt' in v and src in v)
        episodes = sorted({k.rsplit('/', 1)[0] for k in keys})
        first = {ep: by_key[next(k for k in keys if k.startswith(ep + '/'))]['gt'] for ep in episodes}
        picked = spread_sample(episodes, paired_episodes, rng,
                               key=lambda ep: (task_of(first[ep]), first[ep]['split']))
        for ep in picked:
            for k in keys:
                if k.startswith(ep + '/'):
                    for m in ('gt', src):
                        chosen[by_key[k][m]['id']] = by_key[k][m]
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


def continuity(paths, smooth, motion, interval_ok):
    """Motion-normalized VFI gain on activity-weighted triples.

    gain_j = 1 - MSE(VFI(f0,f2), f1) / MSE(copy, f1), copy error = mean of f0/f2 vs f1.
    Copy error grows with motion like VFI error, so the ratio is not a motion-amount
    proxy (global SSIM was: rho=-0.68 vs actor motion in v2). Coherent motion that
    interpolation explains scores near 100; jerky or popping motion scores near 0.

    `weight_sum` is the activity-equivalent triple count. Requiring MIN_WEIGHT_SUM
    stops a single marginal triple from carrying the whole clip: the weighted mean
    is scale-free, so N triples of weight w and one triple of weight w score alike
    (observed: a wrist clip with weight_sum 0.35 over 81 triples reported AMC 83).
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
    value = (float(np.dot(weights, gains) / denom * 100)
             if interval_ok and denom >= MIN_WEIGHT_SUM else None)
    return value, denom, gains, copy_errs


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
    ck = ROOT / 'checkpoints/worldscore'
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
            result['text_condition_applicable'] = row['model'] in TEXT_CONDITIONED_MODELS
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
            amc, weight_sum, gains, copy_errs = continuity(paths, smooth, motion, bool(motion_ok))
            result['active_motion_continuity'] = amc
            result['active_motion_continuity_weight_sum'] = weight_sum
            result['actor_flicker_raw'] = actor.get('actor_flicker_raw') if robot else None
            result['actor_n_arm_dets'] = actor['n_arm_dets']
            result['actor_validity'] = 'robot_detected' if robot else ('object_only' if actor['n_object_dets'] else 'invalid')
            result['paper_scale'] = {
                'visual_integrity': 'CLIP-IQA+ p10 x100; higher is better',
                'text_condition_alignment': 'CLIPScore [0,100]; higher is better; text-conditioned only',
                'actor_motion_raw': 'px/source-frame; static: subject p95, head: ego-compensated subject p95 (bg>=10%), wrist: frame median; descriptive',
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

    for row in rows[args.shard::args.workers]:
        dest = args.out/'results'/f'{row["id"]}.json'
        if dest.exists():
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
            except torch.OutOfMemoryError:
                # Transient when several workers peak together; retry before recording.
                torch.cuda.empty_cache()
                if attempt == 3:
                    dump(args.out/'errors'/f'{row["id"]}.json', {**row, 'traceback':traceback.format_exc()})
                    traceback.print_exc()
                else:
                    print(f'OOM retry {attempt+1} {row["id"]}', flush=True)
                    time.sleep(60 * (attempt + 1))
            except Exception:
                dump(args.out/'errors'/f'{row["id"]}.json', {**row, 'traceback':traceback.format_exc()})
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
    args = ap.parse_args(); args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    if args.shard >= 0:
        worker(args); return
    if args.inventory_only or not (args.out/'manifest.json').exists():
        rows = inventory(args.out)
        if args.smoke_paired_episodes:
            rows = smoke_manifest(rows, args.smoke_paired_episodes, args.smoke_per_group, args.seed)
            dump(args.out/'manifest.json', rows)
            print(f'SMOKE {len(rows)} view-clips', flush=True)
    if args.inventory_only: return
    processes = []
    for i in range(args.workers):
        log = (args.out/f'worker_{i}.log').open('a')
        p = subprocess.Popen([sys.executable, '-u', __file__, '--out', str(args.out),
                              '--workers', str(args.workers), '--shard', str(i)], stdout=log, stderr=subprocess.STDOUT)
        processes.append(p); log.close()
    codes = [p.wait() for p in processes]
    subprocess.run([sys.executable, str(ROOT/'scripts/plot_dense_nr.py'), '--run', str(args.out)], check=True)
    dump(args.out/'queue_finished.json', {'worker_exit_codes':codes, 'time':time.time()})
    if any(codes): raise SystemExit(1)


if __name__ == '__main__': main()
