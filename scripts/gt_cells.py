#!/usr/bin/env python3
"""The six grid cells that have predictions but no ground truth, and how to rebuild it.

One module so the alignment probe and the renderer cannot drift apart: if the probe
validates a mapping, the renderer uses that exact same code path to produce the GT.

Each cell knows three things:
  find_source(clip)  -- the LeRobot source video(s) behind a prediction
  load_probe(clip)   -- small greyscale source frames, for alignment search
  read_source(clip)  -- full-resolution source frames, for rendering

single_arm multiview is the awkward one: its prediction is a 2x2 mosaic, so the source
is three videos plus a black quadrant, and GT is only comparable if it is composed the
same way (prepare_single_arm_multiview_dreamgen_batch.py: TL exterior_1, TR exterior_2,
BL wrist, BR black).
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# --- source lookup -------------------------------------------------------------------

def _find_flat(dataset: str, split: str, cam: str, ep: str) -> Path | None:
    """Datasets whose meta/ sits at the split root (humanoid, single_arm)."""
    hits = glob.glob(str(REPO / 'datasets' / dataset / split / 'videos' / 'chunk-*' / cam / f'{ep}.mp4'))
    return Path(hits[0]) if hits else None


def _find_per_task(dataset: str, split: str, cam: str, task: str, ep: str) -> Path | None:
    """Datasets split into one LeRobot dir per task (bimanual singleview)."""
    hits = glob.glob(str(REPO / 'datasets' / dataset / split / task / 'videos' / 'chunk-*' / cam / f'{ep}.mp4'))
    return Path(hits[0]) if hits else None


# --- frame readers -------------------------------------------------------------------

def _read_ffmpeg(path: Path) -> np.ndarray:
    """Decode every frame via an ffmpeg rawvideo pipe. Fallback for PyAV decode failures.

    Measured need: `datasets/single_arm/multiview/makovian/videos/chunk-010/
    observation.images.exterior_2_left/episode_010407.mp4` is AV1 that PyAV aborts on
    (`InvalidDataError: avcodec_send_packet()`) while ffmpeg's decoder walks it end to
    end and returns all 485 frames -- byte-for-byte 83,808,000 = 485 x 320 x 180 x 3,
    the same length as the two sibling views. Without this fallback that one bad view
    costs the whole episode its GT, which is why single_arm multiview sat at 256/257.
    """
    import subprocess

    w, h = 0, 0
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0:s=,', str(path)],
        capture_output=True, text=True)
    try:
        w, h = (int(x) for x in probe.stdout.strip().split(',')[:2])
    except ValueError:
        raise RuntimeError(f'ffprobe could not size {path}: {probe.stderr.strip()}')

    proc = subprocess.run(
        ['ffmpeg', '-v', 'error', '-i', str(path),
         '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
        capture_output=True)
    frame_bytes = w * h * 3
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError(f'ffmpeg decoded no frames from {path}: {proc.stderr[:200]!r}')
    return np.frombuffer(proc.stdout[:n * frame_bytes], np.uint8).reshape(n, h, w, 3)


def _read_all(path: Path, size=None, grey=False) -> np.ndarray:
    """Every frame as BGR uint8 (or small float32 grey).

    PyAV, not cv2.VideoCapture: the single_arm multiview sources are AV1, which this
    OpenCV build refuses ("Failed to get pixel format") and silently returns zero frames
    for. PyAV decodes them correctly, and using one decoder everywhere keeps the probe
    and the renderer honest about frame counts.

    One AV1 source defeats PyAV too, so a failed PyAV decode falls back to an ffmpeg
    pipe (see `_read_ffmpeg`) instead of losing the episode.
    """
    import av

    out = []
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                out.append(frame.to_ndarray(format='bgr24'))
    except av.error.FFmpegError:
        out = list(_read_ffmpeg(path))
    if not out:
        out = list(_read_ffmpeg(path))

    proc = []
    for img in out:
        if grey:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        proc.append(img.astype(np.float32) if grey else img)
    if not proc:
        raise RuntimeError(f'no frames decoded from {path}')
    return np.stack(proc)


def _dims(path: Path) -> tuple[int, int, int, float]:
    """(n_frames, w, h, fps) with an exact frame count -- container metadata can lie."""
    import av

    n = 0
    w = h = 0
    fps = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        w, h = stream.codec_context.width, stream.codec_context.height
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        try:
            for _ in container.decode(video=0):
                n += 1
        except av.error.FFmpegError:
            n = 0
    if n == 0:                     # same PyAV-hostile AV1 case as _read_all
        n = len(_read_ffmpeg(path))
    return n, w, h, fps


QUAD_W, QUAD_H = 320, 180        # prepare_single_arm_multiview_dreamgen_batch.py


def _mosaic(views: list[np.ndarray | None], quad_w: int, quad_h: int) -> np.ndarray:
    """2x2 grid from up to four aligned frame stacks; None means a black quadrant."""
    n = max(len(v) for v in views if v is not None)
    tiles = []
    for v in views:
        if v is None:
            tiles.append(np.zeros((n, quad_h, quad_w, 3), np.uint8))
        else:
            r = np.stack([cv2.resize(f, (quad_w, quad_h), interpolation=cv2.INTER_AREA)
                          for f in v[:n]])
            if len(r) < n:                       # views can be a frame short; hold the last
                r = np.concatenate([r, np.repeat(r[-1:], n - len(r), axis=0)])
            tiles.append(r)
    top = np.concatenate([tiles[0], tiles[1]], axis=2)
    bottom = np.concatenate([tiles[2], tiles[3]], axis=2)
    return np.concatenate([top, bottom], axis=1)


# --- cell definitions ----------------------------------------------------------------

def _single_source_cell(dataset, cam, per_task, src_fps):
    def find_source(clip):
        if per_task:
            return _find_per_task(dataset, clip['split'], cam, clip['task'], clip['ep'])
        return _find_flat(dataset, clip['split'], cam, clip['ep'])

    def load_probe(clip):
        src = find_source(clip)
        if src is None:
            raise RuntimeError(f'no source video for {clip["key"]}')
        n, w, h, fps = _dims(src)
        return _read_all(src, size=(96, 54), grey=True), n, w, h, fps

    def read_source(clip):
        src = find_source(clip)
        if src is None:
            raise RuntimeError(f'no source video for {clip["key"]}')
        return _read_all(src)

    return {'find_source': find_source, 'load_probe': load_probe,
            'read_source': read_source, 'src_fps': src_fps}


def _mosaic_cell(dataset, cams, src_fps):
    def find_source(clip):
        found = [_find_flat(dataset, clip['split'], cam, clip['ep']) for cam in cams]
        return found if all(f is not None for f in found) else None

    def load_probe(clip):
        srcs = find_source(clip)
        if srcs is None:
            raise RuntimeError(f'missing one of {len(cams)} views for {clip["key"]}')
        n, w, h, fps = _dims(srcs[0])
        views = [_read_all(s) for s in srcs] + [None]
        grid = _mosaic(views, QUAD_W, QUAD_H)
        small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (96, 54),
                                     interpolation=cv2.INTER_AREA).astype(np.float32)
                          for f in grid])
        return small, len(grid), grid.shape[2], grid.shape[1], fps

    def read_source(clip):
        srcs = find_source(clip)
        if srcs is None:
            raise RuntimeError(f'missing one of {len(cams)} views for {clip["key"]}')
        views = [_read_all(s) for s in srcs] + [None]
        return _mosaic(views, QUAD_W, QUAD_H)

    return {'find_source': find_source, 'load_probe': load_probe,
            'read_source': read_source, 'src_fps': src_fps}


CELLS: dict[str, dict] = {
    'dreamgen_bimanual_sv': {
        **_single_source_cell('bimanual/singleview', 'observation.images.cam_high',
                              per_task=True, src_fps=30.0),
        'pred_glob': 'bimanual/output/singleview/dreamgen/{split}/iter_000110000_bimanual16fps/*.mp4',
        'model_fps': 16.0,
        'spatial': 'stretch',
    },
    'dreamgen_humanoid_sv': {
        **_single_source_cell('humanoid/singleview', 'observation.images.ego_view_freq20',
                              per_task=False, src_fps=20.0),
        'pred_glob': 'humanoid/output/singleview/dreamgen/{split}/iter_000090000_static16fps/episode_*.mp4',
        'model_fps': 16.0,
        'spatial': 'stretch',
    },
    'dreamgen_single_arm_sv': {
        **_single_source_cell('single_arm/singleview', 'observation.images.global',
                              per_task=False, src_fps=15.0),
        'pred_glob': 'single_arm/output/singleview/dreamgen/{split}/iter_000090000_static16fps/global/episode_*.mp4',
        'model_fps': 16.0,
        'spatial': 'crop_square',
    },
    'dreamgen_single_arm_mv': {
        **_mosaic_cell('single_arm/multiview',
                       ['observation.images.exterior_1_left',
                        'observation.images.exterior_2_left',
                        'observation.images.wrist_left'], src_fps=15.0),
        'pred_glob': 'single_arm/output/multiview/dreamgen/{split}/episode_*.mp4',
        'model_fps': 10.0,
        'spatial': 'stretch',
    },
}

EP_RE = re.compile(r'(episode_\d+)$')

FRAMES_PER_CHUNK = 93            # prepare_singleview_dreamgen_batch.py
FRAMES_PER_EXTRA_CHUNK = 92      # 93 minus the 1-frame conditioning overlap


def chunks_for(target_frames: float) -> int:
    if target_frames <= FRAMES_PER_CHUNK:
        return 1
    import math
    return math.ceil((target_frames - FRAMES_PER_CHUNK) / FRAMES_PER_EXTRA_CHUNK) + 1


def pred_len_for(n_chunks: int) -> int:
    return FRAMES_PER_CHUNK + (n_chunks - 1) * FRAMES_PER_EXTRA_CHUNK


def solve_convention(n_src: int, n_pred: int, src_fps: float, model_fps: float) -> dict | None:
    """How prediction frame index maps onto source frame index, and how far GT can reach.

    The rate follows from what the model emits, not from prepare's bookkeeping: these
    checkpoints generate `model_fps` video (run_bimanual_singleview_dreamgen_16fps.sh:
    "trained on cam_high view at 16fps (original 30fps -> resampled)"; MODEL_FPS=16 in
    prepare_singleview_dreamgen_batch.py), so one prediction frame is 1/model_fps of real
    time and

        src_idx = round(pred_idx * src_fps / model_fps)

    The alternative -- that the model emits source-rate video (rate 1.0) -- was tested
    directly on pixels over 24 bimanual clips, first chunk only: `resample` won on 21 of
    24 with a +1.13 dB mean margin. Bimanual is the cell with leverage (1.875 vs 1.0); on
    humanoid (1.25 vs 1.0) and single_arm multiview (1.5 vs 1.0) the same test came out
    within 0.2 dB, i.e. not sensitive enough to separate them, and did not contradict the
    rule. So the rate is taken from the documented training fps and is pixel-confirmed
    where confirmation is possible.

    `n_gt` is where the SOURCE runs out, not where the prediction does. Predictions
    routinely outrun their episode -- prepare over-counted chunks for part of the
    bimanual set, leaving 1013-frame rollouts over 33 s of footage. Padding GT with a
    held frame would invent a reference for content that has no ground truth, so GT stops
    at the last real frame and the scorer's own min(len) truncation trims the prediction
    to match.
    """
    if n_src < 1 or n_pred < 1:
        return None
    rate = src_fps / model_fps
    n_gt = min(n_pred, int((n_src - 1) / rate) + 1)
    return {'rule': 'resample', 'rate': rate, 'n_src': n_src, 'n_pred': n_pred,
            'n_gt': n_gt, 'pred_frames_without_gt': n_pred - n_gt,
            'coverage': round(n_gt / n_pred, 4)}


def parse_pred(pred: Path, split: str) -> dict | None:
    """Prediction filename -> the episode (and task, when the name carries one)."""
    stem = pred.stem
    if stem.endswith('_gt'):
        return None
    task = None
    if '__' in stem:
        task, _, stem = stem.partition('__')
    m = EP_RE.match(stem)
    if not m:
        return None
    return {'key': pred.stem, 'ep': m.group(1), 'task': task, 'split': split, 'pred': pred}


def src_index_map(n_gt: int, n_src: int, rate: float) -> np.ndarray:
    """Source frame index for each GT frame, all of them real (see solve_convention)."""
    idx = np.rint(np.arange(n_gt) * rate).astype(np.int64)
    return np.clip(idx, 0, n_src - 1)


def spatial_transform(frames: np.ndarray, w: int, h: int, mode: str) -> np.ndarray:
    """Put source frames into the prediction's framing, using that cell's measured geometry.

    Each cell's transform was identified by comparing prediction frame 0 (the
    conditioning frame) against candidate transforms of source frame 0, over 8 clips per
    cell. The winners are unambiguous -- the right transform scores 32-38 dB and every
    wrong one scores 10-18 dB:

        bimanual SV    stretch      38.1 dB   (vs 17.7 square crop)
        humanoid SV    stretch      32.7 dB   (vs 14.7 aspect crop)
        single_arm MV  stretch      37.4 dB   (source is already 16:9)
        single_arm SV  crop_square  37.9 dB   (vs 17.9 stretch)

    single_arm singleview is the one that needed finding: its prediction is a centred
    480x480 crop of the 640x480 source, stretched back to 640x480. A plain resize scores
    17.9 dB there, which looks like ordinary generation error rather than a geometry bug
    -- an ECC affine fit exposed it (x' = 0.749x + 80, correlation 0.9998). Without the
    crop, every PSNR in that cell would have been understated by ~20 dB.
    """
    if mode == 'stretch':
        pass
    elif mode == 'crop_square':
        sh, sw = frames.shape[1:3]
        m = min(sh, sw)
        y0, x0 = (sh - m) // 2, (sw - m) // 2
        frames = frames[:, y0:y0 + m, x0:x0 + m]
    elif mode == 'crop_pred_aspect':
        sh, sw = frames.shape[1:3]
        ar = w / h
        if sw / sh > ar:
            cw = int(round(sh * ar)); x0 = (sw - cw) // 2
            frames = frames[:, :, x0:x0 + cw]
        else:
            ch = int(round(sw / ar)); y0 = (sh - ch) // 2
            frames = frames[:, y0:y0 + ch]
    else:
        raise ValueError(f'unknown spatial mode {mode!r}')

    if frames.shape[1] == h and frames.shape[2] == w:
        return np.ascontiguousarray(frames)
    interp = cv2.INTER_AREA if frames.shape[1] > h else cv2.INTER_CUBIC
    return np.stack([cv2.resize(f, (w, h), interpolation=interp) for f in frames])


def list_clips(cell: dict, split: str, method: str = 'dense') -> list[dict]:

    pattern = str(REPO / 'sampling_dataset' / method / cell['pred_glob'].format(split=split))
    clips = []
    for p in sorted(glob.glob(pattern)):
        info = parse_pred(Path(p), split)
        if info is not None:
            clips.append(info)
    return clips
