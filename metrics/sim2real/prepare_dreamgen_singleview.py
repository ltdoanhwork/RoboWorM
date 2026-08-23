"""Build paired GT clips + metadata for DreamGen SINGLEVIEW predictions.

The DreamGen singleview output tree is deliberately GT-less: it holds a flat
``episode_XXXXXX.mp4`` plus a ``.txt`` prompt sidecar, and its input tree holds
only the conditioning first frame. The sim2real probe needs a *paired* GT clip on
the SAME pixel grid as the prediction, so this script reconstructs one from the
source LeRobot dataset.

Three things make this non-trivial, and getting any of them wrong silently
produces a meaningless score rather than an error:

1. **Grid mismatch.** GR1 source video is 1280x800 @ 20 fps; the prediction is
   768x432 @ 16 fps. The prepare script that produced the conditioning frame used
   a plain anamorphic ``scale=768:432`` (no crop, no letterbox), so the prediction
   is vertically squashed relative to the source. We reproduce exactly that
   transform - ``scale=768:432,fps=16`` - which is also the validated GT-alignment
   protocol used by ``scripts/run_motion_accuracy_matched.sh``. Cropping instead
   would misalign every mask.

2. **Length mismatch.** Generation length is decoupled from GT length: it is
   ``93 + 92*(n_chunks-1)`` frames (185 for 2 chunks) regardless of how long the
   episode actually is. A 169-frame 20 fps episode is only 135 frames at 16 fps,
   so the tail of the prediction has NO ground truth. We record ``gt_frames_16fps``
   and let the caller cap comparison there (the tracker already truncates to
   ``min(len(gt), len(pred))``), so the unpaired tail is never scored.

3. **Ego-camera motion.** ``observation.images.ego_view_freq20`` is head-mounted.
   Episodes whose task string starts with ``"unlocked waist: "`` pan with the
   robot (measured: ~25 px of drift at 768 wide for episode_000200), which breaks
   both background subtraction and the "object moved but nothing touched it"
   premise of the levitation predicate. Episodes tagged ``"locked waist: "`` are
   genuinely static (measured: 0.0 px drift). ``--waist`` selects which, and the
   emitted metadata carries ``waist`` / ``camera_static`` so downstream code can
   refuse to claim levitation on a moving camera.
"""

import os
import re
import sys
import json
import argparse
import subprocess

_REPO = os.environ.get("REPO")
if _REPO and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

HUMANOID_CAM = "observation.images.ego_view_freq20"
SOURCE_FPS = 20.0
# DreamGen singleview grid, from scripts/prepare_humanoid_singleview_dreamgen_short.py
DG_WIDTH, DG_HEIGHT, DG_FPS = 768, 432, 16


def find_source_video(split_dir, ep_idx, cam=HUMANOID_CAM):
    """episode index -> source mp4, globbing chunks (never hardcode chunk-000)."""
    videos_root = os.path.join(split_dir, "videos")
    if not os.path.isdir(videos_root):
        return None
    for chunk in sorted(os.listdir(videos_root)):
        cand = os.path.join(videos_root, chunk, cam, f"episode_{ep_idx:06d}.mp4")
        if os.path.exists(cand):
            return cand
    return None


def load_episode_meta(split_dir):
    """episode_index -> {'task', 'waist', 'length', 'trajectory_id'}."""
    path = os.path.join(split_dir, "meta", "episodes.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            task = (d.get("tasks") or [""])[0]
            if task.startswith("unlocked waist:"):
                waist = "unlocked"
            elif task.startswith("locked waist:"):
                waist = "locked"
            else:
                waist = "unknown"
            out[int(d["episode_index"])] = {
                "task": re.sub(r"^(un)?locked waist:\s*", "", task),
                "waist": waist,
                "length": d.get("length"),
                "trajectory_id": d.get("trajectory_id"),
            }
    return out


def probe_frames(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except (ValueError, IndexError):
        return None


def build_gt_clip(src, out_mp4, width=DG_WIDTH, height=DG_HEIGHT, fps=DG_FPS):
    """Reproduce the prepare script's transform: anamorphic scale + fps resample."""
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
         "-vf", f"scale={width}:{height}:flags=bicubic,fps={fps}",
         "-an", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out_mp4],
        check=True)
    return out_mp4


def read_prompt(pred_mp4):
    """DreamGen writes the prompt to a .txt sidecar next to the mp4."""
    txt = os.path.splitext(pred_mp4)[0] + ".txt"
    if not os.path.exists(txt):
        return None
    body = open(txt).read()
    m = re.search(r"\[Prompt\]\s*\n(.+?)(?:\n\[|\Z)", body, re.S)
    return m.group(1).strip() if m else None


def prepare_episode(pred_mp4, split_dir, out_dir, cam=HUMANOID_CAM, ep_meta=None):
    """-> dict with gt/pred paths + metadata path, or None if no source GT."""
    ep_idx = int(re.search(r"(\d+)", os.path.basename(pred_mp4)).group(1))
    src = find_source_video(split_dir, ep_idx, cam=cam)
    if src is None:
        return None
    meta_all = ep_meta if ep_meta is not None else load_episode_meta(split_dir)
    info = meta_all.get(ep_idx, {})

    ep_dir = os.path.join(out_dir, f"episode_{ep_idx:06d}")
    os.makedirs(ep_dir, exist_ok=True)
    gt_mp4 = os.path.join(ep_dir, "gt_matched.mp4")
    if not os.path.exists(gt_mp4):
        build_gt_clip(src, gt_mp4)

    n_gt = probe_frames(gt_mp4)
    n_pred = probe_frames(pred_mp4)
    instruction = read_prompt(pred_mp4) or info.get("task") or ""

    meta = {
        "episode_id": f"episode_{ep_idx:06d}",
        "instruction": instruction,
        "fps": DG_FPS,
        "num_frames": n_gt,
        "gt_frames_16fps": n_gt,
        "pred_frames": n_pred,
        # frames of the prediction that have any GT to be compared against
        "paired_frames": None if (n_gt is None or n_pred is None) else min(n_gt, n_pred),
        "source_video": os.path.abspath(src),
        "source_fps": SOURCE_FPS,
        "source_length": info.get("length"),
        "camera": cam,
        "waist": info.get("waist", "unknown"),
        "camera_static": info.get("waist") == "locked",
        "trajectory_id": info.get("trajectory_id"),
    }
    meta_path = os.path.join(ep_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"episode_id": meta["episode_id"], "gt": gt_mp4, "pred": os.path.abspath(pred_mp4),
            "metadata": meta_path, "meta": meta}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True,
                    help="e.g. sampling_dataset/dense/humanoid/output/singleview/"
                         "dreamgen/makovian/iter_000090000_static16fps")
    ap.add_argument("--split_dir", required=True,
                    help="e.g. datasets/humanoid/singleview/makovian")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam", default=HUMANOID_CAM)
    ap.add_argument("--waist", default="locked",
                    choices=["locked", "unlocked", "any"],
                    help="locked = static ego camera (required for bg-subtraction "
                         "and for levitation to be meaningful)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--episodes", default=None, help="comma-separated episode ids")
    args = ap.parse_args()

    ep_meta = load_episode_meta(args.split_dir)
    preds = sorted(f for f in os.listdir(args.pred_dir) if f.endswith(".mp4"))
    if args.episodes:
        want = {int(re.search(r"(\d+)", e).group(1)) for e in args.episodes.split(",")}
        preds = [f for f in preds if int(re.search(r"(\d+)", f).group(1)) in want]

    rows, skipped = [], {"no_source": 0, "waist": 0}
    for f in preds:
        ep_idx = int(re.search(r"(\d+)", f).group(1))
        info = ep_meta.get(ep_idx, {})
        if args.waist != "any" and info.get("waist") != args.waist:
            skipped["waist"] += 1
            continue
        r = prepare_episode(os.path.join(args.pred_dir, f), args.split_dir,
                            args.out, cam=args.cam, ep_meta=ep_meta)
        if r is None:
            skipped["no_source"] += 1
            continue
        rows.append(r)
        m = r["meta"]
        print(f"[prep] {m['episode_id']} waist={m['waist']} "
              f"gt={m['gt_frames_16fps']}f pred={m['pred_frames']}f "
              f"paired={m['paired_frames']}f  {m['instruction'][:58]!r}")
        if args.limit and len(rows) >= args.limit:
            break

    index = os.path.join(args.out, "index.json")
    os.makedirs(args.out, exist_ok=True)
    with open(index, "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "meta"} for r in rows], fh, indent=2)
    print(f"\nprepared {len(rows)} episodes (skipped: {skipped}) -> {index}")


if __name__ == "__main__":
    main()
