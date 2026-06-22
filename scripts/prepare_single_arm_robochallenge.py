#!/usr/bin/env python3
"""Convert Traly/RoboChallenge-lerobot (LeRobot v3.0) tasks into the
single_arm/singleview {makovian,non_makovian} LeRobot-lite layout used by
this benchmark.

RoboChallenge v3.0 packs many episodes into a single mp4 / parquet file.
This script:
  1. Downloads the per-task meta/episodes parquet, data parquet, and the
     global_image video files for the selected episodes.
  2. Slices each episode out of its packed mp4 (by from/to timestamp) and
     downsamples to the target fps (default 15).
  3. Writes one mp4 + one parquet per episode plus aggregated meta files,
     matching the directory convention of datasets/single_arm/singleview.

Output layout (per split):
  {OUT_ROOT}/{split}/
    data/chunk-XXX/episode_NNNNNN.parquet
    videos/chunk-XXX/observation.images.global/episode_NNNNNN.mp4
    meta/{info.json, episodes.jsonl, tasks.jsonl, modality.json}
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "Traly/RoboChallenge-lerobot"
VIDEO_KEY_SRC = "observation.global_image"      # static external camera
VIDEO_KEY_DST = "observation.images.global"     # benchmark naming
CHUNK_SIZE = 1000

# split -> list of (task_name, num_episodes, max_seconds)
SELECTION = {
    "makovian": [
        ("put_cup_on_coaster", 60, 25.0),
        ("turn_on_light_switch", 60, 25.0),
    ],
    "non_makovian": [
        ("press_three_buttons", 60, 28.0),
        ("wipe_the_table", 60, 50.0),
    ],
}


def quat_wxyz_to_euler_xyz(quat):
    """Convert a [w, x, y, z] unit quaternion to XYZ Euler angles (radians)."""
    w, x, y, z = quat
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return [float(roll), float(pitch), float(yaw)]


def canonicalize_pose7(vec):
    """Return a canonical 7D [xyz(3), euler_xyz(3), gripper(1)] vector.

    RoboChallenge tasks come in two layouts:
      - 7D: [xyz(3), euler_xyz(3), gripper(1)] -> returned unchanged.
      - 8D: [xyz(3), quat_wxyz(4), gripper(1)] -> quaternion converted to euler.
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.shape[0] == 7:
        return arr.tolist()
    if arr.shape[0] == 8:
        xyz = arr[0:3].tolist()
        euler = quat_wxyz_to_euler_xyz(arr[3:7])
        gripper = [float(arr[7])]
        return xyz + euler + gripper
    raise ValueError(f"Unexpected state/action dim {arr.shape[0]}, expected 7 or 8")


def canonicalize_column(values):
    """Apply canonicalize_pose7 to every row of a list-of-vectors column."""
    return [canonicalize_pose7(v) for v in values]


def hf(path, cache):
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=cache)


def list_meta_episode_files(task, files):
    return sorted(f for f in files if f.startswith(f"{task}/meta/episodes/"))


def load_episode_meta(task, files, cache):
    """Return list of per-episode dicts with the fields we need."""
    rows = []
    for ef in list_meta_episode_files(task, files):
        p = hf(ef, cache)
        tbl = pq.read_table(p)
        cols = tbl.column_names
        want = [
            "episode_index", "tasks", "length",
            "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index",
            f"videos/{VIDEO_KEY_SRC}/chunk_index",
            f"videos/{VIDEO_KEY_SRC}/file_index",
            f"videos/{VIDEO_KEY_SRC}/from_timestamp",
            f"videos/{VIDEO_KEY_SRC}/to_timestamp",
        ]
        want = [c for c in want if c in cols]
        d = tbl.select(want).to_pylist()
        rows.extend(d)
    return rows


def ffprobe_fps(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    ).stdout.strip()
    if "/" in out:
        a, b = out.split("/")
        return float(a) / float(b)
    return float(out) if out else 0.0


def ffprobe_nframes(video_path):
    """Exact frame count of an mp4 (decodes, so accurate)."""
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    ).stdout.strip()
    try:
        return int(out)
    except ValueError:
        return 0


def align_table_len(tbl, target_len, fps):
    """Trim or pad (repeat last row) a parquet table to target_len rows and
    rebuild the per-frame index columns."""
    n = tbl.num_rows
    if n == target_len:
        return tbl
    d = tbl.to_pydict()
    if target_len < n:
        d = {k: v[:target_len] for k, v in d.items()}
    else:
        pad = target_len - n
        for k, v in d.items():
            d[k] = v + [v[-1]] * pad
    d["frame_index"] = list(range(target_len))
    d["index"] = list(range(target_len))
    d["timestamp"] = [i / fps for i in range(target_len)]
    return pa.table(d)


def slice_episode_video(src_mp4, t0, t1, out_mp4, target_fps):
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.0, t1 - t0)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{t0:.6f}", "-i", str(src_mp4), "-t", f"{dur:.6f}",
        "-vf", f"fps={target_fps}",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_mp4),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4.name}: {r.stderr[-300:]}")


def downsample_frames(arr_len, src_fps, dst_fps):
    """Return integer indices selecting dst_fps frames from a src_fps clip."""
    if dst_fps >= src_fps:
        return np.arange(arr_len)
    n_out = max(1, int(round(arr_len * dst_fps / src_fps)))
    idx = np.linspace(0, arr_len - 1, n_out).round().astype(int)
    return np.unique(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="datasets/single_arm/singleview")
    ap.add_argument("--cache", default="/tmp/robochallenge_dl")
    ap.add_argument("--target-fps", type=float, default=15.0)
    ap.add_argument("--splits", nargs="*", default=list(SELECTION.keys()))
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, override per-task episode count (for testing)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")

    out_root = Path(args.out_root)
    cache = args.cache

    for split in args.splits:
        print(f"\n========== SPLIT: {split} ==========")
        split_dir = out_root / split
        (split_dir / "meta").mkdir(parents=True, exist_ok=True)

        episodes_jsonl = []
        tasks_jsonl = {}          # task_str -> task_index
        out_ep_idx = 0
        src_fps_seen = None

        for task, n_want, max_sec in SELECTION[split]:
            if args.limit > 0:
                n_want = args.limit
            print(f"\n--- {task}: want {n_want} eps (<= {max_sec}s) ---")
            info = json.load(open(hf(f"{task}/meta/info.json", cache)))
            src_fps = float(info["fps"])
            src_fps_seen = src_fps
            meta = load_episode_meta(task, all_files, cache)

            # filter by length, keep shortest-first for clean clips
            cand = [m for m in meta if m["length"] / src_fps <= max_sec]
            cand.sort(key=lambda m: m["length"])
            cand = cand[:n_want]
            print(f"    candidates after filter: {len(cand)}")

            # group selected episodes by their packed data/video file to
            # minimise downloads
            data_files = {}   # (chunk,file) -> local path
            video_files = {}  # (chunk,file) -> local path

            for m in cand:
                tcol = m.get("tasks") or [""]
                task_str = tcol[0] if tcol else ""
                if task_str not in tasks_jsonl:
                    tasks_jsonl[task_str] = len(tasks_jsonl)
                t_idx = tasks_jsonl[task_str]

                # --- ensure data parquet downloaded ---
                dkey = (m["data/chunk_index"], m["data/file_index"])
                if dkey not in data_files:
                    dp = hf(f"{task}/data/chunk-{dkey[0]:03d}/file-{dkey[1]:03d}.parquet", cache)
                    data_files[dkey] = dp
                # --- ensure video downloaded ---
                vk = (m[f"videos/{VIDEO_KEY_SRC}/chunk_index"],
                      m[f"videos/{VIDEO_KEY_SRC}/file_index"])
                if vk not in video_files:
                    vp = hf(f"{task}/videos/{VIDEO_KEY_SRC}/chunk-{vk[0]:03d}/file-{vk[1]:03d}.mp4", cache)
                    video_files[vk] = vp

                chunk = out_ep_idx // CHUNK_SIZE
                out_stem = f"episode_{out_ep_idx:06d}"

                # --- slice + downsample video ---
                out_mp4 = (split_dir / "videos" / f"chunk-{chunk:03d}" /
                           VIDEO_KEY_DST / f"{out_stem}.mp4")
                t0 = m[f"videos/{VIDEO_KEY_SRC}/from_timestamp"]
                t1 = m[f"videos/{VIDEO_KEY_SRC}/to_timestamp"]
                slice_episode_video(video_files[vk], t0, t1, out_mp4, args.target_fps)

                # --- slice + downsample parquet rows ---
                full = pq.read_table(data_files[dkey])
                rows = full.slice(m["dataset_from_index"],
                                  m["dataset_to_index"] - m["dataset_from_index"])
                sel = downsample_frames(rows.num_rows, src_fps, args.target_fps)
                rows = rows.take(pa.array(sel))
                # rewrite frame_index / episode_index / task_index
                n = rows.num_rows
                d = rows.to_pydict()
                # Canonicalize state/action to a uniform 7D layout
                # [xyz(3), euler_xyz(3), gripper(1)]. Some RoboChallenge tasks
                # (e.g. press_three_buttons) store an 8D [xyz, quat_wxyz, gripper]
                # representation; convert the quaternion to Euler so every task
                # shares the same dimensionality required by training.
                for col in ("observation.state", "action"):
                    if col in d:
                        d[col] = canonicalize_column(d[col])
                d["frame_index"] = list(range(n))
                d["episode_index"] = [out_ep_idx] * n
                d["task_index"] = [t_idx] * n
                d["index"] = list(range(n))
                d["timestamp"] = [i / args.target_fps for i in range(n)]
                out_tbl = pa.table(d)
                out_pq = (split_dir / "data" / f"chunk-{chunk:03d}" / f"{out_stem}.parquet")
                out_pq.parent.mkdir(parents=True, exist_ok=True)

                # Align parquet length to the actual number of frames ffmpeg
                # produced (rounding can differ by ±1) so video and states stay
                # frame-aligned for benchmark metrics.
                actual_frames = ffprobe_nframes(out_mp4)
                if actual_frames and actual_frames != n:
                    out_tbl = align_table_len(out_tbl, actual_frames, args.target_fps)
                    n = actual_frames
                pq.write_table(out_tbl, out_pq)

                episodes_jsonl.append({
                    "episode_index": out_ep_idx,
                    "tasks": [task_str],
                    "length": n,
                    "source_task": task,
                    "source_episode": m["episode_index"],
                })
                out_ep_idx += 1
                if out_ep_idx % 20 == 0:
                    print(f"    ... {out_ep_idx} episodes done")

        # ---- write meta ----
        write_meta(split_dir, episodes_jsonl, tasks_jsonl, args.target_fps, src_fps_seen)
        print(f"\n[{split}] TOTAL episodes written: {out_ep_idx}")


def write_meta(split_dir, episodes_jsonl, tasks_jsonl, fps, src_fps):
    meta = split_dir / "meta"
    with (meta / "episodes.jsonl").open("w") as f:
        for e in episodes_jsonl:
            f.write(json.dumps(e) + "\n")
    with (meta / "tasks.jsonl").open("w") as f:
        for task_str, idx in sorted(tasks_jsonl.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": idx, "task": task_str}) + "\n")

    total_frames = sum(e["length"] for e in episodes_jsonl)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "RoboChallenge",
        "total_episodes": len(episodes_jsonl),
        "total_frames": total_frames,
        "total_tasks": len(tasks_jsonl),
        "total_videos": len(episodes_jsonl),
        "total_chunks": (len(episodes_jsonl) // CHUNK_SIZE) + 1,
        "chunks_size": CHUNK_SIZE,
        "fps": fps,
        "source_fps": src_fps,
        "splits": {"train": f"0:{len(episodes_jsonl)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7], "names": None},
            "action": {"dtype": "float32", "shape": [7], "names": None},
            VIDEO_KEY_DST: {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": fps, "video.height": 480, "video.width": 640,
                    "video.channels": 3, "video.codec": "h264",
                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))

    modality = {
        "state": {"state": {"original_key": "observation.state",
                            "start": 0, "end": 7, "rotation_type": None,
                            "absolute": True, "dtype": "float32", "range": None}},
        "action": {"action": {"original_key": "action",
                              "start": 0, "end": 7, "rotation_type": None,
                              "absolute": True, "dtype": "float32", "range": None}},
        "video": {"global": {"original_key": VIDEO_KEY_DST}},
    }
    (meta / "modality.json").write_text(json.dumps(modality, indent=4))


if __name__ == "__main__":
    main()
