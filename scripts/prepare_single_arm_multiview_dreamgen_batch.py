"""
Prepare batch_input.json for DreamGen multiview (GR00T droid) from a LeRobot-format
single-arm multiview dataset under datasets/single_arm/multiview/{makovian,non_makovian}/.

Grid (2×2), same quadrant size as humanoid / bimanual prep:

  ┌─────────────────────┬─────────────────────┐
  │  TL: exterior_1_left│  TR: exterior_2_left│
  ├─────────────────────┼─────────────────────┤
  │  BL: wrist_left     │  BR: black          │
  └─────────────────────┴─────────────────────┘

Camera keys: observation.images.exterior_1_left, exterior_2_left, wrist_left.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CAM_EXT1 = "observation.images.exterior_1_left"
CAM_EXT2 = "observation.images.exterior_2_left"
CAM_WRIST = "observation.images.wrist_left"

CAM_LAYOUT = [
    ("TL", CAM_EXT1),
    ("TR", CAM_EXT2),
    ("BL", CAM_WRIST),
    ("BR", None),
]

PROMPT_TEMPLATE = (
    "A multi-view video shows that a robot {task}. "
    "The video is split into four views: "
    "The top-left view shows the robotic arm from the left side, "
    "the top-right view shows it from another left exterior viewing angle, "
    "the bottom-left view shows a first-person perspective from the "
    "robot's end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

QUAD_W, QUAD_H = 320, 180


def infer_category(dataset_path: Path) -> str | None:
    for part in dataset_path.parts:
        if part == "non_makovian":
            return "non_makovian"
        if part == "makovian":
            return "makovian"
    return None


def ffmpeg_first_frame(video_path: Path, out_jpg: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"scale={QUAD_W}:{QUAD_H}",
            "-vframes",
            "1",
            "-q:v",
            "2",
            str(out_jpg),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_jpg.exists()


def build_2x2_grid(frame_paths: list, out_png: Path) -> bool:
    inputs: list[str] = []
    scale_filters: list[str] = []
    idx = 0
    for slot_path in frame_paths:
        if slot_path is not None:
            inputs += ["-i", str(slot_path)]
        else:
            inputs += [
                "-f",
                "lavfi",
                "-i",
                f"color=c=black:s={QUAD_W}x{QUAD_H}:r=1",
            ]
        scale_filters.append(f"[{idx}:v]scale={QUAD_W}:{QUAD_H}[v{idx}]")
        idx += 1
    filter_complex = (
        ";".join(scale_filters)
        + ";[v0][v1]hstack[top];[v2][v3]hstack[bottom];[top][bottom]vstack[out]"
    )
    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex, "-map", "[out]", "-vframes", "1", str(out_png)]
    )
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and out_png.exists()


def read_episodes_jsonl(meta_dir: Path) -> dict[int, str]:
    ep_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"
    task_map: dict[int, str] = {}
    if tasks_file.exists():
        with open(tasks_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                tidx = obj.get("task_index", obj.get("index"))
                task = obj.get("task", obj.get("description", ""))
                if tidx is not None and task:
                    task_map[int(tidx)] = task

    episodes: dict[int, str] = {}
    if not ep_file.exists():
        return episodes

    with open(ep_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = obj.get("episode_index")
            if ep_idx is None:
                continue
            tasks = obj.get("tasks", [])
            if tasks:
                episodes[int(ep_idx)] = tasks[0]
                continue
            tidx = obj.get("task_index")
            if tidx is not None and int(tidx) in task_map:
                episodes[int(ep_idx)] = task_map[int(tidx)]
                continue
            episodes[int(ep_idx)] = obj.get("description", "performing a manipulation task")

    return episodes


def find_video(dataset_path: Path, cam_key: str, ep_idx: int) -> Path | None:
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return None
    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        candidate = chunk_dir / cam_key / f"episode_{ep_idx:06d}.mp4"
        if candidate.exists():
            return candidate
    return None


def episode_indices_with_all_cameras(dataset_path: Path) -> list[int]:
    """Episode indices that have an mp4 under every required camera (any chunk)."""
    videos_root = dataset_path / "videos"
    required = [CAM_EXT1, CAM_EXT2, CAM_WRIST]
    per_cam: list[set[int]] = []
    for cam in required:
        ids: set[int] = set()
        if videos_root.is_dir():
            for chunk_dir in sorted(videos_root.iterdir()):
                if not chunk_dir.is_dir():
                    continue
                cam_dir = chunk_dir / cam
                if not cam_dir.is_dir():
                    continue
                for p in cam_dir.glob("episode_*.mp4"):
                    try:
                        idx = int(p.stem.split("_", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    ids.add(idx)
        per_cam.append(ids)
    if not per_cam:
        return []
    common = set.intersection(*per_cam)
    return sorted(common)


def output_video_is_done(path: Path) -> bool:
    """Non-empty mp4 on disk counts as already generated."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Root of one category, e.g. datasets/single_arm/multiview/makovian",
    )
    parser.add_argument(
        "--input_dir",
        default="./sampling_dataset/dense/single_arm/input/multiview/dreamgen",
        help="Where to save 2×2 grid PNGs and batch_input.json parent.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for generated videos. Default: <input_dir>/../../output/multiview/dreamgen/<category>/",
    )
    parser.add_argument(
        "--batch_json",
        default="",
        help="Explicit batch_input.json path. Default: <input_dir>/<category>/batch_input.json",
    )
    parser.add_argument(
        "--category",
        default="",
        help="makovian|non_makovian (inferred from --dataset_path if empty).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Consider at most this many episode indices after filtering (sorted order). Omit for all.",
    )
    parser.add_argument(
        "--min_episode_index",
        type=int,
        default=None,
        help="Skip episode indices below this (useful when low-index episodes have no mp4 shards on disk).",
    )
    parser.add_argument(
        "--episode_source",
        choices=("meta", "disk"),
        default="disk",
        help="meta: iterate all episode_index rows in meta/episodes.jsonl (warns if mp4 missing). "
        "disk: only episodes that actually have episode_XXXXXX.mp4 for all three cameras under videos/ "
        "(recommended when meta lists more episodes than you downloaded). Default: disk.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Reuse existing grid PNGs without regenerating.",
    )
    parser.add_argument(
        "--force_regenerate_outputs",
        action="store_true",
        help="Include episodes in batch_input.json even when output mp4 already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    category = args.category.strip() or infer_category(dataset_path)
    if category not in ("makovian", "non_makovian"):
        print(
            "[ERROR] Could not infer category; pass --category makovian or non_makovian.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = Path(args.input_dir).resolve() / category
    batch_json = (
        Path(args.batch_json).resolve()
        if args.batch_json
        else input_dir / "batch_input.json"
    )
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        # <input_dir>/<cat>/ is under .../single_arm/input/multiview/dreamgen/<cat>
        # → mirror to .../single_arm/output/multiview/dreamgen/<cat>
        input_base = Path(args.input_dir).resolve() / category
        single_arm_root = input_base.parent.parent.parent.parent
        output_dir = single_arm_root / "output" / "multiview" / "dreamgen" / category

    meta_dir = dataset_path / "meta"
    episodes_meta = read_episodes_jsonl(meta_dir)

    if args.episode_source == "meta":
        if not episodes_meta:
            print(f"[ERROR] No episodes in {meta_dir}/episodes.jsonl", file=sys.stderr)
            sys.exit(1)
        episode_indices = sorted(episodes_meta.keys())
    else:
        episode_indices = episode_indices_with_all_cameras(dataset_path)
        if not episode_indices:
            print(
                f"[ERROR] episode_source=disk: no episode had all three cameras under {dataset_path / 'videos'}",
                file=sys.stderr,
            )
            sys.exit(1)
        n_meta = len(episodes_meta)
        print(
            f"[INFO] episode_source=disk: {len(episode_indices)} episodes with all cameras on disk; "
            f"meta/episodes.jsonl rows (for captions): {n_meta}",
            flush=True,
        )

    if args.min_episode_index is not None:
        episode_indices = [i for i in episode_indices if i >= args.min_episode_index]
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = input_dir / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset   : {dataset_path}")
    print(f"Category  : {category}")
    print(f"Ep. source: {args.episode_source}")
    print(f"Episodes  : {len(episode_indices)}")
    print(f"Input dir : {input_dir}")
    print(f"Output    : {output_dir}")
    print(f"Batch     : {batch_json}")
    print()

    batch_input: list[dict] = []
    skipped = 0
    skipped_existing_output = 0
    skip_outputs = not args.force_regenerate_outputs
    prompt_template = PROMPT_TEMPLATE

    for ep_idx in episode_indices:
        task_desc = episodes_meta.get(
            ep_idx,
            "performing a manipulation task",
        )
        task_clean = task_desc.rstrip(".,;").strip() or "performing a manipulation task"
        prompt = prompt_template.format(task=task_clean)
        output_video = str(output_dir / f"episode_{ep_idx:06d}.mp4")
        out_path = Path(output_video)
        grid_png = input_dir / f"episode_{ep_idx:06d}.png"

        if skip_outputs and output_video_is_done(out_path):
            print(f"[SKIP] episode_{ep_idx:06d} (output already exists: {out_path.name})")
            skipped_existing_output += 1
            continue

        if args.skip_existing and grid_png.exists():
            batch_input.append(
                {"input_video": str(grid_png), "prompt": prompt, "output_video": output_video}
            )
            print(f"[SKIP] episode_{ep_idx:06d} (grid already exists)")
            continue

        frame_paths: list = []
        all_ok = True
        for _slot, cam_key in CAM_LAYOUT:
            if cam_key is None:
                frame_paths.append(None)
                continue

            video_path = find_video(dataset_path, cam_key, ep_idx)
            if video_path is None:
                print(f"[WARN] episode_{ep_idx:06d}: missing video for {cam_key}")
                all_ok = False
                break

            tmp_jpg = tmp_dir / f"ep{ep_idx:06d}_{cam_key.replace('.', '_')}.jpg"
            if not ffmpeg_first_frame(video_path, tmp_jpg):
                print(f"[WARN] episode_{ep_idx:06d}: ffmpeg failed on {video_path.name}")
                all_ok = False
                break
            frame_paths.append(tmp_jpg)

        if not all_ok:
            skipped += 1
            continue

        if not build_2x2_grid(frame_paths, grid_png):
            print(f"[WARN] episode_{ep_idx:06d}: grid stitch failed")
            skipped += 1
            continue

        print(f"[OK]   episode_{ep_idx:06d} → {grid_png.name}")
        batch_input.append(
            {"input_video": str(grid_png), "prompt": prompt, "output_video": output_video}
        )

    for tmp_file in tmp_dir.glob("*.jpg"):
        tmp_file.unlink(missing_ok=True)

    if not batch_input:
        if skipped_existing_output > 0 and skipped == 0:
            print(
                f"\n[INFO] All {skipped_existing_output} episode(s) already have output video; "
                "nothing to generate.",
                flush=True,
            )
            with open(batch_json, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            print(f"Wrote empty batch: {batch_json}")
            return
        print("\n[ERROR] No batch inputs created.", file=sys.stderr)
        sys.exit(1)

    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_input, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(batch_input)} episodes in batch.")
    if skipped:
        print(f"      {skipped} episodes skipped (input/grid issues).")
    if skipped_existing_output:
        print(f"      {skipped_existing_output} episodes skipped (output already exists).")
    print(f"Batch: {batch_json}")


if __name__ == "__main__":
    main()
