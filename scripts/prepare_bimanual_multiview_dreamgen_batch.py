"""
Prepare batch_input.json for DreamGen multiview (GR00T droid variant) from a
LeRobot-format bimanual ALOHA multiview dataset.

Grid layout (2×2, same quadrant resolution as humanoid/DROID prep scripts):

  Default (2 static + 1 dynamic wrist + black quadrant):
  ┌──────────────┬──────────────┐
  │  TL: high    │  TR: low     │
  ├──────────────┼──────────────┤
  │  BL: wrist*  │  BR: black   │
  └──────────────┴──────────────┘
  * default left wrist; use --dynamic_wrist right for right only.

  With --four_views (both wrists, no black):
  ┌──────────────┬──────────────┐
  │  TL: high    │  TR: low     │
  ├──────────────┼──────────────┤
  │  BL: left    │  BR: right   │
  └──────────────┴────────────┘

Camera keys: observation.images.cam_high, cam_low, cam_left_wrist, cam_right_wrist.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CAM_HIGH = "observation.images.cam_high"
CAM_LOW = "observation.images.cam_low"
CAM_LEFT = "observation.images.cam_left_wrist"
CAM_RIGHT = "observation.images.cam_right_wrist"

QUAD_W, QUAD_H = 320, 180

PROMPT_3VIEW_LEFT = (
    "A multi-view video shows that a bimanual robot {task}. "
    "The video is split into four views: "
    "The top-left view shows an overhead high camera view of the workspace, "
    "the top-right view shows a lower camera perspective, "
    "the bottom-left view shows a first-person perspective from the robot's "
    "left end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

PROMPT_3VIEW_RIGHT = (
    "A multi-view video shows that a bimanual robot {task}. "
    "The video is split into four views: "
    "The top-left view shows an overhead high camera view of the workspace, "
    "the top-right view shows a lower camera perspective, "
    "the bottom-left view shows a first-person perspective from the robot's "
    "right end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

PROMPT_4VIEW = (
    "A multi-view video shows that a bimanual robot {task}. "
    "The video is split into four views: "
    "The top-left view shows a fixed high overhead camera above the workspace, "
    "the top-right view shows a fixed lower camera view of the workspace, "
    "the bottom-left view shows a first-person perspective from the robot's "
    "left end-effector (left gripper camera), "
    "and the bottom-right view shows a first-person perspective from the robot's "
    "right end-effector (right gripper camera). "
    "All four quadrants show live robot-mounted cameras; there is no black or "
    "inactive panel. "
    "The robot {task}."
)


def infer_category(dataset_path: Path) -> str | None:
    for part in dataset_path.parts:
        if part == "non_makovian":
            return "non_makovian"
        if part == "makovian":
            return "makovian"
    return None


def resolve_layout(*, four_views: bool, dynamic_wrist: str) -> tuple[list[tuple[str, str | None]], str]:
    if four_views:
        keys = [CAM_HIGH, CAM_LOW, CAM_LEFT, CAM_RIGHT]
        return [(f"S{i}", k) for i, k in enumerate(keys)], PROMPT_4VIEW
    if dynamic_wrist == "left":
        keys = [CAM_HIGH, CAM_LOW, CAM_LEFT, None]
        return [(f"S{i}", k) for i, k in enumerate(keys)], PROMPT_3VIEW_LEFT
    keys = [CAM_HIGH, CAM_LOW, CAM_RIGHT, None]
    return [(f"S{i}", k) for i, k in enumerate(keys)], PROMPT_3VIEW_RIGHT


def ffmpeg_first_frame(video_path: Path, out_jpg: Path) -> bool:
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"scale={QUAD_W}:{QUAD_H}",
            "-vframes", "1",
            "-q:v", "2",
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
                "-f", "lavfi",
                "-i", f"color=c=black:s={QUAD_W}x{QUAD_H}:r=1",
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


def parse_args():
    repo_hint = "./sampling_dataset/bimanual/input/multiview/dreamgen"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset_path",
        required=True,
        help="LeRobot dataset root for one task (e.g. datasets/bimanual/multiview/makovian/close_toolbox).",
    )
    p.add_argument(
        "--category",
        default="",
        help="Subfolder under input_dir (makovian|non_makovian). If empty, inferred from dataset_path when possible.",
    )
    p.add_argument(
        "--input_dir",
        default=repo_hint,
        help=f"Root for grid PNGs + batch JSON layout. Default: {repo_hint}",
    )
    p.add_argument(
        "--output_dir",
        default="",
        help="Directory for generated DreamGen mp4s. Default: sampling_dataset/bimanual/output/multiview/dreamgen/<category>/<task>/",
    )
    p.add_argument(
        "--batch_json",
        default="",
        help="Path to batch_input.json. Default: <input_dir>/<category>/<task>/batch_input.json",
    )
    p.add_argument(
        "--four_views",
        action="store_true",
        help="Use all four camera streams (left + right wrist); no black quadrant.",
    )
    p.add_argument(
        "--dynamic_wrist",
        choices=["left", "right"],
        default="left",
        help="Which wrist fills the dynamic slot when not using --four_views (default: left).",
    )
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Reuse existing grid PNGs when present (still refreshes batch JSON entries).",
    )
    p.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Always rebuild grid PNGs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    task_name = dataset_path.name
    category = args.category.strip() or infer_category(dataset_path)
    if not category:
        print(
            "[ERROR] Could not infer --category from dataset_path; "
            "pass --category makovian or non_makovian.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_root = Path(args.input_dir).resolve()
    input_task_dir = input_root / category / task_name

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        ip = input_root.as_posix()
        if "/bimanual/input/" in ip:
            output_dir = Path(
                ip.replace("/bimanual/input/", "/bimanual/output/")
            ).resolve() / category / task_name
        else:
            output_dir = (
                Path("./sampling_dataset/bimanual/output/multiview/dreamgen").resolve()
                / category
                / task_name
            )

    batch_json = (
        Path(args.batch_json).resolve()
        if args.batch_json
        else input_task_dir / "batch_input.json"
    )

    cam_layout, prompt_template = resolve_layout(
        four_views=args.four_views,
        dynamic_wrist=args.dynamic_wrist,
    )

    input_task_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = input_task_dir / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = dataset_path / "meta"
    episodes_meta = read_episodes_jsonl(meta_dir)
    if not episodes_meta:
        print(f"[ERROR] No episodes found in {meta_dir}", file=sys.stderr)
        sys.exit(1)

    episode_indices = sorted(episodes_meta.keys())
    if args.max_episodes is not None:
        episode_indices = episode_indices[: args.max_episodes]

    mode = "four_views" if args.four_views else f"3view_{args.dynamic_wrist}_wrist"
    print(f"Dataset   : {dataset_path}")
    print(f"Category  : {category}")
    print(f"Mode      : {mode}")
    print(f"Cam slots : {[c for _, c in cam_layout]}")
    print(f"Episodes  : {len(episode_indices)}")
    print(f"Input dir : {input_task_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Batch JSON: {batch_json}")
    print()

    batch_input: list[dict[str, str]] = []
    skipped = 0

    for ep_idx in episode_indices:
        task_desc = episodes_meta[ep_idx]
        task_clean = task_desc.rstrip(".,;").strip()
        prompt = prompt_template.format(task=task_clean)
        output_video = str(output_dir / f"episode_{ep_idx:06d}.mp4")
        grid_png = input_task_dir / f"episode_{ep_idx:06d}.png"

        if args.skip_existing and grid_png.exists():
            batch_input.append({
                "input_video": str(grid_png),
                "prompt": prompt,
                "output_video": output_video,
            })
            print(f"[SKIP] episode_{ep_idx:06d} (grid already exists)")
            continue

        frame_paths: list = []
        all_ok = True
        for _slot, cam_key in cam_layout:
            if cam_key is None:
                frame_paths.append(None)
                continue
            video_path = find_video(dataset_path, cam_key, ep_idx)
            if video_path is None:
                print(f"[WARN] episode_{ep_idx:06d}: video not found for {cam_key}")
                all_ok = False
                break
            safe = cam_key.replace(".", "_")
            tmp_jpg = tmp_dir / f"ep{ep_idx:06d}_{safe}.jpg"
            if not ffmpeg_first_frame(video_path, tmp_jpg):
                print(f"[WARN] episode_{ep_idx:06d}: ffmpeg failed on {video_path.name}")
                all_ok = False
                break
            frame_paths.append(tmp_jpg)

        if not all_ok:
            skipped += 1
            continue

        if not build_2x2_grid(frame_paths, grid_png):
            print(f"[WARN] episode_{ep_idx:06d}: failed to build grid image")
            skipped += 1
            continue

        print(f"[OK]   episode_{ep_idx:06d} → {grid_png.name}")
        batch_input.append({
            "input_video": str(grid_png),
            "prompt": prompt,
            "output_video": output_video,
        })

    for tmp_file in tmp_dir.glob("*.jpg"):
        tmp_file.unlink(missing_ok=True)

    if not batch_input:
        print("\n[ERROR] No batch inputs created.", file=sys.stderr)
        sys.exit(1)

    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_input, f, indent=2, ensure_ascii=False)

    print(f"\nDone! {len(batch_input)} episodes prepared.")
    if skipped:
        print(f"       {skipped} episodes skipped due to errors.")
    print(f"Batch input JSON: {batch_json}")


if __name__ == "__main__":
    main()
