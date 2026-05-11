"""
Prepare batch_input.json for DreamGen multiview inference
from a LeRobot-format humanoid dataset.

Supports two camera layout groups (auto-detected):

  Group A  (cam_front_rgb available):
  ┌──────────────┬──────────────┐
  │  TL: front   │  TR: head    │  <- two static exterior views
  ├──────────────┼──────────────┤
  │  BL: wrist   │  BR: black   │  <- 1 dynamic gripper view + inactive
  └──────────────┴──────────────┘

  Group B  (cam_high_rgb available, no cam_front_rgb):
  ┌──────────────┬──────────────┐
  │  TL: high    │  TR: 3rd     │  <- two static exterior views
  ├──────────────┼──────────────┤
  │  BL: wrist   │  BR: black   │  <- 1 dynamic gripper view + inactive
  └──────────────┴──────────────┘

Prompt template mirrors the DROID variant prompt so the model receives
the same description style it was trained on.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Camera configurations per group
# ---------------------------------------------------------------------------
# (slot_label, camera_key) – BR slot has camera_key=None → black frame
CAM_LAYOUT_A = [
    ("TL", "observation.images.cam_front_rgb"),       # static, exterior front
    ("TR", "observation.images.cam_head_rgb"),         # static, exterior head
    ("BL", "observation.images.cam_left_wrist_rgb"),   # dynamic, gripper POV
    ("BR", None),                                      # black / inactive
]

CAM_LAYOUT_B = [
    ("TL", "observation.images.cam_high_rgb"),         # static, overhead high
    ("TR", "observation.images.cam_third_view"),        # static, third-person view
    ("BL", "observation.images.cam_left_wrist_rgb"),   # dynamic, gripper POV
    ("BR", None),                                      # black / inactive
]

PROMPT_TEMPLATE_A = (
    "A multi-view video shows that a robot {task}. "
    "The video is split into four views: "
    "The top-left view shows the robotic arm from the front side, "
    "the top-right view shows it from the head/overhead perspective, "
    "the bottom-left view shows a first-person perspective from the "
    "robot's end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

PROMPT_TEMPLATE_B = (
    "A multi-view video shows that a robot {task}. "
    "The video is split into four views: "
    "The top-left view shows the robotic arm from a high overhead perspective, "
    "the top-right view shows it from a third-person side perspective, "
    "the bottom-left view shows a first-person perspective from the "
    "robot's end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

# Each quadrant: 320×180  →  full grid: 640×360 (16:9)
QUAD_W, QUAD_H = 320, 180


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def detect_cam_group(dataset_path: Path) -> str:
    """Auto-detect camera layout group based on available cameras."""
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return "A"  # fallback default

    # Check any chunk directory for available camera subdirs
    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        cam_dirs = {d.name for d in chunk_dir.iterdir() if d.is_dir()}
        if "observation.images.cam_front_rgb" in cam_dirs:
            return "A"
        elif "observation.images.cam_high_rgb" in cam_dirs:
            return "B"
        break  # only need first chunk

    return "A"  # fallback


def ffmpeg_first_frame(video_path: Path, out_jpg: Path) -> bool:
    """Extract the first frame, resize to QUAD_W×QUAD_H, save as JPEG."""
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
    """
    Stitch 4 slots (TL, TR, BL, BR) into a 640×360 PNG.
    frame_paths: list of 4 Path-or-None. None → lavfi black frame.
    """
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


def read_episodes_jsonl(meta_dir: Path) -> dict:
    """Return {episode_index: task_description} from meta/episodes.jsonl.
    Falls back to meta/tasks.jsonl if episodes have no inline tasks."""
    ep_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"

    # Build task_index → description from tasks.jsonl
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
            # Inline tasks list takes priority
            tasks = obj.get("tasks", [])
            if tasks:
                episodes[int(ep_idx)] = tasks[0]
                continue
            # Fallback: task_index → tasks.jsonl
            tidx = obj.get("task_index")
            if tidx is not None and int(tidx) in task_map:
                episodes[int(ep_idx)] = task_map[int(tidx)]
                continue
            episodes[int(ep_idx)] = obj.get("description", "performing a manipulation task")

    return episodes


def find_video(dataset_path: Path, cam_key: str, ep_idx: int) -> Path | None:
    """Scan chunk directories to find episode video for a given camera key."""
    videos_root = dataset_path / "videos"
    if not videos_root.exists():
        return None
    for chunk_dir in sorted(videos_root.iterdir()):
        candidate = chunk_dir / cam_key / f"episode_{ep_idx:06d}.mp4"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_path",
        default="./datasets/humanoid/multiview/non_makovian/Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate",
        help="Root directory of the LeRobot-format dataset.",
    )
    parser.add_argument(
        "--input_dir",
        default="./sampling_dataset/humanoid/multiview/input",
        help="Directory to save the prepared 2×2 grid PNG images.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory where generated videos will be saved. "
             "Defaults to <input_dir>/../non_makovian/dreamgen/<dataset_name>.",
    )
    parser.add_argument(
        "--batch_json",
        default="",
        help="Path for the output batch_input.json. "
             "Defaults to <input_dir>/<dataset_name>/batch_input.json.",
    )
    parser.add_argument(
        "--cam_layout",
        choices=["auto", "A", "B"],
        default="auto",
        help="Camera layout group. 'auto' detects from available cameras.",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=None,
        help="Limit number of episodes processed (useful for quick tests).",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip episodes whose grid PNG already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    dataset_name = dataset_path.name
    input_dir = Path(args.input_dir).resolve() / dataset_name
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(args.input_dir).resolve().parent / "non_makovian" / "dreamgen" / dataset_name
    )
    batch_json = (
        Path(args.batch_json).resolve()
        if args.batch_json
        else input_dir / "batch_input.json"
    )

    # --- Detect or use specified camera layout group ---
    if args.cam_layout == "auto":
        cam_group = detect_cam_group(dataset_path)
    else:
        cam_group = args.cam_layout

    cam_layout = CAM_LAYOUT_A if cam_group == "A" else CAM_LAYOUT_B
    prompt_template = PROMPT_TEMPLATE_A if cam_group == "A" else PROMPT_TEMPLATE_B

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_json.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = input_dir / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Load episode metadata ---
    meta_dir = dataset_path / "meta"
    episodes_meta = read_episodes_jsonl(meta_dir)
    if not episodes_meta:
        print(f"[ERROR] No episodes found in {meta_dir}", file=sys.stderr)
        sys.exit(1)

    episode_indices = sorted(episodes_meta.keys())
    if args.max_episodes:
        episode_indices = episode_indices[: args.max_episodes]

    print(f"Dataset  : {dataset_path}")
    print(f"Cam Group: {cam_group} ({[c for _, c in cam_layout if c]})")
    print(f"Episodes : {len(episode_indices)}")
    print(f"Input dir: {input_dir}")
    print(f"Output   : {output_dir}")
    print(f"Batch    : {batch_json}")
    print()

    batch_input = []
    skipped = 0

    for ep_idx in episode_indices:
        task_desc = episodes_meta[ep_idx]
        # Strip trailing punctuation so the prompt doesn't get ".."
        task_clean = task_desc.rstrip(".,;").strip()
        prompt = prompt_template.format(task=task_clean)
        output_video = str(output_dir / f"episode_{ep_idx:06d}.mp4")
        grid_png = input_dir / f"episode_{ep_idx:06d}.png"

        # --- Skip if grid already exists ---
        if args.skip_existing and grid_png.exists():
            batch_input.append({
                "input_video": str(grid_png),
                "prompt": prompt,
                "output_video": output_video,
            })
            print(f"[SKIP] episode_{ep_idx:06d} (grid already exists)")
            continue

        # --- Extract first frames for each camera slot ---
        frame_paths: list = []
        all_ok = True
        for _slot, cam_key in cam_layout:
            if cam_key is None:
                frame_paths.append(None)  # black frame slot
                continue

            video_path = find_video(dataset_path, cam_key, ep_idx)
            if video_path is None:
                print(f"[WARN] episode_{ep_idx:06d}: video not found for {cam_key}")
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

        # --- Stitch 2×2 grid ---
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

    # Cleanup temp frames
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
