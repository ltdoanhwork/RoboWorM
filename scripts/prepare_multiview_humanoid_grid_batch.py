"""
Prepare 2x2 grid batch_input.json for DreamGen multiview inference.

For each episode, extracts first frame of each camera view, stitches into
a 2x2 grid image (768x432), and generates prompt matching training format.

Output layout:
  {save_root}/{split}/iter_000070000/{dataset_name}/
    batch_input.json
  {input_root}/{split}/{dataset_name}/
    episode_XXXXXX.png  (grid input images)
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

LAYOUT_A = {
    "TL": ("observation.images.cam_front_rgb", "front"),
    "TR": ("observation.images.cam_head_rgb", "head"),
    "BL": ("observation.images.cam_left_wrist_rgb", "left wrist"),
    "BR": ("observation.images.cam_right_wrist_rgb", "right wrist"),
}

LAYOUT_B = {
    "TL": ("observation.images.cam_high_rgb", "high overhead"),
    "TR": ("observation.images.cam_third_view", "third-person side"),
    "BL": ("observation.images.cam_left_wrist_rgb", "left wrist"),
    "BR": ("observation.images.cam_right_wrist_rgb", "right wrist"),
}

QUAD_W, QUAD_H = 384, 216

PROMPT_TEMPLATE = (
    "A multi-view video of an Airbot MMK2 robot performing: {task}. "
    "The video shows four synchronized camera views arranged in a 2x2 grid: "
    "top-left shows the {tl} view, top-right shows the {tr} view, "
    "bottom-left shows the {bl} view, and bottom-right shows the {br} view."
)


def detect_layout(scenario_dir: Path) -> dict | None:
    videos_root = scenario_dir / "videos"
    if not videos_root.exists():
        return None
    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        cam_dirs = {d.name for d in chunk_dir.iterdir() if d.is_dir()}
        if "observation.images.cam_front_rgb" in cam_dirs:
            return LAYOUT_A
        elif "observation.images.cam_high_rgb" in cam_dirs:
            return LAYOUT_B
        break
    return None


def read_episodes_meta(meta_dir: Path) -> dict:
    ep_file = meta_dir / "episodes.jsonl"
    tasks_file = meta_dir / "tasks.jsonl"

    task_map: dict[int, str] = {}
    if tasks_file.exists():
        for line in tasks_file.read_text().strip().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            tidx = obj.get("task_index", obj.get("index"))
            task = obj.get("task", obj.get("description", ""))
            if tidx is not None and task:
                task_map[int(tidx)] = task

    episodes: dict[int, str] = {}
    if not ep_file.exists():
        return episodes

    for line in ep_file.read_text().strip().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        ep_idx = obj.get("episode_index")
        if ep_idx is None:
            continue
        tasks = obj.get("tasks", [])
        if tasks:
            episodes[int(ep_idx)] = tasks[0]
        elif obj.get("task_index") is not None and int(obj["task_index"]) in task_map:
            episodes[int(ep_idx)] = task_map[int(obj["task_index"])]
        else:
            episodes[int(ep_idx)] = "performing a manipulation task"

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


def ffmpeg_first_frame(video_path: Path, out_jpg: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", str(video_path),
         "-vf", f"scale={QUAD_W}:{QUAD_H}",
         "-vframes", "1", "-q:v", "2", str(out_jpg)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_jpg.exists()


def build_grid_image(frame_paths: dict[str, Path], out_png: Path, tmp_dir: Path) -> bool:
    """Stitch TL/TR/BL/BR frames into 768x432 grid."""
    inputs = []
    for slot in ["TL", "TR", "BL", "BR"]:
        inputs += ["-i", str(frame_paths[slot])]

    filter_complex = (
        f"[0:v]scale={QUAD_W}:{QUAD_H}[v0];"
        f"[1:v]scale={QUAD_W}:{QUAD_H}[v1];"
        f"[2:v]scale={QUAD_W}:{QUAD_H}[v2];"
        f"[3:v]scale={QUAD_W}:{QUAD_H}[v3];"
        f"[v0][v1]hstack[top];"
        f"[v2][v3]hstack[bot];"
        f"[top][bot]vstack[out]"
    )

    cmd = ["ffmpeg", "-nostdin", "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[out]", "-vframes", "1", str(out_png)
    ]
    result = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and out_png.exists()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--iter-name", default="iter_000070000")
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    save_root = Path(args.save_root)
    input_root = Path(args.input_root)

    total = 0

    for split in ["makovian", "non_makovian"]:
        split_dir = datasets_root / split
        if not split_dir.exists():
            continue

        for dataset_dir in sorted(split_dir.iterdir()):
            if not dataset_dir.is_dir() or not (dataset_dir / "videos").exists():
                continue

            dataset_name = dataset_dir.name
            layout = detect_layout(dataset_dir)
            if layout is None:
                print(f"[SKIP] {split}/{dataset_name}: cannot detect camera layout")
                continue

            episodes_meta = read_episodes_meta(dataset_dir / "meta")
            if not episodes_meta:
                print(f"[SKIP] {split}/{dataset_name}: no episodes metadata")
                continue

            episode_indices = sorted(episodes_meta.keys())
            if args.max_episodes:
                episode_indices = episode_indices[:args.max_episodes]

            out_dir = save_root / split / args.iter_name / dataset_name
            img_dir = input_root / split / dataset_name
            tmp_dir = img_dir / "_tmp"
            img_dir.mkdir(parents=True, exist_ok=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            batch_input = []

            for ep_idx in episode_indices:
                task = episodes_meta[ep_idx].rstrip(".,;").strip()
                prompt = PROMPT_TEMPLATE.format(
                    task=task,
                    tl=layout["TL"][1],
                    tr=layout["TR"][1],
                    bl=layout["BL"][1],
                    br=layout["BR"][1],
                )

                grid_png = img_dir / f"episode_{ep_idx:06d}.png"
                output_mp4 = str(out_dir / f"episode_{ep_idx:06d}.mp4")

                if grid_png.exists():
                    batch_input.append({
                        "input_video": str(grid_png),
                        "prompt": prompt,
                        "output_video": output_mp4,
                    })
                    continue

                # Extract first frames
                frame_paths = {}
                all_ok = True
                for slot, (cam_key, _) in layout.items():
                    video_path = find_video(dataset_dir, cam_key, ep_idx)
                    if video_path is None:
                        all_ok = False
                        break
                    tmp_jpg = tmp_dir / f"ep{ep_idx:06d}_{slot}.jpg"
                    if not ffmpeg_first_frame(video_path, tmp_jpg):
                        all_ok = False
                        break
                    frame_paths[slot] = tmp_jpg

                if not all_ok:
                    continue

                if not build_grid_image(frame_paths, grid_png, tmp_dir):
                    continue

                batch_input.append({
                    "input_video": str(grid_png),
                    "prompt": prompt,
                    "output_video": output_mp4,
                })

            # Cleanup tmp
            for f in tmp_dir.glob("*"):
                f.unlink(missing_ok=True)

            if batch_input:
                batch_json = out_dir / "batch_input.json"
                with open(batch_json, "w") as f:
                    json.dump(batch_input, f, indent=2)
                total += len(batch_input)
                print(f"[OK] {split}/{dataset_name}: {len(batch_input)} episodes")

    print(f"\nTotal prepared: {total} grid samples")


if __name__ == "__main__":
    main()
