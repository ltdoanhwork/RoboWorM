"""Prepare humanoid multiview GRID dataset in DreamGen flat format.

Each training sample is a 2x2 grid video with all 4 camera views stitched together.
The prompt describes all 4 views.

Grid layout:
  ┌────────────────┬────────────────┐
  │ TL: high/front │ TR: head/third │
  ├────────────────┼────────────────┤
  │ BL: left_wrist │ BR: right_wrist│
  └────────────────┴────────────────┘

Each quadrant is resized to 384x216 (16:9) → total grid: 768x432.
Training resolution: 432x768 (H x W) matches config.

Prompt format:
  "A multi-view video of an Airbot MMK2 robot performing: {task}.
   The video shows four synchronized camera views arranged in a 2x2 grid:
   top-left shows the {tl_view} view, top-right shows the {tr_view} view,
   bottom-left shows the left wrist view, and bottom-right shows the right wrist view."

Reads from:
    datasets/humanoid/multiview/{makovian,non_makovian}/{scenario}/
        videos/chunk-*/{camera_key}/episode_XXXXXX.mp4
        meta/episodes.jsonl or meta/tasks.jsonl

Writes to:
    datasets/humanoid_multiview_grid_dreamgen/
    ├── videos/{split}__{scenario}__grid__episode_{id:06d}.mp4
    └── metas/{split}__{scenario}__grid__episode_{id:06d}.txt

Usage:
    python finetuning/dreamgen/scripts/prepare_humanoid_multiview_grid_dreamgen.py
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

# Camera layout groups (determined by available cameras per dataset)
# Group A: cam_front_rgb, cam_head_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb
# Group B: cam_high_rgb, cam_third_view, cam_left_wrist_rgb, cam_right_wrist_rgb

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

QUAD_W, QUAD_H = 384, 216  # 16:9 per quadrant → grid: 768x432

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


def load_task_caption(scenario_dir: Path) -> str:
    ep_file = scenario_dir / "meta" / "episodes.jsonl"
    tasks_file = scenario_dir / "meta" / "tasks.jsonl"

    if tasks_file.exists():
        for line in tasks_file.read_text().strip().splitlines():
            entry = json.loads(line)
            task = entry.get("task", "")
            if task:
                return task

    if ep_file.exists():
        for line in ep_file.read_text().strip().splitlines():
            entry = json.loads(line)
            tasks = entry.get("tasks", [])
            if tasks:
                return tasks[0]

    return "performing a manipulation task"


def find_episodes(scenario_dir: Path, camera_key: str) -> list[tuple[int, Path]]:
    """Find all episodes for a given camera key. Returns sorted (ep_id, path) list."""
    videos_root = scenario_dir / "videos"
    results = []
    for chunk_dir in sorted(videos_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        cam_dir = chunk_dir / camera_key
        if not cam_dir.exists():
            continue
        for mp4 in sorted(cam_dir.glob("episode_*.mp4")):
            ep_match = re.search(r"(\d+)", mp4.stem)
            if ep_match:
                results.append((int(ep_match.group(1)), mp4))
    return sorted(results, key=lambda x: x[0])


def stitch_grid_video(view_paths: dict[str, Path], output_path: Path, fps: int = 10) -> bool:
    """Stitch 4 view videos into a 2x2 grid video using ffmpeg."""
    tl = view_paths.get("TL")
    tr = view_paths.get("TR")
    bl = view_paths.get("BL")
    br = view_paths.get("BR")

    if not all([tl, tr, bl, br]):
        return False

    filter_complex = (
        f"[0:v]scale={QUAD_W}:{QUAD_H},setsar=1[tl];"
        f"[1:v]scale={QUAD_W}:{QUAD_H},setsar=1[tr];"
        f"[2:v]scale={QUAD_W}:{QUAD_H},setsar=1[bl];"
        f"[3:v]scale={QUAD_W}:{QUAD_H},setsar=1[br];"
        f"[tl][tr]hstack=inputs=2[top];"
        f"[bl][br]hstack=inputs=2[bot];"
        f"[top][bot]vstack=inputs=2[out]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(tl),
        "-i", str(tr),
        "-i", str(bl),
        "-i", str(br),
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        "-an", str(output_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0 and output_path.exists()


def prepare(source: Path, target: Path, fps: int = 10) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    failed = 0

    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        if split_name not in ("makovian", "non_makovian"):
            continue

        for scenario_dir in sorted(split_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario_name = scenario_dir.name

            layout = detect_layout(scenario_dir)
            if layout is None:
                print(f"  WARNING: cannot detect layout for {scenario_dir}, skipping")
                continue

            task_caption = load_task_caption(scenario_dir)
            caption = PROMPT_TEMPLATE.format(
                task=task_caption.rstrip(".,;").strip(),
                tl=layout["TL"][1],
                tr=layout["TR"][1],
                bl=layout["BL"][1],
                br=layout["BR"][1],
            )

            # Find episodes that exist for ALL 4 views
            tl_key = layout["TL"][0]
            episodes_per_view = {}
            for slot, (cam_key, _) in layout.items():
                episodes_per_view[slot] = dict(find_episodes(scenario_dir, cam_key))

            # Only keep episodes present in all 4 views
            common_eps = set(episodes_per_view["TL"].keys())
            for slot in ["TR", "BL", "BR"]:
                common_eps &= set(episodes_per_view[slot].keys())
            common_eps = sorted(common_eps)

            if not common_eps:
                print(f"  WARNING: no episodes with all 4 views in {split_name}/{scenario_name}")
                continue

            print(f"  [{split_name}/{scenario_name}] {len(common_eps)} episodes, layout={'A' if layout is LAYOUT_A else 'B'}")

            for ep_id in common_eps:
                out_stem = f"{split_name}__{scenario_name}__grid__episode_{ep_id:06d}"
                dst_mp4 = videos_out / f"{out_stem}.mp4"
                dst_txt = metas_out / f"{out_stem}.txt"

                if dst_mp4.exists() and dst_txt.exists():
                    skipped += 1
                    continue

                view_paths = {
                    slot: episodes_per_view[slot][ep_id]
                    for slot in ["TL", "TR", "BL", "BR"]
                }

                if not dst_mp4.exists():
                    if not stitch_grid_video(view_paths, dst_mp4, fps=fps):
                        print(f"    FAILED: {out_stem}")
                        failed += 1
                        continue

                if not dst_txt.exists():
                    dst_txt.write_text(caption)

                total += 1
                if total % 50 == 0:
                    print(f"    Processed {total} grid videos...")

    print(f"\nDone. Created {total} grid entries, skipped {skipped} existing, {failed} failed.")
    print(f"Output: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=str, default="datasets/humanoid/multiview",
        help="Root of humanoid multiview dataset",
    )
    parser.add_argument(
        "--target", type=str, default="datasets/humanoid_multiview_grid_dreamgen",
        help="Output directory for DreamGen-format grid dataset",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Output video FPS (default: 10)",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"
    prepare(source, target, fps=args.fps)


if __name__ == "__main__":
    main()
