"""Prepare single-arm RoboChallenge GLOBAL (static) view dataset — 16fps, segmented clips.

Uses the single static camera (observation.images.global) from the new
RoboChallenge-derived dataset at datasets/single_arm/singleview.

Each episode video is split into non-overlapping segments of roughly equal
length, ensuring every segment >= 93 frames so the training dataset class can
sample a continuous 93-frame window from it. This guarantees the model sees ALL
phases of the task (start, middle, completion) rather than only random windows
from a longer video.

Segment logic for a video of N frames (after 16fps transcode):
  - num_segments = N // SEGMENT_TARGET
  - segment_length = N // num_segments  (between 93 and ~140 frames)
  - Each segment becomes a separate training sample with the same caption

Reads from:
    datasets/single_arm/singleview/{makovian,non_makovian}/videos/chunk-*/
        observation.images.global/*.mp4
    captions from {split}/meta/episodes.jsonl (real task prompts)

Writes to (OVERWRITES the prepared training dataset used by the
single_arm_static_singleview_16fps experiment):
    datasets/single_arm_static_dreamgen_16fps/
    ├── videos/{split}__global__ep{id:06d}__seg{i:02d}.mp4
    ├── metas/{split}__global__ep{id:06d}__seg{i:02d}.txt
    └── ...

Usage:
    python finetuning/dreamgen/scripts/prepare_single_arm_robochallenge_dreamgen_16fps.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path

GENERIC_CAPTION = "A robot arm performing a manipulation task."
STATIC_VIEWS = ["observation.images.global"]
VIEW_SHORT_NAMES = {"observation.images.global": "global"}
TARGET_FPS = 16

# ---------------------------------------------------------------------------
# Prompt augmentation
#
# The raw dataset only contains 4 distinct task strings, each repeated verbatim
# for every episode/segment. Training T5 text-conditioning on 4 fixed strings
# makes the text encoder degenerate (it cannot distinguish anything within a
# task and overfits a single embedding per task). To give the model a richer,
# more robust text signal we deterministically build each caption from three
# parts:
#
#   <robot/scene prefix> + <task phrasing variant> + <phase hint>
#
# Everything is deterministic (seeded by episode + segment index) so re-running
# prepare reproduces identical captions and T5 embeddings stay consistent.
# ---------------------------------------------------------------------------

# Consistent prefixes that anchor the clip as a single-arm robot manipulation
# scene (matches the kind of captions the base Cosmos/GR00T model saw).
PROMPT_PREFIXES = [
    "A robot arm",
    "A single robotic arm",
    "A tabletop robot arm",
    "A robot manipulator",
]

# Per-task rephrasings. Keyed by the exact raw task string from episodes.jsonl.
# Each variant is a semantically equivalent way to describe the SAME motion so
# T5 sees diverse surface forms mapping to the same action.
TASK_REPHRASINGS: dict[str, list[str]] = {
    "place the cup on the coaster": [
        "places the cup on the coaster",
        "puts the cup onto the coaster",
        "picks up the cup and sets it down on the coaster",
        "moves the cup onto the coaster",
    ],
    "turn on the light switch": [
        "turns on the light switch",
        "flips the light switch on",
        "presses the light switch to turn it on",
        "reaches for the switch and turns the light on",
    ],
    "press the pink, blue, and green buttons in sequence": [
        "presses the pink, blue, and green buttons in sequence",
        "pushes the pink, then blue, then green buttons in order",
        "presses three buttons one after another: pink, blue, then green",
        "activates the pink, blue, and green buttons in that order",
    ],
    "pull out a tissue, wipe the stains on the table clean, "
    "and then discard the tissue into the trash bin": [
        "pulls out a tissue, wipes the stains on the table clean, "
        "and then discards the tissue into the trash bin",
        "takes a tissue, cleans the stains off the table, "
        "and throws the tissue into the trash bin",
        "grabs a tissue, wipes the table clean, and disposes of it in the bin",
        "pulls a tissue, wipes away the table stains, "
        "then drops the tissue in the trash",
    ],
}

# Phase hints appended based on the segment's position within the episode.
PHASE_HINT_FIRST = "at the start of the task"
PHASE_HINT_MIDDLE = "midway through the task"
PHASE_HINT_LAST = "completing the task"


def _rephrase_task(raw_task: str, variant_idx: int) -> str:
    """Return a deterministic rephrasing variant for a raw task string.

    Falls back to a verb-form of the raw task if it is not in the table.
    """
    variants = TASK_REPHRASINGS.get(raw_task)
    if variants:
        return variants[variant_idx % len(variants)]
    # Unknown task: best-effort, keep the raw text as an action phrase.
    return raw_task


def _phase_hint(seg_idx: int, num_segments: int) -> str:
    if num_segments <= 1:
        return ""
    if seg_idx == 0:
        return PHASE_HINT_FIRST
    if seg_idx == num_segments - 1:
        return PHASE_HINT_LAST
    return PHASE_HINT_MIDDLE


def build_caption(raw_task: str, ep_id: int, seg_idx: int, num_segments: int) -> str:
    """Compose a robot prefix + task rephrasing + phase hint into one caption.

    Deterministic in (ep_id, seg_idx) so reruns reproduce identical captions.
    """
    if not raw_task or raw_task == GENERIC_CAPTION:
        return GENERIC_CAPTION

    prefix = PROMPT_PREFIXES[ep_id % len(PROMPT_PREFIXES)]
    task_phrase = _rephrase_task(raw_task, ep_id + seg_idx)
    phase = _phase_hint(seg_idx, num_segments)

    caption = f"{prefix} {task_phrase}"
    if phase:
        caption = f"{caption}, {phase}"
    return caption + "."
# Training dataset class samples a random window of `sequence_length` (=93)
# frames via np.random.randint(0, total_frames - 93). A clip with EXACTLY 93
# frames gives randint(0, 0) which raises and makes that clip unusable, so we
# require at least 94 frames per segment.
MIN_FRAMES = 94
SEGMENT_TARGET = 94


def get_video_frame_count(video_path: Path) -> int | None:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "stream=nb_frames",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    # Fallback: nb_frames metadata missing — decode and count exactly.
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        try:
            return int(result.stdout.strip().split("\n")[0])
        except ValueError:
            pass
    return None


def transcode_16fps(src: Path, dst: Path) -> bool:
    """Transcode source video to 16fps H.264."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src),
         "-r", str(TARGET_FPS),
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         "-pix_fmt", "yuv420p", "-an", str(dst)],
        capture_output=True,
    )
    return result.returncode == 0


def split_video_segments(src: Path, out_dir: Path, stem_prefix: str,
                         n_frames: int) -> list[Path]:
    """Split a video into roughly equal segments, each >= MIN_FRAMES.

    Returns list of output segment paths.
    """
    if n_frames < MIN_FRAMES:
        return []

    num_segments = max(1, n_frames // SEGMENT_TARGET)
    seg_length = n_frames // num_segments

    if seg_length < MIN_FRAMES:
        num_segments = max(1, n_frames // MIN_FRAMES)
        seg_length = n_frames // num_segments

    outputs = []
    for i in range(num_segments):
        start_frame = i * seg_length
        if i == num_segments - 1:
            end_frame = n_frames
        else:
            end_frame = (i + 1) * seg_length

        seg_frames = end_frame - start_frame
        if seg_frames < MIN_FRAMES:
            continue

        seg_path = out_dir / f"{stem_prefix}__seg{i:02d}.mp4"
        if seg_path.exists():
            outputs.append(seg_path)
            continue

        start_time = start_frame / TARGET_FPS
        duration = seg_frames / TARGET_FPS

        result = subprocess.run(
            ["ffmpeg", "-y",
             "-ss", f"{start_time:.4f}",
             "-i", str(src),
             "-t", f"{duration:.4f}",
             "-c:v", "libx264", "-preset", "fast", "-crf", "18",
             "-pix_fmt", "yuv420p", "-an", str(seg_path)],
            capture_output=True,
        )
        if result.returncode == 0 and seg_path.exists():
            # ffmpeg's time-based cut can be off by +/-1 frame vs seg_frames;
            # verify the real frame count and drop clips that ended up < MIN_FRAMES
            # (a 93-frame clip would crash the training random-window sampler).
            actual = get_video_frame_count(seg_path)
            if actual is not None and actual < MIN_FRAMES:
                seg_path.unlink(missing_ok=True)
                continue
            outputs.append(seg_path)

    return outputs


def load_task_descriptions(split_dir: Path) -> dict[int, str]:
    """Load per-episode task descriptions from episodes.jsonl."""
    episodes_file = split_dir / "meta" / "episodes.jsonl"
    tasks_file = split_dir / "meta" / "tasks.jsonl"

    ep_task: dict[int, str] = {}
    if not episodes_file.exists():
        return ep_task

    with episodes_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep_idx = rec.get("episode_index", -1)
            tasks = rec.get("tasks", [])
            if ep_idx >= 0 and tasks and tasks[0]:
                ep_task[ep_idx] = tasks[0]

    if ep_task:
        return ep_task

    task_map: dict[int, str] = {}
    if tasks_file.exists():
        with tasks_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                task_desc = rec.get("task", "")
                if task_desc:
                    task_map[rec["task_index"]] = task_desc

    if task_map:
        with episodes_file.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ep_idx = rec.get("episode_index", -1)
                t_idx = rec.get("task_index", -1)
                if ep_idx >= 0 and t_idx in task_map:
                    ep_task[ep_idx] = task_map[t_idx]

    return ep_task


def prepare(source: Path, target: Path) -> None:
    videos_out = target / "videos"
    metas_out = target / "metas"
    videos_out.mkdir(parents=True, exist_ok=True)
    metas_out.mkdir(parents=True, exist_ok=True)

    total_clips = 0
    total_episodes = 0
    skipped_existing = 0
    skipped_short = 0

    tmp_dir = Path(tempfile.mkdtemp(prefix="robochallenge_global_"))

    for split_dir in sorted(source.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name

        ep_task = load_task_descriptions(split_dir)
        if ep_task:
            print(f"  [{split_name}] Loaded {len(ep_task)} task descriptions")
        else:
            print(f"  [{split_name}] No task descriptions — using generic caption")

        videos_root = split_dir / "videos"
        if not videos_root.exists():
            print(f"  WARNING: no videos/ dir in {split_dir}, skipping")
            continue

        for view_name in STATIC_VIEWS:
            view_short = VIEW_SHORT_NAMES[view_name]
            mp4_files = sorted(videos_root.glob(f"chunk-*/{view_name}/*.mp4"))

            if not mp4_files:
                print(f"  [{split_name}/{view_short}] No videos found, skipping")
                continue

            for mp4 in mp4_files:
                stem = mp4.stem
                ep_match = re.search(r"(\d+)$", stem)
                ep_id = int(ep_match.group(1)) if ep_match else total_episodes

                out_stem = f"{split_name}__{view_short}__ep{ep_id:06d}"

                # Check if segments already exist
                existing_segs = list(videos_out.glob(f"{out_stem}__seg*.mp4"))
                if existing_segs:
                    skipped_existing += len(existing_segs)
                    total_clips += len(existing_segs)
                    total_episodes += 1
                    continue

                raw_task = ep_task.get(ep_id, GENERIC_CAPTION)

                # Transcode to 16fps temp file
                tmp_16fps = tmp_dir / f"{out_stem}.mp4"
                if not transcode_16fps(mp4, tmp_16fps):
                    print(f"  [WARN] Failed to transcode: {mp4.name}")
                    continue

                n_frames = get_video_frame_count(tmp_16fps)
                if n_frames is None or n_frames < MIN_FRAMES:
                    print(f"  [WARN] {out_stem}: {n_frames} frames, too short")
                    tmp_16fps.unlink(missing_ok=True)
                    skipped_short += 1
                    continue

                # Split into segments
                seg_paths = split_video_segments(
                    tmp_16fps, videos_out, out_stem, n_frames
                )

                # Write a composed caption (prefix + rephrased task + phase hint)
                # for each segment. Phase hint depends on segment position so
                # seg00 reads "at the start of the task" and the final segment
                # reads "completing the task".
                num_segments = len(seg_paths)
                for seg_path in seg_paths:
                    seg_match = re.search(r"__seg(\d+)", seg_path.stem)
                    seg_idx = int(seg_match.group(1)) if seg_match else 0
                    caption = build_caption(raw_task, ep_id, seg_idx, num_segments)
                    txt_path = metas_out / seg_path.name.replace(".mp4", ".txt")
                    if not txt_path.exists():
                        txt_path.write_text(caption)
                    total_clips += 1

                tmp_16fps.unlink(missing_ok=True)
                total_episodes += 1

                if total_episodes % 50 == 0:
                    print(f"  Processed {total_episodes} episodes → {total_clips} clips...")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nDone. {total_episodes} episodes → {total_clips} clips "
          f"(avg {total_clips/max(total_episodes,1):.1f} clips/episode)")
    print(f"  Skipped: {skipped_existing} existing, {skipped_short} too short")
    print(f"Output: {target}")


def rebuild_captions(source: Path, target: Path) -> None:
    """Regenerate metas/*.txt for already-prepared clips WITHOUT re-transcoding.

    Discovers existing segment videos in target/videos, recomputes the segment
    count per episode, and overwrites each .txt with build_caption(). Use this
    after changing the prompt logic so T5 embeddings can be recomputed.
    """
    videos_out = target / "videos"
    metas_out = target / "metas"
    assert videos_out.exists(), f"No prepared videos at {videos_out}"
    metas_out.mkdir(parents=True, exist_ok=True)

    # Map raw task per (split, ep_id) by reloading task descriptions per split.
    split_tasks: dict[str, dict[int, str]] = {}
    for split_dir in sorted(source.iterdir()):
        if split_dir.is_dir():
            split_tasks[split_dir.name] = load_task_descriptions(split_dir)

    seg_re = re.compile(r"^(?P<split>.+)__(?P<view>[^_]+)__ep(?P<ep>\d+)__seg(?P<seg>\d+)$")

    # First pass: count segments per (split, ep).
    seg_counts: dict[tuple[str, int], int] = {}
    clips: list[tuple[Path, str, int, int]] = []
    for mp4 in sorted(videos_out.glob("*.mp4")):
        m = seg_re.match(mp4.stem)
        if not m:
            print(f"  [WARN] cannot parse segment name: {mp4.name}")
            continue
        split = m.group("split")
        ep_id = int(m.group("ep"))
        seg_idx = int(m.group("seg"))
        seg_counts[(split, ep_id)] = max(seg_counts.get((split, ep_id), 0), seg_idx + 1)
        clips.append((mp4, split, ep_id, seg_idx))

    written = 0
    for mp4, split, ep_id, seg_idx in clips:
        raw_task = split_tasks.get(split, {}).get(ep_id, GENERIC_CAPTION)
        num_segments = seg_counts[(split, ep_id)]
        caption = build_caption(raw_task, ep_id, seg_idx, num_segments)
        txt_path = metas_out / (mp4.stem + ".txt")
        txt_path.write_text(caption)
        written += 1

    print(f"\nRebuilt {written} caption files in {metas_out}")
    print("  NOTE: T5 embeddings are now stale — recompute them before training.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare RoboChallenge single-arm GLOBAL view — 16fps segmented clips"
    )
    parser.add_argument(
        "--source", type=str,
        default="datasets/single_arm/singleview",
    )
    parser.add_argument(
        "--target", type=str,
        default="datasets/single_arm_static_dreamgen_16fps",
    )
    parser.add_argument(
        "--rebuild-captions", action="store_true",
        help="Only regenerate metas/*.txt for existing clips (no transcoding), "
             "then exit. Use after changing prompt logic.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    assert source.exists(), f"Source not found: {source}"

    if args.rebuild_captions:
        print(f"Rebuilding captions only for: {target}")
        rebuild_captions(source, target)
        return

    print(f"Preparing RoboChallenge single-arm GLOBAL view (segmented) @ {TARGET_FPS}fps")
    print(f"  Source: {source}")
    print(f"  Target: {target}")
    print(f"  Views: {', '.join(VIEW_SHORT_NAMES.values())}")
    print(f"  Segment target: {SEGMENT_TARGET} frames ({SEGMENT_TARGET/TARGET_FPS:.1f}s)")
    prepare(source, target)


if __name__ == "__main__":
    main()
