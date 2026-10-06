"""
Caption humanoid singleview training videos (makovian / non_makovian) with
Qwen3-VL-30B-A3B-Instruct-FP8 and write DreamGen-ready flat structure.

For each video in datasets/humanoid/singleview/{makovian,non_makovian}/videos/...:
  1. Resolve symlink → identify source (GR1_robot or DreamDojo-HV_Eval).
  2. Fetch the existing task-text hint from gr1robot_task_labels.csv (GR1_robot)
     or gr1_task_labels_cache.jsonl (DreamDojo).
  3. Sample N frames from the video + prepend the task hint in the prompt.
  4. Run Qwen3-VL-30B to produce a natural-language caption.

Outputs (written in parallel):
  a) datasets/humanoid/singleview/{split}/captions/<stem>.txt
     → one-line Qwen caption per episode (intermediate cache).
  b) datasets/humanoid_dreamgen/
       videos/<split>__<stem>.mp4   (symlink)
       metas/<split>__<stem>.txt    (Qwen caption)
     → flat DreamGen training structure, ready for compute_t5_embeddings.sh.

Usage (from project root):
    python scripts/tools/data/caption_humanoid_singleview.py
    python scripts/tools/data/caption_humanoid_singleview.py --splits makovian --dry_run
    python scripts/tools/data/caption_humanoid_singleview.py --overwrite --num_video_frames 12

Prerequisites:
    pip install vllm>=0.8 transformers>=4.57 imageio imageio-ffmpeg pillow
    (run from a venv with the above, NOT the DreamDojo venv)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAMERA = "observation.images.ego_view_freq20"

CAPTION_PROMPT = (
    "You are labelling a short humanoid robot tele-operation clip for a "
    "video-generation model.\n"
    "The robot's task is: \"{task_hint}\"\n"
    "Looking at the provided video frames, write ONE concise English sentence "
    "(max 30 words) describing what the humanoid robot is doing — mention which "
    "arm/hand is used, the manipulated object(s), and the action verb. "
    "Do NOT describe background, camera angle, or lighting. "
    "Output the sentence only, no quotes, no prefix."
)

CAPTION_PROMPT_NO_HINT = (
    "You are labelling a short humanoid robot tele-operation clip for a "
    "video-generation model. In ONE concise English sentence (max 30 words) "
    "describe what the humanoid robot is doing — which arm/hand is used, the "
    "manipulated object(s), and the action verb. Do NOT describe background, "
    "camera angle, or lighting. Output the sentence only, no quotes, no prefix."
)

WS = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Prompt / label loading helpers
# ---------------------------------------------------------------------------

def load_gr1robot_task_hints(ws: Path) -> dict[int, str]:
    """episode_index -> clean task text (no 'locked waist: ' prefix)."""
    ep_file = ws / "datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/meta/episodes.jsonl"
    ep_to_task: dict[int, str] = {}
    if not ep_file.exists():
        return ep_to_task
    with ep_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            raw = (obj.get("tasks") or [""])[0]
            clean = raw.replace("locked waist: ", "").strip()
            ep_to_task[obj["episode_index"]] = clean
    return ep_to_task


def load_dreamdojo_task_hints(ws: Path) -> dict[int, str]:
    """episode_index -> LLM-generated natural language prompt (DreamDojo)."""
    cache = ws / "data/labels/gr1_task_labels_cache.jsonl"
    ep_to_prompt: dict[int, str] = {}
    if not cache.exists():
        return ep_to_prompt
    with cache.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source") != "DreamDojo-HV_Eval":
                continue
            m = re.search(r"(\d+)$", obj.get("episode", ""))
            if m:
                ep_to_prompt[int(m.group(1))] = obj.get("prompt", "")
    return ep_to_prompt


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def sample_frames(video_path: Path, n: int) -> list[Image.Image]:
    frames = iio.imread(str(video_path), plugin="pyav")
    if frames.ndim == 3:
        frames = frames[None]
    T = frames.shape[0]
    if T <= n:
        idx = list(range(T))
    else:
        step = T / n
        idx = [int(i * step) for i in range(n)]
    return [Image.fromarray(frames[i]) for i in idx]


def build_messages(frames: list[Image.Image], task_hint: str) -> list[dict]:
    content = [{"type": "image", "image": img} for img in frames]
    if task_hint:
        prompt_text = CAPTION_PROMPT.format(task_hint=task_hint)
    else:
        prompt_text = CAPTION_PROMPT_NO_HINT
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Caption humanoid singleview training videos with Qwen3-VL."
    )
    p.add_argument(
        "--workspace", type=Path, default=WS,
        help="Project root (default: auto-detected from script location).",
    )
    p.add_argument(
        "--splits", nargs="+", default=["makovian", "non_makovian"],
        help="Which split folders under datasets/humanoid/singleview/ to process.",
    )
    p.add_argument(
        "--dreamgen_target", type=Path,
        default=None,
        help="Where to write humanoid_dreamgen/ flat structure. "
             "Default: <workspace>/datasets/humanoid_dreamgen/",
    )
    p.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    p.add_argument("--num_video_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--overwrite", action="store_true",
        help="Re-caption even if a .txt caption already exists.",
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be done without running the LLM.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ws = args.workspace.resolve()

    sv_root = ws / "datasets/humanoid/singleview"
    dreamgen_target = args.dreamgen_target or (ws / "datasets/humanoid_dreamgen")
    dreamgen_videos = dreamgen_target / "videos"
    dreamgen_metas = dreamgen_target / "metas"
    dreamgen_videos.mkdir(parents=True, exist_ok=True)
    dreamgen_metas.mkdir(parents=True, exist_ok=True)

    # Load hint tables
    print("Loading task hint tables...")
    gr1_hints = load_gr1robot_task_hints(ws)
    dd_hints = load_dreamdojo_task_hints(ws)
    print(f"  GR1_robot hints:   {len(gr1_hints)} episodes")
    print(f"  DreamDojo hints:   {len(dd_hints)} episodes")

    # -----------------------------------------------------------------------
    # Collect all work items
    # -----------------------------------------------------------------------
    # work item: (split, video_path, episode_index, task_hint, src_type, caption_txt, dg_stem)
    work = []

    for split in args.splits:
        split_dir = sv_root / split
        if not split_dir.exists():
            print(f"[warn] split dir not found: {split_dir}")
            continue

        videos = sorted((split_dir / "videos").rglob("*.mp4"))
        print(f"[{split}] found {len(videos)} videos")

        for v in videos:
            # Identify source via symlink target
            try:
                target = os.readlink(v)
            except OSError:
                target = str(v)

            if "GR1_robot" in target:
                src_type = "GR1_robot"
            elif "DreamDojo" in target:
                src_type = "DreamDojo"
            else:
                src_type = "unknown"

            m = re.search(r"(\d+)$", v.stem)
            ep_id = int(m.group(1)) if m else -1

            if src_type == "GR1_robot":
                task_hint = gr1_hints.get(ep_id, "")
            elif src_type == "DreamDojo":
                task_hint = dd_hints.get(ep_id, "")
            else:
                task_hint = ""

            # Intermediate caption cache
            caption_dir = split_dir / "captions"
            caption_txt = caption_dir / f"{v.stem}.txt"

            # DreamGen flat output stem: split__episode_XXXXXX
            dg_stem = f"{split}__{v.stem}"

            work.append((split, v, ep_id, task_hint, src_type, caption_txt, dg_stem))

    if not work:
        print("No videos found — nothing to do.")
        return

    # Items where caption is missing (or overwrite requested)
    todo = [w for w in work if args.overwrite or not w[5].exists()]
    print(f"\nTotal videos: {len(work)}  |  Need caption: {len(todo)}")

    if args.dry_run:
        print("\n[dry_run] Sample items:")
        for w in todo[:5]:
            split, v, ep_id, hint, src, cap_txt, dg_stem = w
            print(f"  [{split}] ep={ep_id:06d} src={src} hint='{hint[:60]}' dg_stem={dg_stem}")
        print("... (dry_run, exiting)")
        return

    # -----------------------------------------------------------------------
    # Run Qwen3-VL captioning
    # -----------------------------------------------------------------------
    if todo:
        print(f"\nLoading {args.model} ...")
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        llm = LLM(
            model=args.model,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            limit_mm_per_prompt={"image": args.num_video_frames},
            dtype="auto",
        )
        sampling = SamplingParams(
            temperature=0.2, top_p=0.9, max_tokens=args.max_new_tokens
        )

        B = args.batch_size
        for i in range(0, len(todo), B):
            chunk = todo[i : i + B]
            prompts_input = []
            for split, v, ep_id, task_hint, src_type, cap_txt, dg_stem in chunk:
                try:
                    frames = sample_frames(v, args.num_video_frames)
                except Exception as e:
                    print(f"  [warn] frame read failed {v.name}: {e}")
                    frames = []
                if not frames:
                    prompts_input.append(None)
                    continue
                msgs = build_messages(frames, task_hint)
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts_input.append({
                    "prompt": text,
                    "multi_modal_data": {"image": frames},
                })

            # Filter out None (frame-read failures)
            valid = [(chunk[j], p) for j, p in enumerate(prompts_input) if p is not None]
            if not valid:
                continue

            outs = llm.generate([p for _, p in valid], sampling)
            for (split, v, ep_id, task_hint, src_type, cap_txt, dg_stem), out in zip(
                [item for item, _ in valid], outs
            ):
                caption = out.outputs[0].text.strip().replace("\n", " ")
                cap_txt.parent.mkdir(parents=True, exist_ok=True)
                cap_txt.write_text(caption)

            done = min(i + B, len(todo))
            print(f"  captioned {done}/{len(todo)}")

    # -----------------------------------------------------------------------
    # Write DreamGen flat structure (videos symlink + metas txt)
    # -----------------------------------------------------------------------
    print("\nWriting humanoid_dreamgen/ flat structure...")
    written = defaultdict(int)

    for split, v, ep_id, task_hint, src_type, cap_txt, dg_stem in work:
        if not cap_txt.exists():
            print(f"  [skip] no caption for {v.stem} ({split})")
            continue

        caption = cap_txt.read_text().strip()
        if not caption:
            continue

        # Symlink video
        dg_vid = dreamgen_videos / f"{dg_stem}.mp4"
        if not dg_vid.exists():
            dg_vid.symlink_to(v.resolve())

        # Write meta txt
        dg_txt = dreamgen_metas / f"{dg_stem}.txt"
        if not dg_txt.exists() or args.overwrite:
            dg_txt.write_text(caption)

        written[split] += 1

    print("\n=== Done ===")
    for split in args.splits:
        n = written[split]
        print(f"  {split}: {n} episodes written to humanoid_dreamgen/")
    print(f"  Target: {dreamgen_target}")
    print("\nNext steps:")
    print("  1. bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh  (DATASETS=humanoid)")
    print("  2. NPROC=8 bash finetuning/dreamgen/launch.sh predict2_video2world_training_2b_humanoid_singleview")


if __name__ == "__main__":
    main()
