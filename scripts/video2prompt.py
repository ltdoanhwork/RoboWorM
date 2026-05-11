"""
Caption robot videos with Qwen3-VL-30B-A3B-Instruct-FP8 (vLLM) and write
DreamGen-compatible inputs.

For every input video <root>/<subset>/videos/.../episode_XXXXXX.mp4 we write
into <out>/<subset>/:
  - episode_XXXXXX.png       first frame (DreamGen image-conditioned input)
  - episode_XXXXXX.txt       Qwen3-VL caption (one-line action prompt)
and finally <out>/<subset>/batch_input.json that points DreamGen at the pair.

Usage:
  python -m scripts.video2prompt \
    --src_root  ./datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1 \
    --out_root  ./sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1 \
    --subsets   DreamDojo-HV_Eval EgoDex_Eval In-lab_Eval GR1_robot \
    --model     Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --output_video_root ./output/dreamgen_singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1
"""

import argparse
import glob
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

CAPTION_PROMPT = (
    "You are labelling a short robot tele-operation clip for a video-generation "
    "model. In ONE concise English sentence (max ~25 words), describe what the "
    "robot is doing — focus on the manipulated object(s), which hand/arm is "
    "used, and the action verb. Do NOT describe the background, camera, or "
    "lighting. Output the sentence only, no quotes, no prefix."
)
import os
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src_root", required=True, type=str,
                   help="Dataset root containing <subset>/videos/chunk-XXX/<view>/episode_*.mp4")
    p.add_argument("--out_root", required=True, type=str)
    p.add_argument("--subsets", nargs="+", default=None,
                   help="Subset folder names. Defaults to all subdirs of src_root.")
    p.add_argument("--view_subdir", default="observation.images.ego_view_freq20",
                   help="Camera-view subfolder under videos/chunk-XXX/")
    p.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    p.add_argument("--output_video_root", default=None,
                   help="Where DreamGen will write generated mp4s; baked into batch_input.json.")
    p.add_argument("--num_video_frames", type=int, default=8,
                   help="Frames passed to Qwen3-VL per clip.")
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=8192)
    p.add_argument("--limit_per_subset", type=int, default=0,
                   help="Cap videos per subset (0 = no cap). Useful for smoke tests.")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-caption even if .txt already exists.")
    return p.parse_args()


def list_videos(src_root: Path, subset: str, view_subdir: str):
    pattern = str(src_root / subset / "videos" / "chunk-*" / view_subdir / "episode_*.mp4")
    return sorted(glob.glob(pattern))


def save_first_frame(video_path: str, png_path: Path):
    png_path.parent.mkdir(parents=True, exist_ok=True)
    if png_path.exists():
        return
    frame = iio.imread(video_path, index=0, plugin="pyav")
    Image.fromarray(frame).save(png_path)


def sample_frames(video_path: str, n: int):
    frames = iio.imread(video_path, plugin="pyav")  # (T, H, W, C)
    if frames.ndim == 3:
        frames = frames[None]
    T = frames.shape[0]
    if T <= n:
        idx = list(range(T))
    else:
        step = T / n
        idx = [int(i * step) for i in range(n)]
    return [Image.fromarray(frames[i]) for i in idx]


def build_messages(video_pil_frames):
    content = [{"type": "image", "image": img} for img in video_pil_frames]
    content.append({"type": "text", "text": CAPTION_PROMPT})
    return [{"role": "user", "content": content}]


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    subsets = args.subsets or [p.name for p in src_root.iterdir() if p.is_dir()]

    # Collect work first so we initialise the LLM only when there is something to do.
    work = []  # (subset, video_path, png_path, txt_path)
    for subset in subsets:
        videos = list_videos(src_root, subset, args.view_subdir)
        if args.limit_per_subset:
            videos = videos[: args.limit_per_subset]
        for v in videos:
            stem = Path(v).stem
            png = out_root / subset / f"{stem}.png"
            txt = out_root / subset / f"{stem}.txt"
            work.append((subset, v, png, txt))
        print(f"[{subset}] {len(videos)} videos")

    if not work:
        print("No videos found. Exiting.")
        return

    todo = [w for w in work if args.overwrite or not w[3].exists()]
    print(f"Total: {len(work)}  to caption: {len(todo)}")

    # First frames are needed for ALL items (they are DreamGen's image input),
    # not only for items missing a caption.
    print("Extracting first frames...")
    for _subset, v, png, _txt in work:
        try:
            save_first_frame(v, png)
        except Exception as e:
            print(f"  [warn] first-frame failed for {v}: {e}")

    if todo:
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
        sampling = SamplingParams(temperature=0.2, top_p=0.9,
                                  max_tokens=args.max_new_tokens)

        BATCH = 8
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            prompts = []
            for _subset, v, _png, _txt in chunk:
                frames = sample_frames(v, args.num_video_frames)
                msgs = build_messages(frames)
                text = processor.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append({
                    "prompt": text,
                    "multi_modal_data": {"image": frames},
                })
            outs = llm.generate(prompts, sampling)
            for (subset, v, _png, txt), out in zip(chunk, outs):
                caption = out.outputs[0].text.strip().replace("\n", " ")
                txt.parent.mkdir(parents=True, exist_ok=True)
                txt.write_text(caption)
            print(f"  captioned {min(i + BATCH, len(todo))}/{len(todo)}")

    # Build batch_input.json per subset.
    out_video_root = args.output_video_root
    for subset in subsets:
        items = []
        sub_out = out_root / subset
        for png in sorted(sub_out.glob("*.png")):
            txt = png.with_suffix(".txt")
            if not txt.exists():
                continue
            prompt = txt.read_text().strip()
            if out_video_root:
                output_video = str(Path(out_video_root) / subset / f"{png.stem}.mp4")
            else:
                output_video = str(sub_out / f"{png.stem}.mp4")
            items.append({
                "input_video": str(png),
                "prompt": prompt,
                "output_video": output_video,
            })
        if items:
            with open(sub_out / "batch_input.json", "w") as f:
                json.dump(items, f, indent=4)
            print(f"[{subset}] wrote batch_input.json with {len(items)} items")


if __name__ == "__main__":
    main()
