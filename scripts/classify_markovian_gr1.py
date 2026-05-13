"""
Classify GR1 humanoid episodes as 'markovian' or 'non_markovian' using
Qwen3-VL-30B-A3B-Instruct-FP8 served via vLLM.

Reads prompts from the two batch_input.json files (DreamDojo-HV_Eval and
EgoDex_Eval), classifies all ~133 episodes, and outputs:
  - scripts/gr1_task_labels.csv          -- all episodes with label + reason
  - scripts/gr1_episode_sampled.csv      -- balanced sample (equal per label
                                            AND equal across the two sources)

Usage:
    # 1. Start the vLLM server first:
    #    bash scripts/launch_qwen3vl_server.sh

    # 2. Run (text-only):
    python scripts/classify_markovian_gr1.py

    # Run with vision (sends input PNG image to model):
    python scripts/classify_markovian_gr1.py --use-vision --batch-size 5

    # Optional flags:
    python scripts/classify_markovian_gr1.py \
        --dreamdojo-json  sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/batch_input.json \
        --egodex-json     sampling_dataset/humanoid/singleview/input/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input.json \
        --cache-file      scripts/gr1_task_labels_cache.jsonl \
        --output-csv      scripts/gr1_task_labels.csv \
        --sampled-csv     scripts/gr1_episode_sampled.csv \
        --sample-per-label-per-source  33 \
        --base-url        http://localhost:8000/v1 \
        --model           Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
        --batch-size      5 \
        --max-image-px    560 \
        --max-retries     2 \
        --seed            42
"""

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a robot manipulation task classifier.

You will be given one or more episodes. Each episode has:
- An image showing the robot's initial state (if provided)
- A task description prompt

Classify each task as exactly one of:
- "markovian": a SINGLE atomic manipulation action. The robot only needs to observe the current state to act. No memory of previous steps is required.
  Typical examples: "picks up a banana", "places a cup on the table", "pushes a ball", "removes a block from the stack", "opens a drawer"

- "non_markovian": involves MULTIPLE sequential steps, OR requires the robot to remember what it has already done, OR involves ongoing/continuous actions that track progress.
  Typical examples: "picks up X then places it into Y while holding Z", "stirs … repeatedly", "topples a line of tiles", "uses both arms to lift and rotate", "sprays water onto plants"

Rules:
- Connectors like "then", "while", "after", "followed by" → non_markovian
- Ongoing / repetitive actions (stir, spray, wipe, fold, rotate) → non_markovian
- Both arms doing different simultaneous things → non_markovian
- Simple single pick, place, push, remove, open/close → markovian
- A pick-and-place is markovian (one action) even with two locations
- Use the image to resolve ambiguity (e.g. verify object count, arm configuration)

Reply with ONLY a valid JSON array (no markdown fences, no extra text):
[{"id": "<id>", "label": "markovian"|"non_markovian", "reason": "<≤12 words>"}]"""

USER_TEMPLATE = "Classify these tasks:\n{tasks_json}"

# ---------------------------------------------------------------------------
# Vision helpers
# ---------------------------------------------------------------------------


def encode_image_b64(image_path: Path, max_px: int = 560) -> str:
    """Load image, resize so longest side <= max_px, return base64 PNG string."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_vision_user_content(batch: list[dict], max_px: int) -> list[dict]:
    """Build multimodal content list: interleave images with per-episode text."""
    content = []
    for i, row in enumerate(batch):
        img_path = Path(row["input_image"])
        if img_path.exists():
            b64 = encode_image_b64(img_path, max_px)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
            content.append({
                "type": "text",
                "text": f'Episode {i+1} (id="{row["id"]}"): {row["prompt"]}',
            })
        else:
            content.append({
                "type": "text",
                "text": f'Episode {i+1} (id="{row["id"]}"): {row["prompt"]} [image not found]',
            })
    content.append({
        "type": "text",
        "text": "\nNow classify all episodes above. Reply with only the JSON array.",
    })
    return content

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_batch_json(path: Path, source: str) -> list[dict]:
    data = json.loads(path.read_text())
    rows = []
    for item in data:
        out_video = item.get("output_video", "")
        episode_name = Path(out_video).stem  # e.g. episode_000000
        rows.append(
            {
                "id": f"{source}/{episode_name}",
                "source": source,
                "episode": episode_name,
                "input_image": item.get("input_video", ""),   # it's a .png despite key name
                "output_video": out_video,
                "prompt": item.get("prompt", "").strip(),
            }
        )
    return rows


def load_cache(cache_file: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not cache_file.exists():
        return cache
    with open(cache_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["id"]] = obj
    return cache


def append_to_cache(cache_file: Path, results: list[dict]) -> None:
    with open(cache_file, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def call_llm(
    base_url: str,
    model: str,
    batch: list[dict],
    max_retries: int,
    use_vision: bool = False,
    max_image_px: int = 560,
) -> list[dict]:
    if use_vision:
        user_content = build_vision_user_content(batch, max_image_px)
    else:
        payload_tasks = [{"id": r["id"], "task": r["prompt"]} for r in batch]
        user_content = USER_TEMPLATE.format(tasks_json=json.dumps(payload_tasks, ensure_ascii=False))

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(1, max_retries + 2):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed: list[dict] = json.loads(raw)
            id_map = {r["id"]: r for r in batch}
            results = []
            for item in parsed:
                rid = item.get("id", "")
                label = item.get("label", "").strip().lower()
                if label not in ("markovian", "non_markovian"):
                    label = "parse_error"
                src_row = id_map.get(rid, {})
                results.append(
                    {
                        "id": rid,
                        "source": src_row.get("source", ""),
                        "episode": src_row.get("episode", ""),
                        "input_image": src_row.get("input_image", ""),
                        "output_video": src_row.get("output_video", ""),
                        "prompt": src_row.get("prompt", ""),
                        "label": label,
                        "reason": item.get("reason", ""),
                    }
                )
            return results

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt <= max_retries:
                time.sleep(2)
            else:
                print(f"  [warn] Parse failed: {e}", file=sys.stderr)
                return [{**r, "label": "parse_error", "reason": str(e)} for r in batch]
        except requests.RequestException as e:
            if attempt <= max_retries:
                time.sleep(5)
            else:
                print(f"  [error] HTTP error: {e}", file=sys.stderr)
                return [{**r, "label": "api_error", "reason": str(e)} for r in batch]

    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GR1 humanoid tasks")
    parser.add_argument(
        "--dreamdojo-json",
        type=Path,
        default=Path(
            "sampling_dataset/humanoid/singleview/input"
            "/PhysicalAI-Robotics-GR00T-Teleop-GR1/DreamDojo-HV_Eval/batch_input.json"
        ),
    )
    parser.add_argument(
        "--egodex-json",
        type=Path,
        default=Path(
            "sampling_dataset/humanoid/singleview/input"
            "/PhysicalAI-Robotics-GR00T-Teleop-GR1/EgoDex_Eval/batch_input.json"
        ),
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("scripts/gr1_task_labels_cache.jsonl"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/gr1_task_labels.csv"),
    )
    parser.add_argument(
        "--sampled-csv",
        type=Path,
        default=Path("scripts/gr1_episode_sampled.csv"),
    )
    parser.add_argument(
        "--sample-per-label-per-source",
        type=int,
        default=33,
        help=(
            "Episodes per (label × source) cell. "
            "e.g. 33 → up to 33 markovian from DreamDojo + 33 from EgoDex, "
            "same for non_markovian → 132 total max. 0 = keep all."
        ),
    )
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-vision",
        action="store_true",
        default=False,
        help="Send input PNG image to the model alongside the prompt (recommended batch-size 5)",
    )
    parser.add_argument(
        "--max-image-px",
        type=int,
        default=560,
        help="Resize images so longest side <= this value before encoding (saves tokens)",
    )
    args = parser.parse_args()

    workspace = Path(__file__).parent.parent

    def resolve(p: Path) -> Path:
        return workspace / p if not p.is_absolute() else p

    dreamdojo_json = resolve(args.dreamdojo_json)
    egodex_json = resolve(args.egodex_json)
    cache_file = resolve(args.cache_file)
    output_csv = resolve(args.output_csv)
    sampled_csv = resolve(args.sampled_csv)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load both sources
    all_rows: list[dict] = []
    all_rows += load_batch_json(dreamdojo_json, "DreamDojo-HV_Eval")
    all_rows += load_batch_json(egodex_json, "EgoDex_Eval")
    print(f"[init] Loaded {len(all_rows)} episodes total")

    # Resume from cache
    cache = load_cache(cache_file)
    print(f"[init] Already classified: {len(cache)}")

    pending = [r for r in all_rows if r["id"] not in cache]
    print(f"[init] Remaining to classify: {len(pending)}")

    if pending:
        batches = [
            pending[i : i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        vision_info = f", vision=ON (max_px={args.max_image_px})" if args.use_vision else ", vision=OFF"
        print(f"[run]  {len(batches)} batch(es), model={args.model}, endpoint={args.base_url}{vision_info}")

        for batch in tqdm(batches, desc="Classifying", unit="batch"):
            results = call_llm(
                args.base_url, args.model, batch, args.max_retries,
                use_vision=args.use_vision, max_image_px=args.max_image_px,
            )
            append_to_cache(cache_file, results)
            for r in results:
                cache[r["id"]] = r

    # Build full CSV
    rows = list(cache.values())
    df = pd.DataFrame(rows)
    # Preserve original ordering (DreamDojo first, then EgoDex, sorted by episode)
    df["_order"] = df["id"].map({r["id"]: i for i, r in enumerate(all_rows)})
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[saved] Full labels → {output_csv}  ({len(df)} rows)")
    print(df.groupby(["source", "label"]).size().to_string())

    # Sample equally per (label × source)
    if args.sample_per_label_per_source > 0:
        parts = []
        for (source, label), group in df.groupby(["source", "label"]):
            if label in ("parse_error", "api_error", "unknown"):
                continue
            n = min(args.sample_per_label_per_source, len(group))
            parts.append(group.sample(n=n, random_state=args.seed))
            print(f"[sample] {source} / {label}: {n}/{len(group)}")

        sampled = (
            pd.concat(parts)
            .sort_values(["source", "episode"])
            .reset_index(drop=True)
        )
        sampled.to_csv(sampled_csv, index=False)
        print(f"\n[saved] Sampled → {sampled_csv}  ({len(sampled)} rows)")
        print(sampled.groupby(["source", "label"]).size().to_string())
    else:
        df.to_csv(sampled_csv, index=False)
        print(f"\n[saved] Sampled (all) → {sampled_csv}")


if __name__ == "__main__":
    main()
