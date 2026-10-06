"""
Classify GR1_robot episodes as 'markovian' or 'non_markovian' using
Qwen3-VL-30B-A3B-Instruct-FP8 served via vLLM (text-only, task strings).

Reads episodes.jsonl from GR1_robot dataset, classifies unique task strings,
then maps back to episodes.

Usage:
    python scripts/tools/data/classify_markovian_gr1robot.py

    python scripts/tools/data/classify_markovian_gr1robot.py \
        --episodes-jsonl datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/meta/episodes.jsonl \
        --cache-file     data/labels/gr1robot_task_labels_cache.jsonl \
        --output-csv     data/labels/gr1robot_task_labels.csv \
        --base-url       http://localhost:8000/v1 \
        --model          Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
        --batch-size     50 \
        --max-retries    2
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt (same rules as gr1.py)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a robot manipulation task classifier.

Classify each task description as exactly one of:
- "markovian": a SINGLE atomic manipulation action. The robot only needs to observe the current state to act. No memory of previous steps is required.
  Typical examples: "pick up red apple to silver pan", "place cup on table", "push ball left", "open drawer", "remove block from stack"

- "non_markovian": involves MULTIPLE sequential steps, OR requires the robot to remember what it has already done, OR involves ongoing/continuous actions.
  Typical examples: "pick up X then place it into Y while holding Z", "stir repeatedly", "sort all blocks", "spray water onto plants", "wipe table surface"

Rules:
- Connectors like "then", "while", "after", "followed by", "and" (joining two actions) → non_markovian
- Ongoing / repetitive actions (stir, spray, wipe, fold, rotate, pour) → non_markovian
- Both arms doing different simultaneous things → non_markovian
- Simple single pick-and-place, push, remove, open/close → markovian
- "unpick up and place X to Y" = pick-and-place = markovian

Reply with ONLY a valid JSON array (no markdown fences, no extra text):
[{"task_index": <int>, "label": "markovian"|"non_markovian", "reason": "<≤12 words>"}]"""

USER_TEMPLATE = "Classify these tasks:\n{tasks_json}"


def clean_prompt(text: str) -> str:
    text = text.replace("locked waist: ", "").replace("grid_", "")
    text = text.replace("_", " ").strip()
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_unique_tasks(episodes_jsonl: Path) -> list[dict]:
    """Return unique cleaned tasks with their first-seen episode index as task_index."""
    seen: dict[str, int] = {}   # task_text -> synthetic task_index
    with open(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for raw in obj.get("tasks", []):
                text = clean_prompt(raw.strip())
                if text and text not in seen:
                    seen[text] = len(seen)
    return [{"task_index": idx, "task": text} for text, idx in seen.items()]


def load_cache(cache_file: Path) -> dict[int, dict]:
    cache: dict[int, dict] = {}
    if not cache_file.exists():
        return cache
    with open(cache_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["task_index"]] = obj
    return cache


def append_to_cache(cache_file: Path, results: list[dict]) -> None:
    with open(cache_file, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def call_llm(base_url: str, model: str, batch: list[dict], max_retries: int) -> list[dict]:
    tasks_json = json.dumps(batch, ensure_ascii=False)
    user_msg = USER_TEMPLATE.format(tasks_json=tasks_json)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    for attempt in range(1, max_retries + 2):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions", json=payload, timeout=120
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed: list[dict] = json.loads(raw)
            index_map = {t["task_index"]: t["task"] for t in batch}
            results = []
            for item in parsed:
                ti = int(item["task_index"])
                label = item.get("label", "").strip().lower()
                if label not in ("markovian", "non_markovian"):
                    label = "parse_error"
                results.append({
                    "task_index": ti,
                    "task": index_map.get(ti, ""),
                    "label": label,
                    "reason": item.get("reason", ""),
                })
            return results

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt <= max_retries:
                time.sleep(2)
            else:
                print(f"  [warn] Parse failed: {e}", file=sys.stderr)
                return [
                    {"task_index": t["task_index"], "task": t["task"],
                     "label": "parse_error", "reason": str(e)}
                    for t in batch
                ]
        except requests.RequestException as e:
            if attempt <= max_retries:
                time.sleep(5)
            else:
                print(f"  [error] HTTP error: {e}", file=sys.stderr)
                return [
                    {"task_index": t["task_index"], "task": t["task"],
                     "label": "api_error", "reason": str(e)}
                    for t in batch
                ]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify GR1_robot tasks via Qwen3-VL")
    parser.add_argument(
        "--episodes-jsonl", type=Path,
        default=Path("datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/meta/episodes.jsonl"),
    )
    parser.add_argument("--cache-file", type=Path, default=Path("data/labels/gr1robot_task_labels_cache.jsonl"))
    parser.add_argument("--output-csv",  type=Path, default=Path("data/labels/gr1robot_task_labels.csv"))
    parser.add_argument("--base-url",    type=str,  default="http://localhost:8000/v1")
    parser.add_argument("--model",       type=str,  default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8")
    parser.add_argument("--batch-size",  type=int,  default=50)
    parser.add_argument("--max-retries", type=int,  default=2)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]

    def resolve(p: Path) -> Path:
        return workspace / p if not p.is_absolute() else p

    episodes_jsonl = resolve(args.episodes_jsonl)
    cache_file     = resolve(args.cache_file)
    output_csv     = resolve(args.output_csv)
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] Loading unique tasks from {episodes_jsonl}")
    all_tasks = load_unique_tasks(episodes_jsonl)
    print(f"[init] Unique tasks: {len(all_tasks)}")

    cache = load_cache(cache_file)
    print(f"[init] Already classified: {len(cache)}")

    pending = [t for t in all_tasks if t["task_index"] not in cache]
    print(f"[init] Remaining to classify: {len(pending)}")

    if pending:
        batches = [pending[i: i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
        print(f"[run]  {len(batches)} batches × {args.batch_size}, model={args.model}")
        for batch in tqdm(batches, desc="Classifying", unit="batch"):
            results = call_llm(args.base_url, args.model, batch, args.max_retries)
            append_to_cache(cache_file, results)
            for r in results:
                cache[r["task_index"]] = r

    rows = list(cache.values())
    df = pd.DataFrame(rows, columns=["task_index", "task", "label", "reason"])
    df = df.sort_values("task_index").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[saved] {output_csv}  ({len(df)} rows)")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
