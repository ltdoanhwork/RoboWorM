"""
Classify DROID task strings as 'markovian' or 'non_markovian' using
Qwen3-VL-30B-A3B-Instruct-FP8 served via vLLM (OpenAI-compatible API).

Only tasks that appear in the first NUM_CHUNKS chunks are classified
(episodes 0 … NUM_CHUNKS*1000-1).

Usage:
    # 1. Start the vLLM server first:
    #    bash scripts/launch_qwen3vl_server.sh

    # 2. Run classification:
    python scripts/classify_markovian_qwen.py

    # Optional flags:
    python scripts/classify_markovian_qwen.py \
        --episodes-jsonl datasets/droid_1.0.1_20chunks/meta/episodes.jsonl \
        --tasks-jsonl    datasets/droid_1.0.1_20chunks/meta/tasks.jsonl \
        --num-chunks     20 \
        --cache-file     scripts/task_labels_cache.jsonl \
        --output-csv     scripts/task_labels.csv \
        --base-url       http://localhost:8000/v1 \
        --model          Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
        --batch-size     50 \
        --max-retries    2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a robot manipulation task classifier.

Classify each task description as exactly one of:
- "markovian": a SINGLE atomic manipulation action. The robot only needs to observe the current state to act. No memory of previous steps is required.
  Typical examples: "Pick up the apple", "Open the drawer", "Push the block to the left", "Close the lid", "Place the cup on the table"

- "non_markovian": involves MULTIPLE sequential steps, OR requires the robot to remember what it has already done. The policy needs history or a plan.
  Typical examples: "Open the bin then put the bottle inside", "First pick up the block, then place it in the bowl", "Sort all blocks by colour", "Move one object then fold the cloth on top"

Rules:
- Connectors like "then", "after", "first … then", "start by … and then", "followed by" → non_markovian
- Ordinal markers: "first", "second", "next", "finally", "lastly" → non_markovian
- Words implying multiple objects / completion: "all", "each", "every", "remaining", "rest of" → non_markovian
- Iterative / repetitive: "repeatedly", "until", "again" → non_markovian
- A single pick-and-place is markovian even if it mentions two locations.

Reply with ONLY a valid JSON array (no markdown fences, no extra text):
[{"task_index": <int>, "label": "markovian"|"non_markovian", "reason": "<≤12 words>"}]"""

USER_TEMPLATE = "Classify these tasks:\n{tasks_json}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_tasks_for_chunks(
    tasks_jsonl: Path,
    episodes_jsonl: Path,
    num_chunks: int,
) -> list[dict]:
    """Return only the unique tasks that appear in the first num_chunks chunks.

    episodes.jsonl stores task strings directly in the 'tasks' list (no task_index).
    We collect active task strings from episodes, then join with tasks.jsonl on text
    to get task_index. Tasks missing from tasks.jsonl get a synthesised negative index.
    """
    chunks_size = 1000
    max_episode = num_chunks * chunks_size

    # Collect unique task strings used by episodes in the target chunks
    active_task_texts: set[str] = set()
    with open(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = obj.get("episode_index", -1)
            if ep_idx < 0 or ep_idx >= max_episode:
                continue
            for t in obj.get("tasks", []):
                t = t.strip()
                if t:
                    active_task_texts.add(t)

    # Build text → task_index map from tasks.jsonl
    text_to_index: dict[str, int] = {}
    with open(tasks_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("task", "").strip()
            if text:
                text_to_index[text] = int(obj["task_index"])

    # Build final list, preserving task_index where known
    tasks = []
    for text in sorted(active_task_texts):
        ti = text_to_index.get(text, -1)
        tasks.append({"task_index": ti, "task": text})

    return tasks


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


def call_llm(
    base_url: str,
    model: str,
    batch: list[dict],
    max_retries: int,
) -> list[dict]:
    tasks_json = json.dumps(batch, ensure_ascii=False, indent=None)
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
                f"{base_url}/chat/completions",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Strip markdown fences if model adds them anyway
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed: list[dict] = json.loads(raw)

            # Validate and normalise
            index_map = {t["task_index"]: t["task"] for t in batch}
            results = []
            for item in parsed:
                ti = int(item["task_index"])
                label = item.get("label", "").strip().lower()
                if label not in ("markovian", "non_markovian"):
                    label = "parse_error"
                results.append(
                    {
                        "task_index": ti,
                        "task": index_map.get(ti, ""),
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
    parser = argparse.ArgumentParser(description="Classify DROID tasks via Qwen3-VL")
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=Path("datasets/droid_1.0.1_20chunks/meta/episodes.jsonl"),
    )
    parser.add_argument(
        "--tasks-jsonl",
        type=Path,
        default=Path("datasets/droid_1.0.1_20chunks/meta/tasks.jsonl"),
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Only classify tasks that appear in the first N chunks",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("scripts/task_labels_cache.jsonl"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/task_labels.csv"),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    # Resolve paths relative to workspace root
    workspace = Path(__file__).parent.parent

    def resolve(p: Path) -> Path:
        return workspace / p if not p.is_absolute() else p

    episodes_jsonl = resolve(args.episodes_jsonl)
    tasks_jsonl = resolve(args.tasks_jsonl)
    cache_file = resolve(args.cache_file)
    output_csv = resolve(args.output_csv)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] Loading tasks for first {args.num_chunks} chunks from {tasks_jsonl}")
    all_tasks = load_tasks_for_chunks(tasks_jsonl, episodes_jsonl, args.num_chunks)
    print(f"[init] Unique non-empty tasks in {args.num_chunks} chunks: {len(all_tasks)}")

    print(f"[init] Loading cache from {cache_file}")
    cache = load_cache(cache_file)
    print(f"[init] Already classified: {len(cache)}")

    pending = [t for t in all_tasks if t["task_index"] not in cache]
    print(f"[init] Remaining to classify: {len(pending)}")

    if not pending:
        print("[done] All tasks already classified. Building CSV ...")
    else:
        batches = [
            pending[i : i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        print(
            f"[run]  {len(batches)} batches × {args.batch_size} tasks, "
            f"model={args.model}, endpoint={args.base_url}"
        )

        for batch in tqdm(batches, desc="Classifying", unit="batch"):
            results = call_llm(args.base_url, args.model, batch, args.max_retries)
            append_to_cache(cache_file, results)
            for r in results:
                cache[r["task_index"]] = r

    # Build final CSV from cache
    rows = list(cache.values())
    df = pd.DataFrame(rows, columns=["task_index", "task", "label", "reason"])
    df = df.sort_values("task_index").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[saved] {output_csv}  ({len(df)} rows)")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
