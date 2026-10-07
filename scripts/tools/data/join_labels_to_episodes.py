"""
Join task_labels.csv back to meta/episodes.jsonl, filter to the first
NUM_CHUNKS chunks, and sample SAMPLE_PER_LABEL episodes per label.

Usage:
    python scripts/tools/data/join_labels_to_episodes.py

    # Optional flags:
    python scripts/tools/data/join_labels_to_episodes.py \
        --episodes-jsonl   datasets/droid_1.0.1_20chunks/meta/episodes.jsonl \
        --task-labels-csv  tmp/20261005_refactor/misc/data_labels/task_labels.csv \
        --output-csv       tmp/20261005_refactor/misc/data_labels/episode_markovian_split.csv \
        --num-chunks       20 \
        --sample-per-label 130 \
        --seed             42
"""

import argparse
import json
from pathlib import Path

import pandas as pd


CHUNKS_SIZE = 1000  # episodes per chunk (from info.json)


def load_episodes(episodes_jsonl: Path, max_episode: int) -> pd.DataFrame:
    rows = []
    with open(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ep_idx = obj.get("episode_index", -1)
            if ep_idx < 0 or ep_idx >= max_episode:
                continue
            length = obj.get("length")
            tasks = obj.get("tasks", [])
            task_text = next((t for t in tasks if t and t.strip()), "")
            rows.append(
                {
                    "episode_index": ep_idx,
                    "length": length,
                    "task": task_text,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Join task labels to episodes and sample")
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        default=Path("datasets/droid_1.0.1_20chunks/meta/episodes.jsonl"),
    )
    parser.add_argument(
        "--task-labels-csv",
        type=Path,
        default=Path("tmp/20261005_refactor/misc/data_labels/task_labels.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("tmp/20261005_refactor/misc/data_labels/episode_markovian_split.csv"),
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=20,
        help="Only include episodes from the first N chunks",
    )
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=130,
        help="Number of episodes to sample per label (0 = keep all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]

    def resolve(p: Path) -> Path:
        return workspace / p if not p.is_absolute() else p

    episodes_jsonl = resolve(args.episodes_jsonl)
    task_labels_csv = resolve(args.task_labels_csv)
    output_csv = resolve(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    max_episode = args.num_chunks * CHUNKS_SIZE
    print(f"[load] episodes (chunks 0–{args.num_chunks - 1}, episode_index < {max_episode})")
    episodes_df = load_episodes(episodes_jsonl, max_episode)
    print(f"       {len(episodes_df)} episodes loaded")

    print(f"[load] task labels: {task_labels_csv}")
    labels_df = pd.read_csv(task_labels_csv)
    print(f"       {len(labels_df)} task labels loaded")
    print(f"       label counts:\n{labels_df['label'].value_counts().to_string()}")

    # Join on task text (episodes.jsonl has no task_index field)
    merged = episodes_df.merge(
        labels_df[["task_index", "task", "label", "reason"]],
        on="task",
        how="left",
    )

    unmatched = merged["label"].isna().sum()
    if unmatched:
        print(f"[warn] {unmatched} episodes have no matching label (empty/unknown task)")
    merged["label"] = merged["label"].fillna("unknown")
    merged["reason"] = merged["reason"].fillna("")

    # Save full joined table before sampling
    full_csv = output_csv.with_name(output_csv.stem + "_full.csv")
    merged.sort_values("episode_index").to_csv(full_csv, index=False)
    print(f"\n[saved] full join → {full_csv}  ({len(merged)} rows)")
    print(merged["label"].value_counts().to_string())

    # Sample per label
    if args.sample_per_label > 0:
        sampled_parts = []
        for label, group in merged.groupby("label"):
            if label in ("unknown", "parse_error", "api_error"):
                continue
            n = min(args.sample_per_label, len(group))
            sampled_parts.append(group.sample(n=n, random_state=args.seed))
            print(f"[sample] {label}: {n}/{len(group)} episodes selected")

        sampled = pd.concat(sampled_parts).sort_values("episode_index").reset_index(drop=True)
        sampled.to_csv(output_csv, index=False)
        print(f"\n[saved] sampled → {output_csv}  ({len(sampled)} rows)")
        print(sampled["label"].value_counts().to_string())
    else:
        merged.sort_values("episode_index").to_csv(output_csv, index=False)
        print(f"\n[saved] {output_csv}  ({len(merged)} rows)")


if __name__ == "__main__":
    main()
