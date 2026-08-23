#!/usr/bin/env python3
"""Resolve a human-readable "prompt" per item in data_for_web/manifest.json.

Cascades per model family (see .cursor exploration notes):
  - Ctrl-World (ctrlworld / ctrlworld_4view_grid): `metadata.json["instruction"]`
    in the matching input episode dir (own method, else dense, else augment).
  - DreamGen singleview: `.txt` sidecar next to the pred mp4, `[Prompt]` block.
  - DreamGen multiview: `batch_input*.json` (list of {input_video,prompt,...})
    anywhere under the input tree; match by episode number + split in path.
  - DreamDojo: no prompt is ever fed to the model (empty string at inference);
    best-effort human label from the source LeRobot dataset's tasks/episodes.
  - Fallback: humanize the `task` field (underscores -> spaces).

Writes the enriched manifest back in place and reports per-source coverage.

Usage:
    python scripts/resolve_prompts.py
    python scripts/resolve_prompts.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLING = REPO / "sampling_dataset"
DATASETS = REPO / "datasets"

EP_NUM_RE = re.compile(r"episode_?(\d{5,7})")


def humanize(s: str) -> str:
    s = (s or "").replace("_", " ").strip()
    return (s[:1].upper() + s[1:]) if s else ""


def ctrlworld_instruction(item: dict) -> str | None:
    model = item["model"]
    for cand in (item["source_method"], "dense", "augment"):
        p = (
            SAMPLING
            / cand
            / item["embodiment"]
            / "input"
            / "multiview"
            / model
            / item["split"]
            / item["unit_key"]
            / "metadata.json"
        )
        if p.is_file():
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            instr = (data.get("instruction") or "").strip()
            if instr:
                return instr
    return None


def dreamgen_singleview_prompt(item: dict) -> str | None:
    src = SAMPLING / item["source_path"]
    txt = src.with_suffix(".txt")
    if not txt.is_file():
        return None
    try:
        content = txt.read_text()
    except Exception:
        return None
    m = re.search(r"\[Prompt\]\s*\n(.*?)\n\[Negative Prompt\]", content, re.S)
    if m:
        return m.group(1).strip()
    return None


@lru_cache(maxsize=None)
def _dreamgen_multiview_index(cand_method: str, embodiment: str) -> tuple:
    root = SAMPLING / cand_method / embodiment / "input" / "multiview" / "dreamgen"
    entries = []
    if not root.is_dir():
        return tuple(entries)
    for jf in root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue
        rows = data if isinstance(data, list) else [data]
        for row in rows:
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            vid = row.get("input_video") or row.get("output_video") or ""
            if prompt and vid:
                entries.append((str(jf), vid, prompt))
    return tuple(entries)


def dreamgen_multiview_prompt(item: dict) -> str | None:
    m = EP_NUM_RE.search(item["episode"])
    if not m:
        return None
    ep_num = m.group(1).lstrip("0") or "0"
    ep_pat = re.compile(rf"episode_?0*{re.escape(ep_num)}(?:[._]|$)")
    split = item["split"]
    for cand in (item["source_method"], "dense", "augment"):
        for _jf, vid, prompt in _dreamgen_multiview_index(cand, item["embodiment"]):
            if split not in vid:
                continue
            if ep_pat.search(vid):
                return prompt.strip()
    return None


@lru_cache(maxsize=None)
def _episodes_index(embodiment: str, split: str) -> dict:
    p = DATASETS / embodiment / "singleview" / split / "meta" / "episodes.jsonl"
    out = {}
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        tasks = d.get("tasks") or []
        if "episode_index" in d and tasks:
            out[d["episode_index"]] = tasks[0]
    return out


@lru_cache(maxsize=None)
def _bimanual_task_label(task_name: str, split: str) -> str | None:
    for cand_split in (split, "makovian", "non_makovian"):
        p = DATASETS / "bimanual" / "singleview" / cand_split / task_name / "meta" / "tasks.jsonl"
        if p.is_file():
            try:
                d = json.loads(p.read_text().splitlines()[0])
                return d.get("task")
            except Exception:
                continue
    return None


def dreamdojo_prompt(item: dict) -> str | None:
    emb = item["embodiment"]
    if emb == "bimanual":
        label = _bimanual_task_label(item["task"], item["split"])
        return label.strip().capitalize() if label else None
    m = EP_NUM_RE.search(item["episode"])
    if not m:
        return None
    ep_idx = int(m.group(1))
    idx = _episodes_index(emb, item["split"])
    task = idx.get(ep_idx)
    return task.strip() if task else None


def resolve_prompt(item: dict) -> str:
    model = item["model"]
    view = item["view"]
    try:
        if model in ("ctrlworld", "ctrlworld_4view_grid"):
            p = ctrlworld_instruction(item)
        elif model == "dreamgen" and view == "singleview":
            p = dreamgen_singleview_prompt(item)
        elif model == "dreamgen" and view == "multiview":
            p = dreamgen_multiview_prompt(item)
        elif model == "dreamdojo":
            p = dreamdojo_prompt(item)
        else:
            p = None
    except Exception:
        p = None
    if p:
        return p.strip()
    return humanize(item.get("task", ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=SAMPLING / "data_for_web" / "manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    items = manifest["items"]

    by_source = defaultdict(lambda: [0, 0])  # key -> [n_real, n_fallback]
    for it in items:
        prompt = resolve_prompt(it)
        it["prompt"] = prompt
        key = f"{it['model']}/{it['view']}"
        fallback = humanize(it.get("task", ""))
        by_source[key][0 if prompt != fallback else 1] += 1

    print("=== prompt resolution coverage (model/view: n_specific / n_fallback-to-task) ===")
    for k in sorted(by_source):
        real, fb = by_source[k]
        total = real + fb
        print(f"  {k:28s} specific={real:4d}  fallback={fb:4d}  ({real/total:.0%} specific)")

    print("\n=== sample ===")
    for it in items[:3] + items[200:203]:
        print(f"  [{it['id']}] {it['model']}/{it['view']} :: {it['prompt']!r}")

    if args.dry_run:
        print("\n[--dry-run] not writing")
        return 0

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote prompts into {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
