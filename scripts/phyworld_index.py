"""Walk sampling_dataset and emit one JSONL row per (method, cell, episode).

Resolves pred/dense video paths and the task instruction across the two
output conventions (per-episode folder vs flat episode_*.mp4).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from metrics.phyworld.schema import EMBODIMENTS, METHODS, MODELS, SPLITS, VIEWS

PRED_CANDIDATES = ("pred_all_views.mp4", "full_pred.mp4")


def _instruction_from_dir(ep_dir: Path) -> str:
    mj = ep_dir / "metrics.json"
    if mj.exists():
        try:
            data = json.loads(mj.read_text())
            for k in ("instruction", "prompt", "text"):
                if data.get(k):
                    return str(data[k]).strip()
        except (json.JSONDecodeError, OSError):
            pass
    meta = ep_dir / "metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            if data.get("instruction"):
                return str(data["instruction"]).strip()
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def _instruction_from_txt(txt: Path) -> str:
    if not txt.exists():
        return ""
    text = txt.read_text(errors="ignore")
    # DreamGen sidecar: [Prompt] ... [Negative Prompt]
    if "[Prompt]" in text:
        body = text.split("[Prompt]", 1)[1]
        if "[Negative" in body:
            body = body.split("[Negative", 1)[0]
        return body.strip()
    return text.strip().splitlines()[0] if text.strip() else ""


def resolve_pred(ep_dir_or_mp4: Path) -> Path | None:
    if ep_dir_or_mp4.is_file() and ep_dir_or_mp4.suffix == ".mp4":
        return ep_dir_or_mp4
    if ep_dir_or_mp4.is_dir():
        for name in PRED_CANDIDATES:
            p = ep_dir_or_mp4 / name
            if p.exists():
                return p
    return None


def _dense_counterpart(pred_path: Path, method: str, ds_root: Path) -> Path | None:
    try:
        rel = pred_path.relative_to(ds_root / method)
    except ValueError:
        return None
    cand = ds_root / "dense" / rel
    return cand if cand.exists() else None


def iter_cell(ds_root: Path, method: str, embodiment: str, view: str, model: str, split: str):
    cell = ds_root / method / embodiment / "output" / view / model / split
    if not cell.is_dir():
        return

    # Per-episode folders
    for ep in sorted(cell.glob("episode_*")):
        if ep.is_dir():
            pred = resolve_pred(ep)
            if pred is None:
                continue
            yield {
                "method": method,
                "embodiment": embodiment,
                "view": view,
                "model": model,
                "split": split,
                "episode": ep.name,
                "pred_path": str(pred),
                "dense_path": str(p) if (p := _dense_counterpart(pred, method, ds_root)) else None,
                "instruction": _instruction_from_dir(ep),
                "episode_dir": str(ep),
                "n_views": 3 if pred.name == "pred_all_views.mp4" else 1,
            }
        elif ep.is_file() and ep.suffix == ".mp4":
            yield {
                "method": method,
                "embodiment": embodiment,
                "view": view,
                "model": model,
                "split": split,
                "episode": ep.stem,
                "pred_path": str(ep),
                "dense_path": str(p) if (p := _dense_counterpart(ep, method, ds_root)) else None,
                "instruction": _instruction_from_txt(ep.with_suffix(".txt")),
                "episode_dir": str(ep.parent),
                "n_views": 1,
            }

    # Nested iter_*/task/episode_* (dreamgen multiview / dreamdojo)
    for ep in sorted(cell.glob("**/episode_*")):
        if not ep.is_dir():
            continue
        # already yielded as a direct child
        if ep.parent == cell:
            continue
        pred = resolve_pred(ep)
        if pred is None:
            continue
        yield {
            "method": method,
            "embodiment": embodiment,
            "view": view,
            "model": model,
            "split": split,
            "episode": ep.name,
            "pred_path": str(pred),
            "dense_path": str(p) if (p := _dense_counterpart(pred, method, ds_root)) else None,
            "instruction": _instruction_from_dir(ep),
            "episode_dir": str(ep),
            "n_views": 3 if pred.name == "pred_all_views.mp4" else 1,
            "task": ep.parent.name,
        }


def walk(
    ds_root: Path,
    methods=None,
    embodiment=None,
    view=None,
    model=None,
    split=None,
):
    methods = list(methods or METHODS)
    embodiments = [embodiment] if embodiment else list(EMBODIMENTS)
    views = [view] if view else list(VIEWS)
    models = [model] if model else list(MODELS)
    splits = [split] if split else list(SPLITS)
    for m in methods:
        for e in embodiments:
            for v in views:
                for mo in models:
                    for s in splits:
                        yield from iter_cell(ds_root, m, e, v, mo, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=str(_REPO / "sampling_dataset"))
    ap.add_argument("--methods", nargs="+", default=["dense"])
    ap.add_argument("--embodiment", default="single_arm")
    ap.add_argument("--view", default="multiview")
    ap.add_argument("--model", default="ctrlworld")
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    rows = list(walk(
        Path(args.dataset_root),
        methods=args.methods,
        embodiment=args.embodiment,
        view=args.view,
        model=args.model,
        split=args.split,
    ))
    if args.out == "-":
        for r in rows:
            print(json.dumps(r))
    else:
        Path(args.out).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"[phyworld_index] {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
