#!/usr/bin/env python3
"""Build sampling_dataset/data_for_web for the human-labeling website.

Layout matches sampling_dataset (method is top-level, model/view/task kept):

    data_for_web/{method}/{embodiment}/output/{view}/{model}/{split}/...

~400 items, even across (embodiment, view, model, split).
Ctrl-World bimanual/humanoid = 4-view grid only.
Multiview copies **prediction only**:
  * Ctrl-World: ``pred_all_views.mp4`` (``full_pred.mp4`` is GT stacked on pred)
  * DreamGen: ``full_pred.mp4`` (never ``full_merged`` / GT)

Augment:
  * take ALL same-image task changes: task_swap, object_relocation
  * a small random sample of the rest (edit_only / prompt_only / combo / recolor)

Usage:
    python scripts/build_data_for_web.py
    python scripts/build_data_for_web.py --target 400 --seed 42 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLING = REPO / "sampling_dataset"
DEST = SAMPLING / "data_for_web"

SKIP_METHODS = {".cache", "data_for_web"}
SKIP_FNAMES = {
    "full_gt.mp4",
    "gt_all_views.mp4",
    "full_merged.mp4",
    "pred_all_views.mp4",
}
SKIP_PARTS = {"_fps_compare", "input"}
SKIP_DIRNAMES = {".cache", "__pycache__", ".ipynb_checkpoints"}
EMBODIMENTS = ("bimanual", "humanoid", "single_arm")
SPLITS = ("makovian", "non_makovian")
STRONG_AUG_TAGS = {"task_swap", "object_relocation"}
AUG_TAG_RE = re.compile(
    r"__(edit_only|prompt_only|combo|task_swap|object_relocation|v\d+_[a-z]+)$"
)


def is_pred_mp4(name: str) -> bool:
    return name.endswith(".mp4") and name not in SKIP_FNAMES and not name.startswith("seg_")


def keep_setting(emb: str, view: str, model: str, split: str) -> bool:
    if emb not in EMBODIMENTS or split not in SPLITS:
        return False
    if model == "ctrlworld" and emb in ("bimanual", "humanoid"):
        return False
    return True


def parse_setting(rel_parts: tuple[str, ...]):
    try:
        i = rel_parts.index("output")
    except ValueError:
        return None
    if i < 1 or len(rel_parts) < i + 4:
        return None
    emb, view, model, split = rel_parts[i - 1], rel_parts[i + 1], rel_parts[i + 2], rel_parts[i + 3]
    if any(p in SKIP_PARTS for p in rel_parts):
        return None
    if not keep_setting(emb, view, model, split):
        return None
    return emb, view, model, split


def strip_aug(name: str) -> str:
    return AUG_TAG_RE.sub("", name)


def extract_tag(unit_key: str) -> str:
    stem = unit_key.replace("\\", "/")
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    m = AUG_TAG_RE.search(stem)
    return m.group(1) if m else ""


def extract_task(unit_key: str, view: str = "", model: str = "") -> str:
    parts = Path(unit_key).parts
    for p in parts:
        stem = p[:-4] if p.endswith(".mp4") else p
        stem = strip_aug(stem)
        if stem.startswith("iter_") or stem in {"global"}:
            continue
        if "__episode_" in stem:
            return stem.split("__episode_")[0]
        if stem.startswith("episode_") and "__" in stem:
            return stem[len("episode_") :].rsplit("__", 1)[0]
        if stem.startswith("episode_"):
            continue
        return stem
    stem = Path(unit_key).name
    if stem.endswith(".mp4"):
        stem = stem[:-4]
    stem = strip_aug(stem)
    if "__episode_" in stem:
        return stem.split("__episode_")[0]
    if view and model:
        return f"{view}/{model}"
    return "unknown"


def extract_episode(unit_key: str) -> str:
    for p in reversed(Path(unit_key).parts):
        stem = p[:-4] if p.endswith(".mp4") else p
        if "episode" in stem:
            return strip_aug(stem)
    return strip_aug(Path(unit_key).name)


def pred_file(src: Path, kind: str) -> Path:
    """Return the pred-only video for a discovered unit.

    Ctrl-World episode dirs write ``full_pred.mp4`` as a GT-on-top / pred-below
    comparison (the "label" the raters should not see). ``pred_all_views.mp4``
    is the prediction grid alone. DreamGen dirs have no pred_all_views; their
    ``full_pred.mp4`` is already pred-only (``full_merged.mp4`` is GT|pred).
    """
    if kind != "dir":
        return src
    pred_views = src / "pred_all_views.mp4"
    if pred_views.is_file():
        return pred_views
    return src / "full_pred.mp4"


def discover(method_dir: Path) -> list[dict]:
    out = []
    seen: set[tuple] = set()
    for dirpath, dirnames, filenames in os.walk(method_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRNAMES]
        rel = Path(dirpath).relative_to(method_dir)
        parts = rel.parts
        if any(p in SKIP_PARTS for p in parts):
            dirnames[:] = []
            continue
        st = parse_setting(parts)
        if st is None:
            continue
        emb, view, model, split = st
        split_dir = method_dir.joinpath(emb, "output", view, model, split)
        for fn in filenames:
            if not is_pred_mp4(fn):
                continue
            full = Path(dirpath) / fn
            if fn == "full_pred.mp4":
                unit_path, kind = Path(dirpath), "dir"
            else:
                unit_path, kind = full, "file"
            try:
                unit_key = str(unit_path.relative_to(split_dir))
            except ValueError:
                continue
            stamp = (emb, view, model, split, unit_key)
            if stamp in seen:
                continue
            seen.add(stamp)
            out.append(
                {
                    "embodiment": emb,
                    "view": view,
                    "model": model,
                    "split": split,
                    "unit_key": unit_key,
                    "src": unit_path,
                    "kind": kind,
                    "task": extract_task(unit_key, view, model),
                    "episode": extract_episode(unit_key),
                    "aug_tag": extract_tag(unit_key),
                    "cell": f"{emb}/{view}/{model}/{split}",
                }
            )
    return out


def sample_diverse(cands: list[dict], n: int, seed: int, tag: str) -> list[dict]:
    if n <= 0 or not cands:
        return []
    by_task: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_task[c["task"]].append(c)
    rng = random.Random(f"{seed}:{tag}")
    for t in by_task:
        rng.shuffle(by_task[t])
    tasks = list(by_task)
    rng.shuffle(tasks)
    picked = []
    i = 0
    while len(picked) < n and any(by_task[t] for t in tasks):
        t = tasks[i % len(tasks)]
        if by_task[t]:
            picked.append(by_task[t].pop())
        i += 1
        if i > n * max(len(tasks), 1) + 80:
            break
    return picked


def copy_pred(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.copy2(src, dest)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--weak-aug-share", type=float, default=0.08, help="Random non-strong augment share of target")
    ap.add_argument("--fast-share", type=float, default=0.22, help="worldcache share of the fill slots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dest: Path = args.dest
    methods_present = sorted(
        p.name for p in SAMPLING.iterdir() if p.is_dir() and p.name not in SKIP_METHODS
    )
    print(f"sampling_dataset methods: {methods_present}")

    by_method: dict[str, list[dict]] = {}
    for method in methods_present:
        units = discover(SAMPLING / method)
        by_method[method] = units
        print(f"  [{method:12s}] {len(units):5d} pred units")

    def with_method(method: str, rows: list[dict]) -> list[dict]:
        return [{**r, "source_method": method} for r in rows]

    dense = with_method("dense", by_method.get("dense", []))
    aug = with_method("augment", by_method.get("augment", []))
    fast = with_method("worldcache", by_method.get("worldcache", []))

    strong = [r for r in aug if r["aug_tag"] in STRONG_AUG_TAGS]
    weak_aug = [r for r in aug if r["aug_tag"] not in STRONG_AUG_TAGS]

    cells = sorted({r["cell"] for r in dense + aug + fast})
    n_cells = len(cells)
    base, rem = divmod(args.target, n_cells)
    cell_budget = {c: base + (1 if i < rem else 0) for i, c in enumerate(cells)}

    print(f"\n{n_cells} cells, base {base}/cell (+{rem} extras) -> target {args.target}")
    print(f"strong augment (task_swap/object_relocation): {len(strong)} — taking ALL")

    planned: list[dict] = []
    taken_eps: set[str] = set()

    def ep_key(r: dict) -> str:
        return f"{r['cell']}|{r['episode']}"

    # 1) must-take strong augment
    for r in sorted(strong, key=lambda x: (x["cell"], x["task"], x["episode"])):
        planned.append({**r, "role": "strong_augment"})
        taken_eps.add(ep_key(r))

    strong_per_cell = defaultdict(int)
    for r in planned:
        strong_per_cell[r["cell"]] += 1

    # 2) fill each cell up to budget
    n_weak_budget = int(round(args.target * args.weak_aug_share))
    weak_by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in weak_aug:
        if ep_key(r) not in taken_eps:
            weak_by_cell[r["cell"]].append(r)

    # even weak-aug across cells that have any
    weak_cells = [c for c in cells if weak_by_cell[c]]
    weak_quota = {}
    if weak_cells:
        wb, wr = divmod(n_weak_budget, len(weak_cells))
        weak_quota = {c: wb + (1 if i < wr else 0) for i, c in enumerate(weak_cells)}

    fill_stats = []
    for cell in cells:
        already = strong_per_cell[cell]
        need = max(0, cell_budget[cell] - already)
        n_weak = min(weak_quota.get(cell, 0), need, len(weak_by_cell[cell]))
        weak_got = sample_diverse(weak_by_cell[cell], n_weak, args.seed, f"weak:{cell}")
        for r in weak_got:
            planned.append({**r, "role": "weak_augment"})
            taken_eps.add(ep_key(r))
        need -= len(weak_got)

        n_fast = min(int(round(need * args.fast_share)), need)
        dense_c = [r for r in dense if r["cell"] == cell and ep_key(r) not in taken_eps]
        fast_c = [r for r in fast if r["cell"] == cell and ep_key(r) not in taken_eps]

        fast_got = sample_diverse(fast_c, n_fast, args.seed, f"fast:{cell}")
        for r in fast_got:
            planned.append({**r, "role": "fast"})
            taken_eps.add(ep_key(r))
        need -= len(fast_got)

        dense_got = sample_diverse(dense_c, need, args.seed, f"dense:{cell}")
        for r in dense_got:
            planned.append({**r, "role": "dense"})
            taken_eps.add(ep_key(r))
        leftover = need - len(dense_got)
        if leftover > 0:
            extra_fast = [r for r in fast if r["cell"] == cell and ep_key(r) not in taken_eps]
            more = sample_diverse(extra_fast, leftover, args.seed, f"fast2:{cell}")
            for r in more:
                planned.append({**r, "role": "fast"})
                taken_eps.add(ep_key(r))
            leftover -= len(more)

        fill_stats.append(
            (
                cell,
                cell_budget[cell],
                already,
                len(weak_got),
                len(fast_got) + leftover,  # will recompute below
                len(dense_got),
            )
        )

    planned.sort(key=lambda r: (r["cell"], r["source_method"], r["task"], r["episode"]))

    def count(pred):
        return sum(1 for r in planned if pred(r))

    print("\n=== per cell ===")
    print(f"{'cell':48s}  budg  strong weak  fast dense  total tasks")
    for cell in cells:
        rows = [r for r in planned if r["cell"] == cell]
        n_s = count(lambda r, c=cell: r["cell"] == c and r["role"] == "strong_augment")
        n_w = count(lambda r, c=cell: r["cell"] == c and r["role"] == "weak_augment")
        n_f = count(lambda r, c=cell: r["cell"] == c and r["role"] == "fast")
        n_d = count(lambda r, c=cell: r["cell"] == c and r["role"] == "dense")
        n_t = len({r["task"] for r in rows})
        print(f"{cell:48s}  {cell_budget[cell]:4d}  {n_s:6d} {n_w:4d} {n_f:5d} {n_d:5d}  {len(rows):5d} {n_t:5d}")

    print("\n=== totals ===")
    print(f"  items          {len(planned)}")
    print("  by role:")
    by_role = defaultdict(int)
    by_method_c = defaultdict(int)
    by_emb = defaultdict(int)
    by_view = defaultdict(int)
    by_split = defaultdict(int)
    by_model = defaultdict(int)
    by_tag = defaultdict(int)
    by_emb_view = defaultdict(int)
    for r in planned:
        by_role[r["role"]] += 1
        by_method_c[r["source_method"]] += 1
        by_emb[r["embodiment"]] += 1
        by_view[r["view"]] += 1
        by_split[r["split"]] += 1
        by_model[f"{r['embodiment']}/{r['model']}"] += 1
        by_emb_view[f"{r['embodiment']}/{r['view']}"] += 1
        if r["aug_tag"]:
            by_tag[r["aug_tag"]] += 1
    for k, n in sorted(by_role.items()):
        print(f"    {k:20s} {n:4d}")
    print("  by source_method:")
    for k, n in sorted(by_method_c.items(), key=lambda x: -x[1]):
        print(f"    {k:20s} {n:4d}")
    print("  by embodiment:")
    for k, n in sorted(by_emb.items()):
        print(f"    {k:20s} {n:4d}")
    print("  by view:")
    for k, n in sorted(by_view.items()):
        print(f"    {k:20s} {n:4d}")
    print("  by split:")
    for k, n in sorted(by_split.items()):
        print(f"    {k:20s} {n:4d}")
    print("  by embodiment/view:")
    for k, n in sorted(by_emb_view.items()):
        print(f"    {k:20s} {n:4d}")
    print("  by embodiment/model:")
    for k, n in sorted(by_model.items()):
        print(f"    {k:32s} {n:4d}")
    if by_tag:
        print("  augment tags taken:")
        for k, n in sorted(by_tag.items()):
            print(f"    {k:20s} {n:4d}")

    if args.dry_run:
        print("\n[--dry-run] not writing")
        return 0

    if dest.exists():
        for child in dest.iterdir():
            if child.name == "labels.json":
                continue
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    dest.mkdir(parents=True, exist_ok=True)
    items = []
    n_ok = n_miss = 0
    for i, row in enumerate(planned):
        src = pred_file(Path(row["src"]), row["kind"])
        if not src.is_file():
            n_miss += 1
            continue
        rel = (
            Path(row["source_method"])
            / row["embodiment"]
            / "output"
            / row["view"]
            / row["model"]
            / row["split"]
            / row["unit_key"]
        )
        if row["kind"] == "dir":
            dest_pred = dest / rel / src.name
        else:
            dest_pred = dest / rel
        copy_pred(src, dest_pred)
        item_id = f"{i:05d}"
        items.append(
            {
                "id": item_id,
                "embodiment": row["embodiment"],
                "view": row["view"],
                "model": row["model"],
                "split": row["split"],
                "task": row["task"],
                "source_method": row["source_method"],
                "role": row["role"],
                "aug_tag": row["aug_tag"],
                "episode": row["episode"],
                "unit_key": row["unit_key"],
                "cell": row["cell"],
                "web_path": str(dest_pred.relative_to(dest)),
                "video": f"videos/{dest_pred.relative_to(dest).as_posix()}",
                "source_path": str(src.relative_to(SAMPLING)),
            }
        )
        n_ok += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "target": args.target,
        "layout": "{method}/{embodiment}/output/{view}/{model}/{split}/...",
        "policy": {
            "ctrlworld": "bimanual/humanoid use 4-view grid only",
            "multiview": "pred only: Ctrl-World pred_all_views.mp4; DreamGen full_pred.mp4 (never GT/merged stack)",
            "strong_augment": "all task_swap + object_relocation",
            "weak_augment": f"random ~{args.weak_aug_share:.0%} of target",
            "even": "even slots across embodiment/view/model/split cells",
        },
        "counts": {
            "items": n_ok,
            "missing_src": n_miss,
            "by_role": dict(by_role),
            "by_method": dict(by_method_c),
            "by_embodiment": dict(by_emb),
            "by_view": dict(by_view),
            "by_split": dict(by_split),
            "by_model": dict(by_model),
            "by_cell": {c: sum(1 for it in items if it["cell"] == c) for c in cells},
        },
        "items": items,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    public_items = [
        {
            "id": it["id"],
            "video": it["video"],
            "embodiment": it["embodiment"],
            "view": it["view"],
            "model": it["model"],
            "split": it["split"],
            "task": it["task"],
        }
        for it in items
    ]
    (dest / "catalog.json").write_text(json.dumps({"n": len(public_items), "items": public_items}, indent=2) + "\n")
    if not (dest / "labels.json").exists():
        (dest / "labels.json").write_text(
            json.dumps({"description": "Human labels keyed by item id", "labels": {}}, indent=2) + "\n"
        )

    webpage = REPO / "webpage"
    if webpage.is_dir():
        videos_link = webpage / "videos"
        if videos_link.exists() or videos_link.is_symlink():
            videos_link.unlink()
        videos_link.symlink_to(os.path.relpath(dest, webpage))
        for name in ("manifest.json", "catalog.json"):
            dest_link = webpage / name
            if dest_link.exists() or dest_link.is_symlink():
                dest_link.unlink()
            dest_link.symlink_to(os.path.relpath(dest / name, webpage))

    (dest / "README.md").write_text(
        f"""# data_for_web

Same layout as `sampling_dataset/`:

```
{{method}}/{{embodiment}}/output/{{view}}/{{model}}/{{split}}/{{task_or_episode}}/full_pred.mp4
```

Multiview stores **prediction only** (Ctrl-World `pred_all_views.mp4`,
DreamGen `full_pred.mp4` — never the GT|pred stack). Strong augment
(`task_swap`, `object_relocation`) is included in full; other augment is a
small random sample. ~{args.target} clips, even across cells.

```bash
python scripts/build_data_for_web.py --target 400 --seed 42
```
"""
    )
    print(f"\nWrote {n_ok} pred files -> {dest} ({n_miss} missing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
