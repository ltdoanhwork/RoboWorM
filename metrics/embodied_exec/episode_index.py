"""Universal resolver: any generated mp4 under ``sampling_dataset`` -> its GT episode.

The execution-grounded tier needs, for every generated video, the *action-labelled*
GT episode it was generated from. Nothing else in the tier can run without that
mapping, and "applies to the whole benchmark" is exactly the claim this file has to
earn, so it reports coverage over every mp4 it finds instead of assuming.

Two indices are built:

``build_dataset_index()``
    walks ``datasets/`` once and maps
    ``(embodiment, view_type, split, task, episode_id) -> parquet path``.
    Doing this once matters: a per-record ``glob`` over ~50k videos re-scans the
    chunk dirs 50k times.

``parse_output_path()``
    turns an mp4 path into a record. Six naming conventions coexist in the tree
    (see ``PATTERNS`` below); they accumulated as models were added and are not
    going to be unified retroactively, so the resolver absorbs the difference.

Episode ids are NOT globally unique: ``episode_000004`` exists under many bimanual
task dirs. Only ``(embodiment, view_type, split, task, episode_id)`` identifies an
episode, which is why ``task`` is parsed rather than ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLITS = ("makovian", "non_makovian")

# mp4 basenames that are renderings, not the prediction itself
ROLE_BY_BASENAME = {
    "full_pred": "pred",
    "full_gt": "gt",
    "full_merged": "merged",
    "pred_all_views": "pred_strip",
    "gt_all_views": "gt_strip",
}

EPISODE_RE = re.compile(r"^episode_(\d+)$")
SEG_RE = re.compile(r"^seg_(\d+)$")
# bimanual singleview flattens task and episode into one filename
TASK_EPISODE_RE = re.compile(r"^(?P<task>.+?)__episode_(?P<eid>\d+)$")
# ctrlworld bimanual/humanoid flattens them the other way round
EPISODE_TASK_RE = re.compile(r"^episode_(?P<task>.+?)__(?P<eid>\d+)$")

PATTERNS = """
1  {split}/iter_*/{view}/episode_N.mp4              single_arm  sv  dreamgen
2  {split}/iter_*/episode_N/{role}.mp4              single_arm  sv  dreamdojo
3  {split}/episode_N.mp4                            single_arm  mv  dreamgen
4  {split}/iter_*/{task}__episode_N.mp4             bimanual    sv  dreamgen
5  {split}/iter_*/{task}/episode_N/{role|seg}.mp4   bimanual/humanoid mv dreamgen
6  {split}/episode_N/{role}.mp4                     any         mv  ctrlworld
7  {split}/iter_*/{task}__episode_N/{role}.mp4      bimanual    sv  dreamdojo
8  {split}/episode_{task}__N/{role}.mp4             bimanual/humanoid mv ctrlworld
""".strip()

# only these mp4 roles are model predictions worth scoring
PRED_ROLES = ("pred", "pred_strip", "bare", "seg")


@dataclass
class OutputRecord:
    """One generated mp4, located within the benchmark's taxonomy."""

    mp4: str
    method: str          # dense, worldcache, sito, pisa, ...
    embodiment: str      # single_arm, bimanual, humanoid
    view_type: str       # singleview, multiview
    model: str           # dreamgen, dreamdojo, ctrlworld
    split: str | None    # makovian, non_makovian
    task: str | None     # only for bimanual/humanoid; single_arm has none
    episode_id: str | None
    iter_tag: str | None
    view: str | None     # e.g. global, exterior_1_left (pattern 1 only)
    role: str            # pred, gt, merged, pred_strip, gt_strip, seg, bare
    seg_index: int | None
    pattern: int | None

    @property
    def key(self) -> tuple:
        return (self.embodiment, self.view_type, self.split, self.task, self.episode_id)


def _norm_eid(eid: str) -> int:
    """Episode ids are written with 4 or 6 digits depending on the script that wrote
    them (``episode_0023`` vs ``episode_000023``). Compare them as integers."""
    return int(eid)


def build_dataset_index(datasets_root: Path) -> dict[tuple, str]:
    """Map ``(embodiment, view_type, split, task, episode_id) -> parquet path``.

    ``task`` is None for the flat layouts (single_arm, humanoid singleview) and the
    task directory name for the nested ones (bimanual, humanoid multiview).
    """
    index: dict[tuple, str] = {}
    for emb_dir in sorted(datasets_root.iterdir()):
        if not emb_dir.is_dir():
            continue
        emb = emb_dir.name
        for view_type in ("singleview", "multiview"):
            vdir = emb_dir / view_type
            if not vdir.is_dir():
                continue
            for split in SPLITS:
                sdir = vdir / split
                if not sdir.is_dir():
                    continue
                # flat layout: split/data/chunk-*/episode_*.parquet
                _harvest(index, sdir, emb, view_type, split, task=None)
                # nested layout: split/{task}/data/chunk-*/episode_*.parquet
                for task_dir in sorted(sdir.iterdir()):
                    if task_dir.is_dir() and (task_dir / "data").is_dir():
                        _harvest(index, task_dir, emb, view_type, split, task=task_dir.name)
    return index


def _harvest(index, base: Path, emb, view_type, split, task) -> None:
    data_dir = base / "data"
    if not data_dir.is_dir():
        return
    for chunk in sorted(data_dir.glob("chunk-*")):
        for pq in chunk.glob("episode_*.parquet"):
            m = EPISODE_RE.match(pq.stem)
            if m:
                index[(emb, view_type, split, task, _norm_eid(m.group(1)))] = str(pq)


def parse_output_path(mp4: Path, sampling_root: Path) -> OutputRecord | None:
    """Locate one mp4 in the taxonomy. Returns None for paths outside the schema."""
    try:
        rel = mp4.relative_to(sampling_root)
    except ValueError:
        return None
    parts = rel.parts
    # {method}/{embodiment}/output/{view_type}/{model}/...
    if len(parts) < 6 or parts[2] != "output":
        return None
    method, embodiment, _, view_type, model = parts[:5]
    tail = list(parts[5:])

    rec = OutputRecord(
        mp4=str(mp4), method=method, embodiment=embodiment, view_type=view_type,
        model=model, split=None, task=None, episode_id=None, iter_tag=None,
        view=None, role="bare", seg_index=None, pattern=None,
    )

    if tail and tail[0] in SPLITS:
        rec.split = tail.pop(0)
    elif tail and tail[0].startswith("_"):
        return None  # scratch dirs such as _fps_compare

    if tail and tail[0].startswith("iter_"):
        rec.iter_tag = tail.pop(0)

    if not tail:
        return None

    stem = Path(tail[-1]).stem

    if len(tail) == 1:
        # pattern 3/4: a single file directly under split[/iter]
        m = EPISODE_RE.match(stem)
        if m:
            rec.episode_id, rec.pattern = m.group(1), 3
            return rec
        m = TASK_EPISODE_RE.match(stem)
        if m:
            rec.task, rec.episode_id, rec.pattern = m.group("task"), m.group("eid"), 4
            return rec
        return None

    if len(tail) == 2:
        head = tail[0]
        m = EPISODE_RE.match(head)
        if m:
            # pattern 2/6: episode dir + role file
            rec.episode_id = m.group(1)
            rec.role = ROLE_BY_BASENAME.get(stem, "other")
            rec.pattern = 6 if model == "ctrlworld" else 2
            return rec
        m = TASK_EPISODE_RE.match(head)
        if m:
            # pattern 7: {task}__episode_N dir + role file
            rec.task, rec.episode_id = m.group("task"), m.group("eid")
            rec.role = ROLE_BY_BASENAME.get(stem, "other")
            rec.pattern = 7
            return rec
        m = EPISODE_TASK_RE.match(head)
        if m:
            # pattern 8: episode_{task}__N dir + role file (ctrlworld, task-first id-last)
            rec.task, rec.episode_id = m.group("task"), m.group("eid")
            rec.role = ROLE_BY_BASENAME.get(stem, "other")
            rec.pattern = 8
            return rec
        m = EPISODE_RE.match(stem)
        if m:
            # pattern 1: view dir + episode file
            rec.view, rec.episode_id, rec.pattern = head, m.group(1), 1
            return rec
        return None

    if len(tail) >= 3:
        # pattern 5: {task}/episode_N/{role|seg}.mp4
        m = EPISODE_RE.match(tail[-2])
        if not m:
            return None
        rec.task = "/".join(tail[:-2])
        rec.episode_id = m.group(1)
        seg = SEG_RE.match(stem)
        if seg:
            rec.role, rec.seg_index = "seg", int(seg.group(1))
        else:
            rec.role = ROLE_BY_BASENAME.get(stem, "other")
        rec.pattern = 5
        return rec

    return None


def resolve_gt(rec: OutputRecord, ds_index: dict[tuple, str]) -> str | None:
    """Parquet path of the GT episode for this record, or None.

    Falls back across ``view_type`` and ``task``-presence because a generated video's
    location does not always match where its GT lives: single_arm multiview dreamgen
    predictions are indexed by a global episode id that only exists in one of the two
    view_type trees, and a task-less key can still resolve if exactly one task holds
    that episode id.
    """
    if rec.episode_id is None or rec.split is None:
        return None

    eid = _norm_eid(rec.episode_id)
    candidates = [
        (rec.embodiment, rec.view_type, rec.split, rec.task, eid),
        (rec.embodiment, rec.view_type, rec.split, None, eid),
    ]
    other_view = "multiview" if rec.view_type == "singleview" else "singleview"
    candidates += [
        (rec.embodiment, other_view, rec.split, rec.task, eid),
        (rec.embodiment, other_view, rec.split, None, eid),
    ]
    for k in candidates:
        hit = ds_index.get(k)
        if hit:
            return hit

    # last resort: unique task match for this (embodiment, split, episode_id)
    if rec.task is None:
        matches = [
            v for (e, vt, s, t, i), v in ds_index.items()
            if e == rec.embodiment and s == rec.split and i == eid
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def scan(sampling_root: Path, datasets_root: Path, methods: list[str] | None = None):
    """Walk every mp4 under ``sampling_root`` and resolve it. Returns (records, stats)."""
    ds_index = build_dataset_index(datasets_root)
    records: list[OutputRecord] = []
    stats = {
        "mp4_seen": 0, "parsed": 0, "unparsed": [], "resolved": 0,
        "unresolved_examples": [], "by_pattern": Counter(),
        "by_group": defaultdict(lambda: {"total": 0, "resolved": 0}),
    }

    method_dirs = sorted(d for d in sampling_root.iterdir() if d.is_dir())
    if methods:
        method_dirs = [d for d in method_dirs if d.name in methods]

    for mdir in method_dirs:
        for mp4 in (mdir).rglob("*.mp4"):
            # `input/` holds GT copies, not predictions
            if "/output/" not in str(mp4):
                continue
            stats["mp4_seen"] += 1
            rec = parse_output_path(mp4, sampling_root)
            if rec is None:
                if len(stats["unparsed"]) < 20:
                    stats["unparsed"].append(str(mp4.relative_to(sampling_root)))
                continue
            stats["parsed"] += 1
            stats["by_pattern"][rec.pattern] += 1
            gt = resolve_gt(rec, ds_index)
            group = f"{rec.method}/{rec.embodiment}/{rec.view_type}/{rec.model}"
            stats["by_group"][group]["total"] += 1
            if gt:
                stats["resolved"] += 1
                stats["by_group"][group]["resolved"] += 1
            elif len(stats["unresolved_examples"]) < 20:
                stats["unresolved_examples"].append(str(mp4.relative_to(sampling_root)))
            records.append((rec, gt))
    stats["dataset_index_size"] = len(ds_index)
    return records, stats

def main() -> None:
    ap = argparse.ArgumentParser(description=f"Resolve generated mp4s to GT episodes.\n\n{PATTERNS}",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sampling-root", default=str(_REPO_ROOT / "sampling_dataset"))
    ap.add_argument("--datasets-root", default=str(_REPO_ROOT / "datasets"))
    ap.add_argument("--methods", nargs="*", default=None, help="limit to these method dirs")
    ap.add_argument("--out", default=None, help="write index as JSONL")
    ap.add_argument("--roles", nargs="*", default=None,
                    help="keep only these roles in the JSONL (e.g. pred pred_strip bare)")
    ap.add_argument("--min-coverage", type=float, default=None,
                    help="exit non-zero if resolved/parsed falls below this fraction")
    args = ap.parse_args()

    records, stats = scan(Path(args.sampling_root), Path(args.datasets_root), args.methods)

    print(f"dataset index      : {stats['dataset_index_size']} GT episodes")
    print(f"mp4 seen           : {stats['mp4_seen']}")
    print(f"parsed             : {stats['parsed']}")
    print(f"resolved to GT     : {stats['resolved']}"
          + (f"  ({100.0 * stats['resolved'] / stats['parsed']:.1f}% of parsed)" if stats["parsed"] else ""))
    print(f"by pattern         : {dict(sorted(stats['by_pattern'].items(), key=lambda kv: (kv[0] is None, kv[0])))}")

    if stats["unparsed"]:
        print(f"\nunparsed ({len(stats['unparsed'])} shown):")
        for p in stats["unparsed"]:
            print("  ", p)
    if stats["unresolved_examples"]:
        print(f"\nunresolved ({len(stats['unresolved_examples'])} shown):")
        for p in stats["unresolved_examples"]:
            print("  ", p)

    print("\n{:<52} {:>8} {:>8} {:>7}".format("group", "total", "resolved", "pct"))
    for group, d in sorted(stats["by_group"].items()):
        pct = 100.0 * d["resolved"] / d["total"] if d["total"] else 0.0
        print("{:<52} {:>8} {:>8} {:>6.1f}%".format(group, d["total"], d["resolved"], pct))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        n = 0
        with open(args.out, "w") as f:
            for rec, gt in records:
                if args.roles and rec.role not in args.roles:
                    continue
                f.write(json.dumps({**asdict(rec), "gt_parquet": gt}) + "\n")
                n += 1
        print(f"\nwrote {n} records -> {args.out}")

    if args.min_coverage is not None and stats["parsed"]:
        cov = stats["resolved"] / stats["parsed"]
        if cov < args.min_coverage:
            print(f"\nFAIL: coverage {cov:.4f} < {args.min_coverage}")
            raise SystemExit(1)
        print(f"\nOK: coverage {cov:.4f} >= {args.min_coverage}")


if __name__ == "__main__":
    main()
