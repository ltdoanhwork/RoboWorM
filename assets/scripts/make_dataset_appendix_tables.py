#!/usr/bin/env python3
"""Regenerate the dataset-appendix evidence JSON and LaTeX tables.

Every number in ``assets/paper/appendix/1_dataset.tex`` must come from this
script so the appendix stays recomputable from the on-disk benchmark.

Usage
-----
    python3 assets/scripts/make_dataset_appendix_tables.py \
        --json metrics/reports/dataset_appendix_evidence.json \
        --tex-cells assets/paper/appendix/tables/dataset_cells.tex \
        --tex-taxonomy assets/paper/appendix/tables/dataset_taxonomy.tex

Selectors (kept in sync with the JSON ``selectors`` block):

* ``episodes_selected``  -- ``data/**/episode_*.parquet`` actually on disk
* ``frames_selected``    -- ``meta/episodes.jsonl`` ``length`` summed over
  the selected episode indices only (NOT ``info.json:total_frames``, which
  describes the upstream pool)
* ``episodes_available`` -- ``meta/episodes.jsonl`` line count
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBODIMENTS = ("single_arm", "bimanual", "humanoid")
VIEWS = ("singleview", "multiview")
SPLITS = ("makovian", "non_makovian")

# Paper-facing names for the two temporal regimes. The on-disk directories keep
# the historical (misspelled) "makovian" spelling; it must never reach the paper.
SPLIT_NAME = {"makovian": "Markovian", "non_makovian": "Non-Markovian"}
SPLIT_SHORT = {"makovian": "Mk.", "non_makovian": "Non-mk."}
EMB_NAME = {"single_arm": "Single-arm", "bimanual": "Bimanual",
            "humanoid": "Humanoid"}

EP_RE = re.compile(r"episode_(\d+)\.parquet$")
EP_MP4_RE = re.compile(r"episode_(\d+)\.mp4$")


def complete_episodes(tdir: str, view_names: Iterable[str]) -> set[int] | None:
    """Episode indices that carry EVERY view of this task's camera rig.

    The preparation pipeline drops an episode missing any required stream rather
    than padding it (see appendix ``supp:dataset_processing``), so counting
    ``data/**/episode_*.parquet`` alone OVERSTATES the evaluated set: a handful
    of DROID episodes keep their parquet but ship only 2 of 3 videos, and those
    never reach a rollout. Returns ``None`` when the cell has no video features,
    in which case the parquet count stands.
    """
    views = [v for v in view_names]
    if not views:
        return None
    per_view: list[set[int]] = []
    for short in views:
        found: set[int] = set()
        for p in glob.glob(os.path.join(tdir, "videos", "**",
                                        f"observation.images.{short}",
                                        "episode_*.mp4"), recursive=True):
            m = EP_MP4_RE.search(p)
            if m:
                found.add(int(m.group(1)))
        per_view.append(found)
    if not any(per_view):
        return None
    return set.intersection(*per_view)

# Human-facing corpus names. Source: docs/dense_generation_config.md.
CORPUS = {
    ("single_arm", "singleview"): "RoboChallenge",
    ("single_arm", "multiview"): "DROID",
    ("bimanual", "singleview"): "AIST Bimanual (ALOHA)",
    ("bimanual", "multiview"): "AIST Bimanual (ALOHA)",
    ("humanoid", "singleview"): "GR00T Teleop GR1",
    ("humanoid", "multiview"): "RoboCOIN (AIRBOT MMK2)",
}


@dataclass
class Cell:
    embodiment: str
    view: str
    split: str
    robot_types: list[str] = field(default_factory=list)
    source_fps: list[float] = field(default_factory=list)
    source_resolution: list[str] = field(default_factory=list)
    views_available: list[str] = field(default_factory=list)
    views_per_episode: int = 0
    camera_rigs: int = 0
    task_dirs: int = 0
    episodes_selected: int = 0
    episodes_available: int = 0
    # Episodes whose parquet exists but which lack at least one required view
    # video; excluded from episodes_selected because preparation drops them.
    episodes_incomplete_views: int = 0
    dropped_incomplete_views: list = field(default_factory=list, repr=False)
    instructions_distinct: int = 0
    frames_selected: int = 0
    length_median: int = 0
    length_min: int = 0
    length_max: int = 0
    # Full per-episode frame counts of the SELECTED episodes, so that duration
    # distributions can be plotted without rescanning the corpora.
    lengths: list = field(default_factory=list, repr=False)
    episode_keys: set = field(default_factory=set, repr=False)
    per_task: list = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("episode_keys",)}
        d["corpus"] = CORPUS[(self.embodiment, self.view)]
        return d


def task_dirs_for(base: str) -> list[str]:
    """Return per-task LeRobot dirs, or [base] when the cell is flat."""
    nested = [d for d in sorted(glob.glob(os.path.join(base, "*")))
              if os.path.isdir(os.path.join(d, "meta"))]
    return nested or [base]


def scan_cell(embodiment: str, view: str, split: str) -> Cell | None:
    base = os.path.join(REPO, "datasets", embodiment, view, split)
    if not os.path.isdir(base):
        return None

    cell = Cell(embodiment, view, split)
    robots, fps, res, views = set(), set(), set(), set()
    rigs: set[tuple[str, ...]] = set()
    per_dir_view_counts: list[int] = []
    instructions: set[str] = set()
    lengths: list[int] = []

    for tdir in task_dirs_for(base):
        parquets = glob.glob(os.path.join(tdir, "data", "**", "episode_*.parquet"),
                             recursive=True)
        selected = sorted(int(EP_RE.search(p).group(1)) for p in parquets
                          if EP_RE.search(p))
        if not selected:
            continue

        info = json.load(open(os.path.join(tdir, "meta", "info.json")))
        robots.add(str(info.get("robot_type")))
        fps.add(info.get("fps"))
        dir_views = []
        for name, feat in info.get("features", {}).items():
            if feat.get("dtype") != "video":
                continue
            short = name.replace("observation.images.", "")
            views.add(short)
            dir_views.append(short)
            shape = feat.get("shape") or []
            if len(shape) >= 2:
                res.add(f"{shape[1]}x{shape[0]}")  # W x H
        if dir_views:
            rigs.add(tuple(sorted(dir_views)))
            per_dir_view_counts.append(len(dir_views))

        # Drop episodes missing a required view stream: they are filtered out by
        # the preparation step and never produce a rollout, so counting them
        # would make the appendix disagree with the generated data.
        usable = complete_episodes(tdir, dir_views)
        if usable is not None:
            dropped = [e for e in selected if e not in usable]
            if dropped:
                cell.episodes_incomplete_views += len(dropped)
                cell.dropped_incomplete_views.extend(
                    f"{os.path.basename(tdir)}/episode_{e:06d}" for e in dropped)
            selected = [e for e in selected if e in usable]
            if not selected:
                continue

        cell.task_dirs += 1
        cell.episodes_selected += len(selected)

        episodes = {}
        with open(os.path.join(tdir, "meta", "episodes.jsonl")) as fh:
            for line in fh:
                rec = json.loads(line)
                episodes[rec["episode_index"]] = rec
        cell.episodes_available += len(episodes)

        task_lengths, task_instr = [], set()
        for idx in selected:
            rec = episodes.get(idx, {})
            length = int(rec.get("length", 0))
            lengths.append(length)
            task_lengths.append(length)
            instr = (rec.get("tasks") or [""])[0]
            instructions.add(instr)
            task_instr.add(instr)
            cell.episode_keys.add((os.path.basename(tdir), idx))

        cell.per_task.append({
            "task": os.path.basename(tdir),
            "episodes_selected": len(selected),
            "episodes_available": len(episodes),
            "instruction": sorted(task_instr)[0] if task_instr else "",
            "length_min": min(task_lengths) if task_lengths else 0,
            "length_max": max(task_lengths) if task_lengths else 0,
        })

    if not cell.episodes_selected:
        return None

    cell.robot_types = sorted(robots)
    cell.source_fps = sorted(f for f in fps if f is not None)
    cell.source_resolution = sorted(res)
    cell.views_available = sorted(views)
    # Views a single episode actually carries. The humanoid multiview cells span
    # two camera rigs, so the UNION of view names (6) overstates per-episode
    # synchronisation (4). Report the max per-directory count instead.
    cell.views_per_episode = max(per_dir_view_counts) if per_dir_view_counts else 0
    cell.camera_rigs = len(rigs)
    cell.instructions_distinct = len(instructions)
    cell.frames_selected = sum(lengths)
    cell.length_median = int(statistics.median(lengths))
    cell.length_min = min(lengths)
    cell.length_max = max(lengths)
    cell.lengths = sorted(lengths)
    return cell


def tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fmt_fps(values: Iterable[float]) -> str:
    out = []
    for v in values:
        out.append(str(int(v)) if float(v).is_integer() else f"{v:g}")
    return "/".join(out)


def cells_table(cells: list[Cell], total_instances: int, total_distinct: int) -> str:
    lines = [
        r"% Generated by assets/scripts/make_dataset_appendix_tables.py -- do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Composition of the twelve \ours{} evaluation cells, recomputed "
        r"from the source metadata. \emph{Eps.} is the number of episodes we "
        r"evaluate and \emph{Pool} the number available upstream; \emph{Views} is "
        r"the synchronised cameras per episode and \emph{Instr.} the distinct "
        r"instructions. Frame rate, resolution and view count are reported as "
        r"found in the source, before the per-generator resampling and grid "
        r"composition of \cref{supp:dataset_processing}. Bimanual single-view is "
        r"the head camera of the same episodes as bimanual multi-view, hence the "
        r"identical counts.}",
        r"\label{tab:supp_dataset_cells}",
        r"\begin{tabular}{@{}lllrrrrrrr@{}}",
        r"\toprule",
        r"Embodiment & Track & Regime & Robot & Views & FPS & Resolution & "
        r"Eps. & Pool & Instr. \\",
        r"\midrule",
    ]
    for c in cells:
        lines.append(
            " & ".join([
                EMB_NAME[c.embodiment],
                c.view.replace("singleview", "single").replace("multiview", "multi"),
                SPLIT_SHORT[c.split],
                tex_escape(c.robot_types[0].split("/")[0]) if c.robot_types else "",
                str(c.views_per_episode),
                fmt_fps(c.source_fps),
                r"$\times$".join(c.source_resolution[0].split("x"))
                if c.source_resolution else "",
                str(c.episodes_selected),
                f"{c.episodes_available:,}",
                str(c.instructions_distinct),
            ]) + r" \\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{7}{@{}l}{\textbf{Total evaluation conditions "
        r"(episode $\times$ camera track)}} & \textbf{" + str(total_instances)
        + r"} & & \\",
        r"\multicolumn{7}{@{}l}{\textbf{Distinct robot trajectories}} & \textbf{"
        + str(total_distinct) + r"} & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def taxonomy_table(cells: list[Cell]) -> str:
    """Emit the taxonomy as a longtable: it is longer than one page, and a plain
    `table` float would overflow the page bottom."""
    head = (
        r"Embodiment & Regime & Instruction & Eps. \\" "\n"
        r"\midrule"
    )
    lines = [
        r"% Generated by assets/scripts/make_dataset_appendix_tables.py -- do not edit by hand.",
        r"% Requires \usepackage{longtable} in the preamble.",
        r"{\small",
        r"\begin{longtable}{@{}l l >{\raggedright\arraybackslash}p{0.54\linewidth} r@{}}",
        r"\caption{Complete instruction set for the cells whose instructions are "
        r"defined per task, quoted verbatim from the source corpus. The two "
        r"bimanual camera tracks share these tasks and are listed once. The "
        r"remaining cells (DROID, RoboChallenge, GR1) carry a near-unique "
        r"instruction per episode and are summarised in "
        r"\cref{tab:supp_dataset_cells}.}"
        r"\label{tab:supp_dataset_taxonomy} \\",
        r"\toprule",
        head,
        r"\endfirsthead",
        r"\multicolumn{4}{@{}l}{\emph{\cref{tab:supp_dataset_taxonomy} continued "
        r"from the previous page.}} \\",
        r"\toprule",
        head,
        r"\endhead",
        r"\midrule",
        r"\multicolumn{4}{r@{}}{\emph{continued on the next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    emitted: set[tuple[str, str, str]] = set()
    groups: list[Cell] = []
    for c in cells:
        if c.task_dirs <= 1:
            continue
        # Bimanual singleview/multiview expose the same task dirs; emit once.
        signature = (c.embodiment, c.split,
                     ",".join(sorted(t["task"] for t in c.per_task)))
        if signature in emitted:
            continue
        emitted.add(signature)
        groups.append(c)

    for gi, c in enumerate(groups):
        if gi:
            lines.append(r"\midrule")
        rows = sorted(c.per_task, key=lambda x: x["instruction"].lower())
        for i, t in enumerate(rows):
            emb = EMB_NAME[c.embodiment] if i == 0 else ""
            spl = SPLIT_NAME[c.split] if i == 0 else ""
            lines.append(" & ".join([
                emb, spl,
                tex_escape(t["instruction"]),
                str(t["episodes_selected"]),
            ]) + r" \\")

    lines += [r"\end{longtable}", r"}", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default="metrics/reports/dataset_appendix_evidence.json")
    ap.add_argument("--tex-cells", default="assets/paper/appendix/tables/dataset_cells.tex")
    ap.add_argument("--tex-taxonomy", default="assets/paper/appendix/tables/dataset_taxonomy.tex")
    args = ap.parse_args()

    cells: list[Cell] = []
    for emb in EMBODIMENTS:
        for view in VIEWS:
            for split in SPLITS:
                c = scan_cell(emb, view, split)
                if c:
                    cells.append(c)

    total_instances = sum(c.episodes_selected for c in cells)
    # Distinct source trajectories. Key on the CORPUS, not the view mode:
    # bimanual singleview is cam_high of the same ALOHA episodes as multiview
    # (same corpus -> collapse), whereas single_arm/humanoid singleview come
    # from different corpora than their multiview counterparts (RoboChallenge
    # vs DROID, GR1 vs RoboCOIN) and must NOT collapse even when their episode
    # indices happen to coincide.
    distinct = set()
    for c in cells:
        corpus = CORPUS[(c.embodiment, c.view)]
        distinct |= {(corpus, c.embodiment, c.split, t, e) for t, e in c.episode_keys}
    total_distinct = len(distinct)

    # Which (embodiment, split) pairs reuse the same episodes across view modes?
    reuse = []
    for emb in EMBODIMENTS:
        for split in SPLITS:
            sv = next((c for c in cells if (c.embodiment, c.view, c.split) == (emb, "singleview", split)), None)
            mv = next((c for c in cells if (c.embodiment, c.view, c.split) == (emb, "multiview", split)), None)
            if sv and mv:
                reuse.append({
                    "embodiment": emb, "split": split,
                    "singleview_episodes": sv.episodes_selected,
                    "multiview_episodes": mv.episodes_selected,
                    "identical_episode_sets": sv.episode_keys == mv.episode_keys,
                    "shared_episodes": len(sv.episode_keys & mv.episode_keys),
                })

    payload = {
        "schema_version": "roboworm/dataset-appendix-evidence-v1",
        "generated_by": "assets/scripts/make_dataset_appendix_tables.py",
        "purpose": ("Numeric evidence backing appendix/1_dataset.tex. Every number in "
                    "that section must appear here with a selector pointing at an "
                    "on-disk source."),
        "selectors": {
            "episodes_selected": "count of datasets/{emb}/{view}/{split}/**/data/**/episode_*.parquet, MINUS episodes missing any required view video (those are dropped by preparation and never rolled out)",
            "frames_selected": "sum of meta/episodes.jsonl 'length' over the selected episode indices only",
            "episodes_available": "line count of meta/episodes.jsonl (pool declared by the LeRobot metadata)",
            "instructions_distinct": "distinct meta/episodes.jsonl tasks[0] over selected episodes",
            "source_fps": "meta/info.json fps",
            "source_resolution": "meta/info.json features[video].shape as WxH",
            "views_available": "meta/info.json features with dtype == video",
        },
        "totals": {
            "cells": len(cells),
            "episode_instances": total_instances,
            "distinct_source_episodes": total_distinct,
        },
        "notes": [
            "bimanual singleview exposes the cam_high stream of the SAME ALOHA "
            "episodes as bimanual multiview (symlinked by "
            "scripts/export_bimanual_singleview_cam_high.py), so those two cells "
            "share episodes.",
            "single_arm/humanoid singleview come from DIFFERENT corpora than their "
            "multiview counterparts (RoboChallenge vs DROID; GR1 vs RoboCOIN); "
            "episode-index collisions between them are coincidental and are NOT "
            "collapsed when counting distinct trajectories.",
            "episode_instances counts (episode x camera-track) evaluation "
            "conditions; distinct_source_episodes counts unique source "
            "trajectories.",
            "humanoid multiview spans two camera rigs (cam_front/cam_head vs "
            "cam_high/cam_third_view). The union of view names is 6, but each "
            "episode carries 4 synchronised views; views_per_episode reports the "
            "latter.",
            "frames_selected sums meta/episodes.jsonl length over selected "
            "episodes; it is NOT info.json total_frames, which describes the "
            "upstream pool.",
            "episodes_selected EXCLUDES episodes that keep their parquet but "
            "lack a required view video (3 DROID single_arm multiview episodes: "
            "2 in makovian, 1 in non_makovian). Preparation drops them, so they "
            "produce no rollout; see episodes_incomplete_views per cell. This is "
            "why the single_arm multiview cells report 128/129 rather than "
            "130/130, and why the benchmark totals 1,605 not 1,608.",
            "PROVENANCE DISCREPANCY (bimanual): the acquisition manifest in "
            "scripts/download_bimanual_multiview_dropbox.sh labels "
            "place_the_chinese_spoons_on_the_cloth as 'makovian', but on disk it "
            "lives under non_makovian/ (3 episodes). This scan is disk-authoritative, "
            "so the tables follow the on-disk placement. Resolve the manifest before "
            "release.",
            "PROVENANCE GAP (humanoid multiview): the makovian/non_makovian "
            "assignment for the 6 RoboCOIN task dirs is materialised as a directory "
            "partition but no script or manifest in-tree records the rule.",
            "datasets/bimanual/multiview/non_makovian/ contains download residue "
            "('gr00t?rlkey=...', 'wget-log'). It carries no meta/ dir so it is not "
            "counted as a task, but it should be cleaned before release.",
        ],
        "view_mode_reuse": reuse,
        "cells": [c.to_json() for c in cells],
    }

    for path, text in (
        (args.json, json.dumps(payload, indent=2) + "\n"),
        (args.tex_cells, cells_table(cells, total_instances, total_distinct)),
        (args.tex_taxonomy, taxonomy_table(cells)),
    ):
        full = path if os.path.isabs(path) else os.path.join(REPO, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as fh:
            fh.write(text)
        print(f"wrote {path}")

    print(f"\ncells={len(cells)} episode_instances={total_instances} "
          f"distinct_source_episodes={total_distinct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
