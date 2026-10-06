"""
Rebuild meta/episodes.jsonl (and patch meta/info.json) for humanoid singleview

`makovian` / `non_makovian` were created with symlinked DreamDojo-HV_Eval meta
while data/ mixes GR1_robot + DreamDojo episodes. Episode indices overlap (0–67)
across sources, so we resolve each parquet file by schema:

  - GR1_robot parquets include column `annotation.human.action.verb`
  - DreamDojo-HV_Eval parquets do not

Usage:
  python scripts/prepare/repair_humanoid_singleview_split_meta.py
  python scripts/prepare/repair_humanoid_singleview_split_meta.py --only makovian
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

PHYSICAL_NAME = "PhysicalAI-Robotics-GR00T-Teleop-GR1"


def load_episode_lines(episodes_jsonl: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(episodes_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[int(obj["episode_index"])] = line
    return out


def source_for_parquet(parquet_path: Path) -> str:
    cols = pq.read_schema(parquet_path).names
    if "annotation.human.action.verb" in cols:
        return "GR1_robot"
    return "DreamDojo-HV_Eval"


def repair_split(
    split_dir: Path,
    gr1_base: Path,
    dry_run: bool,
) -> None:
    dd_lines = load_episode_lines(gr1_base / "DreamDojo-HV_Eval" / "meta" / "episodes.jsonl")
    gr1_lines = load_episode_lines(gr1_base / "GR1_robot" / "meta" / "episodes.jsonl")

    parquets = sorted(split_dir.rglob("*.parquet"))
    if not parquets:
        print(f"[{split_dir.name}] no parquet files, skip")
        return

    rows: list[dict] = []
    missing: list[tuple[int, str, Path]] = []

    for p in parquets:
        stem = p.stem
        if not stem.startswith("episode_"):
            continue
        idx = int(stem.replace("episode_", ""))
        src = source_for_parquet(p)
        line = gr1_lines.get(idx) if src == "GR1_robot" else dd_lines.get(idx)
        if line is None:
            missing.append((idx, src, p))
            continue
        rows.append(json.loads(line))

    rows.sort(key=lambda r: r["episode_index"])

    if missing:
        print(f"[{split_dir.name}] WARNING: {len(missing)} parquet files could not be matched:")
        for idx, src, p in missing[:12]:
            print(f"    episode_index={idx} inferred={src} path={p}")
        if len(missing) > 12:
            print(f"    ... and {len(missing) - 12} more")

    meta_dir = split_dir / "meta"
    ep_path = meta_dir / "episodes.jsonl"
    info_path = meta_dir / "info.json"

    total_frames = sum(int(r["length"]) for r in rows)

    print(
        f"[{split_dir.name}] parquets_on_disk={len(parquets)} "
        f"episodes_written={len(rows)} total_frames={total_frames}"
    )

    if dry_run:
        return

    meta_dir.mkdir(parents=True, exist_ok=True)
    if ep_path.exists() and not (meta_dir / "episodes.jsonl.bak").exists():
        shutil.copy2(ep_path, meta_dir / "episodes.jsonl.bak")
    if info_path.exists() and not (meta_dir / "info.json.bak").exists():
        shutil.copy2(info_path, meta_dir / "info.json.bak")

    with open(ep_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    info: dict = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    info["total_episodes"] = len(rows)
    info["total_frames"] = total_frames
    info["total_videos"] = len(rows)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("makovian", "non_makovian"),
        help="Repair only these splits (default: both).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ws: Path = args.workspace
    gr1_base = ws / "datasets" / "humanoid" / "singleview" / PHYSICAL_NAME
    if not gr1_base.is_dir():
        # This repair backfills split meta FROM the original PhysicalAI release.
        # Once the splits carry complete meta the source is no longer needed, and
        # it is not kept in the workspace. Hard-failing here turned a finished
        # one-off migration into a permanent startup error for every caller
        # (run_humanoid_singleview_dreamdojo_{dense,sito,itm}.sh and the grid
        # searches), which is why SiTo/ITM on this cell had never run. Treat an
        # absent source as "nothing left to repair" when the splits already look
        # complete, and only fail when they do not.
        sv_root = ws / "datasets" / "humanoid" / "singleview"
        incomplete = []
        for name in (args.only if args.only else ["makovian", "non_makovian"]):
            episodes = sv_root / name / "meta" / "episodes.jsonl"
            n_meta = len(load_episode_lines(episodes)) if episodes.is_file() else 0
            n_data = len(list((sv_root / name / "data").rglob("*.parquet")))
            if n_meta == 0 or n_meta != n_data:
                incomplete.append(f"{name} (meta={n_meta}, parquet={n_data})")
        if incomplete:
            raise SystemExit(
                f"Missing dataset root: {gr1_base}\n"
                f"and the splits are NOT already repaired: {', '.join(incomplete)}.\n"
                "Restore the PhysicalAI release to repair them."
            )
        print(f"[skip] {gr1_base} absent, but split meta is already complete "
              "-> nothing to repair")
        return

    sv_root = ws / "datasets" / "humanoid" / "singleview"
    splits = args.only if args.only else ["makovian", "non_makovian"]

    for name in splits:
        repair_split(sv_root / name, gr1_base, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
