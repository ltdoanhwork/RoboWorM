"""One-shot migration: fix bimanual/humanoid multiview Ctrl-World layout.

The bimanual/humanoid inference runs saved flat episode dirs like
`.../ctrlworld/episode_makovian_close_toolbox__000026/`, baking the
makovian/non_makovian category into the episode name. Per the
sampling-dataset-layout rule, `{category}` must be its own path segment:
`.../ctrlworld/makovian/episode_close_toolbox__000026/`.

This moves existing output + input episode dirs into the correct
`{category}/episode_<name>` layout, deletes the stale top-level
`all_summary.json`, and regenerates a per-category `all_summary.json` from the
moved output `metrics.json` files.

Usage:
    python scripts/migrate_ctrlworld_multiview_layout.py            # dry run
    python scripts/migrate_ctrlworld_multiview_layout.py --apply    # do it
"""

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics")
EMBODIMENTS = ["bimanual", "humanoid"]
IOS = ["output", "input"]


def split_category(name):
    if name.startswith("non_makovian_"):
        return "non_makovian", name[len("non_makovian_"):]
    if name.startswith("makovian_"):
        return "makovian", name[len("makovian_"):]
    return None, name


def migrate_dir(base: Path, apply: bool):
    """Move flat episode_<category>_<rest> dirs into <category>/episode_<rest>."""
    if not base.exists():
        print(f"  [skip] {base} does not exist")
        return 0
    moved = 0
    for child in sorted(base.iterdir()):
        if not child.is_dir() or not child.name.startswith("episode_"):
            continue
        stem = child.name[len("episode_"):]
        category, rest = split_category(stem)
        if category is None:
            print(f"  [warn] no category prefix, leaving as-is: {child.name}")
            continue
        dst = base / category / f"episode_{rest}"
        if dst.exists():
            print(f"  [conflict] dst exists, skipping: {dst}")
            continue
        print(f"  {child.name}  ->  {category}/episode_{rest}")
        if apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(child), str(dst))
        moved += 1
    return moved


def regenerate_summaries(output_base: Path, apply: bool):
    """Rebuild per-category all_summary.json from moved metrics.json files."""
    if not output_base.exists():
        return
    for category_dir in sorted(output_base.iterdir()):
        if not category_dir.is_dir() or category_dir.name not in ("makovian", "non_makovian"):
            continue
        metrics_list = []
        for ep_dir in sorted(category_dir.iterdir()):
            mpath = ep_dir / "metrics.json"
            if mpath.exists():
                with open(mpath) as f:
                    metrics_list.append(json.load(f))
        if not metrics_list:
            continue
        summary = {
            "mean_psnr": sum(m["psnr"] for m in metrics_list) / len(metrics_list),
            "mean_ssim": sum(m["ssim"] for m in metrics_list) / len(metrics_list),
            "mean_lpips": sum(m["lpips"] for m in metrics_list) / len(metrics_list),
            "num_episodes": len(metrics_list),
        }
        out = category_dir / "all_summary.json"
        print(f"  summary {out.relative_to(output_base)}: "
              f"n={summary['num_episodes']} PSNR={summary['mean_psnr']:.3f}")
        if apply:
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Actually move files (default: dry run).")
    args = ap.parse_args()

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== Ctrl-World multiview layout migration [{mode}] ===\n")

    for emb in EMBODIMENTS:
        for io in IOS:
            base = ROOT / "sampling_dataset" / "dense" / emb / io / "multiview" / "ctrlworld"
            print(f"[{emb}/{io}] {base}")
            n = migrate_dir(base, args.apply)
            print(f"  -> {n} dirs to move")

            stale = base / "all_summary.json"
            if stale.exists():
                print(f"  [stale] removing top-level {stale.name} (regenerated per-category)")
                if args.apply:
                    stale.unlink()
            print()

    print("=== Regenerating per-category summaries (output only) ===")
    for emb in EMBODIMENTS:
        output_base = ROOT / "sampling_dataset" / "dense" / emb / "output" / "multiview" / "ctrlworld"
        print(f"[{emb}/output]")
        regenerate_summaries(output_base, args.apply)
        print()

    if not args.apply:
        print("Dry run complete. Re-run with --apply to perform the migration.")


if __name__ == "__main__":
    main()
