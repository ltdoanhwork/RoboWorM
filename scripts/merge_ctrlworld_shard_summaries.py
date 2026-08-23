"""
Aggregate a FINAL per-category `all_summary.json` from the per-episode
`metrics.json` files of a Ctrl-World multiview output tree.

Needed when a full run was sharded across GPUs (`--num_shards`): each shard
writes `all_summary.shard{i}of{N}.json` covering only its own episodes. This
script rebuilds the authoritative summary from the per-episode metrics (the
source of truth), so it is also safe to run after a resumed / partial run.

Usage:
  python scripts/merge_ctrlworld_shard_summaries.py \
      --output-root sampling_dataset/dense/bimanual/output/multiview/ctrlworld_4view_grid

Writes {output_root}/{category}/all_summary.json (+ an overall summary at the
root) and removes the per-shard files unless --keep-shard-files is given.
"""

import argparse
import json
from pathlib import Path


def summarize(metrics_list):
    n = len(metrics_list)
    summary = {
        "mean_psnr": sum(m["psnr"] for m in metrics_list) / n,
        "mean_ssim": sum(m["ssim"] for m in metrics_list) / n,
        "mean_lpips": sum(m["lpips"] for m in metrics_list) / n,
        "num_episodes": n,
    }
    times = [m["wall_time_s"] for m in metrics_list if "wall_time_s" in m]
    if times:
        summary["mean_wall_time_s"] = sum(times) / len(times)

    # Per-view means (all episodes share the same view names).
    views = {}
    for m in metrics_list:
        for view, vm in (m.get("per_view") or {}).items():
            views.setdefault(view, []).append(vm)
    if views:
        summary["per_view"] = {
            view: {
                "psnr": sum(x["psnr"] for x in vals) / len(vals),
                "ssim": sum(x["ssim"] for x in vals) / len(vals),
                "lpips": sum(x["lpips"] for x in vals) / len(vals),
                "num_episodes": len(vals),
            }
            for view, vals in sorted(views.items())
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=str, required=True,
                        help="Dir containing {category}/episode_*/metrics.json")
    parser.add_argument("--keep-shard-files", action="store_true",
                        help="Keep the all_summary.shard*.json files.")
    args = parser.parse_args()

    root = Path(args.output_root)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    by_category = {}
    for mf in sorted(root.glob("*/episode_*/metrics.json")):
        category = mf.parent.parent.name
        with open(mf) as f:
            by_category.setdefault(category, []).append(json.load(f))
    # Flat layout fallback (no category dir).
    for mf in sorted(root.glob("episode_*/metrics.json")):
        by_category.setdefault(None, []).append(json.load(open(mf)))

    if not by_category:
        raise SystemExit(f"No per-episode metrics.json found under {root}")

    all_metrics = []
    for category, metrics_list in sorted(by_category.items(), key=lambda kv: kv[0] or ""):
        all_metrics += metrics_list
        out_dir = root / category if category is not None else root
        summary = summarize(metrics_list)
        with open(out_dir / "all_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{category or 'all'}] {summary['num_episodes']} episodes  "
              f"PSNR {summary['mean_psnr']:.3f}  SSIM {summary['mean_ssim']:.4f}  "
              f"LPIPS {summary['mean_lpips']:.4f}")
        for view, vm in (summary.get("per_view") or {}).items():
            print(f"    {view:<18} PSNR {vm['psnr']:.2f}  SSIM {vm['ssim']:.4f}  LPIPS {vm['lpips']:.4f}")

    if len(by_category) > 1:
        overall = summarize(all_metrics)
        with open(root / "all_summary.json", "w") as f:
            json.dump(overall, f, indent=2)
        print(f"\n[OVERALL] {overall['num_episodes']} episodes  "
              f"PSNR {overall['mean_psnr']:.3f}  SSIM {overall['mean_ssim']:.4f}  "
              f"LPIPS {overall['mean_lpips']:.4f}")

    if not args.keep_shard_files:
        removed = 0
        for sf in root.glob("**/all_summary.shard*.json"):
            sf.unlink()
            removed += 1
        if removed:
            print(f"\nremoved {removed} per-shard summary file(s)")


if __name__ == "__main__":
    main()
