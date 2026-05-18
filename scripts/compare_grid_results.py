"""Compare grid search results across ITM and SiTo configs.

Usage:
    python scripts/compare_grid_results.py
    python scripts/compare_grid_results.py --grid-root tmp_2/grid_search tmp_2/grid_search_sito
"""

import argparse
import json
from pathlib import Path


def find_summaries(grid_root: Path):
    results = []
    for summary_file in sorted(grid_root.rglob("all_summary.json")):
        rel = summary_file.relative_to(grid_root)
        config_name = rel.parts[0]
        with open(summary_file) as f:
            data = json.load(f)
        results.append((config_name, str(rel.parent), data))
    return results


def parse_timing_log(timing_log: Path):
    timings = {}
    current_config = None
    if not timing_log.exists():
        return timings
    with open(timing_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("--- Config:"):
                current_config = line.split("Config:")[1].strip().rstrip(" ---")
            elif line.startswith("s/it:") and current_config:
                timings[current_config] = float(line.split(":")[1].strip())
            elif line.startswith("Total time:") and current_config:
                timings.setdefault(current_config, {})
                if isinstance(timings[current_config], dict):
                    timings[current_config] = None
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid-root",
        nargs="+",
        default=["tmp_2/grid_search", "tmp_2/grid_search_sito"],
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parent.parent
    all_results = []
    all_timings = {}

    for grid_dir in args.grid_root:
        grid_path = workspace / grid_dir
        if not grid_path.exists():
            continue
        results = find_summaries(grid_path)
        all_results.extend([(grid_dir, *r) for r in results])
        timing_log = grid_path / "timing_log.txt"
        if timing_log.exists():
            with open(timing_log) as f:
                content = f.read()
            current_config = None
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("--- Config:"):
                    current_config = line.split("Config:")[1].strip().rstrip(" ---")
                elif "s/it:" in line and current_config:
                    try:
                        s_per_it = float(line.split("s/it:")[1].strip())
                        all_timings[f"{grid_dir}/{current_config}"] = s_per_it
                    except ValueError:
                        pass

    if not all_results:
        print("No results found. Run grid search scripts first.")
        return

    print("=" * 90)
    print(f"{'Config':<35} {'PSNR↑':>8} {'SSIM↑':>8} {'LPIPS↓':>8} {'s/it':>8} {'Speedup':>8}")
    print("=" * 90)

    baseline_time = None
    rows = []

    for grid_dir, config_name, rel_path, data in all_results:
        psnr = data.get("psnr_mean", data.get("psnr", "N/A"))
        ssim = data.get("ssim_mean", data.get("ssim", "N/A"))
        lpips = data.get("lpips_mean", data.get("lpips", "N/A"))

        key = f"{grid_dir}/{config_name}"
        s_it = all_timings.get(key)

        if "dense_baseline" in config_name and s_it is not None:
            baseline_time = s_it

        rows.append((config_name, psnr, ssim, lpips, s_it, grid_dir))

    for config_name, psnr, ssim, lpips, s_it, grid_dir in rows:
        psnr_str = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else str(psnr)
        ssim_str = f"{ssim:.4f}" if isinstance(ssim, (int, float)) else str(ssim)
        lpips_str = f"{lpips:.4f}" if isinstance(lpips, (int, float)) else str(lpips)
        s_it_str = f"{s_it:.2f}s" if s_it else "N/A"
        speedup_str = ""
        if s_it and baseline_time and baseline_time > 0:
            speedup = baseline_time / s_it
            speedup_str = f"{speedup:.2f}x"

        label = config_name
        if "sito" in grid_dir:
            label = f"[SiTo] {config_name}"
        elif "grid_search" in grid_dir and "sito" not in grid_dir:
            label = f"[ITM]  {config_name}" if "itm" in config_name else f"       {config_name}"

        print(f"{label:<35} {psnr_str:>8} {ssim_str:>8} {lpips_str:>8} {s_it_str:>8} {speedup_str:>8}")

    print("=" * 90)
    print("\n↑ = higher is better, ↓ = lower is better")
    if baseline_time:
        print(f"Baseline s/it: {baseline_time:.2f}s")


if __name__ == "__main__":
    main()
