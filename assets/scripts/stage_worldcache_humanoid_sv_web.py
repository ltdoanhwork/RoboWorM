#!/usr/bin/env python3
"""Stage the selected humanoid-singleview WorldCache operating point for the web demo.

Reads metrics/reports/worldcache_humanoid_sv_tuning.json, picks the operating
point, and copies the accelerated videos plus the matched dense control into
sampling_dataset/data_for_web/{worldcache,dense}/humanoid/output/singleview/
dreamgen/<split>/<iter>/, each with a timing.json sidecar.

Selection rule: among candidates whose fidelity to the dense control is at
least on par with the shipped default (SSIM no worse, LPIPS no worse, within
tolerance), take the fastest. A candidate that is both faster AND better than
the shipped default therefore wins outright.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "metrics/reports/worldcache_humanoid_sv_tuning.json"
WEB = ROOT / "sampling_dataset/data_for_web"
SHIPPED_TAG = "A_t0.15_r0.30_p4"  # current script default = the shipped optimum
SSIM_TOL = 0.005
LPIPS_TOL = 0.005

TIMING_NOTE = (
    "per-episode 'LVG Total inference time' from the inference log "
    "(excludes model load, VAE decode, ffmpeg and metric computation)"
)
MEASURED_NOTE = (
    "single idle GPU, strictly sequential runs, identical episode set for the dense "
    "control and every configuration; the dense control is the same run script with "
    "WorldCache disabled, so only the cache flags differ"
)


def select(cands: list[dict]) -> dict:
    shipped = next((c for c in cands if c["tag"] == SHIPPED_TAG), None)
    if shipped is None:
        raise SystemExit(f"ERROR: shipped reference config {SHIPPED_TAG} not in report")

    eligible = []
    for c in cands:
        if not c.get("speedup_total"):
            continue
        d_ssim = c["vs_dense_ssim"] - shipped["vs_dense_ssim"]
        d_lpips = c["vs_dense_lpips"] - shipped["vs_dense_lpips"]
        if d_ssim >= -SSIM_TOL and d_lpips <= LPIPS_TOL:
            eligible.append(c)
            print(
                f"  eligible: {c['tag']:<20} speedup {c['speedup_total']:.2f}x  "
                f"dSSIM {d_ssim:+.4f}  dLPIPS {d_lpips:+.4f}"
            )
        else:
            print(
                f"  rejected: {c['tag']:<20} speedup {c['speedup_total']:.2f}x  "
                f"dSSIM {d_ssim:+.4f}  dLPIPS {d_lpips:+.4f}  (below shipped fidelity)"
            )

    if not eligible:
        raise SystemExit("ERROR: no candidate matched shipped fidelity")
    best = max(eligible, key=lambda c: c["speedup_total"])
    print(f"\n  -> selected {best['tag']} ({best['speedup_total']:.2f}x)")
    return best


def stage(rep: dict, best: dict, dense: dict, dry_run: bool, canonical: bool) -> None:
    cell = rep["cell"]
    sub = Path(cell["embodiment"]) / "output" / cell["view"] / cell["family"] / cell["split"] / cell["checkpoint_iter"]
    if canonical:
        # Writes straight into the curated web trees. This OVERWRITES any
        # same-named episode already published there, which mixes provenance,
        # so it is opt-in only.
        wc_dir = WEB / "worldcache" / sub
        dn_dir = WEB / "dense" / sub
    else:
        # Default: a self-contained, non-destructive folder for this qualitative
        # comparison. Nothing already published in data_for_web is touched.
        base = WEB / "qualitative" / "worldcache_humanoid_singleview_dreamgen"
        wc_dir = base / "worldcache"
        dn_dir = base / "dense"

    for tag, row, dest, extra in (
        ("worldcache", best, wc_dir, True),
        ("dense", dense, dn_dir, False),
    ):
        if not dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        timing: dict = {
            "method": tag,
            "mean_episode_s": row["mean_episode_s"],
            "median_episode_s": row["median_episode_s"],
            "per_episode_s": row["per_episode_s"],
            "total_s": row["total_s"],
            "n_episodes": row["n_timed_episodes"],
            "timing_note": TIMING_NOTE,
            "measured": MEASURED_NOTE,
            "cell": cell,
            "episodes": rep["protocol"]["episodes"],
        }
        if extra:
            timing["config"] = row["knobs"]
            timing["speedup_vs_dense"] = row["speedup_total"]
            timing["fidelity_to_dense"] = {
                "psnr": row["vs_dense_psnr"],
                "ssim": row["vs_dense_ssim"],
                "lpips": row["vs_dense_lpips"],
                "reference": "matched dense control prediction of the same episode",
                "per_episode": row["per_episode_quality"],
            }
            timing["shipped_default_for_comparison"] = SHIPPED_TAG
        else:
            timing["config"] = {"worldcache": "disabled"}
            timing["speedup_vs_dense"] = 1.0

        for src in row["videos"]:
            src_p = Path(src)
            dst_p = dest / src_p.name
            clobber = " (OVERWRITES existing)" if dst_p.exists() else ""
            print(f"  {'[dry] ' if dry_run else ''}{src_p.name} -> {dest}{clobber}")
            if not dry_run:
                shutil.copy2(src_p, dst_p)
        if not dry_run:
            (dest / "timing.json").write_text(json.dumps(timing, indent=2))
        print(f"  {'[dry] ' if dry_run else ''}timing.json -> {dest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--canonical",
        action="store_true",
        help="write into data_for_web/{worldcache,dense}/... instead of the scoped "
        "qualitative/ folder; may overwrite already-published episodes",
    )
    args = ap.parse_args()

    rep = json.loads(Path(args.report).read_text())
    print("Selecting operating point:")
    best = select(rep["candidates"])
    print()
    stage(rep, best, rep["dense_control"], args.dry_run, args.canonical)
    print("\nDone.")


if __name__ == "__main__":
    main()
