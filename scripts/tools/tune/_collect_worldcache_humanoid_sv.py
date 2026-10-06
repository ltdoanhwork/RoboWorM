#!/usr/bin/env python3
"""Collect the humanoid-singleview WorldCache sweep into one report.

Timing comes from the per-episode "LVG Total inference time" lines in each run
log (never wall clock). Quality comes from the per-config quality_vs_dense.json
written by scripts/metrics/_score_vs_dense.py. Speedup is computed against the matched
dense control produced through the identical entrypoint.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

TIME_RE = re.compile(r"LVG Total inference time:\s*([0-9.]+)\s*seconds for\s*(\d+)\s*chunks")


def parse_times(log: Path) -> list[dict]:
    if not log.is_file():
        return []
    out = []
    for m in TIME_RE.finditer(log.read_text(errors="replace")):
        out.append({"seconds": float(m.group(1)), "chunks": int(m.group(2))})
    return out


def summarize(tag: str, scratch: Path, split: str, out_iter: str, knobs: dict | None) -> dict:
    times = parse_times(scratch / "logs" / f"{tag}.log")
    secs = [t["seconds"] for t in times]
    row: dict = {
        "tag": tag,
        "knobs": knobs,
        "n_timed_episodes": len(secs),
        "per_episode_s": [round(s, 3) for s in secs],
        "total_s": round(sum(secs), 3) if secs else None,
        "mean_episode_s": round(st.mean(secs), 3) if secs else None,
        "median_episode_s": round(st.median(secs), 3) if secs else None,
    }
    q = scratch / tag / "quality_vs_dense.json"
    if q.is_file():
        d = json.loads(q.read_text())
        row.update(
            {
                "vs_dense_psnr": round(d["mean_psnr"], 3) if d.get("mean_psnr") is not None else None,
                "vs_dense_ssim": round(d["mean_ssim"], 4),
                "vs_dense_lpips": round(d["mean_lpips"], 4),
                "n_scored_episodes": d["n_episodes"],
                "n_bit_identical": d.get("n_identical"),
                "per_episode_quality": {
                    k: {
                        "psnr": round(v["psnr"], 3) if v.get("psnr") is not None else None,
                        "ssim": round(v["ssim"], 4),
                        "lpips": round(v["lpips"], 4),
                        "frames": v["frames"],
                        "identical": v.get("identical"),
                    }
                    for k, v in d["per_episode"].items()
                },
            }
        )
    mp4s = sorted((scratch / tag / split / out_iter).glob("*.mp4"))
    row["videos"] = [str(p) for p in mp4s]
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-iter", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--configs", required=True, help="semicolon list of tag|thresh|ret|probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scratch = Path(args.scratch)
    episodes = [e for e in args.episodes.split(",") if e]

    dense = summarize("dense_control", scratch, args.split, args.out_iter, {"worldcache": "disabled"})

    cands = []
    for spec in args.configs.split(";"):
        if not spec.strip():
            continue
        tag, thresh, ret, probe = spec.split("|")
        cands.append(
            summarize(
                tag,
                scratch,
                args.split,
                args.out_iter,
                {"rel_l1_thresh": float(thresh), "ret_ratio": float(ret), "probe_depth": int(probe)},
            )
        )

    base = dense.get("total_s")
    base_mean = dense.get("mean_episode_s")
    for c in cands:
        c["speedup_total"] = round(base / c["total_s"], 3) if base and c.get("total_s") else None
        c["speedup_mean_episode"] = (
            round(base_mean / c["mean_episode_s"], 3) if base_mean and c.get("mean_episode_s") else None
        )

    floor_path = scratch / "dense_control" / "quality_vs_shipped_dense.json"
    floor = None
    if floor_path.is_file():
        d = json.loads(floor_path.read_text())
        floor = {
            "psnr": round(d["mean_psnr"], 3) if d.get("mean_psnr") is not None else None,
            "ssim": round(d["mean_ssim"], 4),
            "lpips": round(d["mean_lpips"], 4),
            "n_episodes": d.get("n_episodes"),
            "n_bit_identical": d.get("n_identical"),
            "per_episode": {
                k: {
                    "psnr": round(v["psnr"], 3) if v.get("psnr") is not None else None,
                    "ssim": round(v["ssim"], 4),
                    "lpips": round(v["lpips"], 4),
                    "identical": v.get("identical"),
                }
                for k, v in d["per_episode"].items()
            },
            "interpretation": (
                "Dense control vs the shipped dense sampling_dataset output. Episodes "
                "that are BIT-IDENTICAL confirm this single-GPU dense path reproduces "
                "the shipped pass exactly, so candidate deviations below are caused by "
                "the cache and not by run-to-run noise. Episodes that differ were "
                "generated by an older bulk pass under a different launch configuration "
                "(multi-GPU context parallelism), so their deviation reflects that "
                "configuration difference, NOT sampler nondeterminism."
            ),
        }

    report = {
        "cell": {
            "family": "dreamgen",
            "backbone": "Cosmos-Predict2 2B DiT",
            "embodiment": "humanoid",
            "view": "singleview",
            "entrypoint": "models.dreamgen.examples.video2world_lvg",
            "checkpoint_iter": args.out_iter,
            "split": args.split,
            "chunks": 2,
        },
        "protocol": {
            "episodes": episodes,
            "timing_source": "per-episode 'LVG Total inference time' log line (not wall clock)",
            "dense_control": (
                "same run script with WORLDCACHE_ENABLED=0, so control and candidates "
                "differ only by the cache flags"
            ),
            "quality_reference": "matched dense prediction of the same episode (PhyFid-DR)",
        },
        "dense_control": dense,
        "dense_control_vs_shipped_dense": floor,
        "candidates": cands,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\n{'tag':<20}{'total_s':>9}{'mean_s':>9}{'speedup':>9}{'PSNR':>9}{'SSIM':>8}{'LPIPS':>8}")
    print("-" * 72)
    print(f"{'dense_control':<20}{dense.get('total_s') or 0:>9.1f}{dense.get('mean_episode_s') or 0:>9.1f}{'1.00x':>9}{'-':>9}{'-':>8}{'-':>8}")
    for c in cands:
        print(
            f"{c['tag']:<20}{c.get('total_s') or 0:>9.1f}{c.get('mean_episode_s') or 0:>9.1f}"
            f"{(('%.2fx' % c['speedup_total']) if c.get('speedup_total') else '-'):>9}"
            f"{(('%.2f' % c['vs_dense_psnr']) if c.get('vs_dense_psnr') else '-'):>9}"
            f"{(('%.4f' % c['vs_dense_ssim']) if c.get('vs_dense_ssim') else '-'):>8}"
            f"{(('%.4f' % c['vs_dense_lpips']) if c.get('vs_dense_lpips') else '-'):>8}"
        )
    if floor:
        ident = floor.get("n_bit_identical") or 0
        print(
            f"\ndense control vs shipped dense: {ident}/{floor.get('n_episodes')} bit-identical"
            f"  (differing episodes: SSIM={floor['ssim']:.4f} LPIPS={floor['lpips']:.4f}"
            " -- older bulk pass, different launch config)"
        )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
