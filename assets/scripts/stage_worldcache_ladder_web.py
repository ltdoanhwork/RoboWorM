#!/usr/bin/env python3
"""Consolidate the WorldCache quality ladder into ONE qualitative folder.

Produces a single tree under
  sampling_dataset/data_for_web/qualitative/worldcache_quality_ladder/
holding, for both humanoid DreamGen cells, the four rungs

  01_dense      reference (no caching)
  02_ok         shipped/selected operating point -- quality parity
  03_degraded   visibly soft, still coherent
  04_broken     collapsed

Rungs 01/02 are copied from what the two tuning passes already published; 03/04/05
come from the scratch produced by scripts/_make_worldcache_ladder.sh. Every rung
uses the same episodes, so files with the same name are directly comparable.

Idempotent: rungs whose upstream source has been retired (the singleview tuning
pass used to publish into its own folder, now consolidated here) are preserved
from the copy already in this tree, so a re-run never empties them.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "sampling_dataset/data_for_web"
DEST = WEB / "qualitative/worldcache_quality_ladder"
SCRATCH = Path(os.environ.get("LADDER_SCRATCH", ROOT / "tmp/wc_ladder"))

SV_PUB = WEB / "qualitative/worldcache_humanoid_singleview_dreamgen"  # legacy, retired
MV_PUB_SUB = Path("humanoid/output/multiview/dreamgen/non_makovian/iter_000100000")
MV_TASKS = [
    "AIRBOT_MMK2_boxs_storage__episode_000046",
    "Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate__episode_000066",
]
# Bright multiview set. The two MV_TASKS above are the DARKEST in the dataset
# (grid mean luma ~51/50 of 255, ~36% near-black) because their wrist cameras are
# unlit; these three are the brightest distinct tasks (138.6 / 137.6 / 128.6) and
# span both grid layouts and both splits. Measured by
# scripts/_rank_multiview_brightness.py. Rungs come wholly from ladder scratch,
# including their own dense reference.
MVB_CELL = "humanoid_multiview_bright_dreamgen"
MVB_RUNGS = ["01_dense", "02_ok", "04_degraded", "05_broken"]
MVB_KNOBS = {
    "01_dense": None,
    "02_ok": dict(rel_l1_thresh=0.25, ret_ratio=0.2, probe_depth=2),
    "04_degraded": dict(rel_l1_thresh=0.30, ret_ratio=0.10, probe_depth=2),
    "05_broken": dict(rel_l1_thresh=1.00, ret_ratio=0.00, probe_depth=2),
}
MVB_TAG = {"01_dense": "dense", "02_ok": "ok",
           "04_degraded": "degraded", "05_broken": "broken"}

# Known-good measurements from the two tuning passes (see
# metrics/reports/worldcache_humanoid_{sv,mv}_tuning.json).
SV_KNOWN = {
    "01_dense": dict(knobs=None, s_per_ep=259.37, speedup=1.00),
    "02_ok": dict(knobs=dict(rel_l1_thresh=0.12, ret_ratio=0.2, probe_depth=4),
                  s_per_ep=121.82, speedup=2.13,
                  fidelity=dict(mean_psnr=16.95, mean_ssim=0.6445, mean_lpips=0.2866)),
}
MV_KNOWN = {
    "01_dense": dict(knobs=None, s_per_seg=118.84, speedup=1.00),
    "02_ok": dict(knobs=dict(rel_l1_thresh=0.25, ret_ratio=0.2, probe_depth=2),
                  s_per_seg=53.29, speedup=2.23,
                  fidelity=dict(mean_psnr=18.684, mean_ssim=0.7850, mean_lpips=0.2716)),
}
LADDER_KNOBS = {
    "03_near_ok": {"sv": dict(rel_l1_thresh=0.20, ret_ratio=0.15, probe_depth=4)},
    "04_degraded": {"sv": dict(rel_l1_thresh=0.30, ret_ratio=0.10, probe_depth=4),
                    "mv": dict(rel_l1_thresh=0.30, ret_ratio=0.10, probe_depth=2)},
    "05_broken": {"sv": dict(rel_l1_thresh=1.00, ret_ratio=0.00, probe_depth=4),
                  "mv": dict(rel_l1_thresh=1.00, ret_ratio=0.00, probe_depth=2)},
}
# Measured values for the ladder rungs, so metadata survives the scratch being
# deleted (staging must never overwrite good numbers with None). Scratch, when
# present, is still preferred -- these are the fallback of record.
MEASURED = {
    "sv": {
        "03_near_ok": dict(t=97.77, speedup=2.653,
                           fid=dict(mean_psnr=15.155, mean_ssim=0.5407, mean_lpips=0.4037)),
        "04_degraded": dict(t=87.17, speedup=2.975,
                            fid=dict(mean_psnr=11.643, mean_ssim=0.3765, mean_lpips=0.5705)),
        "05_broken": dict(t=63.45, speedup=4.088,
                          fid=dict(mean_psnr=12.036, mean_ssim=0.3951, mean_lpips=0.5517)),
    },
    "mv": {
        "04_degraded": dict(t=42.86, speedup=2.773,
                            fid=dict(mean_psnr=15.049, mean_ssim=0.6228, mean_lpips=0.3868)),
        "05_broken": dict(t=18.11, speedup=6.562,
                          fid=dict(mean_psnr=16.345, mean_ssim=0.6469, mean_lpips=0.3857)),
    },
}
# scratch tag for each rung (scripts/_make_worldcache_ladder.sh writes sv_<tag>)
RUNG_TAG = {"03_near_ok": "near_ok", "04_degraded": "degraded", "05_broken": "broken"}
RUNG_DESC = {
    "01_dense": "reference: caching disabled",
    "02_ok": "selected operating point: fidelity at parity with dense",
    "03_near_ok": "almost fine: slight softening, all structure intact",
    "04_degraded": "visibly softened / smeared, subject still locatable",
    "05_broken": "collapsed: no dense warm-up and every step skipped",
}
SV_RUNGS = ["01_dense", "02_ok", "03_near_ok", "04_degraded", "05_broken"]
MV_RUNGS = ["01_dense", "02_ok", "04_degraded", "05_broken"]


def seg_times(log: Path, pattern: str) -> list[float]:
    if not log.is_file():
        return []
    return [float(x) for x in re.findall(pattern, log.read_text(errors="replace"))]


def copy(src: Path, dst_dir: Path, name: str | None = None) -> bool:
    if not src.is_file():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / (name or src.name))
    return True


def write_meta(d: Path, **kw) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "info.json").write_text(json.dumps(kw, indent=2))


def fidelity(prefix: str, tag: str) -> dict | None:
    """Fidelity to the dense prediction of the same episode (PhyFid-DR), as
    measured by scripts/_score_vs_dense.py during the ladder pass."""
    p = SCRATCH / f"score_{prefix}_{tag}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    out = {}
    for k, nd in (("mean_psnr", 3), ("mean_ssim", 4), ("mean_lpips", 4)):
        v = d.get(k)
        if v is not None:
            out[k] = round(v, nd)
    return out or None


def do_sv() -> list[str]:
    notes = []
    cell = DEST / "humanoid_singleview_dreamgen"
    dense_s = SV_KNOWN["01_dense"]["s_per_ep"]
    for rung in SV_RUNGS:
        if rung in SV_KNOWN:  # published rungs: dense + selected operating point
            srcdir = SV_PUB / ("dense" if rung == "01_dense" else "worldcache")
            n = sum(copy(p, cell / rung) for p in sorted(srcdir.glob("*.mp4")))
            if n == 0:  # already consolidated here on an earlier run
                n = len(list((cell / rung).glob("*.mp4")))
            k = SV_KNOWN[rung]
            write_meta(cell / rung, rung=rung, description=RUNG_DESC[rung],
                       method="dense" if rung == "01_dense" else "worldcache",
                       config=k["knobs"], mean_s_per_episode=k["s_per_ep"],
                       speedup_vs_dense=k["speedup"], n_episodes=n,
                       fidelity_vs_dense=k.get("fidelity"),
                       source="metrics/reports/worldcache_humanoid_sv_tuning.json")
            notes.append(f"  SV {rung}: {n} videos, {k['s_per_ep']}s/ep, {k['speedup']}x")
            continue
        tag = RUNG_TAG[rung]
        out = SCRATCH / f"sv_{tag}"
        n = sum(copy(p, cell / rung) for p in sorted(out.rglob("*.mp4")))
        if n == 0:  # scratch gone; keep what is already consolidated
            n = len(list((cell / rung).glob("*.mp4")))
        t = seg_times(SCRATCH / f"sv_{tag}.log", r"LVG Total inference time:\s*([0-9.]+)")
        m = MEASURED["sv"][rung]
        mean = round(sum(t) / len(t), 2) if t else m["t"]
        speed = round(dense_s / mean, 3) if t else m["speedup"]
        write_meta(cell / rung, rung=rung, description=RUNG_DESC[rung],
                   method="worldcache", config=LADDER_KNOBS[rung]["sv"],
                   mean_s_per_episode=mean, speedup_vs_dense=speed,
                   fidelity_vs_dense=fidelity("sv", tag) or m["fid"],
                   per_episode_s=[round(x, 2) for x in t], n_episodes=n)
        notes.append(f"  SV {rung}: {n} videos, {mean}s/ep, {speed}x")
    return notes


def do_mv() -> list[str]:
    notes = []
    cell = DEST / "humanoid_multiview_dreamgen"
    dense_s = MV_KNOWN["01_dense"]["s_per_seg"]
    for rung in MV_RUNGS:
        if rung in MV_KNOWN:
            method = "dense" if rung == "01_dense" else "worldcache"
            n = 0
            for task in MV_TASKS:
                src = WEB / method / MV_PUB_SUB / task
                for f in ("full_pred.mp4", "full_gt.mp4", "full_merged.mp4", "metrics.json"):
                    copy(src / f, cell / rung / task, f)
                n += (src / "full_pred.mp4").is_file()
            if n == 0:  # source retired; already consolidated here
                n = len(list((cell / rung).glob("*/full_pred.mp4")))
            k = MV_KNOWN[rung]
            write_meta(cell / rung, rung=rung, description=RUNG_DESC[rung], method=method,
                       config=k["knobs"], steady_s_per_segment=k["s_per_seg"],
                       speedup_vs_dense=k["speedup"], n_episodes=n,
                       fidelity_vs_dense=k.get("fidelity"),
                       source="metrics/reports/worldcache_humanoid_mv_tuning.json")
            notes.append(f"  MV {rung}: {n} episodes, {k['s_per_seg']}s/seg, {k['speedup']}x")
            continue
        tag = RUNG_TAG[rung]
        base = SCRATCH / f"mv_{tag}"
        n, qual = 0, []
        for mj in sorted(base.rglob("metrics.json")):
            epdir = mj.parent
            task = f"{epdir.parent.name}__{epdir.name}"
            for f in ("full_pred.mp4", "full_gt.mp4", "full_merged.mp4", "metrics.json"):
                copy(epdir / f, cell / rung / task, f)
            try:
                m = json.loads(mj.read_text())
                qual.append((m.get("psnr"), m.get("ssim")))
            except Exception:
                pass
            n += 1
        if n == 0:  # scratch gone; count what is already consolidated
            n = len(list((cell / rung).glob("*/full_pred.mp4")))
        t = seg_times(SCRATCH / f"mv_{tag}.log", r"Total inference time:\s*([0-9.]+)")
        # Segment 0 of each episode pays one-off warm-up; steady state excludes it.
        m = MEASURED["mv"][rung]
        steady = round(sum(t[1:]) / len(t[1:]), 2) if len(t) > 1 else m["t"]
        speed = round(dense_s / steady, 3) if len(t) > 1 else m["speedup"]
        vs_gt = None
        if qual and all(a is not None and b is not None for a, b in qual):
            vs_gt = {"mean_psnr": round(sum(a for a, _ in qual) / len(qual), 3),
                     "mean_ssim": round(sum(b for _, b in qual) / len(qual), 4)}
        write_meta(cell / rung, rung=rung, description=RUNG_DESC[rung],
                   method="worldcache", config=LADDER_KNOBS[rung]["mv"],
                   steady_s_per_segment=steady, speedup_vs_dense=speed,
                   fidelity_vs_dense=fidelity("mv", rung) or m["fid"],
                   quality_vs_gt=vs_gt,
                   per_segment_s=[round(x, 2) for x in t], n_episodes=n)
        notes.append(f"  MV {rung}: {n} episodes, {steady}s/seg, {speed}x")
    return notes


def do_mvb() -> list[str]:
    """Bright multiview cell: every rung, dense included, from ladder scratch."""
    notes = []
    cell = DEST / MVB_CELL
    dense_steady = None
    for rung in MVB_RUNGS:
        tag = MVB_TAG[rung]
        base = SCRATCH / f"mvb_{tag}"
        n = 0
        for mj in sorted(base.rglob("metrics.json")):
            epdir = mj.parent
            task = f"{epdir.parent.name}__{epdir.name}"
            for f in ("full_pred.mp4", "full_gt.mp4", "full_merged.mp4", "metrics.json"):
                copy(epdir / f, cell / rung / task, f)
            n += 1
        if n == 0:  # scratch gone; keep what is already consolidated
            n = len(list((cell / rung).glob("*/full_pred.mp4")))
        t = seg_times(SCRATCH / f"mvb_{tag}.log", r"Total inference time:\s*([0-9.]+)")
        steady = round(sum(t[1:]) / len(t[1:]), 2) if len(t) > 1 else None
        if rung == "01_dense":
            dense_steady = steady
        write_meta(cell / rung, rung=rung, description=RUNG_DESC[rung],
                   method="dense" if rung == "01_dense" else "worldcache",
                   config=MVB_KNOBS[rung], steady_s_per_segment=steady,
                   speedup_vs_dense=(round(dense_steady / steady, 3)
                                     if steady and dense_steady else None),
                   fidelity_vs_dense=fidelity("mvb", tag),
                   per_segment_s=[round(x, 2) for x in t], n_episodes=n)
        notes.append(f"  MVB {rung}: {n} episodes, {steady}s/seg, "
                     f"{round(dense_steady / steady, 2) if steady and dense_steady else '?'}x")
    return notes


def contact_sheet(cell_dir: Path, rel_video: str, out_png: Path) -> bool:
    """Stack 4 timepoints per rung as rows so the ladder is visible at a glance."""
    try:
        import imageio.v3 as iio
        import numpy as np
    except Exception:
        return False
    rows, labels = [], []
    for rung in sorted(p.name for p in cell_dir.iterdir() if p.is_dir()):
        v = cell_dir / rung / rel_video
        if not v.is_file():
            continue
        try:
            frames = np.stack([np.asarray(f) for f in iio.imiter(v)])
        except Exception:
            continue
        idx = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
        rows.append(np.concatenate([frames[i] for i in idx], axis=1))
        labels.append(rung)
    if not rows:
        return False
    w = min(r.shape[1] for r in rows)
    grid = np.concatenate([r[:, :w] for r in rows], axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(out_png, grid)
    print(f"  contact sheet ({' / '.join(labels)} top-to-bottom): {out_png}")
    return True


def main() -> None:
    # Clean per rung (not the whole tree): rungs whose source has been retired
    # must survive a re-run from their already-consolidated copy.
    print("SINGLEVIEW")
    for n in do_sv():
        print(n)
    print("MULTIVIEW")
    for n in do_mv():
        print(n)
    print("MULTIVIEW (bright)")
    for n in do_mvb():
        print(n)

    readme = DEST / "README.md"
    readme.write_text(
        "# WorldCache quality ladder (qualitative)\n\n"
        "Two DreamGen humanoid cells, one folder, ordered rungs from *fine* to\n"
        "*completely broken*. Every rung uses the same episodes, so identically-named\n"
        "files are directly comparable across rungs.\n\n"
        "| rung | meaning |\n|---|---|\n"
        + "".join(f"| `{k}` | {v} |\n" for k, v in RUNG_DESC.items())
        + "\nEach rung folder carries an `info.json` with the WorldCache knobs, the\n"
        "measured time, the speedup vs dense and the fidelity to the dense prediction.\n\n"
        "- `humanoid_singleview_dreamgen/` — one mp4 per episode (prediction only),\n"
        "  5 rungs.\n"
        "- `humanoid_multiview_dreamgen/` — per-episode folder with `full_pred.mp4`,\n"
        "  `full_gt.mp4` and `full_merged.mp4` (GT | pred side by side; easiest to\n"
        "  read, since a single multiview frame is itself a 2x2 camera grid). 4 rungs:\n"
        "  the singleview-only `03_near_ok` point was not run for this cell.\n"
        "- `humanoid_multiview_bright_dreamgen/` — same 4 rungs on 3 **brighter**\n"
        "  tasks. The two tasks in the cell above are the darkest in the dataset\n"
        "  (grid mean luma ~51/255, ~36% near-black pixels, because their wrist\n"
        "  cameras are unlit); these three are the brightest distinct tasks\n"
        "  (~138/137/129) and span both grid layouts and both splits. Prefer this\n"
        "  cell for figures. Ranked by `scripts/_rank_multiview_brightness.py`.\n\n"
        "## How to read the numbers\n\n"
        "`figures/<cell>__<episode>.png` is the ready-made ladder grid (rows = rungs\n"
        "top-to-bottom dense->broken, columns = timepoints, speedup on each row);\n"
        "`frames/` holds the same frames as individual lossless PNGs, with the exact\n"
        "indices recorded in `frames/index.json`. Indices are matched across rungs --\n"
        "every rung of an episode has identical length, so index i is the same moment.\n\n"
        "Timing: singleview from the per-episode `LVG Total inference time` log line;\n"
        "multiview from the per-segment `Total inference time`, steady state (segment 0\n"
        "excluded because it pays one-off warm-up).\n\n"
        "Quality for the accelerated rungs is fidelity to the **dense prediction** of\n"
        "the same episode (PhyFid-DR), not to ground truth.\n\n"
        "**Caveat — the metrics stop ranking the two worst rungs.** Past the point of\n"
        "collapse, reference metrics saturate and can even invert: multiview\n"
        "`05_broken` scores *above* `04_degraded` (SSIM 0.647 vs 0.623 against the\n"
        "dense prediction; 0.585 vs 0.558 against GT) while being visibly destroyed,\n"
        "because a low-detail washed-out frame correlates better with the reference\n"
        "than a frame full of wrong high-frequency structure. The same happens in\n"
        "singleview (0.395 vs 0.377). Rungs 04 and 05 are therefore both \"broken\" and\n"
        "must be judged visually; only rungs 01-03 are ordered by the metrics.\n\n"
        "## Reproduce\n\n"
        "```bash\n"
        "bash scripts/_make_worldcache_ladder.sh --estimate   # cost before GPU time\n"
        "bash scripts/_make_worldcache_ladder.sh 0            # ONLY=sv|mv|mvb for one pass\n"
        "python3 assets/scripts/stage_worldcache_ladder_web.py\n"
        "python3 assets/scripts/extract_ladder_frames.py      # frames/ + figures/\n"
        "```\n"
    )
    print(f"\nwrote {readme}")

    print("\ncontact sheets")
    contact_sheet(DEST / "humanoid_singleview_dreamgen", "episode_000829.mp4",
                  DEST / "contact_sheet_singleview.png")
    # full_pred, not full_merged: a merged frame is GT|pred side by side, which
    # halves each panel again and becomes unreadable in a contact sheet. Use
    # full_merged.mp4 in the folder itself for side-by-side viewing.
    contact_sheet(DEST / "humanoid_multiview_dreamgen",
                  f"{MV_TASKS[0]}/full_pred.mp4",
                  DEST / "contact_sheet_multiview.png")
    bright_ep = "Airbot_MMK2_doodled_line__episode_000001/full_pred.mp4"
    contact_sheet(DEST / MVB_CELL, bright_ep,
                  DEST / "contact_sheet_multiview_bright.png")

    total = sum(1 for _ in DEST.rglob("*.mp4"))
    print(f"\nDone: {total} videos under {DEST}")


if __name__ == "__main__":
    main()
