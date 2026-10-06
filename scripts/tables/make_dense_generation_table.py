#!/usr/bin/env python3
"""Generation-config table for sampling_dataset/dense — measured, not estimated.

One column per dense cell (embodiment x view track x generator). Every video
property (resolution, fps, frames per clip, duration, #clips) is MEASURED with
ffprobe from the prediction videos actually on disk. Sampler settings
(guidance / steps / seed) come from the run config recorded next to the output
when it exists (`run_config.json`), otherwise from the canonical run-script
default, with the provenance string carried in the JSON output.

Usage:
    python3 scripts/tables/make_dense_generation_table.py                 # markdown to stdout
    python3 scripts/tables/make_dense_generation_table.py --json out.json --tex out.tex
    python3 scripts/tables/make_dense_generation_table.py --max-clips 40   # faster sampling
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DENSE = ROOT / "sampling_dataset" / "dense"

# Which cells the paper reports lives in ONE place (assets/scripts/benchmark_cells.py)
# so this table, the accel tables, the speedup CSV and the figures cannot drift
# apart. The Ctrl-World 3-view strip is excluded for bimanual/humanoid there
# (superseded by their 4-view grid); single_arm keeps its strip.
sys.path.insert(0, str(ROOT / "assets" / "scripts"))
from benchmark_cells import cell_is_excluded as is_excluded  # noqa: E402
from benchmark_cells import describe as describe_exclusions  # noqa: E402

# --- Static, auditable per-cell facts -----------------------------------------
# `pred` = glob (relative to <cell>/<split>/) of the video that IS the model
# prediction at generation resolution. NOTE for Ctrl-World: `full_pred.mp4` is
# GT-strip-over-pred-strip (see infer_bimanual_multiview_ctrlworld.py:441), so
# the real prediction canvas is `pred_all_views.mp4`.
CELLS = [
    dict(
        cell="single_arm/multiview/ctrlworld", short="DROID-MV",
        embodiment="single_arm", track="multiview", family="Ctrl-World",
        source="DROID (cadene/droid_1.0.1)",
        generator="Ctrl-World (SVD img2vid U-Net, action-cond.)",
        ckpt="Ctrl-World/checkpoint-10000.pt (released DROID pretrain)",
        guidance="1.0", steps="50", seed="unseeded",
        cfg_src="models/Ctrl-World/config.py:57-58; run_single_arm_multiview_ctrlworld.sh:23",
        views=3, layout="3x1 latent strip", saved="strip",
        pred="*/pred_all_views.mp4",
    ),
    dict(
        cell="bimanual/multiview/ctrlworld", short="ALOHA-MV",
        embodiment="bimanual", track="multiview", family="Ctrl-World",
        source="AIST Bimanual (ALOHA)",
        generator="Ctrl-World (SVD img2vid U-Net, action-cond.)",
        ckpt="bimanual_multiview_5fps/checkpoint-280000.pt",
        guidance="1.0", steps="50", seed="unseeded",
        cfg_src="models/Ctrl-World/config_bimanual.py:58-59",
        views=3, layout="3x1 latent strip", saved="strip",
        pred="*/pred_all_views.mp4",
    ),
    dict(
        cell="bimanual/multiview/ctrlworld_4view_grid", short="ALOHA-MV-grid",
        embodiment="bimanual", track="multiview", family="Ctrl-World",
        source="AIST Bimanual (ALOHA)",
        generator="Ctrl-World (4-view 2x2 latent canvas)",
        ckpt="bimanual_multiview_4view_grid/checkpoint-110000.pt",
        guidance="1.0", steps="50", seed="unseeded",
        cfg_src="models/Ctrl-World/config_bimanual_grid.py:67-68",
        views=4, layout="2x2 latent grid", saved="strip",
        pred="*/pred_all_views.mp4",
    ),
    dict(
        cell="humanoid/multiview/ctrlworld", short="RoboCOIN-MV",
        embodiment="humanoid", track="multiview", family="Ctrl-World",
        source="RoboCOIN (AIRBOT MMK2)",
        generator="Ctrl-World (SVD img2vid U-Net, action-cond.)",
        ckpt="humanoid_multiview/checkpoint-200000.pt",
        guidance="1.0", steps="50", seed="unseeded",
        cfg_src="models/Ctrl-World/config_humanoid.py:52-53",
        views=3, layout="3x1 latent strip", saved="strip",
        pred="*/pred_all_views.mp4",
    ),
    dict(
        cell="humanoid/multiview/ctrlworld_4view_grid", short="RoboCOIN-MV-grid",
        embodiment="humanoid", track="multiview", family="Ctrl-World",
        source="RoboCOIN (AIRBOT MMK2)",
        generator="Ctrl-World (4-view 2x2 latent canvas)",
        ckpt="humanoid_multiview_4view_grid/checkpoint-140000.pt",
        guidance="1.0", steps="50", seed="unseeded",
        cfg_src="models/Ctrl-World/config_humanoid_grid.py:64-65",
        views=4, layout="2x2 latent grid", saved="strip",
        pred="*/pred_all_views.mp4",
    ),
    dict(
        cell="single_arm/multiview/dreamgen", short="DROID-MV",
        embodiment="single_arm", track="multiview", family="DreamGen",
        source="DROID (cadene/droid_1.0.1)",
        generator="Cosmos-Predict2 14B (GR00T-Dreams-DROID)",
        ckpt="GR00T-dreams-DROID 14B (video2world_gr00t, --gr00t_variant droid)",
        guidance="7", steps="35", seed="0",
        cfg_src="examples/video2world_gr00t.py:145-146; pipelines/video2world.py:926",
        views=3, layout="2x2 grid (3 active + blank)", saved="grid2x2",
        pred="episode_*.mp4",
    ),
    dict(
        cell="bimanual/multiview/dreamgen", short="ALOHA-MV",
        embodiment="bimanual", track="multiview", family="DreamGen",
        source="AIST Bimanual (ALOHA)",
        generator="Cosmos-Predict2 2B (post-trained)",
        ckpt="bimanual_multiview_seg16/iter_000120000",
        guidance="7", steps="35", seed="0 + segment index",
        cfg_src="run_bimanual_multiview_dreamgen_full.sh:62,67; infer_bimanual_multiview_rollout.py:352",
        views=4, layout="2x2 grid", saved="grid2x2",
        pred="*/*/*/full_pred.mp4",
    ),
    dict(
        cell="humanoid/multiview/dreamgen", short="RoboCOIN-MV",
        embodiment="humanoid", track="multiview", family="DreamGen",
        source="RoboCOIN (AIRBOT MMK2)",
        generator="Cosmos-Predict2 2B (post-trained)",
        ckpt="humanoid_multiview_seg16/iter_000100000",
        guidance="7", steps="35", seed="0 + segment index",
        cfg_src="run_humanoid_multiview_dreamgen_full.sh:63,70; infer_bimanual_multiview_rollout.py:352",
        views=4, layout="2x2 grid", saved="grid2x2",
        pred="*/*/*/full_pred.mp4",
    ),
    dict(
        cell="single_arm/singleview/dreamgen", short="RoboChallenge-SV",
        embodiment="single_arm", track="singleview", family="DreamGen",
        source="RoboChallenge (Traly/RoboChallenge-lerobot)",
        generator="Cosmos-Predict2 2B (post-trained, static 16fps)",
        ckpt="single_arm_static_singleview_16fps_v2/iter_000090000",
        guidance="7", steps="35", seed="0",
        cfg_src="run_single_arm_singleview_dreamgen_static16fps.sh:60-62",
        views=1, layout="single view", saved="single",
        pred="*/global/episode_*.mp4",
    ),
    dict(
        cell="bimanual/singleview/dreamgen", short="ALOHA-SV",
        embodiment="bimanual", track="singleview", family="DreamGen",
        source="AIST Bimanual (ALOHA), cam_high",
        generator="Cosmos-Predict2 2B (post-trained, 16fps)",
        ckpt="bimanual_singleview_16fps/iter_000110000",
        guidance="7", steps="35", seed="0",
        cfg_src="run_bimanual_singleview_dreamgen_16fps.sh:42-44",
        views=1, layout="single view", saved="single",
        pred="*/*.mp4",
    ),
    dict(
        cell="humanoid/singleview/dreamgen", short="GR1-SV",
        embodiment="humanoid", track="singleview", family="DreamGen",
        source="GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1)",
        generator="Cosmos-Predict2 2B (post-trained, static 16fps)",
        ckpt="humanoid_singleview_16fps/iter_000090000",
        guidance="7", steps="35", seed="0",
        cfg_src="run_humanoid_singleview_dreamgen_static16fps.sh:51-53",
        views=1, layout="single view", saved="single",
        pred="*/episode_*.mp4",
    ),
    dict(
        cell="single_arm/singleview/dreamdojo", short="RoboChallenge-SV",
        embodiment="single_arm", track="singleview", family="DreamDojo",
        source="RoboChallenge (Traly/RoboChallenge-lerobot)",
        generator="DreamDojo 2B action-cond. DiT (autoregressive)",
        ckpt="dreamdojo_2b_480_640_single_arm_sv/iter_000030000",
        guidance="1.5", steps="35", seed="per-chunk index",
        cfg_src="run_config.json (sibling cells); infer_single_arm_singleview_autoregressive.py:355",
        views=1, layout="single view", saved="single",
        pred="*/episode_*/full_pred.mp4",
    ),
    dict(
        cell="bimanual/singleview/dreamdojo", short="ALOHA-SV",
        embodiment="bimanual", track="singleview", family="DreamDojo",
        source="AIST Bimanual (ALOHA), cam_high",
        generator="DreamDojo 2B action-cond. DiT (autoregressive)",
        ckpt="dreamdojo_2b_480_640_aloha_resume70k_bimanual_sv/iter_000100000",
        guidance="1.5", steps="35", seed="per-chunk index",
        cfg_src="run_bimanual_singleview_dreamdojo_full_episode.sh:21; infer_..._single_segment.py",
        views=1, layout="single view", saved="single",
        pred="*/*/full_pred.mp4",
    ),
    dict(
        cell="humanoid/singleview/dreamdojo", short="GR1-SV",
        embodiment="humanoid", track="singleview", family="DreamDojo",
        source="GR00T Teleop GR1 (nvidia/PhysicalAI-...-GR1)",
        generator="DreamDojo 2B action-cond. DiT (autoregressive)",
        ckpt="dreamdojo/2B_GR1_post-train/iter_000050000",
        guidance="1.5", steps="35", seed="per-chunk index",
        cfg_src="run_config.json (recorded next to output)",
        views=1, layout="single view", saved="single",
        pred="*/episode_*/full_pred.mp4",
    ),
]

SPLITS = ("makovian", "non_makovian")


def probe(path: Path) -> dict | None:
    """Measured stream properties of one video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets,duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        st = json.loads(out.stdout)["streams"][0]
    except Exception:
        return None
    num, _, den = st.get("avg_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    frames = int(st.get("nb_read_packets") or 0)
    return dict(
        w=int(st["width"]), h=int(st["height"]), fps=fps, frames=frames,
        duration=frames / fps if fps else 0.0,
    )


def cell_videos(cell: dict) -> list[Path]:
    base = DENSE / cell["cell"].split("/")[0] / "output" / "/".join(cell["cell"].split("/")[1:])
    vids: list[Path] = []
    for split in SPLITS:
        sd = base / split
        if sd.is_dir():
            vids += sorted(sd.glob(cell["pred"]))
    return vids


def mode(values):
    return statistics.mode(values) if values else None


def measure(cell: dict, max_clips: int, workers: int) -> dict:
    vids = cell_videos(cell)
    n_total = len(vids)
    sample = vids if (max_clips <= 0 or n_total <= max_clips) else vids[:: max(1, n_total // max_clips)][:max_clips]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(probe, sample))
    probes = [p for p in results if p and p["frames"]]
    unreadable = [str(v.relative_to(DENSE)) for v, p in zip(sample, results) if not (p and p["frames"])]
    if not probes:
        return dict(**cell, clips=n_total, measured=0, unreadable=unreadable)
    res = mode([(p["w"], p["h"]) for p in probes])
    fps = mode([round(p["fps"], 4) for p in probes])
    frames = sorted(p["frames"] for p in probes)
    durs = sorted(p["duration"] for p in probes)
    if cell["saved"] == "strip":          # N views concatenated horizontally
        per_view = (res[0] // cell["views"], res[1])
    elif cell["saved"] == "grid2x2":      # 2x2 canvas
        per_view = (res[0] // 2, res[1] // 2)
    else:
        per_view = res
    return dict(
        **cell,
        clips=n_total, measured=len(probes), unreadable=unreadable,
        canvas=f"{res[0]}x{res[1]}",
        per_view=f"{per_view[0]}x{per_view[1]}",
        fps=fps,
        frames_median=int(statistics.median(frames)),
        frames_min=frames[0], frames_max=frames[-1],
        dur_median=round(statistics.median(durs), 2),
        dur_min=round(durs[0], 2), dur_max=round(durs[-1], 2),
    )


ROWS = [
    ("Source data", lambda r: r["source"]),
    ("Generator", lambda r: r["generator"]),
    ("Checkpoint", lambda r: r["ckpt"]),
    ("Views / clip", lambda r: f"{r['views']} ({r['layout']})"),
    ("Resolution (per view)", lambda r: r["per_view"]),
    ("Resolution (canvas)", lambda r: r["canvas"]),
    ("FPS", lambda r: f"{r['fps']:g}"),
    ("Frames per clip", lambda r: f"{r['frames_median']} [{r['frames_min']}-{r['frames_max']}]"),
    ("Duration per clip", lambda r: f"{r['dur_median']:.2f} s [{r['dur_min']:.2f}-{r['dur_max']:.2f}]"),
    ("Guidance scale", lambda r: r["guidance"]),
    ("Sampling steps", lambda r: r["steps"]),
    ("Seed", lambda r: r["seed"]),
    ("Clips on disk", lambda r: str(r["clips"])),
]


def md_family_tables(rows: list[dict]) -> str:
    out = []
    for fam in ("Ctrl-World", "DreamGen", "DreamDojo"):
        sel = [r for r in rows if r["family"] == fam and r.get("measured")]
        if not sel:
            continue
        head = [f"{r['short']}<br>({r['embodiment']}/{r['track']})" for r in sel]
        out.append(f"### {fam}\n")
        out.append("| | " + " | ".join(head) + " |")
        out.append("|---|" + "---|" * len(sel))
        for label, fn in ROWS:
            out.append(f"| **{label}** | " + " | ".join(fn(r) for r in sel) + " |")
        out.append("")
    return "\n".join(out)


def md_flat_table(rows: list[dict]) -> str:
    cols = ["Cell", "Source data", "Generator", "Views", "Res (per view)", "Res (canvas)",
            "FPS", "Frames/clip", "Duration/clip", "Guid.", "Steps", "Seed", "Clips"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        if not r.get("measured"):
            out.append(f"| `{r['cell']}` | {r['source']} | {r['generator']} | " + " | ".join(["n/a"] * 9) + " | 0 |")
            continue
        out.append("| " + " | ".join([
            f"`{r['cell']}`", r["source"], r["generator"], f"{r['views']} ({r['layout']})",
            r["per_view"], r["canvas"], f"{r['fps']:g}",
            f"{r['frames_median']} [{r['frames_min']}–{r['frames_max']}]",
            f"{r['dur_median']:.2f} s", r["guidance"], r["steps"], r["seed"], str(r["clips"]),
        ]) + " |")
    return "\n".join(out)


def tex_family_tables(rows: list[dict]) -> str:
    def esc(s):
        return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    out = []
    for fam in ("Ctrl-World", "DreamGen", "DreamDojo"):
        sel = [r for r in rows if r["family"] == fam and r.get("measured")]
        if not sel:
            continue
        out.append(r"\begin{table}[t]\centering\small")
        out.append(rf"\caption{{Generation configuration of the \textsc{{{esc(fam)}}} cells of the dense benchmark. "
                   r"Video properties are measured with \texttt{ffprobe} over every prediction clip on disk; "
                   r"frames per clip are median [min--max] because clip length is matched to the ground-truth episode.}")
        out.append(rf"\label{{tab:dense-config-{fam.lower().replace('-', '')}}}")
        out.append(r"\begin{tabular}{l" + "c" * len(sel) + "}\\toprule")
        out.append(" & " + " & ".join(rf"\textbf{{{esc(r['short'])}}}" for r in sel) + r" \\")
        out.append(" & " + " & ".join(esc(f"{r['embodiment']}/{r['track']}") for r in sel) + r" \\\midrule")
        for label, fn in ROWS:
            out.append(esc(label) + " & " + " & ".join(esc(fn(r)) for r in sel) + r" \\")
        out.append(r"\bottomrule\end{tabular}\end{table}")
        out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-clips", type=int, default=0,
                    help="Probe at most N clips per cell (0 = all, evenly sampled otherwise).")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--tex", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()

    dropped = [c["cell"] for c in CELLS if is_excluded(c["cell"])]
    if dropped:
        # Printed, not silent: a cell disappearing from the paper should be
        # visible in the log of the script that dropped it.
        print("benchmark_cells: %s" % describe_exclusions())
        for c in dropped:
            print("  dropped %s" % c)
    rows = [measure(c, args.max_clips, args.workers) for c in CELLS
            if not is_excluded(c["cell"])]
    bad = [u for r in rows for u in r.get("unreadable", [])]

    md = ["# Dense generation configuration (measured)", "",
          "Every video property below is measured with `ffprobe` from the prediction",
          "clips in `sampling_dataset/dense/`; sampler settings come from the run",
          "config / canonical run script (`cfg_src` in the JSON). Regenerate with:", "",
          "```bash",
          "python3 scripts/tables/make_dense_generation_table.py \\",
          "  --md docs/dense_generation_config.md \\",
          "  --json metrics/reports/dense_generation_config.json \\",
          "  --tex metrics/reports/dense_generation_config.tex",
          "```", "",
          f"Cells: {len(rows)} | clips probed: {sum(r.get('measured', 0) for r in rows)} "
          f"/ {sum(r['clips'] for r in rows)} on disk"
          + (f" | **{len(bad)} unreadable**: " + ", ".join(f"`{b}`" for b in bad) if bad else ""), "",
          "Frames/duration per clip are **median [min–max]**, not a constant: every clip",
          "is generated to match its ground-truth episode length, so clip length varies",
          "within a cell.", "",
          "## Per generator family (paper layout)", "", md_family_tables(rows),
          "## All cells (flat)", "", md_flat_table(rows), ""]
    text = "\n".join(md)
    print(text)
    if args.md:
        args.md.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    if args.tex:
        args.tex.write_text(tex_family_tables(rows))


if __name__ == "__main__":
    main()
