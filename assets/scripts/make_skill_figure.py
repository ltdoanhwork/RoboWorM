#!/usr/bin/env python3
"""Paper figure: N skills x M frames, one row per manipulation primitive.

Illustrates the measured skill taxonomy (`assets/scripts/make_benchmark_stats.py`) with
real GT frames, captioned with the verbatim dataset instruction and the source
embodiment, so the figure doubles as a diversity claim (different skills come
from different embodiments / source datasets).

    python3 assets/scripts/make_skill_figure.py
    python3 assets/scripts/make_skill_figure.py --cols 5 --tile-h 150

Writes `assets/figures/fig_skills.{png,pdf}`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
OUT = REPO / "assets" / "figures"

# (skill, unit under datasets/, episode_index, embodiment label, camera glob)
# chosen from the measured taxonomy — 4 different skills from 3 embodiments
ROWS = [
    ("pick & place",
     "single_arm/singleview/makovian", 0,
     "single-arm · RoboChallenge", None),
    ("press / actuate",
     "single_arm/singleview/non_makovian", 0,
     "single-arm · RoboChallenge", None),
    ("pour",
     "single_arm/multiview/makovian", 6672,
     "single-arm · DROID", "exterior_1_left"),
    ("fold / unfold",
     "bimanual/multiview/non_makovian/fold_yellow_towel", 0,
     "bimanual · AIST (ALOHA)", "cam_high"),
    ("insert / assemble",
     "bimanual/multiview/makovian/find_insert_large_gear_shaft", 22,
     "bimanual · AIST (ALOHA)", "cam_high"),
    ("handover",
     "bimanual/multiview/non_makovian/handover_and_wipe_sponge", 0,
     "bimanual · AIST (ALOHA)", "cam_high"),
    ("pick & place (humanoid)",
     "humanoid/singleview/makovian", 200,
     "humanoid · GR00T GR1", None),
]

WAIST = re.compile(r"^(un)?locked waist:\s*", re.I)
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"]


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in FONT_DIRS:
        p = Path(d) / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def meta(unit: Path, ep: int) -> dict:
    for line in (unit / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if int(o["episode_index"]) == ep:
            return o
    raise SystemExit(f"episode {ep} not in {unit}")


def resolve_ep(unit: Path, ep: int, cam: str | None) -> int:
    """Only a subset of episodes ship a video; fall back to the first that does."""
    have = sorted(int(re.search(r"episode_(\d+)", p.name).group(1))
                  for p in (unit / "videos").rglob("episode_*.mp4")
                  if cam is None or cam in str(p))
    if not have:
        raise SystemExit(f"no videos under {unit} (cam={cam})")
    if ep in have:
        return ep
    print(f"    [note] ep{ep:06d} has no video in {unit.name}; using "
          f"ep{have[0]:06d} instead")
    return have[0]


def video(unit: Path, ep: int, cam: str | None) -> Path:
    hits = sorted(p for p in (unit / "videos").rglob(f"episode_{ep:06d}.mp4")
                  if cam is None or cam in str(p))
    if not hits:
        raise SystemExit(f"no video: {unit} ep{ep} cam={cam}")
    return hits[0]


def grab(v: Path, idxs: list, tmp: Path) -> list:
    tmp.mkdir(parents=True, exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(v),
                    "-vf", f"select='{expr}'", "-vsync", "0",
                    str(tmp / "f_%03d.png")], check=True)
    return [Image.open(p).convert("RGB") for p in sorted(tmp.glob("f_*.png"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--tile-h", type=int, default=124)
    ap.add_argument("--rows", type=int, default=len(ROWS))
    args = ap.parse_args()

    rows = ROWS[:args.rows]
    built = []
    for skill, rel, ep, label, cam in rows:
        unit = DS / rel
        ep = resolve_ep(unit, ep, cam)
        o = meta(unit, ep)
        n = int(o["length"])
        instr = WAIST.sub("", (o.get("tasks") or [""])[0] or
                          (o.get("source_task") or "").replace("_", " ")).strip()
        idxs = [round(i * (n - 1) / (args.cols - 1)) for i in range(args.cols)]
        ims = grab(video(unit, ep, cam), idxs, Path("/tmp/skillfig") / skill[:12])
        ims = ims[:args.cols]
        built.append((skill, instr, label, n, ims))
        print(f"  {skill:24s} {rel.split('/')[0]:11s} ep{ep:06d} {n:4d}f  "
              f"{len(ims)} frames  \"{instr[:52]}\"")

    PAD, LEFT, TXT_W = 4, 8, 250
    tile_h = args.tile_h
    tw = max(2, round(built[0][4][0].width * tile_h / built[0][4][0].height))
    tw = min(tw, 190)
    W = LEFT + TXT_W + args.cols * (tw + PAD) + LEFT
    TITLE_H, ROW_GAP = 30, PAD
    H = TITLE_H + len(built) * (tile_h + ROW_GAP) + 8

    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)
    d.text((LEFT, 7), "Manipulation skills covered by the benchmark "
           "(ground-truth frames, verbatim instructions)",
           font=font(14, True), fill="#111111")

    palette = ["#1F6FB2", "#B2431F", "#2E7D4F", "#6A3D9A", "#A38200", "#0F6E75"]
    y = TITLE_H
    for i, (skill, instr, label, n, ims) in enumerate(built):
        col = palette[i % len(palette)]
        d.rectangle((0, y, 4, y + tile_h - 1), fill=col)
        d.text((LEFT + 4, y + 4), skill, font=font(13, True), fill=col)
        d.text((LEFT + 4, y + 22), label, font=font(10), fill="#888888")
        # wrap the instruction to the label column
        words, line, ly = instr.split(), "", y + 40
        for wd in words:
            t = (line + " " + wd).strip()
            if d.textlength(f'"{t}"', font=font(10)) > TXT_W - 16 and line:
                d.text((LEFT + 4, ly), line, font=font(10), fill="#333333")
                ly += 12
                line = wd
            else:
                line = t
        if line:
            d.text((LEFT + 4, ly), line, font=font(10), fill="#333333")

        for c, im in enumerate(ims):
            fig.paste(im.resize((tw, tile_h), Image.LANCZOS),
                      (LEFT + TXT_W + c * (tw + PAD), y))
        y += tile_h + ROW_GAP

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "fig_skills.png"
    fig.save(p)
    fig.save(OUT / "fig_skills.pdf", "PDF", resolution=300)
    print(f"\n[ok] {p.relative_to(REPO)}  {fig.size[0]}x{fig.size[1]}")


if __name__ == "__main__":
    main()
