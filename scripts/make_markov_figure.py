#!/usr/bin/env python3
"""Paper figure contrasting a Markovian vs a non-Markovian task.

Two rows of real GT frames from `single_arm/singleview` (same embodiment, same
camera, so the ONLY thing that differs is the task structure), captioned with
the verbatim dataset instruction from `datasets/.../meta/tasks.jsonl`.

    python3 scripts/make_markov_figure.py
    python3 scripts/make_markov_figure.py --non-markov tissue --cols 6

Writes `assets/figures/fig_markov_vs_nonmarkov.{png,pdf}`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets" / "single_arm" / "singleview"
OUT = REPO / "assets" / "figures"

# (split, verbatim instruction, episode_index, sub-step captions for the strip)
MARKOV = ("makovian", "place the cup on the coaster", 0, None)
NONMARKOV = {
    "buttons": ("non_makovian",
                "press the pink, blue, and green buttons in sequence", 0,
                ["press pink", "press blue", "press green"]),
    "tissue": ("non_makovian",
               "pull out a tissue, wipe the stains on the table clean, "
               "and then discard the tissue into the trash bin", 60,
               ["pull out tissue", "wipe the table", "discard in bin"]),
}

FONTS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-%s.ttf",
         "/usr/share/fonts/dejavu/DejaVuSans-%s.ttf"]


def font(size: int, bold: bool = False):
    for pat in FONTS:
        p = Path(pat % ("Bold" if bold else "Book" if "dejavu/D" in pat else "Bold"))
        if p.exists():
            return ImageFont.truetype(str(p), size)
    for pat in FONTS:                      # regular fallback
        p = Path(pat.replace("-%s", ""))
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def episode_len(split: str, ep: int) -> int:
    f = DS / split / "meta" / "episodes.jsonl"
    for line in f.read_text().splitlines():
        o = json.loads(line)
        if int(o["episode_index"]) == ep:
            return int(o["length"])
    raise SystemExit(f"episode {ep} not in {f}")


def video_path(split: str, ep: int) -> Path:
    root = DS / split / "videos"
    hits = sorted(root.rglob(f"episode_{ep:06d}.mp4"))
    if not hits:
        raise SystemExit(f"no video for {split} episode {ep}")
    return hits[0]


def grab(video: Path, idxs: list, tmp: Path) -> list:
    import subprocess
    tmp.mkdir(parents=True, exist_ok=True)
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i",
                    str(video), "-vf", f"select='{expr}'", "-vsync", "0",
                    str(tmp / "f_%03d.png")], check=True)
    return [Image.open(p).convert("RGB")
            for p in sorted(tmp.glob("f_*.png"))][:len(idxs)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--non-markov", choices=sorted(NONMARKOV), default="buttons")
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile-h", type=int, default=168)
    args = ap.parse_args()

    rows = [("Markovian", *MARKOV), ("Non-Markovian", *NONMARKOV[args.non_markov])]

    PAD, LEFT = 5, 14
    TITLE_H, CAP_H, TS_H, ROW_GAP = 26, 40, 15, 18
    BR_H = 20                      # bracket band, only for non-Markovian rows
    tile_h = args.tile_h

    built = []
    for kind, split, instr, ep, steps in rows:
        n = episode_len(split, ep)
        idxs = [round(i * (n - 1) / (args.cols - 1)) for i in range(args.cols)]
        ims = grab(video_path(split, ep), idxs, Path(f"/tmp/mkfig/{kind}"))
        tw = max(2, round(ims[0].width * tile_h / ims[0].height))
        built.append((kind, instr, ep, n, idxs, [im.resize((tw, tile_h),
                     Image.LANCZOS) for im in ims], steps, tw))

    tw = min(b[7] for b in built)
    W = LEFT + args.cols * (tw + PAD) - PAD + LEFT
    heights = [CAP_H + tile_h + TS_H + (BR_H if b[6] else 0) for b in built]
    H = TITLE_H + sum(heights) + (len(built) - 1) * ROW_GAP + 10
    fig = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(fig)

    d.text((LEFT, 6), "Markovian vs. non-Markovian tasks "
           "(single-arm, global view, ground truth)", font=font(15, True),
           fill="#111111")

    y = TITLE_H
    for (kind, instr, ep, n, idxs, ims, steps, _), row_h in zip(built, heights):
        colour = "#1F6FB2" if kind == "Markovian" else "#B2431F"
        d.rectangle((0, y, 4, y + row_h - 6), fill=colour)
        d.text((LEFT, y + 2), kind, font=font(14, True), fill=colour)
        d.text((LEFT, y + 20), f'"{instr}"', font=font(13), fill="#111111")
        meta = f"episode {ep:06d} · {n} frames"
        d.text((W - LEFT - d.textlength(meta, font=font(11)), y + 4),
               meta, font=font(11), fill="#888888")

        ty = y + CAP_H
        for c, im in enumerate(ims):
            x = LEFT + c * (tw + PAD)
            fig.paste(im.resize((tw, tile_h), Image.LANCZOS), (x, ty))
            lab = f"t={idxs[c]}"
            d.text((x + tw / 2 - d.textlength(lab, font=font(10)) / 2,
                    ty + tile_h + 2), lab, font=font(10), fill="#999999")
        if steps:  # bracket the sub-goals on their own band under the t= labels
            span = (args.cols * (tw + PAD) - PAD) / len(steps)
            by = ty + tile_h + TS_H + BR_H - 6
            for k, s in enumerate(steps):
                x0, x1 = LEFT + k * span, LEFT + (k + 1) * span - PAD
                d.line((x0, by, x1, by), fill=colour, width=2)
                d.line((x0, by - 4, x0, by), fill=colour, width=2)
                d.line((x1, by - 4, x1, by), fill=colour, width=2)
                d.text(((x0 + x1) / 2 - d.textlength(s, font=font(11, True)) / 2,
                        by - 17), s, font=font(11, True), fill=colour)
        y += row_h + ROW_GAP

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "fig_markov_vs_nonmarkov.png"
    fig.save(png)
    fig.save(OUT / "fig_markov_vs_nonmarkov.pdf", "PDF", resolution=300)
    print(f"[ok] {png.relative_to(REPO)}  {fig.size[0]}x{fig.size[1]}")
    for kind, instr, ep, n, *_ in built:
        print(f"     {kind:14s} ep{ep:06d}  {n:4d} frames  \"{instr}\"")


if __name__ == "__main__":
    main()
