#!/usr/bin/env python3
"""Build the paper figures that illustrate the three embodiments of the benchmark.

Outputs (assets/figures/):
  embodiments/<embodiment>_<track>_<view>.png   INDIVIDUAL bare images (default)
  embodiments/manifest.json                     what each file is (camera, task, source)
  fig_embodiments.png / .pdf                    composite 3 rows x (single|multi view)
  fig_embodiments_teaser.png / .pdf             composite 1x3 teaser

Frames are extracted once with ffmpeg into assets/figures/frames/ and cached, so
re-running only recomposes the output (fast iteration on styling).

Usage:
  python3 assets/scripts/make_fig_embodiments.py --only singles              # just the loose images
  python3 assets/scripts/make_fig_embodiments.py --only singles --crop 4:3   # uniform aspect
  python3 assets/scripts/make_fig_embodiments.py --only singles --height 480 --pdf
  python3 assets/scripts/make_fig_embodiments.py --only composites           # just the big figures
  python3 assets/scripts/make_fig_embodiments.py --force-frames              # re-extract frames

To change which episode / camera / frame represents an embodiment, edit build_rows().
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
OUT_DIR = REPO / "assets" / "figures"
FRAME_DIR = OUT_DIR / "frames"

# --------------------------------------------------------------------------------------
# Content spec: which episode / camera / frame represents each embodiment
# --------------------------------------------------------------------------------------


@dataclass
class Panel:
    key: str  # filename stem for the cached frame
    video: Path  # source mp4
    frame: int  # frame index to grab
    caption: str  # label drawn under the panel


@dataclass
class Row:
    name: str  # embodiment display name
    accent: str  # accent colour (hex)
    sv_robot: str  # robot / source dataset of the single-view stream
    mv_robot: str  # robot / source dataset of the multi-view stream
    sv_meta: str  # resolution / fps / #episodes of single-view stream
    mv_meta: str  # resolution / fps / #episodes of multi-view stream
    sv_task: str  # instruction of the shown single-view episode
    mv_task: str  # instruction of the shown multi-view episode
    dof: str  # action space summary
    sv: Panel = None  # single-view panel
    mv: list = field(default_factory=list)  # multi-view panels
    hero: Panel = None  # frame used by the compact teaser figure
    hero_task: str = ""  # instruction of the hero episode


def _sa_mv(view: str) -> Path:
    return (
        DS
        / "single_arm/multiview/makovian/videos/chunk-000"
        / f"observation.images.{view}/episode_000903.mp4"
    )


def _bi_mv(view: str) -> Path:
    return (
        DS
        / "bimanual/multiview/makovian/close_toolbox/videos/chunk-000"
        / f"observation.images.{view}/episode_000002.mp4"
    )


def _hu_mv(view: str) -> Path:
    return (
        DS
        / "humanoid/multiview/makovian/Airbot_MMK2_move_pan/videos/chunk-000"
        / f"observation.images.{view}/episode_000000.mp4"
    )


def build_rows() -> list:
    sa = Row(
        name="Single-Arm",
        accent="#1F6FB2",
        sv_robot="RoboChallenge",
        mv_robot="DROID (Franka)",
        sv_meta="480x640 | 15 fps | 240 eps",
        mv_meta="180x320 | 15 fps | 260 eps",
        sv_task="place the cup on the coaster",
        mv_task="turn on the third switch from the left",
        dof="7-DoF arm + gripper",
        sv=Panel(
            "sa_sv_global",
            DS
            / "single_arm/singleview/makovian/videos/chunk-000"
            / "observation.images.global/episode_000001.mp4",
            80,
            "global static",
        ),
        mv=[
            Panel("sa_mv_ext1", _sa_mv("exterior_1_left"), 80, "exterior 1"),
            Panel("sa_mv_ext2", _sa_mv("exterior_2_left"), 80, "exterior 2"),
            Panel("sa_mv_wrist", _sa_mv("wrist_left"), 40, "wrist"),
        ],
    )
    bi = Row(
        name="Bimanual",
        accent="#B24A1F",
        sv_robot="ALOHA (2x ViperX)",
        mv_robot="ALOHA (2x ViperX)",
        sv_meta="480x640 | 30 fps | 188 eps",
        mv_meta="480x640 | 30 fps | 188 eps",
        sv_task="close the toolbox",
        mv_task="close the toolbox",
        dof="2 x (6-DoF arm + gripper)",
        sv=Panel("bi_sv_high", _bi_mv("cam_high"), 140, "overhead (cam_high)"),
        mv=[
            Panel("bi_mv_high", _bi_mv("cam_high"), 140, "overhead"),
            Panel("bi_mv_low", _bi_mv("cam_low"), 60, "low front"),
            Panel("bi_mv_lw", _bi_mv("cam_left_wrist"), 200, "left wrist"),
            Panel("bi_mv_rw", _bi_mv("cam_right_wrist"), 200, "right wrist"),
        ],
    )
    hu = Row(
        name="Humanoid",
        accent="#2E7D4F",
        sv_robot="Fourier GR1-T2",
        mv_robot="AIRBOT MMK2",
        sv_meta="800x1280 | 20 fps | 260 eps",
        mv_meta="480x640 | 30 fps | 149 eps",
        sv_task="pick up green bokchoy into wire basket",
        mv_task="move the pan on the table",
        dof="dual arm + waist / mobile base",
        sv=Panel(
            "hu_sv_ego",
            DS
            / "humanoid/singleview/makovian/videos/chunk-000"
            / "observation.images.ego_view_freq20/episode_000829.mp4",
            30,
            "ego view",
        ),
        mv=[
            Panel("hu_mv_head", _hu_mv("cam_head_rgb"), 30, "head"),
            Panel("hu_mv_front", _hu_mv("cam_front_rgb"), 30, "front"),
            Panel("hu_mv_lw", _hu_mv("cam_left_wrist_rgb"), 30, "left wrist"),
            Panel("hu_mv_rw", _hu_mv("cam_right_wrist_rgb"), 30, "right wrist"),
        ],
    )
    sa.hero = Panel("hero_sa", _sa_mv("exterior_2_left"), 80, "")
    sa.hero_task = sa.mv_task
    bi.hero = Panel("hero_bi", _bi_mv("cam_high"), 230, "")
    bi.hero_task = bi.mv_task
    hu.hero = Panel("hero_hu", _hu_mv("cam_head_rgb"), 90, "")
    hu.hero_task = hu.mv_task
    return [sa, bi, hu]


# --------------------------------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------------------------------


def extract(panel: Panel, force: bool = False) -> Path:
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    dst = FRAME_DIR / f"{panel.key}.png"
    if dst.exists() and not force:
        return dst
    if not panel.video.exists():
        raise FileNotFoundError(panel.video)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(panel.video),
            "-vf", f"select=eq(n\\,{panel.frame})",
            "-vsync", "0", "-frames:v", "1",
            str(dst),
        ],
        check=True,
    )
    return dst


# --------------------------------------------------------------------------------------
# Drawing helpers
# --------------------------------------------------------------------------------------

FONT_DIRS = [
    Path("/usr/share/fonts/truetype/dejavu"),
    Path.home() / ".local/share/fonts",
    Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/share/fonts"),
]


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        names = ["DejaVuSans-BoldOblique.ttf", "DejaVuSans-Bold.ttf"]
    elif italic:
        names = ["DejaVuSans-Oblique.ttf", "DejaVuSans.ttf"]
    else:
        names = ["DejaVuSans.ttf"]
    if bold and not italic:
        names = ["DejaVuSans-Bold.ttf"]
    for n in names:
        for dirp in FONT_DIRS:
            p = dirp / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def fit_crop(path: Path, w: int, h: int) -> Image.Image:
    """Center-crop to the target aspect ratio, then resize (no distortion)."""
    im = Image.open(path).convert("RGB")
    tgt, src = w / h, im.width / im.height
    if src > tgt:  # too wide -> crop sides
        nw = int(round(im.height * tgt))
        x0 = (im.width - nw) // 2
        im = im.crop((x0, 0, x0 + nw, im.height))
    elif src < tgt:  # too tall -> crop top/bottom
        nh = int(round(im.width / tgt))
        y0 = (im.height - nh) // 2
        im = im.crop((0, y0, im.width, y0 + nh))
    return im.resize((w, h), Image.LANCZOS)


def paste_panel(canvas, path, x, y, w, h, border, bw=3, radius=8):
    """Rounded-corner image tile with a coloured border."""
    im = fit_crop(path, w, h)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    canvas.paste(im, (x, y), mask)
    ImageDraw.Draw(canvas).rounded_rectangle(
        (x, y, x + w - 1, y + h - 1), radius=radius, outline=border, width=bw
    )


def text_center(draw, cx, y, s, f, fill="#1a1a1a"):
    w = draw.textlength(s, font=f)
    draw.text((cx - w / 2, y), s, font=f, fill=fill)


def wrap(draw, s: str, f, max_w: float) -> list:
    """Greedy word wrap to a pixel width."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if draw.textlength(cand, font=f) <= max_w or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


# --------------------------------------------------------------------------------------
# Figure 1: full embodiment / camera overview
# --------------------------------------------------------------------------------------


def fig_overview(rows: list, out_stem: Path, scale: int = 2):
    S = scale
    PW, PH = 200 * S, 150 * S          # panel size (4:3)
    GAP = 7 * S                        # gap between panels
    CAP = 15 * S                       # caption strip under a panel
    TASK = 15 * S                      # task line under the captions
    LABEL_W = 200 * S                  # left label column
    GROUP_GAP = 26 * S                 # gap between the two tracks
    MV_SLOTS = 4                       # multi-view area is 4 panels wide
    MARGIN = 16 * S
    HDR = 32 * S                       # top group-header band
    ROW_H = PH + CAP + TASK + 16 * S

    mv_area = MV_SLOTS * PW + (MV_SLOTS - 1) * GAP
    W = MARGIN * 2 + LABEL_W + PW + GROUP_GAP + mv_area
    H = MARGIN * 2 + HDR + len(rows) * ROW_H

    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)

    f_row = font(18 * S, bold=True)
    f_sub = font(10 * S, bold=True)
    f_meta = font(9 * S)
    f_cap = font(9 * S)
    f_task = font(9 * S, italic=True)
    f_hdr = font(12 * S, bold=True)

    sv_x = MARGIN + LABEL_W
    mv_x = sv_x + PW + GROUP_GAP

    # ---- group headers -------------------------------------------------------------
    y = MARGIN
    d.rounded_rectangle((sv_x, y, sv_x + PW, y + 21 * S), radius=5 * S, fill="#E7EFF7")
    text_center(d, sv_x + PW / 2, y + 4 * S, "Single-view track", f_hdr, "#1F4E79")
    d.rounded_rectangle(
        (mv_x, y, mv_x + mv_area, y + 21 * S), radius=5 * S, fill="#F6EFE8"
    )
    text_center(d, mv_x + mv_area / 2, y + 4 * S, "Multi-view track", f_hdr, "#7A4A20")

    # ---- rows ----------------------------------------------------------------------
    y = MARGIN + HDR
    for ri, r in enumerate(rows):
        if ri:  # thin separator between embodiments
            d.line((MARGIN, y - 8 * S, W - MARGIN, y - 8 * S), fill="#E4E4E4", width=S)

        # ---- left label block
        d.rounded_rectangle(
            (MARGIN, y + 4 * S, MARGIN + 4 * S, y + PH - 4 * S),
            radius=2 * S,
            fill=r.accent,
        )
        tx = MARGIN + 13 * S
        tw = LABEL_W - 20 * S
        ty = y + 6 * S
        d.text((tx, ty), r.name, font=f_row, fill="#111111")
        ty += 25 * S
        for lbl, robot, meta in (
            ("SV", r.sv_robot, r.sv_meta),
            ("MV", r.mv_robot, r.mv_meta),
        ):
            d.text((tx, ty), f"{lbl}  {robot}", font=f_sub, fill=r.accent)
            ty += 12 * S
            d.text((tx + 17 * S, ty), meta, font=f_meta, fill="#666666")
            ty += 17 * S
        ty += 3 * S
        for line in wrap(d, r.dof, f_meta, tw):
            d.text((tx, ty), line, font=f_meta, fill="#8A8A8A")
            ty += 12 * S

        # ---- single-view panel
        paste_panel(canvas, extract(r.sv), sv_x, y, PW, PH, r.accent, bw=3 * S)
        text_center(d, sv_x + PW / 2, y + PH + 4 * S, r.sv.caption, f_cap, "#333333")
        text_center(
            d, sv_x + PW / 2, y + PH + CAP + 3 * S, f"\u201c{r.sv_task}\u201d",
            f_task, "#8A8A8A",
        )

        # ---- multi-view panels (centered inside the 4-slot area if fewer)
        n = len(r.mv)
        used = n * PW + (n - 1) * GAP
        x = mv_x + (mv_area - used) / 2
        for p in r.mv:
            paste_panel(canvas, extract(p), int(x), y, PW, PH, "#BFBFBF", bw=2 * S)
            text_center(d, x + PW / 2, y + PH + 4 * S, p.caption, f_cap, "#333333")
            x += PW + GAP
        text_center(
            d, mv_x + mv_area / 2, y + PH + CAP + 3 * S, f"\u201c{r.mv_task}\u201d",
            f_task, "#8A8A8A",
        )

        y += ROW_H

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(out_stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(out_stem.with_suffix(".pdf"), "PDF", resolution=300)
    print(f"[ok] {out_stem.with_suffix('.png')}  ({canvas.width}x{canvas.height})")
    print(f"[ok] {out_stem.with_suffix('.pdf')}")


# --------------------------------------------------------------------------------------
# Figure 2: compact teaser (one hero frame per embodiment)
# --------------------------------------------------------------------------------------


def fig_teaser(rows: list, out_stem: Path, scale: int = 2):
    S = scale
    PW, PH = 264 * S, 198 * S
    GAP = 12 * S
    MARGIN = 12 * S
    TITLE = 46 * S

    W = MARGIN * 2 + 3 * PW + 2 * GAP
    H = MARGIN * 2 + PH + TITLE
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    f_name = font(16 * S, bold=True)
    f_meta = font(10 * S)
    f_task = font(9 * S, italic=True)

    # hero frame per embodiment: the view that best shows the whole robot
    x = MARGIN
    for r in rows:
        hero, task = r.hero, r.hero_task
        paste_panel(canvas, extract(hero), x, MARGIN, PW, PH, r.accent, bw=3 * S)
        yy = MARGIN + PH + 6 * S
        text_center(d, x + PW / 2, yy, r.name, f_name, "#111111")
        yy += 20 * S
        robots = r.sv_robot if r.sv_robot == r.mv_robot else f"{r.sv_robot} / {r.mv_robot}"
        text_center(d, x + PW / 2, yy, robots, f_meta, "#5A5A5A")
        yy += 14 * S
        text_center(d, x + PW / 2, yy, f"\u201c{task}\u201d", f_task, "#9A9A9A")
        x += PW + GAP

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(out_stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(out_stem.with_suffix(".pdf"), "PDF", resolution=300)
    print(f"[ok] {out_stem.with_suffix('.png')}  ({canvas.width}x{canvas.height})")


# --------------------------------------------------------------------------------------
# Individual images (no border, no label) — one clean file per camera view
# --------------------------------------------------------------------------------------

SINGLE_DIR = OUT_DIR / "embodiments"

ASPECTS = {"none": None, "4:3": 4 / 3, "16:9": 16 / 9, "1:1": 1.0, "3:2": 3 / 2}


def _slug(s: str) -> str:
    return s.lower().replace("-", "_").replace(" ", "_")


def export_singles(rows: list, crop: str = "none", height: int = 0, pdf: bool = False):
    """Write one bare image per camera view, named <embodiment>_<track>_<view>.png.

    No borders, captions or padding are drawn — these are meant to be placed
    individually with \\includegraphics. `crop` optionally normalises the aspect
    ratio so a row of subfigures lines up; `height` optionally downscales.
    """
    if crop not in ASPECTS:
        raise SystemExit(f"--crop must be one of {sorted(ASPECTS)}")
    ratio = ASPECTS[crop]

    SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for r in rows:
        emb = _slug(r.name)
        items = [("sv", r.sv, r.sv_task), ("hero", r.hero, r.hero_task)]
        items += [("mv", p, r.mv_task) for p in r.mv]
        for track, panel, task in items:
            src = extract(panel)
            im = Image.open(src).convert("RGB")
            if ratio is not None:
                w = im.width
                h = int(round(w / ratio))
                if h > im.height:  # source not tall enough -> derive from height
                    h = im.height
                    w = int(round(h * ratio))
                im = fit_crop(src, w, h)
            if height and im.height != height:
                im = im.resize(
                    (int(round(im.width * height / im.height)), height), Image.LANCZOS
                )

            view = panel.key.split("_", 2)[-1] if track != "hero" else "hero"
            name = f"{emb}_{track}_{view}" if track != "hero" else f"{emb}_hero"
            dst = SINGLE_DIR / f"{name}.png"
            im.save(dst, dpi=(300, 300))
            if pdf:
                im.save(dst.with_suffix(".pdf"), "PDF", resolution=300)
            manifest.append(
                {
                    "file": dst.name,
                    "embodiment": r.name,
                    "track": {"sv": "singleview", "mv": "multiview", "hero": "hero"}[track],
                    "camera": panel.caption or view,
                    "size": f"{im.width}x{im.height}",
                    "task": task,
                    "source_video": str(panel.video.relative_to(REPO)),
                    "frame": panel.frame,
                }
            )
            print(f"[ok] {dst.relative_to(REPO)}  ({im.width}x{im.height})  {panel.caption}")

    (SINGLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[ok] {(SINGLE_DIR / 'manifest.json').relative_to(REPO)}  ({len(manifest)} images)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-frames", action="store_true")
    ap.add_argument("--scale", type=int, default=2, help="pixel scale (2 ~ 300dpi print)")
    ap.add_argument(
        "--only",
        choices=["all", "singles", "composites"],
        default="all",
        help="what to build: individual images, the composite figures, or both",
    )
    ap.add_argument(
        "--crop",
        default="none",
        help='aspect for the individual images: "none" (native frame), "4:3", "16:9", "1:1"',
    )
    ap.add_argument(
        "--height",
        type=int,
        default=0,
        help="resize individual images to this height (0 = keep native)",
    )
    ap.add_argument("--pdf", action="store_true", help="also write .pdf for each single image")
    args = ap.parse_args()

    rows = build_rows()
    for r in rows:
        for p in [r.sv, r.hero] + r.mv:
            extract(p, force=args.force_frames)

    if args.only in ("all", "singles"):
        export_singles(rows, crop=args.crop, height=args.height, pdf=args.pdf)
    if args.only in ("all", "composites"):
        fig_overview(rows, OUT_DIR / "fig_embodiments", scale=args.scale)
        fig_teaser(rows, OUT_DIR / "fig_embodiments_teaser", scale=args.scale)


if __name__ == "__main__":
    main()
