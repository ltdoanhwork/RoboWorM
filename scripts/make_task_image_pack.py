#!/usr/bin/env python3
"""Task-keyed image pack: ONE folder per task, timeline frames as loose PNGs.

Third view of the same data, for assembling qualitative figures by hand:

  * `real_videos/`   -> grouped by (embodiment x track x split), ships mp4s
  * `figure_drawio/` -> grouped by camera slot, ONE instant per embodiment
  * `figure_tasks/`  -> grouped by TASK, a temporal sequence per task  <-- this

Each task dir holds `f1..fN_<frame>.png` (uniformly spaced through the episode,
so dragging them left-to-right in draw.io reads as the task unfolding), a
`strip.png` preview, `prompt.txt` (verbatim, copy-paste into the figure) and
`info.json`. Tasks are chosen to cover every measured skill in the taxonomy
(`scripts/make_benchmark_stats.py`) across all 3 embodiments and both
markovian / non_makovian splits.

    python3 scripts/make_task_image_pack.py --list     # inspect selection
    python3 scripts/make_task_image_pack.py            # write the pack
    python3 scripts/make_task_image_pack.py --frames 6 --width 720

Writes `assets/drive/figure_tasks/<NN>_<skill>__<embodiment>_<split>/`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_benchmark_stats as S  # noqa: E402
from make_teaser_tiles import meta, prompt_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets"
OUT = REPO / "assets" / "drive" / "figure_tasks"

# Camera to draw the timeline from, per track. First match wins; multiview units
# get their most legible overview camera (wrist views are useless as a timeline).
PREF_CAM = ["global", "ego_view_freq20", "cam_high", "cam_head_rgb",
            "cam_front_rgb", "exterior_1_left", "cam_high_rgb", "cam_third_view"]

# (skill, unit under datasets/, episode) — every entry verified present on disk.
# Optional 4th element forces a camera when the default overview view hides the
# actual manipulation (e.g. the DROID arm occludes the tabletop).
# Ordered so the pack reads skill-by-skill; `NN_` prefix keeps that order in a
# file picker. Covers all 14 measured primitives.
TASKS = [
    ("pick & place", "single_arm/singleview/makovian", 0),
    ("pick & place", "humanoid/singleview/makovian", 12318),
    ("pick & place", "bimanual/singleview/makovian/put_cup_moved_red_handle_holder", 4),
    ("push / slide", "single_arm/multiview/makovian", 397),
    ("push / slide", "humanoid/singleview/non_makovian", 527),
    ("insert / assemble", "bimanual/singleview/makovian/find_hole_and_insert_into_gear", 29),
    ("press / actuate", "single_arm/singleview/makovian", 60),
    ("press / actuate", "single_arm/singleview/non_makovian", 0),
    ("press / actuate", "bimanual/singleview/makovian/hit_mark_with_hammer", 36),
    ("wipe / clean", "bimanual/singleview/makovian/wipe_the_desk_with_a_blue_cloth", 19),
    ("wipe / clean", "single_arm/singleview/non_makovian", 60),
    ("pour", "single_arm/multiview/makovian", 6672),
    ("pour", "bimanual/singleview/makovian/spoon_red_block_bowl", 1),
    ("handover", "bimanual/singleview/non_makovian/handover_and_wipe_sponge", 0),
    ("fold / unfold", "bimanual/singleview/non_makovian/fold_bath_towel", 10),
    ("fold / unfold", "single_arm/multiview/non_makovian", 1602),
    ("remove / extract", "single_arm/multiview/non_makovian", 582),
    ("open / close", "single_arm/multiview/non_makovian", 5380),
    ("stack", "single_arm/multiview/non_makovian", 827),
    ("sort", "single_arm/multiview/non_makovian", 1688),
    ("draw / trace", "single_arm/multiview/non_makovian", 14890, "wrist_left"),
    ("hang / cover", "single_arm/multiview/non_makovian", 11705),
    ("pick & place", "humanoid/multiview/non_makovian/"
     "Airbot_MMK2_storage_bottle_part", 3),
]

SRC = {("single_arm", "singleview"): "RoboChallenge",
       ("single_arm", "multiview"): "DROID",
       ("bimanual", "singleview"): "AIST Bimanual (ALOHA)",
       ("bimanual", "multiview"): "AIST Bimanual (ALOHA)",
       ("humanoid", "singleview"): "GR00T-Teleop-GR1",
       ("humanoid", "multiview"): "RoboCOIN (AIRBOT MMK2)"}


def slug(s: str, keep: str = "-_") -> str:
    out = "".join(c if (c.isalnum() or c in keep) else "_" for c in s.lower())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def pick_camera(unit: Path) -> str:
    cams = [c.split(".")[-1] for c in S.views(unit)]
    for p in PREF_CAM:
        if p in cams:
            return p
    if not cams:
        raise SystemExit(f"no video features in {unit}")
    return cams[0]


def video_of(unit: Path, ep: int, cam: str) -> Path:
    hits = [p for p in (unit / "videos").rglob(f"episode_{ep:06d}.mp4")
            if f".{cam}/" in str(p) or f"/{cam}/" in str(p)]
    if not hits:
        raise SystemExit(f"missing {cam} ep{ep} in {unit}")
    return hits[0]


def grab(video: Path, idx: int, dst: Path, width: int) -> Image.Image:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".raw.png")
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"select='eq(n\\,{idx})'", "-vsync", "0",
                    "-frames:v", "1", str(tmp)], check=True)
    im = Image.open(tmp).convert("RGB")
    im = im.resize((width, round(width * im.height / im.width)), Image.LANCZOS)
    im.save(dst)
    tmp.unlink()
    return im


def font(sz: int, bold: bool = False):
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{n}", sz)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5, help="timeline frames per task")
    ap.add_argument("--width", type=int, default=640, help="frame width px")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-extract existing frames")
    args = ap.parse_args()

    rows = []
    for i, spec in enumerate(TASKS, 1):
        skill, rel, ep = spec[0], spec[1], spec[2]
        force_cam = spec[3] if len(spec) > 3 else None
        unit = DS / rel
        parts = rel.split("/")
        emb, track, split = parts[0], parts[1], parts[2]
        task_key = parts[3] if len(parts) > 3 else None
        o = meta(unit, ep)
        n = int(o["length"])
        rows.append(dict(
            n=i, skill=skill, unit=rel, embodiment=emb, track=track, split=split,
            task_key=task_key, episode_index=ep, episode_length=n,
            prompt=prompt_of(o), camera=force_cam or pick_camera(unit),
            source=SRC[(emb, track)],
            dir=f"{i:02d}_{slug(skill)}__{emb}_{split}"))

    if args.list:
        for r in rows:
            print(f"{r['n']:02d} {r['skill']:<18} {r['embodiment']:<10} "
                  f"{r['split']:<12} {r['camera']:<16} ep{r['episode_index']:<6} "
                  f"{r['episode_length']:>5}f")
            print(f"    \"{r['prompt'][:100]}\"")
        print(f"\n{len(rows)} tasks, "
              f"{len({r['skill'] for r in rows})} distinct skills")
        return

    for r in rows:
        unit = DS / r["unit"]
        d = OUT / r["dir"]
        d.mkdir(parents=True, exist_ok=True)
        n, k = r["episode_length"], args.frames
        idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
        vid = video_of(unit, r["episode_index"], r["camera"])
        ims, files = [], []
        for j, idx in enumerate(idxs, 1):
            f = d / f"f{j}_{idx:05d}.png"
            if f.exists() and not args.force:
                im = Image.open(f).convert("RGB")
            else:
                im = grab(vid, idx, f, args.width)
            ims.append(im)
            files.append(dict(slot=j, frame=idx, file=f"{r['dir']}/{f.name}"))
        r["frames"] = files

        # filmstrip preview, white gutter so frames stay distinguishable
        tw = 260
        th = round(tw * ims[0].height / ims[0].width)
        g = 6
        strip = Image.new("RGB", (k * tw + (k - 1) * g, th), "white")
        for j, im in enumerate(ims):
            strip.paste(im.resize((tw, th), Image.LANCZOS), (j * (tw + g), 0))
        strip.save(d / "strip.png")

        (d / "prompt.txt").write_text(r["prompt"] + "\n")
        (d / "info.json").write_text(json.dumps(r, indent=2) + "\n")
        print(f"{r['n']:02d} {r['dir']:<52} {k} frames  {r['camera']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")

    # ---- INDEX.md ----
    L = ["# `figure_tasks` — one folder per task, timeline frames as loose PNGs",
         "",
         "Built by `scripts/make_task_image_pack.py`. Each dir has "
         f"`f1..f{args.frames}_<frame>.png` (uniformly spaced through the episode, "
         "so left-to-right = the task unfolding), `strip.png`, `prompt.txt` and "
         "`info.json`.",
         "",
         f"{len(rows)} tasks covering "
         f"{len({r['skill'] for r in rows})} skills, "
         f"{len({r['embodiment'] for r in rows})} embodiments, both splits.",
         "",
         "| # | skill | embodiment | split | source | camera | ep | len | prompt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']:02d} | {r['skill']} | {r['embodiment']} | "
                 f"{r['split']} | {r['source']} | `{r['camera']}` | "
                 f"{r['episode_index']} | {r['episode_length']} | "
                 f"*\"{r['prompt'][:92]}\"* |")
    L += ["", "## Notes", "",
          "- Timeline camera is the most legible overview view per unit "
          "(`global` / `cam_high` / `ego_view_freq20` / `exterior_1_left`); wrist "
          "views make poor timelines. All cameras of one instant live in "
          "`../figure_drawio/` instead.",
          "- `non_makovian` prompts are the compositional ones (12% of the "
          "benchmark chains 2+ primitives) — e.g. #08 presses three buttons in "
          "sequence, #11 pulls a tissue, wipes, then discards it. Those are the "
          "rows worth showing to justify the markovian / non-markovian split.",
          "- Regenerate: `python3 scripts/make_task_image_pack.py`; frames are "
          "reused unless `--force`."]
    (OUT / "INDEX.md").write_text("\n".join(L) + "\n")

    # ---- contact sheet: one strip row per task ----
    tw, th, g, hdr = 190, None, 5, 34
    first = Image.open(OUT / rows[0]["dir"] / rows[0]["frames"][0]["file"].split("/")[-1])
    th = round(tw * first.height / first.width)
    W = 8 + args.frames * (tw + g)
    H = sum(hdr + th + 8 for _ in rows) + 8
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, fs = font(13, True), font(10)
    y = 8
    for r in rows:
        dr.text((6, y), f"{r['n']:02d}  {r['skill']}  ·  {r['embodiment']}"
                f"/{r['split']}  ·  {r['source']}", font=fb, fill="#111111")
        dr.text((6, y + 17), f'"{r["prompt"][:150]}"', font=fs, fill="#555555")
        for f in r["frames"]:
            im = Image.open(OUT / r["dir"] / f["file"].split("/")[-1])
            sheet.paste(im.resize((tw, th), Image.LANCZOS),
                        (8 + (f["slot"] - 1) * (tw + g), y + hdr))
        y += hdr + th + 8
    sheet.save(OUT / "contact_sheet.png")
    print(f"\n[ok] {OUT.relative_to(REPO)}  {len(rows)} tasks x {args.frames} frames")


if __name__ == "__main__":
    main()
