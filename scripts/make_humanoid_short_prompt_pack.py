#!/usr/bin/env python3
"""Humanoid SHORT-PROMPT pack: the briefest real instructions we have, sv + mv.

Follow-up to `make_task_image_pack.py`, which put three humanoid tasks in
`figure_tasks/` with 26 / 55 / 60-character prompts — too long to sit under a
figure tile. This pack picks the shortest instruction that actually exists in
the data for each view condition, so a humanoid row can be captioned without
wrapping.

Measured floors (every humanoid prompt in `datasets/humanoid` was enumerated):

  singleview  24 chars  — beats the 26 already in figure_tasks. Shorter strings
                          exist but are junk: "invalid" (annotation rejects) and
                          "grid_Bar" / "grid_Home" (scene tags, not tasks).
  multiview   54 chars  — this is the HARD floor. Humanoid multiview ships only
                          6 tasks / 12 unique prompts and every one is >= 54,
                          so it cannot be made meaningfully shorter than the
                          55-char task already in figure_tasks. `tasks.jsonl`
                          holds no shorter alias either.

Because of that floor each dir also gets `prompt_short.txt`: an ABRIDGED label
for figure use, produced by dropping the trailing which-hand clause. It is
display-only — `prompt.txt` stays byte-identical to the dataset instruction and
is what any quantitative claim must cite.

    python3 scripts/make_humanoid_short_prompt_pack.py --list
    python3 scripts/make_humanoid_short_prompt_pack.py

Writes `assets/drive/figure_tasks_humanoid_short/`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_benchmark_stats as S  # noqa: E402
from make_task_image_pack import (  # noqa: E402
    SRC,
    grab,
    pick_camera,
    slug,
    video_of,
)
from make_teaser_tiles import meta, prompt_of  # noqa: E402

# Raw LeRobot camera key -> the label to print under a figure tile. Without this
# a reader cannot tell `cam_third_view` (a room camera) from `cam_head_rgb` (on
# the robot) or `cam_*_wrist_rgb` (in-hand).
VIEW_LABEL = {
    "ego_view_freq20": "egocentric view",
    "cam_front_rgb": "front view",
    "cam_head_rgb": "head view",
    "cam_high_rgb": "overhead view",
    "cam_third_view": "third-person view",
    "cam_left_wrist_rgb": "left wrist view",
    "cam_right_wrist_rgb": "right wrist view",
}


def view_label(cam: str) -> str:
    return VIEW_LABEL.get(cam, cam.replace("cam_", "").replace("_rgb", "")
                          .replace("_", " ") + " view")


def cameras_of(unit: Path) -> list:
    return [c.split(".")[-1] for c in S.views(unit)]


def _views_row(d: Path, r: dict) -> None:
    """The multiview presentation: ALL cameras at ONE instant, each labelled.

    A multiview row must be one synchronized moment across cameras — showing a
    timeline of a single camera instead would say nothing about the view setup.
    Column order follows `views` in info.json.
    """
    tw, pad, hdr, bar = 300, 6, 34, 22
    vs = r["views"]
    first = Image.open(d / Path(vs[0]["file"]).name)
    th = round(tw * first.height / first.width)
    W = pad + len(vs) * (tw + pad)
    sheet = Image.new("RGB", (W, hdr + th + bar + pad), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 4), f'{r["dir"]}   "{r["prompt_short"]}"',
            font=font(13, True), fill="#111111")
    dr.text((pad, 19), f"{len(vs)} synchronized cameras, ONE instant: frame "
            f"{r['frame']} of {r['episode_length']}",
            font=font(11), fill="#B03030")
    for i, v in enumerate(vs):
        x = pad + i * (tw + pad)
        sheet.paste(Image.open(d / Path(v["file"]).name)
                    .resize((tw, th), Image.LANCZOS), (x, hdr))
        dr.rectangle([x, hdr + th, x + tw, hdr + th + bar], fill="#111111")
        dr.text((x + 4, hdr + th + 4), f"v{i + 1}  {v['view']}",
                font=font(12, True), fill="#FFEE55")
        dr.text((x + tw - 4 - dr.textlength(v["camera"], font=font(10)),
                 hdr + th + 6), v["camera"], font=font(10), fill="#AAAAAA")
    sheet.save(d / "all_views.png")

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets"
OUT = REPO / "assets" / "drive" / "figure_tasks_humanoid_short"

# (rel unit, episode, abridged label or None, forced camera or None, at)
# `at` = timestamp fraction; for multiview it is THE instant all cameras are
# sampled at, tuned so every view of that row is legible at once.
# Ordered singleview-first so the pack reads sv, sv, mv, mv.
TASKS = [
    ("humanoid/singleview/makovian", 880, None, None, 0.45),
    ("humanoid/singleview/makovian", 1065, None, None, 0.45),
    ("humanoid/multiview/makovian/AIRBOT_MMK2_mobile_car", 0,
     "push the yellow cart right", None, 0.50),
    ("humanoid/multiview/non_makovian/Airbot_MMK2_storage_bottle_part", 3,
     "put the bottle in the box", None, 0.25),
]


def font(sz: int, bold: bool = False):
    n = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{n}", sz)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    rows = []
    for i, (rel, ep, short, force_cam, at) in enumerate(TASKS, 1):
        unit = DS / rel
        parts = rel.split("/")
        emb, track, split = parts[0], parts[1], parts[2]
        task_key = parts[3] if len(parts) > 3 else None
        o = meta(unit, ep)
        prompt = prompt_of(o)
        rows.append(dict(
            n=i, unit=rel, embodiment=emb, track=track, split=split,
            task_key=task_key, episode_index=ep, episode_length=int(o["length"]),
            prompt=prompt, prompt_chars=len(prompt),
            prompt_short=short or prompt,
            prompt_short_chars=len(short or prompt),
            prompt_short_is_abridged=short is not None,
            camera=force_cam or pick_camera(unit), source=SRC[(emb, track)],
            at=at,
            dir=f"{i:02d}_{track}_{split}__" + slug(task_key or f"ep{ep}"),
        ))

    if args.list:
        for r in rows:
            print(f"{r['n']:02d} {r['track']:<10} {r['split']:<12} "
                  f"{r['camera']:<16} ep{r['episode_index']:<6} "
                  f"{r['episode_length']:>4}f  prompt={r['prompt_chars']:>3}ch "
                  f"short={r['prompt_short_chars']:>3}ch")
            print(f'    verbatim : "{r["prompt"]}"')
            if r["prompt_short_is_abridged"]:
                print(f'    abridged : "{r["prompt_short"]}"')
        return

    for r in rows:
        unit = DS / r["unit"]
        d = OUT / r["dir"]
        d.mkdir(parents=True, exist_ok=True)
        n, k = r["episode_length"], args.frames

        # stale files from earlier naming schemes
        for old in list(d.glob("f?_*.png")) + list(d.glob("v?_*.png")):
            old.unlink()
        if (d / "views").is_dir():
            for old in (d / "views").rglob("*"):
                if old.is_file():
                    old.unlink()

        if r["track"] == "multiview":
            # MULTIVIEW = every synchronized camera at ONE instant.
            idx = round((n - 1) * r["at"])
            r["frame"] = idx
            r["views"] = []
            for i, cam in enumerate(cameras_of(unit), 1):
                f = d / f"v{i}_{cam}.png"
                try:
                    vid = video_of(unit, r["episode_index"], cam)
                except SystemExit:
                    continue                     # episode lacks that camera
                if not (f.exists() and not args.force):
                    grab(vid, idx, f, args.width)
                r["views"].append(dict(slot=i, camera=cam,
                                       view=view_label(cam), frame=idx,
                                       file=f"{r['dir']}/{f.name}"))
            _views_row(d, r)
            r["camera"] = None                   # no single timeline camera
            r["view_label"] = f"{len(r['views'])} synchronized views"
            (d / "views.txt").write_text(
                f"{len(r['views'])} synchronized cameras at frame {idx}/{n}:\n" +
                "".join(f"  v{v['slot']}  {v['view']:<20}({v['camera']})\n"
                        for v in r["views"]))
            print(f"{r['n']:02d} {r['dir']:<48} {len(r['views'])} views "
                  f"@f{idx}  " +
                  ", ".join(f"v{v['slot']}={v['view']}" for v in r["views"]))
        else:
            # SINGLEVIEW = one camera, so a timeline is the only thing to show.
            idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [0]
            cam = r["camera"]
            r["view_label"] = view_label(cam)
            vid = video_of(unit, r["episode_index"], cam)
            ims, recs = [], []
            for j, idx in enumerate(idxs, 1):
                f = d / f"f{j}_{idx:05d}__{cam}.png"
                if f.exists() and not args.force:
                    im = Image.open(f).convert("RGB")
                else:
                    im = grab(vid, idx, f, args.width)
                ims.append(im)
                recs.append(dict(slot=j, frame=idx, camera=cam,
                                 view=view_label(cam),
                                 file=f"{r['dir']}/{f.name}"))
            r["frames"] = recs
            tw = 260
            th = round(tw * ims[0].height / ims[0].width)
            g, bar = 6, 20
            strip = Image.new("RGB", (k * tw + (k - 1) * g, th + bar), "white")
            dr = ImageDraw.Draw(strip)
            for j, im in enumerate(ims):
                x = j * (tw + g)
                strip.paste(im.resize((tw, th), Image.LANCZOS), (x, bar))
                dr.text((x + 3, 4), f"f{j + 1}", font=font(12, True),
                        fill="#111111")
            dr.text((32, 4), f"{view_label(cam)} ({cam})  ·  f1..f{k} = frames "
                    f"{idxs[0]}..{idxs[-1]} of {n} (one camera, timeline)",
                    font=font(12), fill="#B03030")
            strip.save(d / "strip.png")
            (d / "view.txt").write_text(
                f"{view_label(cam)}\t({cam})\n"
                f"single camera; f1..f{k} = frames "
                f"{', '.join(str(i) for i in idxs)} of {n} (timeline).\n")
            print(f"{r['n']:02d} {r['dir']:<48} {k}f timeline  "
                  f"{r['view_label']} ({cam})")

        (d / "prompt.txt").write_text(r["prompt"] + "\n")
        (d / "prompt_short.txt").write_text(r["prompt_short"] + "\n")
        (d / "info.json").write_text(json.dumps(r, indent=2) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")

    L = ["# `figure_tasks_humanoid_short` — humanoid tasks with the shortest real prompts",
         "",
         "Built by `scripts/make_humanoid_short_prompt_pack.py`.",
         "",
         "## What the loose PNGs are (read this before using them)",
         "",
         "The two tracks are presented DIFFERENTLY, because the thing worth "
         "showing differs:",
         "",
         "| track | files | axis | meaning |",
         "|---|---|---|---|",
         "| **multiview** | `v1..v4_<camera>.png` | **camera** | all synchronized "
         "cameras at **ONE instant** — this is what makes it multi-view |",
         "| **singleview** | `f1..f5_<frame>__<camera>.png` | **time** | one "
         "camera, uniformly spaced frames — a timeline of the task |",
         "",
         "The camera key is in every filename, and the human-readable view name "
         "(`front view`, `left wrist view`, …) is burnt into the bottom bar of "
         "each tile in `all_views.png` / `strip.png`, so a loose PNG is never "
         "ambiguous. `views.txt` (multiview) / `view.txt` (singleview) spell out "
         "the mapping in text; `info.json` carries it machine-readably.",
         "",
         "| # | track | split | source | cameras / view | ep | len | chars | prompt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["track"] == "multiview":
            what = ", ".join(f"v{v['slot']}=`{v['camera']}`" for v in r["views"])
            what += f" @f{r['frame']}"
        else:
            what = f"`{r['camera']}` timeline"
        L.append(f"| {r['n']:02d} | {r['track']} | {r['split']} | {r['source']} | "
                 f"{what} | {r['episode_index']} | {r['episode_length']} | "
                 f"{r['prompt_chars']} | *\"{r['prompt']}\"* |")
    L += ["", "### View names", "",
          "| camera key | label used in figures |", "|---|---|"]
    used = []
    for r in rows:
        for c in ([v["camera"] for v in r["views"]] if r["track"] == "multiview"
                  else [r["camera"]]):
            if c not in used:
                used.append(c)
    for c in used:
        L.append(f"| `{c}` | {view_label(c)} |")
    L += ["", "## Why these four", "",
          "Every humanoid prompt in `datasets/humanoid` was enumerated and sorted "
          "by length. The floors are **24 chars for singleview** (better than the "
          "26-char task already in `figure_tasks/`) and **54 chars for "
          "multiview**.",
          "",
          "The multiview number is a hard limit, not a search failure: humanoid "
          "multiview ships only 6 tasks / 12 unique prompts and all of them are "
          ">= 54 chars, so it cannot be made meaningfully shorter than the "
          "55-char task already in `figure_tasks/` (#23). Shorter singleview "
          "strings do exist but are junk — `invalid` (rejected annotations) and "
          "`grid_Bar` / `grid_Home` (scene tags, not instructions).",
          "",
          "## `prompt.txt` vs `prompt_short.txt`", "",
          "- `prompt.txt` — **verbatim** dataset instruction. Cite this one.",
          "- `prompt_short.txt` — abridged label for figure use, made by dropping "
          "the trailing which-hand clause (multiview only; the two singleview "
          "tasks are already short so both files match). Display-only.",
          "",
          "| # | verbatim | abridged |", "|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']:02d} | *\"{r['prompt']}\"* ({r['prompt_chars']}ch) | "
                 f"*\"{r['prompt_short']}\"* ({r['prompt_short_chars']}ch)"
                 f"{'' if r['prompt_short_is_abridged'] else ' — unchanged'} |")
    L += ["", "Regenerate: `python3 scripts/make_humanoid_short_prompt_pack.py` "
          "(frames reused unless `--force`).",
          "",
          "Camera note: multiview rows sample EVERY camera declared in that "
          "unit's `meta/info.json` (4 each). A few extra mp4 dirs exist on disk "
          "that `info.json` does not declare — those are ignored on purpose, the "
          "declared set is the benchmark's view configuration."]
    (OUT / "INDEX.md").write_text("\n".join(L) + "\n")

    # contact sheet: singleview rows show their timeline, multiview rows show
    # their synchronized cameras — each tile labelled with what it actually is
    tw, g, hdr, bar = 190, 5, 36, 18
    ncol = max(len(r.get("frames") or r.get("views")) for r in rows)
    first = Image.open(OUT / rows[0]["dir"] /
                       (rows[0].get("frames") or rows[0]["views"])[0]["file"]
                       .split("/")[-1])
    th = round(tw * first.height / first.width)
    W = 8 + ncol * (tw + g)
    H = sum(hdr + th + bar + 8 for _ in rows) + 8
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, fs, fx = font(13, True), font(10), font(10, True)
    y = 8
    for r in rows:
        mv = r["track"] == "multiview"
        tiles = r["views"] if mv else r["frames"]
        dr.text((6, y), f"{r['n']:02d}  humanoid/{r['track']}/{r['split']}  ·  "
                f"{r['source']}  ·  {r['prompt_chars']}ch", font=fb, fill="#111111")
        dr.text((6, y + 16),
                (f"{len(tiles)} synchronized cameras @ frame {r['frame']}"
                 f"/{r['episode_length']}" if mv else
                 f"1 camera ({r['camera']}), timeline f1..f{len(tiles)} "
                 f"of {r['episode_length']} frames"),
                font=fs, fill="#B03030")
        dr.text((6, y + 26), f'"{r["prompt"]}"', font=fs, fill="#555555")
        for i, t in enumerate(tiles):
            x = 8 + i * (tw + g)
            im = Image.open(OUT / r["dir"] / t["file"].split("/")[-1])
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + hdr))
            dr.rectangle([x, y + hdr + th, x + tw, y + hdr + th + bar],
                         fill="#111111")
            dr.text((x + 3, y + hdr + th + 3),
                    f"v{t['slot']}  {t['view']}" if mv
                    else f"f{t['slot']}  frame {t['frame']}",
                    font=fx, fill="#FFEE55")
        y += hdr + th + bar + 8
    sheet.save(OUT / "contact_sheet.png")
    print(f"\n[ok] {OUT.relative_to(REPO)}  {len(rows)} tasks")


if __name__ == "__main__":
    main()
