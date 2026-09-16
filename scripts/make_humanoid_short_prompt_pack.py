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
from make_task_image_pack import (  # noqa: E402
    SRC,
    grab,
    pick_camera,
    slug,
    video_of,
)
from make_teaser_tiles import meta, prompt_of  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets"
OUT = REPO / "assets" / "drive" / "figure_tasks_humanoid_short"

# (rel unit, episode, abridged label or None, forced camera or None)
# Ordered singleview-first so the pack reads sv, sv, mv, mv.
# mobile_car forces `cam_third_view`: its default `cam_high_rgb` is mounted so
# low that the frame is mostly blown-out tabletop — third view is the only one
# showing both hands, the yellow cart AND the push direction.
TASKS = [
    ("humanoid/singleview/makovian", 880, None, None),
    ("humanoid/singleview/makovian", 1065, None, None),
    ("humanoid/multiview/makovian/AIRBOT_MMK2_mobile_car", 0,
     "push the yellow cart right", "cam_third_view"),
    ("humanoid/multiview/non_makovian/Airbot_MMK2_storage_bottle_part", 3,
     "put the bottle in the box", None),
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
    for i, (rel, ep, short, force_cam) in enumerate(TASKS, 1):
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

        tw = 260
        th = round(tw * ims[0].height / ims[0].width)
        g = 6
        strip = Image.new("RGB", (k * tw + (k - 1) * g, th), "white")
        for j, im in enumerate(ims):
            strip.paste(im.resize((tw, th), Image.LANCZOS), (j * (tw + g), 0))
        strip.save(d / "strip.png")

        (d / "prompt.txt").write_text(r["prompt"] + "\n")
        (d / "prompt_short.txt").write_text(r["prompt_short"] + "\n")
        (d / "info.json").write_text(json.dumps(r, indent=2) + "\n")
        print(f"{r['n']:02d} {r['dir']:<48} {k} frames  {r['camera']:<18} "
              f"{r['prompt_chars']}ch -> {r['prompt_short_chars']}ch")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")

    L = ["# `figure_tasks_humanoid_short` — humanoid tasks with the shortest real prompts",
         "",
         "Built by `scripts/make_humanoid_short_prompt_pack.py`. Same layout as "
         f"`figure_tasks/`: `f1..f{args.frames}_<frame>.png` (uniformly spaced, "
         "left-to-right = the task unfolding), `strip.png`, `prompt.txt`, "
         "`info.json` — plus `prompt_short.txt`.",
         "",
         "| # | track | split | source | camera | ep | len | chars | prompt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']:02d} | {r['track']} | {r['split']} | {r['source']} | "
                 f"`{r['camera']}` | {r['episode_index']} | {r['episode_length']} | "
                 f"{r['prompt_chars']} | *\"{r['prompt']}\"* |")
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
          "Camera note: #03 is forced to `cam_third_view`. Its default overview "
          "camera (`cam_high_rgb`) sits so low that the frame is mostly "
          "blown-out tabletop; third view is the only one showing both hands, "
          "the cart and the push direction."]
    (OUT / "INDEX.md").write_text("\n".join(L) + "\n")

    # contact sheet: one strip row per task
    tw, g, hdr = 190, 5, 34
    first = Image.open(OUT / rows[0]["dir"] / rows[0]["frames"][0]["file"].split("/")[-1])
    th = round(tw * first.height / first.width)
    W = 8 + args.frames * (tw + g)
    H = sum(hdr + th + 8 for _ in rows) + 8
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, fs = font(13, True), font(10)
    y = 8
    for r in rows:
        dr.text((6, y), f"{r['n']:02d}  humanoid/{r['track']}/{r['split']}  ·  "
                f"{r['source']}  ·  {r['prompt_chars']}ch", font=fb, fill="#111111")
        dr.text((6, y + 17), f'"{r["prompt"]}"', font=fs, fill="#555555")
        for f in r["frames"]:
            im = Image.open(OUT / r["dir"] / f["file"].split("/")[-1])
            sheet.paste(im.resize((tw, th), Image.LANCZOS),
                        (8 + (f["slot"] - 1) * (tw + g), y + hdr))
        y += hdr + th + 8
    sheet.save(OUT / "contact_sheet.png")
    print(f"\n[ok] {OUT.relative_to(REPO)}  {len(rows)} tasks x {args.frames} frames")


if __name__ == "__main__":
    main()
