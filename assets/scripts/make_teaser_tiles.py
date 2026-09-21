#!/usr/bin/env python3
"""Teaser tiles: ONE episode, ONE timestamp, one frame per camera.

Fills the "(a) Dataset Collection" slot row under each embodiment: the tiles are
deliberately the SAME instant of the SAME episode across all of that
embodiment's synchronized cameras, so the row reads as one coherent scene
(3 tiles single-arm, 4 bimanual, 4 humanoid). Diversity is shown later in the
qualitative figures, not here.

    python3 assets/scripts/make_teaser_tiles.py --list          # candidate episodes
    python3 assets/scripts/make_teaser_tiles.py                 # write the tiles
    python3 assets/scripts/make_teaser_tiles.py --at 0.45 --contact-sheet

Writes `assets/figures/teaser_tiles/<embodiment>/<camera>.png` (one tile per
camera, cropped to a square so it drops straight into a rounded slot) plus
`tiles.json` recording episode / frame / prompt, and optionally a labelled
`contact_sheet.png` for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
OUT = REPO / "assets" / "figures" / "teaser_tiles"

# (embodiment, unit under datasets/, episode, camera order = slot order, source,
#  at = timestamp as a fraction of episode length — tuned PER embodiment so every
#  camera of that row is readable at the same instant)
PICKS = {
    "single_arm": (
        "single_arm/multiview/makovian", 1533,
        ["exterior_1_left", "exterior_2_left", "wrist_left"],
        "DROID (Franka)", 0.55),
    "bimanual": (
        "bimanual/multiview/makovian/tape_closed_cardboard", 46,
        ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        "AIST Bimanual (ALOHA)", 0.30),
    "humanoid": (
        "humanoid/multiview/makovian/Airbot_MMK2_move_pan", 0,
        ["cam_front_rgb", "cam_head_rgb", "cam_left_wrist_rgb", "cam_right_wrist_rgb"],
        "RoboCOIN (AIRBOT MMK2)", 0.40),
}

# alternates worth trying via --pick, all verified to have every camera on disk
ALTS = {
    "single_arm": [("single_arm/multiview/makovian", 8397,
                    "Remove the spray bottle from the sink."),
                   ("single_arm/multiview/makovian", 1872,
                    "Push down the toaster lever")],
    "bimanual": [("bimanual/multiview/makovian/insert_washer_shaft", 73, None),
                 ("bimanual/multiview/non_makovian/fold_yellow_towel", 7, None)],
    "humanoid": [("humanoid/multiview/makovian/Airbot_MMK2_move_pan", 0, None),
                 ("humanoid/multiview/non_makovian/AIRBOT_MMK2_boxs_storage", 0, None)],
}

EPNUM = re.compile(r"episode_(\d+)")
WAIST = re.compile(r"^(un)?locked waist:\s*", re.I)


def meta(unit: Path, ep: int) -> dict:
    for line in (unit / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip() and int(json.loads(line)["episode_index"]) == ep:
            return json.loads(line)
    raise SystemExit(f"episode {ep} not in {unit}")


def prompt_of(o: dict) -> str:
    ts = [t for t in (o.get("tasks") or [])
          if t and t.strip().lower() not in {"valid", "invalid"}]
    t = ts[0] if ts else (o.get("source_task") or "").replace("_", " ")
    return WAIST.sub("", t).strip()


def cam_video(unit: Path, ep: int, cam: str) -> Path:
    hits = [p for p in (unit / "videos").rglob(f"episode_{ep:06d}.mp4")
            if f".{cam}/" in str(p) or f"/{cam}/" in str(p)]
    if not hits:
        raise SystemExit(f"missing camera {cam} for {unit} ep{ep}")
    return hits[0]


def frame_at(video: Path, idx: int, dst: Path) -> Image.Image:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
                    "-vf", f"select='eq(n\\,{idx})'", "-vsync", "0",
                    "-frames:v", "1", str(dst)], check=True)
    return Image.open(dst).convert("RGB")


def square(im: Image.Image, size: int) -> Image.Image:
    """Centre-crop to a square then resize — slots in the teaser are square."""
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=float, default=None,
                    help="override the per-embodiment timestamp fraction")
    ap.add_argument("--size", type=int, default=384, help="tile px (square)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--list", action="store_true", help="print candidates and exit")
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument("--no-square", action="store_true", help="keep native aspect")
    ap.add_argument("--pick", nargs="*", default=None, metavar="EMB=ALT_IDX",
                    help="use an ALTS entry instead, e.g. humanoid=0 bimanual=1")
    args = ap.parse_args()

    picks = dict(PICKS)
    for spec in args.pick or []:
        emb, _, k = spec.partition("=")
        if emb not in picks or not k.isdigit() or int(k) >= len(ALTS.get(emb, [])):
            raise SystemExit(f"bad --pick {spec!r}; alts: "
                             f"{ {e: len(v) for e, v in ALTS.items()} }")
        rel, ep, _ = ALTS[emb][int(k)]
        picks[emb] = (rel, ep, picks[emb][2], picks[emb][3], picks[emb][4])

    if args.list:
        for emb, (rel, ep, cams, src, at0) in picks.items():
            at = args.at if args.at is not None else at0
            o = meta(DS / rel, ep)
            print(f"\n{emb}  [{src}]  {len(cams)} cameras")
            print(f"  unit    : {rel}")
            print(f"  episode : {ep:06d}   length {o['length']}   "
                  f"frame @{at:.2f} = {round((o['length'] - 1) * at)}")
            print(f"  prompt  : \"{prompt_of(o)}\"")
            print(f"  cameras : {cams}")
            for rel2, ep2, note in ALTS.get(emb, []):
                o2 = meta(DS / rel2, ep2)
                print(f"  alt     : {rel2} ep{ep2:06d} ({o2['length']}f) "
                      f"\"{note or prompt_of(o2)}\"")
        return

    manifest = {}
    for emb, (rel, ep, cams, src, at0) in picks.items():
        if args.only and emb not in args.only:
            continue
        at = args.at if args.at is not None else at0
        unit = DS / rel
        o = meta(unit, ep)
        n = int(o["length"])
        idx = round((n - 1) * at)
        d = OUT / emb
        tiles = []
        for cam in cams:
            raw = d / f"_raw_{cam}.png"
            im = frame_at(cam_video(unit, ep, cam), idx, raw)
            out = d / f"{cam}.png"
            (im if args.no_square else square(im, args.size)).save(out)
            raw.unlink()
            tiles.append({"camera": cam, "file": f"{emb}/{out.name}"})
        manifest[emb] = {
            "source": src, "unit": rel, "episode_index": ep, "episode_length": n,
            "frame": idx, "at": at, "prompt": prompt_of(o),
            "n_cameras": len(cams), "tiles": tiles,
        }
        print(f"{emb:11s} ep{ep:06d} frame {idx:4d}/{n:<5d} {len(cams)} tiles  "
              f"\"{prompt_of(o)[:56]}\"")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tiles.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.contact_sheet:
        from PIL import ImageDraw, ImageFont
        try:
            f = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
            fs = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            f = fs = ImageFont.load_default()
        S, PAD, HDR = 200, 6, 46
        ncol = max(m["n_cameras"] for m in manifest.values())
        W = PAD + ncol * (S + PAD)
        H = sum(HDR + S + PAD for _ in manifest) + PAD
        sheet = Image.new("RGB", (W, H), "white")
        dr = ImageDraw.Draw(sheet)
        y = PAD
        for emb, m in manifest.items():
            dr.text((PAD, y), f"{emb}  ·  {m['n_cameras']} synced cameras  ·  "
                    f"ep{m['episode_index']:06d} frame {m['frame']}/{m['episode_length']}",
                    font=f, fill="#111111")
            dr.text((PAD, y + 19), f'"{m["prompt"][:96]}"', font=fs, fill="#555555")
            for c, t in enumerate(m["tiles"]):
                im = Image.open(OUT / Path(t["file"]).parent.name / Path(t["file"]).name)
                sheet.paste(im.resize((S, S), Image.LANCZOS),
                            (PAD + c * (S + PAD), y + HDR))
                dr.text((PAD + c * (S + PAD) + 3, y + HDR + S - 15),
                        t["camera"], font=fs, fill="#FFEE55")
            y += HDR + S + PAD
        sheet.save(OUT / "contact_sheet.png")
        print(f"[ok] {(OUT / 'contact_sheet.png').relative_to(REPO)}")

    print(f"[ok] {(OUT / 'tiles.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
