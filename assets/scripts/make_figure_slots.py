#!/usr/bin/env python3
"""Figure asset pack for the draw.io teaser: one PNG per camera slot.

Fills the "(a) Dataset Collection" block of the teaser: for EACH embodiment it
emits the single-view row (1 camera) and the multi-view row (3-4 synchronized
cameras). Every tile in a row is the SAME instant of the SAME episode, so the
row reads as one coherent scene; diversity is shown later in the qualitative
figures, not here.

Files are numbered in slot order (`sv_1_*`, `mv_1_*`, `mv_2_*`, ...) so they drop
into the draw.io canvas left-to-right without guessing.

    python3 assets/scripts/make_figure_slots.py --list      # inspect picks + prompts
    python3 assets/scripts/make_figure_slots.py             # write the pack
    python3 assets/scripts/make_figure_slots.py --square    # also square-cropped copies

Writes `assets/drive/figure_drawio/<embodiment>/{sv,mv}_<n>_<camera>.png`,
a `slots.json` manifest (episode / frame / prompt / source per row) and a
labelled `contact_sheet.png`.

IMPORTANT provenance note baked into the manifest: single-view and multi-view are
DIFFERENT source datasets for single_arm and humanoid, but for bimanual the
single-view track IS `cam_high` of the same ALOHA multiview episode (symlinked by
scripts/tools/data/export_bimanual_singleview_cam_high.py) — not a separate robot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_teaser_tiles import cam_video, frame_at, meta, prompt_of, square  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DS = REPO / "datasets"
OUT = REPO / "assets" / "drive" / "figure_drawio"

# One entry per (embodiment, track): unit under datasets/, episode, camera order
# = slot order, source dataset label, and `at` = timestamp as a fraction of the
# episode, tuned so every camera of that row is readable at the same instant.
SLOTS = {
    ("single_arm", "sv"): dict(
        unit="single_arm/singleview/makovian", ep=None,
        cams=["global"], source="RoboChallenge (Traly/RoboChallenge-lerobot)",
        robot="single Franka-class arm, 1 static global camera", at=0.55),
    ("single_arm", "mv"): dict(
        unit="single_arm/multiview/makovian", ep=1533,
        cams=["exterior_1_left", "exterior_2_left", "wrist_left"],
        source="DROID (cadene/droid_1.0.1)",
        robot="Franka Panda, 2 exterior + 1 wrist camera", at=0.55),
    ("bimanual", "sv"): dict(
        unit="bimanual/singleview/makovian/tape_closed_cardboard", ep=None,
        cams=["cam_high"], source="AIST Bimanual Manipulation (ALOHA)",
        robot="ALOHA static, `cam_high` of the SAME multiview episode", at=0.55),
    ("bimanual", "mv"): dict(
        unit="bimanual/multiview/makovian/tape_closed_cardboard", ep=None,
        cams=["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"],
        source="AIST Bimanual Manipulation (ALOHA)",
        robot="ALOHA static, 2 overview + 2 wrist cameras", at=0.55),
    ("humanoid", "sv"): dict(
        # short prompt (26 chars): "Pick up corn to pink plate" — the corn, the
        # pink plate and the hand are all clearly visible, unlike the previous
        # 55-char starfruit/shelf episode.
        unit="humanoid/singleview/makovian", ep=12318,
        cams=["ego_view_freq20"],
        source="NVIDIA PhysicalAI-Robotics-GR00T-Teleop-GR1",
        robot="Fourier GR1 humanoid, 1 egocentric camera", at=0.45),
    ("humanoid", "mv"): dict(
        # Chosen for a SHORT prompt (55 chars) so the teaser caption fits:
        # "put the water bottle and handle into the cardboard box."
        # humanoid multiview only has 8 prompts in total; the others are 114-138
        # chars. move_pan / mobile_car were also rejected earlier (wrist cameras
        # blurred/blown out at every timestamp, mobile_car's table nearly empty),
        # and doodled_line's wrist views are fully occluded by the hand.
        # Trade-off accepted: cam_head/wrist here are darker and less legible than
        # the 138-char storage_mango_pomegranate episode.
        unit="humanoid/multiview/non_makovian/Airbot_MMK2_storage_bottle_part",
        ep=3,
        cams=["cam_front_rgb", "cam_head_rgb",
              "cam_left_wrist_rgb", "cam_right_wrist_rgb"],
        source="RoboCOIN (AIRBOT MMK2)",
        robot="AIRBOT MMK2 mobile humanoid, 2 head + 2 wrist cameras", at=0.25),
}

ORDER = ["single_arm", "bimanual", "humanoid"]
TRACK_NAME = {"sv": "single-view", "mv": "multi-view"}


def episodes_with(unit: Path, cams: list) -> list:
    """Episode indices that have a video for EVERY requested camera."""
    per = []
    for c in cams:
        per.append({int(p.stem.split("_")[-1])
                    for p in (unit / "videos").rglob("episode_*.mp4")
                    if f".{c}/" in str(p) or f"/{c}/" in str(p)})
    return sorted(set.intersection(*per)) if per else []


def resolve(spec: dict) -> tuple:
    """Pick an episode that really has every camera on disk."""
    unit = DS / spec["unit"]
    avail = episodes_with(unit, spec["cams"])
    if not avail:
        raise SystemExit(f"no episode has all of {spec['cams']} in {spec['unit']}")
    ep = spec["ep"] if spec["ep"] in avail else avail[0]
    return unit, ep


def font(sz: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", sz)
    except OSError:
        return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print picks and exit")
    ap.add_argument("--width", type=int, default=640, help="tile width px")
    ap.add_argument("--square", action="store_true",
                    help="also write square-cropped copies under _square/")
    ap.add_argument("--only", nargs="*", default=None, help="embodiments to build")
    args = ap.parse_args()

    if args.list:
        for emb in ORDER:
            print(f"\n=== {emb} ===")
            for tr in ("sv", "mv"):
                s = SLOTS[(emb, tr)]
                unit, ep = resolve(s)
                o = meta(unit, ep)
                n = int(o["length"])
                print(f"  {TRACK_NAME[tr]:<12} {len(s['cams'])} cam(s)  "
                      f"{s['source']}")
                print(f"    unit    : {s['unit']}")
                print(f"    episode : {ep:06d}  ({n} frames)  "
                      f"frame @{s['at']:.2f} = {round((n - 1) * s['at'])}")
                print(f"    prompt  : \"{prompt_of(o)}\"")
                print(f"    cams    : {s['cams']}")
        return

    manifest = {}
    for emb in ORDER:
        if args.only and emb not in args.only:
            continue
        manifest[emb] = {}
        for tr in ("sv", "mv"):
            s = SLOTS[(emb, tr)]
            unit, ep = resolve(s)
            o = meta(unit, ep)
            n = int(o["length"])
            idx = round((n - 1) * s["at"])
            d = OUT / emb
            d.mkdir(parents=True, exist_ok=True)
            tiles = []
            for i, cam in enumerate(s["cams"], 1):
                raw = d / f"_raw_{tr}_{cam}.png"
                im = frame_at(cam_video(unit, ep, cam), idx, raw)
                w = args.width
                im = im.resize((w, round(w * im.height / im.width)), Image.LANCZOS)
                name = f"{tr}_{i}_{cam}.png"
                im.save(d / name)
                if args.square:
                    sq = OUT / "_square" / emb
                    sq.mkdir(parents=True, exist_ok=True)
                    square(im, min(w, 512)).save(sq / name)
                raw.unlink()
                tiles.append(dict(slot=i, camera=cam, file=f"{emb}/{name}",
                                  size=list(im.size)))
            manifest[emb][tr] = dict(
                track=TRACK_NAME[tr], source=s["source"], robot=s["robot"],
                unit=s["unit"], episode_index=ep, episode_length=n,
                frame=idx, at=s["at"], prompt=prompt_of(o),
                n_cameras=len(s["cams"]), tiles=tiles)
            print(f"{emb:11s} {tr}  ep{ep:06d} frame {idx:4d}/{n:<5d} "
                  f"{len(s['cams'])} tile(s)  \"{prompt_of(o)[:52]}\"")

    # bimanual sv/mv share the source: record the relationship explicitly so the
    # figure is not captioned as two different robots
    for emb, m in manifest.items():
        if "sv" in m and "mv" in m:
            m["same_source"] = m["sv"]["source"] == m["mv"]["source"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "slots.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # ---- contact sheet: 2 rows per embodiment (sv then mv), labelled ----
    TW, PAD, HDR = 300, 8, 52
    ncol = max(m[t]["n_cameras"] for m in manifest.values() for t in ("sv", "mv"))
    rows = [(emb, tr) for emb in manifest for tr in ("sv", "mv")]
    th = round(TW * 9 / 16)
    W = PAD + ncol * (TW + PAD)
    H = sum(HDR + th + PAD for _ in rows) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    fb, fs, fx = font(16, True), font(12), font(11)
    y = PAD
    for emb, tr in rows:
        m = manifest[emb][tr]
        dr.text((PAD, y), f"{emb}  ·  {m['track']}  ·  {m['n_cameras']} cam  ·  "
                f"ep{m['episode_index']:06d} f{m['frame']}/{m['episode_length']}",
                font=fb, fill="#111111")
        dr.text((PAD, y + 20), f"{m['source']}", font=fx, fill="#B03030")
        dr.text((PAD, y + 34), f'"{m["prompt"][:110]}"', font=fs, fill="#555555")
        for t in m["tiles"]:
            im = Image.open(OUT / t["file"]).resize((TW, th), Image.LANCZOS)
            x = PAD + (t["slot"] - 1) * (TW + PAD)
            sheet.paste(im, (x, y + HDR))
            dr.rectangle([x, y + HDR + th - 17, x + TW, y + HDR + th], fill="#000000")
            dr.text((x + 4, y + HDR + th - 16), t["camera"], font=fx, fill="#FFEE55")
        y += HDR + th + PAD
    sheet.save(OUT / "contact_sheet.png")
    print(f"\n[ok] {OUT.relative_to(REPO)}  "
          f"({sum(m[t]['n_cameras'] for m in manifest.values() for t in ('sv','mv'))} tiles)")


if __name__ == "__main__":
    main()
