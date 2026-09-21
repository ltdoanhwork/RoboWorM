#!/usr/bin/env python3
"""Hero / teaser figure: a dimmed montage of real dataset frames behind four
statistics panels.

Layout follows the convention of recent multi-embodiment dataset papers (RoboMIND
Fig. 1, RoboInter Fig. 1): real frames establish visual diversity, and the panels
carry the quantitative claim. Unlike those, the panels here are bars/matrices
rather than pie charts -- a pie cannot show the (method x model) coverage grid,
which is the thing this benchmark is actually about.

    (1) source corpus            (3) episode duration
    (2) distinct instructions    (4) rollouts: dense + acceleration coverage

Frames are sampled straight out of `datasets/` and cached, so re-running is cheap.

    python3 assets/scripts/audit_sampling_distribution.py --json metrics/reports/sampling_audit.json
    python3 assets/scripts/make_benchmark_stats.py --json metrics/reports/benchmark_stats.json
    python3 assets/scripts/make_dataset_appendix_tables.py
    python3 assets/scripts/make_hero_figure.py

Writes assets/paper/figures/fig_hero.{png,pdf}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import subprocess
from pathlib import Path

from PIL import (Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont)

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
OUT = REPO / "assets/paper/figures"
CACHE = REPO / "metrics/reports/_hero_frames"
AUDIT = REPO / "metrics/reports/sampling_audit.json"
STATS = REPO / "metrics/reports/benchmark_stats.json"
EVID = REPO / "metrics/reports/dataset_appendix_evidence.json"


def units() -> list[Path]:
    """Every LeRobot unit under datasets/ (a dir holding meta/episodes.jsonl)."""
    return sorted(p.parent.parent for p in DATASETS.rglob("meta/episodes.jsonl"))


def episodes_of(unit: Path) -> list[int]:
    out = []
    for line in (unit / "meta" / "episodes.jsonl").read_text().splitlines():
        if line.strip():
            try:
                out.append(int(json.loads(line)["episode_index"]))
            except (KeyError, ValueError):
                pass
    return out


def any_video(unit: Path, ep: int) -> Path | None:
    hits = sorted((unit / "videos").rglob(f"episode_{ep:06d}.mp4"))
    return hits[0] if hits else None


# Camera preference for the illustration frames, best first. Plain sorting picked
# `cam_front_rgb` on the humanoid rig -- a body-mounted camera angled down at the
# robot's own grippers, which framed the table diagonally and made the action
# unreadable. Head / high / exterior cameras show the scene as a person would.
VIEW_RANK = ("cam_head", "cam_high", "head_rgb", "high_rgb", "exterior_1",
             "exterior_2", "exterior", "cam_third", "third_view", "global",
             "ego_view", "cam_low", "cam_front")


def best_video(unit: Path, ep: int) -> Path | None:
    """The most legible camera for this episode (wrist views come last)."""
    hits = sorted((unit / "videos").rglob(f"episode_{ep:06d}.mp4"))
    if not hits:
        return None

    def rank(p: Path) -> tuple[int, int, str]:
        name = p.parent.name.lower()
        wrist = 1 if "wrist" in name else 0        # never a scene-level view
        for i, key in enumerate(VIEW_RANK):
            if key in name:
                return (wrist, i, name)
        return (wrist, len(VIEW_RANK), name)

    return sorted(hits, key=rank)[0]


def grab(video: Path, dst: Path, frac: float = 0.45) -> Image.Image | None:
    """Pull one frame from ~45% into the clip (mid-episode = object in hand)."""
    if dst.exists():
        try:
            return Image.open(dst).convert("RGB")
        except OSError:
            dst.unlink(missing_ok=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -ss before -i is the fast seek; we only need a representative frame.
    dur = probe_duration(video)
    ts = max(0.0, (dur or 4.0) * frac)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{ts:.2f}",
         "-i", str(video), "-frames:v", "1", "-q:v", "3", str(dst)],
        capture_output=True)
    if r.returncode != 0 or not dst.exists():
        return None
    try:
        return Image.open(dst).convert("RGB")
    except OSError:
        return None


def probe_duration(video: Path) -> float | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(video)],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def square(im: Image.Image, size: int) -> Image.Image:
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


def sharpest_frame(video: Path, key: str, size: int,
                   fracs=(0.35, 0.45, 0.55, 0.65)) -> Image.Image | None:
    """Best of several candidate timestamps, by variance-of-Laplacian sharpness.

    A single fixed timestamp lands on motion blur often enough to matter: the
    wipe/clean example caught the arm mid-sweep and the cloth smeared into the
    table. Sampling a few moments and keeping the crispest one is cheap (ffmpeg
    seeks are fast) and removes that failure.
    """
    best, best_score = None, -1.0
    for f in fracs:
        im = grab(video, CACHE / f"{key}__f{int(f * 100)}.jpg", frac=f)
        if im is None:
            continue
        if (v := _sharpness(im)) > best_score:
            best, best_score = im, v
    return best


def _sharpness(im: Image.Image) -> float:
    """Variance of a 3x3 Laplacian -- the standard blur proxy."""
    g = im.convert("L").resize((160, 160))
    lap = g.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0],
                                      scale=1))
    px = list(lap.getdata())
    mean = sum(px) / len(px)
    return sum((p - mean) ** 2 for p in px) / len(px)


def _motion(a: Image.Image, b: Image.Image, thr: int = 26
            ) -> tuple[float, float, float]:
    """(moved fraction, centrality, compactness) of the changed region, in [0,1].

    This is what decides whether a viewer can SEE the action. Two failures drove
    the third term: the DROID "pour" frame was sharp and on-label but the arm was
    a few percent of a wide living-room shot (small + off-centre), while a
    "slide the curtain" frame scored high on motion with NO robot visible,
    because an entire curtain swaying is a big but diffuse change. Compactness --
    how tightly the moving pixels cluster -- separates a manipulator doing
    something from a whole surface drifting.
    """
    N = 96
    ga = a.convert("L").resize((N, N))
    gb = b.convert("L").resize((N, N))
    pa, pb = ga.getdata(), gb.getdata()
    hits = [i for i, (x, y) in enumerate(zip(pa, pb)) if abs(x - y) > thr]
    if not hits:
        return 0.0, 0.0, 0.0
    frac = len(hits) / (N * N)
    xs = [i % N for i in hits]
    ys = [i // N for i in hits]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    # 1 at the centre, 0 at a corner
    central = 1.0 - min(1.0, (abs(cx / N - 0.5) + abs(cy / N - 0.5)))
    # mean distance from the centroid, normalised; tight cluster -> near 1
    spread = sum(((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                 for x, y in zip(xs, ys)) / len(xs) / N
    compact = max(0.0, 1.0 - spread / 0.32)
    return frac, central, compact


def legibility(video: Path, key: str,
               fracs=(0.30, 0.42, 0.54, 0.66, 0.78)
               ) -> tuple[float, Image.Image | None]:
    """Score how understandable this clip's best frame is, and return that frame.

    score = sharpness x motion-size-band x centrality x compactness. The motion
    band is 2%-35% of the frame: below it the action is invisible, above it the
    camera itself is moving (ego views) and nothing is stable to look at.
    """
    ims = []
    for f in fracs:
        im = grab(video, CACHE / f"{key}__f{int(f * 100)}.jpg", frac=f)
        if im is not None:
            ims.append(im)
    if len(ims) < 2:
        return 0.0, (ims[0] if ims else None)

    sharps = [_sharpness(im) for im in ims]
    smax = max(sharps) or 1.0
    best, best_score = ims[0], -1.0
    for i in range(len(ims)):
        nb = ims[i + 1] if i + 1 < len(ims) else ims[i - 1]
        frac, central, compact = _motion(ims[i], nb)
        if frac < 0.02:
            band = frac / 0.02 * 0.35          # action too small to read
        elif frac <= 0.35:
            band = 1.0
        else:
            band = max(0.15, 1.0 - (frac - 0.35) * 2.0)   # whole frame moving
        s = ((sharps[i] / smax) * band * (0.35 + 0.65 * central)
             * (0.30 + 0.70 * compact))
        if s > best_score:
            best, best_score = ims[i], s
    return best_score, best


def label_frame(im: Image.Image, title: str, subtitle: str) -> Image.Image:
    """Burn a caption bar onto the frame so it is self-explanatory standalone."""
    from matplotlib import font_manager
    out = im.convert("RGB").copy()
    W, H = out.size
    bar = int(H * 0.20)
    # translucent dark bar (BridgeData V2 uses the same device)
    strip = Image.new("RGB", (W, bar), (17, 24, 39))
    out.paste(Image.blend(out.crop((0, H - bar, W, H)), strip, 0.72),
              (0, H - bar))
    d = ImageDraw.Draw(out)

    def font(px, bold=False):
        fam = "DejaVu Sans"
        p = font_manager.findfont(font_manager.FontProperties(
            family=fam, weight="bold" if bold else "normal"))
        return ImageFont.truetype(p, px)

    d.text((int(W * 0.035), H - bar + int(bar * 0.13)), title,
           font=font(int(H * 0.062), bold=True), fill=(255, 255, 255))
    sub = subtitle if len(subtitle) <= 46 else subtitle[:45].rstrip() + "\u2026"
    d.text((int(W * 0.035), H - bar + int(bar * 0.58)), sub,
           font=font(int(H * 0.040)), fill=(209, 213, 219))
    return out


def cell_of(unit: Path) -> tuple[str, str]:
    """(embodiment, view) for a unit, from its path under datasets/."""
    parts = unit.relative_to(DATASETS).parts
    return (parts[0], parts[1]) if len(parts) >= 2 else ("?", "?")


def _stats_module():
    """Load make_benchmark_stats as a module.

    The primitive labels in panel (b) come from its classifier, so the filmstrip
    must use THE SAME classifier -- re-implementing the verb rules here would let
    the pictures drift away from the counts they illustrate.
    """
    import importlib.util
    src = Path(__file__).with_name("make_benchmark_stats.py")
    spec = importlib.util.spec_from_file_location("_bstats", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _label_verbs(skill: str) -> list[str]:
    """Verbs that appear in the primitive's own name, e.g. 'wipe / clean' ->
    ['wipe', 'clean'].

    The classifier maps synonyms onto a primitive ('spoon' -> pour, 'brush' ->
    wipe / clean). That is right for COUNTING, but it picked pictures that
    contradict their label: the 'pour' frame was a gripper dropping a block into
    a bowl (instruction "spoon yellow block bowl"). For the illustration we
    therefore prefer episodes whose instruction uses the primitive's own verb.
    """
    return [p for p in (w.strip() for w in re.split(r"[/&]", skill.lower())) if p]


def primitive_frames(n_prim: int, size: int, seed: int, n_try: int = 12
                     ) -> list[tuple[str, int, Image.Image]]:
    """One ground-truth frame per manipulation primitive, most common first.

    Returns (primitive, distinct-instruction count, frame). For each primitive up
    to `n_try` on-label episodes are scored by `legibility` and the clearest one
    wins -- taking the first candidate gave technically-correct but unreadable
    pictures (a 2%-of-frame arm in a wide living-room shot for "pour").
    """
    bs = _stats_module()
    stats = json.loads(STATS.read_text())
    order = [k for k, _ in sorted(stats["skills_primary"].items(),
                                  key=lambda kv: -kv[1])][:n_prim]
    counts = stats["skills_primary"]

    cand: dict[str, list[tuple[Path, int, str, str]]] = collections.defaultdict(list)
    eps_per_skill: collections.Counter = collections.Counter()
    eps_seen = eps_unmapped = 0
    for emb in ("single_arm", "bimanual", "humanoid"):
        for track in ("multiview", "singleview"):
            for split in ("makovian", "non_makovian"):
                for u in bs.units(emb, track, split):
                    have = bs.present(u)
                    f = u / "meta" / "episodes.jsonl"
                    for line in f.read_text().splitlines():
                        if not line.strip():
                            continue
                        o = json.loads(line)
                        if o.get("episode_index") not in have:
                            continue
                        eps_seen += 1
                        prompt = bs.instruction(o)
                        sk = bs.skill_of(prompt) if prompt else ""
                        if not sk:
                            eps_unmapped += 1
                            continue
                        eps_per_skill[sk] += 1
                        if sk in order:
                            cand[sk].append((u, o["episode_index"], emb, prompt))

    rng = random.Random(seed)
    out: list[tuple[str, int, Image.Image]] = []
    used_emb: collections.Counter = collections.Counter()
    # Exported alongside the figure so each picture in the strip can be traced
    # back to the episode it came from -- a frame with no provenance is not
    # evidence.
    exp = CACHE / "primitives"
    exp.mkdir(parents=True, exist_ok=True)
    for old in exp.glob("*"):
        old.unlink()
    index: list[str] = []
    caps: list[str] = []
    total_instr = sum(counts.values())
    for rank, sk in enumerate(order, 1):
        pool = cand.get(sk, [])
        if not pool:
            print(f"  WARNING: no episode found for primitive '{sk}'")
            continue
        verbs = _label_verbs(sk)
        tried: list[tuple] = []

        def on_label(prompt: str) -> bool:
            p = prompt.lower()
            return any(re.search(rf"\b{v}", p) for v in verbs)

        def score(t):
            _, _, emb, prompt = t
            p = prompt.lower()
            # a human hand in frame makes a robot primitive read ambiguously
            human = 1 if re.search(r"\bhuman\b|\bhandover\b", p) else 0
            return (human, used_emb[emb], rng.random())

        # On-label is a HARD FILTER, not one term of the score: letting legibility
        # outvote it brought back "spoon yellow block bowl" as the picture for
        # `pour`. Only if no on-label episode exists do we fall back to the rest.
        strict = [t for t in pool if on_label(t[3])]
        ordered = sorted(strict or pool, key=score)

        for u, ep, emb, prompt in ordered:
            v = best_video(u, ep)
            if v is None:
                continue
            cam = re.sub(r"^observation\.images\.", "", v.parent.name)
            key = f"{u.relative_to(DATASETS)}".replace(os.sep, "__")
            sc, im = legibility(v, f"prim__{key}__{cam}__{ep:06d}")
            if im is None:
                continue
            tried.append((sc, im, u, ep, cam, prompt, emb))
            if len(tried) >= n_try:
                break
        if not tried:
            print(f"  WARNING: no usable video for primitive '{sk}'")
            continue

        # Pick the most LEGIBLE candidate, not merely the first on-label one.
        sc, im, u, ep, cam, prompt, emb = max(tried, key=lambda t: t[0])
        sq = square(im, size)
        out.append((sk, counts[sk], sq))
        slug = re.sub(r"[^a-z0-9]+", "_", sk.lower()).strip("_")
        name = f"{rank}_{slug}.jpg"
        sq.save(exp / name, quality=95)
        # A second, self-explanatory copy with the action and count burnt in, so
        # a viewer needs no surrounding caption to read the picture.
        label_frame(sq, sk.title(),
                    f"{counts[sk]} instructions · {eps_per_skill[sk]} episodes"
                    ).save(exp / f"labeled_{name}", quality=95)
        index.append(f"{name}\t{sk}\t{counts[sk]}\t{eps_per_skill[sk]}\t"
                     f"{u.relative_to(DATASETS)}\t{ep}\t{cam}\t{sc:.3f}\t"
                     f"{prompt}")
        caps.append(
            f"### {rank}. {sk}\n\n"
            f"![{sk}]({name})\n\n"
            f"**{counts[sk]} distinct instructions** "
            f"({100 * counts[sk] / total_instr:.1f}% of {total_instr}), "
            f"**{eps_per_skill[sk]} episode instances**.\n\n"
            f"Frame shown: `{u.relative_to(DATASETS)}`, episode {ep}, "
            f"camera `{cam}` --- *\u201c{prompt}\u201d*  "
            f"(legibility {sc:.2f} of {len(tried)} candidates)\n")
        used_emb[emb] += 1
    (exp / "index.tsv").write_text(
        "file\tprimitive\tdistinct_instructions\tepisode_instances\t"
        "source_unit\tepisode\tcamera\tlegibility\tinstruction\n"
        + "\n".join(index) + "\n")
    # Reconcile the per-picture episode counts against the benchmark total.
    # A reader WILL sum the numbers burnt into the strip, so the residual has
    # to be stated rather than left as an apparent inconsistency.
    shown = {sk for sk, _, _ in out}
    shown_eps = sum(eps_per_skill[sk] for sk in shown)
    other_list = [(k, v) for k, v in eps_per_skill.most_common()
                  if k not in shown]
    other_eps = sum(v for _, v in other_list)
    (exp / "captions.md").write_text(
        "# Manipulation primitives --- action examples\n\n"
        "One ground-truth frame per primitive, ordered by how many distinct "
        "instructions map to it. Counts come from "
        "`metrics/reports/benchmark_stats.json`; the frame is taken from the "
        "episode named under each picture.\n\n" + "\n".join(caps)
        + "\n\n## Why the episode counts above do not add up to "
        f"{eps_seen}\n\n"
        "Three separate gaps, all measured, not estimated:\n\n"
        f"- `{shown_eps}` episode instances are covered by the "
        f"{len(out)} primitives pictured here.\n"
        f"- `{other_eps}` fall under the {len(other_list)} remaining "
        "primitives in the taxonomy, which the strip has no room to show "
        f"({', '.join(f'{k} {v}' for k, v in other_list)}).\n"
        f"- `{eps_unmapped}` carry no usable instruction at all -- humanoid "
        "singleview episodes labelled by scene rather than by task, plus a "
        "couple of placeholders -- so no primitive can be assigned.\n\n"
        f"`{shown_eps} + {other_eps} + {eps_unmapped} = {eps_seen}`. "
        "Each episode is counted once: an instruction that mentions several "
        "primitives is attributed to its primary one, so these three groups "
        "partition the benchmark rather than overlap. The multi-label view "
        "(where one episode may appear under several primitives) is reported "
        "separately in the appendix and is the reason the per-primitive "
        "mention counts there are larger.\n")
    print(f"  primitive strip: {len(out)} frames -> "
          f"{exp.relative_to(REPO)}/  (embodiments {dict(used_emb)})")
    print(f"  episode reconciliation: {shown_eps} shown + {other_eps} other "
          f"primitives + {eps_unmapped} unlabelled = {eps_seen} instances")
    return out


def collect_frames(n: int, tile: int, seed: int) -> list[Image.Image]:
    """Sample frames with an EQUAL QUOTA PER (embodiment, view) cell.

    Sampling per unit skews badly: bimanual is split into ~51 task directories
    while humanoid/single_arm have one or two, so a per-unit quota gave 56/60
    bimanual frames and made the montage look like a single-corpus dataset.
    """
    us = units()
    if not us:
        raise SystemExit(f"no LeRobot units under {DATASETS}")
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[tuple[Path, int]]] = {}
    for u in us:
        eps = episodes_of(u)
        if eps:
            cells.setdefault(cell_of(u), []).extend((u, e) for e in eps)
    if not cells:
        raise SystemExit("no episodes found")
    for v in cells.values():
        rng.shuffle(v)

    # Round-robin over cells so every embodiment/view appears, then top up from
    # whichever cells still have candidates.
    order = sorted(cells)
    out: list[Image.Image] = []
    got = collections.Counter()
    idx = {k: 0 for k in order}
    while len(out) < n and any(idx[k] < len(cells[k]) for k in order):
        for k in order:
            if len(out) >= n:
                break
            while idx[k] < len(cells[k]):
                u, ep = cells[k][idx[k]]
                idx[k] += 1
                v = any_video(u, ep)
                if v is None:
                    continue
                key = f"{u.relative_to(DATASETS)}".replace(os.sep, "__")
                im = grab(v, CACHE / f"{key}__{ep:06d}.jpg")
                if im is not None:
                    out.append(square(im, tile))
                    got[k] += 1
                    break
    if not out:
        raise SystemExit("could not extract any frame -- is ffmpeg working?")
    print("  frames per cell: "
          + ", ".join(f"{a}/{b}={got[(a, b)]}" for a, b in order))
    rng.shuffle(out)
    return out


def montage(frames: list[Image.Image], cols: int, rows: int, tile: int,
            dim: float, blur: float) -> Image.Image:
    """Dense grid variant: tile the frames, then knock the contrast/brightness
    down so overlaid text stays legible."""
    canvas = Image.new("RGB", (cols * tile, rows * tile), "white")
    for i in range(cols * rows):
        canvas.paste(frames[i % len(frames)], ((i % cols) * tile,
                                               (i // cols) * tile))
    if blur > 0:
        canvas = canvas.filter(ImageFilter.GaussianBlur(blur))
    canvas = ImageEnhance.Brightness(canvas).enhance(dim)
    canvas = ImageEnhance.Color(canvas).enhance(0.92)
    return canvas


def scatter_montage(frames: list[Image.Image], width: int, height: int,
                    count: int, seed: int, fade: float,
                    min_frac: float = 0.030, max_frac: float = 0.062
                    ) -> Image.Image:
    """Sparse scattered-thumbnail cloud on white (BridgeData V2 teaser style).

    Small tiles at random positions and slight random rotation, dense in the
    middle and thinning toward the edges. This reads much better than a dense
    grid behind panels: the white gaps give the foreground cards somewhere to
    sit, so no scrim is needed and the frames stay fully visible.
    """
    rng = random.Random(seed)
    canvas = Image.new("RGB", (width, height), "white")
    cx, cy = width / 2, height / 2
    # The panels occupy the middle, so tiles there would be hidden anyway. Place
    # them in an annulus that starts just inside the cards and thins outward:
    # radius is uniform-in-area over [inner, 1] and tile size AND opacity decay
    # with radius, which is what gives the reference figure its sparse fade
    # instead of a hard ring of thumbnails at the border.
    inner = 0.62
    placed = 0
    for i in range(count):
        src = frames[i % len(frames)]
        u = rng.random()
        rad = math.sqrt(inner ** 2 + u * (1.0 - inner ** 2))
        # near the inside of the annulus tiles are bigger; far out they shrink
        decay = 1.0 - 0.45 * (rad - inner) / (1.0 - inner)
        size = max(8, int(width * rng.uniform(min_frac, max_frac) * decay))
        tileim = src.resize((size, size), Image.LANCZOS)
        # fade grows with radius so the cloud dissolves toward the edges
        f = min(0.92, fade + 0.55 * (rad - inner) / (1.0 - inner))
        if f > 0:
            tileim = ImageEnhance.Brightness(tileim).enhance(1.0 + f * 0.9)
            tileim = ImageEnhance.Color(tileim).enhance(max(0.0, 1.0 - f * 0.7))
        # thin white keyline so overlapping tiles stay separable
        framed = Image.new("RGB", (size + 4, size + 4), "white")
        framed.paste(tileim, (2, 2))
        angle = rng.uniform(-22, 22)
        rot = framed.convert("RGBA").rotate(angle, expand=True,
                                            resample=Image.BICUBIC)
        # rotate() leaves transparent corners; build a matching alpha mask from a
        # solid rectangle so those corners are NOT pasted as opaque white squares
        # over neighbouring tiles.
        mask = Image.new("L", framed.size, 255).rotate(
            angle, expand=True, resample=Image.BICUBIC)
        ang = rng.uniform(0, 2 * math.pi)
        # stretch horizontally: the canvas is 16:10, a circular cloud would leave
        # the left/right margins empty
        x = cx + math.cos(ang) * rad * width * 0.60 - rot.width / 2
        y = cy + math.sin(ang) * rad * height * 0.62 - rot.height / 2
        if (x < 0 or y < 0 or x + rot.width > width
                or y + rot.height > height):
            continue        # keep whole tiles only -- no half-cut thumbnails
        canvas.paste(rot.convert("RGB"), (int(x), int(y)), mask)
        placed += 1
    print(f"  scatter: {placed} thumbnails placed in an annulus "
          f"(inner radius {inner})")
    return canvas


def load_evidence():
    audit = json.load(open(AUDIT))
    stats = json.load(open(STATS))
    evid = json.load(open(EVID))
    if not audit.get("scope_matches_design"):
        raise SystemExit("audit does not match design scope -- fix the data, "
                         "not the figure")
    return audit, stats, evid


def draw_panels(bg_path: Path, bg_size: tuple[int, int],
                bg_style: str = "scatter",
                strip: list[tuple[str, int, "Image.Image"]] | None = None
                ) -> None:
    """Four panels over the montage. Imported lazily so the montage step above can
    be tested without matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    audit, stats, evid = load_evidence()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "pdf.fonttype": 42,
    })

    W = 5.5      # = ICLR \linewidth; do NOT exceed it or LaTeX downscales text
    H = W * bg_size[1] / bg_size[0]
    fig = plt.figure(figsize=(W, H))
    base = fig.add_axes([0, 0, 1, 1])
    base.imshow(plt.imread(str(bg_path)), aspect="auto", interpolation="bilinear")
    base.set_axis_off()

    # No full-canvas scrim: a 0.86-alpha white sheet over 97% of the area hid the
    # montage almost entirely. Each panel now carries its own card, so the frames
    # stay visible in the gutters and margins.
    # On the scatter background the tiles already sit on white, so the cards can
    # be fully opaque with a dark keyline (BridgeData V2 does exactly this) --
    # that reads far crisper than a translucent card over a busy mosaic.
    SCATTER = bg_style == "scatter"
    CARD_ALPHA = 1.0 if SCATTER else 0.90
    CARD_EDGE = "#1A1A1A" if SCATTER else "#9E9E9E"
    CARD_LW = 1.3 if SCATTER else 0.5

    def card(x, y, w, h, alpha=None):
        base.add_patch(FancyBboxPatch(
            (x, y), w, h, transform=base.transAxes,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            facecolor="white",
            alpha=CARD_ALPHA if alpha is None else alpha,
            edgecolor=CARD_EDGE, lw=CARD_LW, zorder=3))

    def panel(rect, tag, heading):
        ax = fig.add_axes(rect, zorder=5)
        ax.patch.set_alpha(0.0)          # the card behind it supplies the white
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(labelsize=6.2, length=1.8, color="#AAAAAA")
        ax.set_title(f"{tag}  {heading}", fontsize=7.4, loc="left", pad=11)
        return ax

    # ------------------------------------------------------------------ layout
    # Everything vertical is laid out in INCHES and converted at the end. The
    # earlier version hardcoded axes fractions tuned for one canvas aspect, so
    # adding the strip (taller canvas) silently overlapped every card.
    FH = W * bg_size[1] / bg_size[0]        # figure height, inches
    band_top = 0.30                        # cloud visible above the title card
    t_h = 0.36                             # title card
    gap = 0.10
    p_axes = 0.80                          # plotting area of a panel
    p_pad_t, p_pad_b = 0.25, 0.30          # panel title / xlabel inside the card
    p_card = p_axes + p_pad_t + p_pad_b
    s_h = 0.62 if strip else 0.0           # strip frames
    s_lab = 0.20 if strip else 0.0         # strip labels
    s_head = 0.20 if strip else 0.0        # strip heading
    band_bot = 0.28

    need = (band_top + t_h + gap + p_card + gap + p_card
            + (gap + s_head + s_h + s_lab if strip else 0.0) + band_bot)
    if need > FH + 1e-6:
        raise SystemExit(
            f"layout needs {need:.2f}in but the canvas is {FH:.2f}in tall; "
            f"raise --rows (currently {bg_size[1] // (bg_size[0] // 10)})")
    slack = FH - need                      # spread leftovers into the cloud bands
    band_top += slack * 0.5
    band_bot += slack * 0.5

    def fy(inches_from_bottom: float) -> float:
        return inches_from_bottom / FH

    def fh(inches: float) -> float:
        return inches / FH

    y = band_bot
    s_frame_y = y + s_lab
    y += s_lab + s_h + s_head + (gap if strip else 0.0)
    bot_card_y = y
    y += p_card + gap
    top_card_y = y
    y += p_card + gap * 0.0                # title sits directly above
    t_card_y = FH - band_top - t_h

    T_Y, T_H = fy(t_card_y), fh(t_h)
    STRIPY, S_HEAD_Y = fy(s_frame_y), fy(s_frame_y + s_h + 0.03)
    ASPECT = bg_size[0] / bg_size[1]        # width fraction -> height fraction

    L, R, BW = 0.158, 0.623, 0.297
    BH = fh(p_axes)
    TOPY, BOTY = fy(top_card_y + p_pad_b), fy(bot_card_y + p_pad_b)
    CH = fh(p_card)
    card(0.048, fy(top_card_y), 0.437, CH)
    card(0.515, fy(top_card_y), 0.437, CH)
    card(0.048, fy(bot_card_y), 0.437, CH)
    card(0.515, fy(bot_card_y), 0.437, CH)

    title = (f"{evid['totals']['episode_instances']:,} episode instances  ·  "
             f"3 world models  ·  9 acceleration methods  ·  "
             f"{audit['grand_total']:,} rollouts")
    card(0.048, T_Y, 0.904, T_H)
    fig.text(0.5, T_Y + T_H * 0.62,
             "A physics benchmark for accelerated video world models",
             ha="center", va="center", fontsize=9.6, zorder=6)
    fig.text(0.5, T_Y + T_H * 0.24, title, ha="center", va="center",
             fontsize=6.8, color="#444444", zorder=6)

    # ---- (1) source corpus -------------------------------------------------
    ax = panel([L, TOPY, BW, BH], "(a)", "Source corpora")
    per = collections.Counter()
    for c in evid["cells"]:
        per[c["corpus"]] += c["episodes_selected"]
    items = per.most_common()[::-1]
    SHORT = {"AIST Bimanual (ALOHA)": "AIST Bimanual",
             "RoboCOIN (AIRBOT MMK2)": "RoboCOIN",
             "GR00T Teleop GR1": "GR00T GR1"}
    names = [SHORT.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(range(len(vals)), vals, color="#0072B2", alpha=0.85, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, f"{v}", va="center", fontsize=6.0,
                color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.0)
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("episodes", fontsize=6.4, labelpad=1)
    ax.spines["left"].set_visible(False)

    # ---- (2) distinct instructions per primitive --------------------------
    ax = panel([R, TOPY, BW, BH], "(b)", "Manipulation primitives")
    sk = sorted(stats["skills_primary"].items(), key=lambda kv: -kv[1])[:9][::-1]
    labs = [k for k, _ in sk]
    vals = [v for _, v in sk]
    ax.barh(range(len(vals)), vals, color="#009E73", alpha=0.85, height=0.62)
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v}", va="center", fontsize=6.0, color="#333333")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(labs, fontsize=6.0)
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(vals) * 3.0)
    ax.set_xlabel(f"distinct instructions of "
                  f"{stats['distinct_prompts']} total (log)", fontsize=6.4,
                  labelpad=1)
    ax.spines["left"].set_visible(False)

    # ---- (3) episode duration ---------------------------------------------
    ax = panel([L, BOTY, BW, BH], "(c)", "Episode duration")
    EMB = ["humanoid", "bimanual", "single_arm"]
    LAB = {"humanoid": "Humanoid", "bimanual": "Bimanual",
           "single_arm": "Single-arm"}
    COL = {"humanoid": "#009E73", "bimanual": "#D55E00",
           "single_arm": "#0072B2"}
    secs = {e: [] for e in EMB}
    for c in evid["cells"]:
        fps = (c["source_fps"] or [15.0])[0] or 15.0
        secs[c["embodiment"]].extend(n / fps for n in c["lengths"])
    bp = ax.boxplot([secs[e] for e in EMB], vert=False, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="#222222", lw=1.0),
                    whiskerprops=dict(lw=0.6, color="#999999"),
                    capprops=dict(lw=0.6, color="#999999"),
                    boxprops=dict(lw=0.5, edgecolor="#777777"))
    for patch, e in zip(bp["boxes"], EMB):
        patch.set_facecolor(COL[e])
        patch.set_alpha(0.5)
    ax.set_yticks(range(1, len(EMB) + 1))
    ax.set_yticklabels([LAB[e] for e in EMB], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xscale("log")
    allv = [x for v in secs.values() for x in v]
    ax.set_xlim(min(allv) * 0.65, max(allv) * 1.9)
    ax.set_xlabel(f"seconds, {sum(allv) / 3600:.1f} h total (log)",
                  fontsize=6.4, labelpad=1)
    ax.spines["left"].set_visible(False)

    # ---- (4) dense rollouts + acceleration coverage ------------------------
    ax = panel([R, BOTY, BW, BH], "(d)",
               "Rollouts per model and method")
    MODELS = ["dreamgen", "dreamdojo", "ctrlworld"]
    MLAB = {"dreamgen": "DreamGen", "dreamdojo": "DreamDojo",
            "ctrlworld": "Ctrl-World"}
    FAM = [("Dense", ["dense"], "#555555"),
           ("Token red.", ["sito", "itm"], "#CC79A7"),
           ("Caching", ["worldcache", "dicache", "fastercache"], "#0072B2"),
           ("Sparse attn.", ["pisa", "radial", "svg1", "svg2"], "#009E73")]
    by = collections.defaultdict(int)
    for c in audit["method_cells"]:
        by[(c["method"], c["model"])] += c["total"]
    ax.set_axis_off()
    nrow, ncol = len(FAM), len(MODELS)
    for j, m in enumerate(MODELS):
        ax.text((j + 0.5) / ncol, 1.005, MLAB[m], ha="center", va="bottom",
                fontsize=6.0, transform=ax.transAxes)
    for i, (fname, members, col) in enumerate(FAM):
        yc = 1 - (i + 0.5) / nrow
        ax.text(-0.02, yc, fname, ha="right", va="center", fontsize=6.0,
                color=col, transform=ax.transAxes)
        for j, m in enumerate(MODELS):
            n = sum(by[(x, m)] for x in members)
            xc = (j + 0.5) / ncol
            if n == 0:
                ax.text(xc, yc, "–", ha="center", va="center", fontsize=7,
                        color="#CCCCCC", transform=ax.transAxes)
                continue
            ax.add_patch(plt.Rectangle(
                ((j + 0.04) / ncol, 1 - (i + 0.92) / nrow), 0.92 / ncol,
                0.84 / nrow, transform=ax.transAxes, facecolor=col, alpha=0.14,
                edgecolor=col, lw=0.5))
            ax.text(xc, yc, f"{n:,}", ha="center", va="center", fontsize=6.6,
                    color=col, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, -0.17, "– = method not applicable to that model",
            ha="center", va="top", fontsize=5.8, color="#888888",
            transform=ax.transAxes)

    # ---- (e) qualitative strip: one GT frame per primitive -----------------
    # Panel (b) says HOW MANY instructions each primitive has; this says what it
    # LOOKS like. The strip is full-width because a per-row thumbnail inside (b)
    # would be ~2.4 mm tall in print -- a colour smudge, not a picture.
    if strip:
        n = len(strip)
        SX, SW = 0.048, 0.904
        tw = SW / n
        # The frames are square, so a width of `tw` (figure fractions) becomes a
        # height of tw * (W/FH) -- using the canvas aspect here would stretch them.
        th = tw * (W / FH)
        card(SX - 0.006, STRIPY - fh(s_lab) - 0.006, SW + 0.012,
             fh(s_lab) + th + fh(s_head) + 0.010)
        fig.text(SX + 0.004, S_HEAD_Y,
                 "(e)  One ground-truth frame per primitive, ordered as in (b)",
                 fontsize=7.4, ha="left", va="bottom", zorder=6)
        for i, (name, cnt, im) in enumerate(strip):
            iax = fig.add_axes([SX + i * tw + tw * 0.022, STRIPY,
                                tw * 0.956, th], zorder=5)
            iax.imshow(im)
            iax.set_xticks([])
            iax.set_yticks([])
            for s in iax.spines.values():
                s.set_color("#BBBBBB")
                s.set_linewidth(0.5)
            iax.set_xlabel(f"{name}\n{cnt} instr.", fontsize=5.3, labelpad=1.6,
                           color="#333333", linespacing=1.15)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_hero.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print("wrote figures/fig_hero.png + .pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg-style", choices=("scatter", "grid"), default="scatter",
                    help="scatter = sparse rotated-thumbnail cloud on white "
                         "(BridgeData V2 teaser look); grid = dense tiled mosaic")
    ap.add_argument("--frames", type=int, default=60,
                    help="distinct episodes sampled (equal quota per cell)")
    ap.add_argument("--scatter-count", type=int, default=210,
                    help="thumbnails placed in scatter mode (frames are reused)")
    ap.add_argument("--fade", type=float, default=0.30,
                    help="scatter mode: 0 = full colour, higher = washed out")
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--rows", type=int, default=10,
                    help="canvas height in tiles; 10 leaves room for the "
                         "qualitative strip, use 7 with --primitives 0")
    ap.add_argument("--primitives", type=int, default=9,
                    help="frames in the qualitative strip (0 disables it)")
    ap.add_argument("--tile", type=int, default=240)
    ap.add_argument("--dim", type=float, default=1.0,
                    help="grid mode only: >1 brightens toward white")
    ap.add_argument("--blur", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    frames = collect_frames(args.frames, args.tile, args.seed)
    if args.bg_style == "scatter":
        W, H = args.cols * args.tile, args.rows * args.tile
        bg = scatter_montage(frames, W, H, args.scatter_count, args.seed,
                             args.fade)
    else:
        bg = montage(frames, args.cols, args.rows, args.tile, args.dim,
                     args.blur)
    CACHE.mkdir(parents=True, exist_ok=True)
    bg_path = CACHE / "_montage.jpg"
    bg.save(bg_path, quality=92)
    print(f"montage[{args.bg_style}]: {len(frames)} distinct frames "
          f"-> {bg.size[0]}x{bg.size[1]}")

    strip = (primitive_frames(args.primitives, 420, args.seed)
             if args.primitives > 0 else None)

    draw_panels(bg_path, bg.size, args.bg_style, strip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
