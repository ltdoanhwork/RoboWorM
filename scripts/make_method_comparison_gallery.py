#!/usr/bin/env python3
"""Build the "same video across every acceleration method" comparison gallery.

Picks the BEST-looking `dense` predictions in `sampling_dataset/` (ranked by the
PhyWorld NR `phy_qual` score computed in `metrics/WorldScore/results/`), then
pulls the SAME task + episode index from every other acceleration method
(dicache / worldcache / fastercache / pisa / radial / svg1 / svg2 / sito / itm)
so each output dir is a like-for-like comparison of one episode.

Cells are (embodiment, view track, world model, split) so the gallery also spans
models: dreamgen, dreamdojo, ctrlworld and ctrlworld_4view_grid.

Output (default `assets/drive/method_comparison`):

    <embodiment>/<track>/<model>/<split>/<NN>_<task>__ep<idx>/
        gt.mp4                  ground truth (when the model tree ships one)
        dense.mp4               the reference prediction
        <method>.mp4            same episode, one file per acceleration method
        frames/<name>_fNNNNN.png    same frame indices for every method
        compare.png             labelled sheet: rows = methods, cols = frames
        info.json               task, episode, iter, phy_qual, per-method paths

    manifest.json / INDEX.md

Usage:
    python3 scripts/make_method_comparison_gallery.py --dry-run
    python3 scripts/make_method_comparison_gallery.py                 # 2 per cell
    python3 scripts/make_method_comparison_gallery.py --per-cell 3 --frames 8
    python3 scripts/make_method_comparison_gallery.py --only humanoid
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_real_video_gallery import (  # noqa: E402  (shared helpers)
    even, extract_frames, frame_indices, probe, slug, transcode,
)

REPO = Path(__file__).resolve().parents[1]
SD = REPO / "sampling_dataset"
NR_RESULTS = REPO / "metrics/WorldScore/results/dense_nr_v3_div30/results"
OUT_DEFAULT = REPO / "assets" / "drive" / "method_comparison"

# order = how rows are stacked in compare.png (family by family)
METHODS = [
    "dense",
    "dicache", "worldcache", "fastercache",   # caching
    "pisa", "radial", "svg1", "svg2",         # sparse attention
    "sito", "itm",                            # token pruning
]
FAMILY = {"dense": "reference", "dicache": "caching", "worldcache": "caching",
          "fastercache": "caching", "pisa": "sparse-attn", "radial": "sparse-attn",
          "svg1": "sparse-attn", "svg2": "sparse-attn", "sito": "pruning",
          "itm": "pruning"}

# preferred prediction file per model family, first match wins
PRED_NAMES = ["pred_all_views.mp4", "full_pred.mp4"]
GT_NAMES = ["gt_all_views.mp4", "full_gt.mp4"]

EP_RE = re.compile(r"episode_(\d+)")
# ctrlworld names its dirs episode_<task>__<index>
EP_TASK_RE = re.compile(r"^episode_(?P<task>.+)__(?P<ep>\d+)$")
# dreamdojo / dreamgen-singleview use <task>__episode_<index>
TASK_EP_RE = re.compile(r"^(?P<task>.+)__episode_(?P<ep>\d+)$")


@dataclass
class Rec:
    method: str
    video: Path
    gt: Path = None
    task: str = ""
    episode: int = -1
    iter: str = "-"


@dataclass
class Pick:
    embodiment: str
    track: str
    model: str
    split: str
    task: str
    episode: int
    iter: str
    score: float = None
    recs: dict = field(default_factory=dict)   # method -> Rec


def parse_video(path: Path, method: str):
    """Map any prediction mp4 in the sampling_dataset to a canonical key."""
    parts = list(path.parts)
    if "output" not in parts:
        return None
    i = parts.index("output")
    emb, rest = parts[i - 1], parts[i + 1:]
    if len(rest) < 3:
        return None
    track, model, tail = rest[0], rest[1], rest[2:]
    split = tail[0]
    if split not in ("makovian", "non_makovian"):
        return None
    tail = tail[1:]
    if not tail:
        return None

    it = next((c for c in tail if c.startswith("iter_")), "-")
    body = [c for c in tail if c != it]
    ep, task, view = None, "", ""

    for comp in body:
        stem = comp[:-4] if comp.endswith(".mp4") else comp
        m = TASK_EP_RE.match(stem) or EP_TASK_RE.match(stem)
        if m:
            task, ep = m.group("task"), int(m.group("ep"))
            break
        m = EP_RE.search(stem)
        if m and stem[: m.start()].rstrip("_") in ("", "episode"):
            ep = int(m.group(1))
            break
    if ep is None:
        return None

    # dreamgen multiview keeps the task in its own dir: iter/<task>/episode_N/file
    if not task:
        cands = [c for c in body if not c.endswith(".mp4") and not EP_RE.search(c)]
        if cands:
            task = cands[-1]
    # single_arm singleview dreamgen: iter/<view>/episode_N.mp4
    if track == "singleview" and model == "dreamgen" and emb == "single_arm":
        v = [c for c in body if not c.endswith(".mp4") and not EP_RE.search(c)]
        view = v[-1] if v else ""
        task = ""

    gt = None
    if path.name in PRED_NAMES:
        for g in GT_NAMES:
            if (path.parent / g).exists():
                gt = path.parent / g
                break
    key = (emb, track, model, split, it, task, ep, view)
    return key, Rec(method=method, video=path, gt=gt, task=task, episode=ep, iter=it)


def read_prompt(video: Path) -> str:
    """DreamGen writes a `<episode>.txt` sidecar with the conditioning prompt."""
    txt = video.with_suffix(".txt")
    if not txt.exists():
        return ""
    lines = [l.strip() for l in txt.read_text(errors="ignore").splitlines()]
    for i, l in enumerate(lines):
        if l.startswith("[Prompt]"):
            for nxt in lines[i + 1:]:
                if nxt and not nxt.startswith("["):
                    return nxt
    return next((l for l in lines if l and not l.startswith("[")), "")


# the multiview prompts wrap the instruction in a long grid-layout template
_TEMPLATES = [
    re.compile(r"shows that a robot (.+?)\.\s+The video is split", re.S),
    re.compile(r"robot performing:\s*(.+?)\.\s+The video shows", re.S),
]


def clean_prompt(prompt: str) -> str:
    for rx in _TEMPLATES:
        m = rx.search(prompt)
        if m:
            return m.group(1).strip()
    return prompt.strip()


_META_CACHE: dict = {}


def dataset_task(emb: str, track: str, split: str, ep: int) -> str:
    """Task of an episode from the source LeRobot meta (flat datasets only)."""
    key = (emb, track, split)
    if key not in _META_CACHE:
        f = REPO / "datasets" / emb / track / split / "meta" / "episodes.jsonl"
        table = {}
        if f.exists():
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                o = json.loads(line)
                t = o.get("source_task")
                if not t:
                    tasks = [x for x in (o.get("tasks") or [])
                             if x and x.strip().lower() not in {"valid", "invalid"}]
                    t = tasks[0] if tasks else ""
                table[int(o["episode_index"])] = str(t).replace("_", " ")
        _META_CACHE[key] = table
    return _META_CACHE[key].get(ep, "")


def resolve_task(p: "Pick") -> str:
    """Best human-readable label: dir name > source dataset meta > prompt."""
    if p.task and p.task not in ("episode", "global", ""):
        return p.task
    t = dataset_task(p.embodiment, p.track, p.split, p.episode)
    if t:
        return t
    pr = clean_prompt(read_prompt(p.recs["dense"].video))
    return re.sub(r"^(A|The)\s+(multi-view\s+)?(video|robot(ic)?\s+arm|robot)\s+",
                  "", pr).rstrip(".") or "episode"


def index_methods(methods: list) -> dict:
    """key -> {method: Rec} for every prediction video in the sampling_dataset."""
    index: dict = collections.defaultdict(dict)
    for m in methods:
        root = SD / m
        if not root.exists():
            continue
        for p in root.rglob("*.mp4"):
            n = p.name
            if n in GT_NAMES or n == "full_merged.mp4" or n.startswith("seg_"):
                continue
            parsed = parse_video(p, m)
            if not parsed:
                continue
            key, rec = parsed
            cur = index[key].get(m)
            # prefer pred_all_views over full_pred for the multi-view models
            if cur is None or (n in PRED_NAMES and cur.video.name in PRED_NAMES
                               and PRED_NAMES.index(n) < PRED_NAMES.index(cur.video.name)):
                index[key][m] = rec
    return index


def load_scores() -> dict:
    """dense video path (relative to sampling_dataset/dense) -> mean phy_qual."""
    if not NR_RESULTS.exists():
        return {}
    acc: dict = collections.defaultdict(list)
    for f in NR_RESULTS.glob("*.json"):
        try:
            x = json.loads(f.read_text())
        except Exception:
            continue
        if x.get("method") != "dense" or x.get("phy_qual") is None:
            continue
        acc[x["relative"]].append(float(x["phy_qual"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def select(index: dict, scores: dict, n: int, min_methods: int) -> list:
    """Best-scoring dense episodes per cell that also exist for other methods."""
    cells: dict = collections.defaultdict(list)
    for key, per in index.items():
        if "dense" not in per or len(per) - 1 < min_methods:
            continue
        emb, track, model, split, it, task, ep, view = key
        dense = per["dense"]
        try:
            rel = str(dense.video.relative_to(SD / "dense"))
        except ValueError:
            rel = None
        cells[(emb, track, model, split)].append(Pick(
            embodiment=emb, track=track, model=model, split=split,
            task=task or view or "episode", episode=ep, iter=it,
            score=scores.get(rel), recs=per,
        ))

    out = []
    for cell in sorted(cells):
        cands = cells[cell]
        for c in cands:  # readable, task-diverse labels for the flat cells
            c.task = resolve_task(c)
        scored = [c for c in cands if c.score is not None]
        pool = scored or cands
        # best quality first, then de-duplicate by task so a cell isn't 5x the
        # same task; fall back to more episodes of a task only if needed
        pool.sort(key=lambda c: (-(c.score or 0), c.task, c.episode))
        picked, used = [], set()
        for c in pool:
            if c.task in used:
                continue
            picked.append(c)
            used.add(c.task)
            if len(picked) == n:
                break
        for c in pool:
            if len(picked) == n:
                break
            if c not in picked:
                picked.append(c)
        out += picked
    return out


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

FONT_CANDIDATES = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/.local/share/fonts/DejaVuSans-Bold.ttf"),
]


def _font(size: int):
    from PIL import ImageFont
    for p in FONT_CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def build_compare_sheet(order: list, frames: dict, fdir: Path, dst: Path,
                        title: str, tile_h: int = 190) -> bool:
    """Labelled sheet: one row per method, one column per shared frame index."""
    from PIL import Image, ImageDraw

    rows = [(name, frames[name]) for name in order if frames.get(name)]
    if len(rows) < 2:
        return False
    ncol = min(len(f) for _, f in rows)
    LABEL_W, PAD = 118, 3
    TITLE_H, COLHDR_H = 30, 18

    first = Image.open(fdir / Path(rows[0][1][0]["file"]).name)
    tile_w = even(max(2, round(first.width * tile_h / first.height)))
    step_x, step_y = tile_w + PAD, tile_h + PAD

    W = LABEL_W + ncol * step_x + PAD
    H = TITLE_H + COLHDR_H + len(rows) * step_y + PAD
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), title, font=_font(15), fill="#111111")

    # column header = the shared sampling position. Frame numbers are only
    # meaningful when every clip has the same length; otherwise the stills are
    # matched by relative position in the rollout (0% .. 100%), so say that.
    ref = rows[0][1]
    exact = len({tuple(f["frame"] for f in fl[:ncol]) for _, fl in rows}) == 1
    for c in range(ncol):
        cx = LABEL_W + c * step_x + tile_w / 2
        s = (f"frame {ref[c]['frame']}" if exact else
             f"{round(100 * c / max(1, ncol - 1))}%")
        d.text((cx - d.textlength(s, font=_font(11)) / 2, TITLE_H + 3),
               s, font=_font(11), fill="#777777")

    y0 = TITLE_H + COLHDR_H
    for r, (name, fl) in enumerate(rows):
        y = y0 + r * step_y
        colour = {"reference": "#1F6FB2", "caching": "#2E7D4F",
                  "sparse-attn": "#B24A1F", "pruning": "#6A3D9A",
                  "gt": "#333333"}.get(FAMILY.get(name, "gt"), "#333333")
        d.rectangle((0, y, 4, y + tile_h - 1), fill=colour)
        d.text((11, y + 4), name, font=_font(14), fill=colour)
        fam = FAMILY.get(name, "ground truth")
        d.text((11, y + 22), fam if name != "gt" else "ground truth",
               font=_font(10), fill="#999999")
        for c in range(ncol):
            im = Image.open(fdir / Path(fl[c]["file"]).name).convert("RGB")
            sheet.paste(im.resize((tile_w, tile_h), Image.LANCZOS),
                        (LABEL_W + c * step_x, y))
        if r:  # separator above every row but the first
            d.line((LABEL_W, y - 2, W - PAD, y - 2), fill="#E2E2E2", width=1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dst)
    return True


def write_pick(p: Pick, out_root: Path, rank: int, args) -> dict:
    name = f"{rank:02d}_{slug(p.task, 40)}__ep{p.episode:06d}"
    d = out_root / p.embodiment / p.track / p.model / p.split / name
    d.mkdir(parents=True, exist_ok=True)

    dense = p.recs["dense"]
    entries: dict = {}
    order: list = []

    # ground truth first (same episode, straight from the dense tree)
    if dense.gt and dense.gt.exists():
        dst = d / "gt.mp4"
        if dst.exists() or transcode(dense.gt, dst, args.max_height, args.crf):
            entries["gt"] = {"file": dst.name, "source": str(dense.gt.relative_to(REPO)),
                             "family": "gt", **probe(dst)}
            order.append("gt")

    for m in METHODS:
        rec = p.recs.get(m)
        if rec is None:
            continue
        dst = d / f"{m}.mp4"
        if not dst.exists() and not transcode(rec.video, dst, args.max_height, args.crf):
            print(f"    [WARN] transcode failed: {rec.video}")
            continue
        entries[m] = {"file": dst.name, "source": str(rec.video.relative_to(REPO)),
                      "family": FAMILY[m], **probe(dst)}
        order.append(m)

    # ---- frames: stills at the SAME relative rollout position for every method
    # (identical frame numbers whenever the clips share a length, which is the
    # normal case; otherwise matched by position — see `frames_exact`)
    frames: dict = {}
    if args.frames > 0:
        fdir = d / "frames"
        for nm in order:
            vid = d / entries[nm]["file"]
            n = probe(vid).get("frames") or 0
            want = frame_indices(n, args.frames)
            have = [{"file": f"frames/{nm}_f{i:05d}.png", "frame": i,
                     "t": round(i / (probe(vid).get("fps") or 1), 3)}
                    for i in want]
            if want and all((fdir / Path(h["file"]).name).exists() for h in have):
                fr = have                        # already extracted, reuse
            else:
                fr = extract_frames(vid, fdir, args.frames, nm)
            if fr:
                frames[nm] = fr
                entries[nm]["frames"] = [f["frame"] for f in fr]
    exact = len({tuple(v.get("frames") or ()) for v in entries.values()}) == 1

    sheet = None
    if frames and len(frames) >= 2:
        s = d / "compare.png"
        title = (f"{p.embodiment} / {p.track} / {p.model} / {p.split}   "
                 f"{p.task}  (ep {p.episode:06d}"
                 + (f", phy_qual {p.score:.1f})" if p.score is not None else ")"))
        if s.exists() or build_compare_sheet(order, frames, fdir, s, title,
                                             args.tile_height):
            sheet = s.name

    prompt = clean_prompt(read_prompt(dense.video))
    info = {
        "embodiment": p.embodiment, "track": p.track, "model": p.model,
        "split": p.split, "task": p.task, "episode_index": p.episode,
        "iter": p.iter, "phy_qual_dense": p.score, "prompt": prompt,
        "n_methods": len([m for m in order if m != "gt"]),
        "frames_exact": exact,
        "compare_sheet": sheet, "order": order, "clips": entries,
        "dir": str(d.relative_to(out_root)),
    }
    (d / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def write_index(manifest: list, out: Path):
    lines = [
        "# Same-episode comparison across acceleration methods",
        "",
        "Each dir holds ONE episode rendered by every acceleration method that has",
        "it, so a row-wise read is a like-for-like comparison. `dense.mp4` is the",
        "reference; `gt.mp4` is the ground truth where the model tree ships one.",
        "",
        "`compare.png` is the quick look: rows = methods (colour-coded by family —",
        "blue reference, green caching, orange sparse-attention, purple pruning),",
        "columns = the SAME frame indices for every method. `frames/<method>_fNNNNN.png`",
        "are those stills individually.",
        "",
        "Where a cell's clips all share a length the column header is the exact frame",
        "number; in the few cells where the dense rollout was rendered at a different",
        "fps/length than the accelerated ones the header is the relative rollout",
        "position (0%..100%) instead — `info.json` records this as `frames_exact`.",
        "",
        "Episodes were ranked by the PhyWorld NR `phy_qual` of the DENSE prediction",
        "(`metrics/WorldScore/results/dense_nr_v3_div30`), so these are the cleanest",
        "dense outputs — a method degrading here is a real degradation, not a bad",
        "starting point. Cells span embodiment x view-track x world model x split.",
        "",
        "Regenerate with `python3 scripts/make_method_comparison_gallery.py`.",
        "",
    ]
    cells = collections.defaultdict(list)
    for m in manifest:
        cells[(m["embodiment"], m["track"], m["model"], m["split"])].append(m)
    for k in sorted(cells):
        lines += [f"## {' / '.join(k)}", "",
                  "| # | task | episode | phy_qual (dense) | methods | dir |",
                  "|---|------|---------|------------------|---------|-----|"]
        for i, m in enumerate(cells[k], 1):
            q = "-" if m["phy_qual_dense"] is None else f"{m['phy_qual_dense']:.1f}"
            ms = ", ".join(x for x in m["order"] if x != "gt")
            lines.append(f"| {i} | {m['task']} | {m['episode_index']:06d} | {q} | "
                         f"{ms} | `{m['dir']}` |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--per-cell", type=int, default=2,
                    help="episodes per (embodiment, track, model, split) cell")
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these embodiments")
    ap.add_argument("--models", nargs="*", default=None,
                    help="restrict to these world models (dreamgen, dreamdojo, ...)")
    ap.add_argument("--min-methods", type=int, default=4,
                    help="skip episodes covered by fewer acceleration methods")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--tile-height", type=int, default=190)
    ap.add_argument("--max-height", type=int, default=480)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print("[1/3] indexing sampling_dataset ...")
    index = index_methods(METHODS)
    print(f"      {len(index)} episode keys")
    print("[2/3] loading PhyWorld NR dense scores ...")
    scores = load_scores()
    print(f"      {len(scores)} scored dense videos")

    picks = select(index, scores, args.per_cell, args.min_methods)
    if args.only:
        picks = [p for p in picks if p.embodiment in args.only]
    if args.models:
        picks = [p for p in picks if p.model in args.models]

    print(f"[3/3] {len(picks)} episodes selected")
    manifest, rank = [], collections.Counter()
    for p in picks:
        cell = (p.embodiment, p.track, p.model, p.split)
        rank[cell] += 1
        ms = [m for m in METHODS if m in p.recs]
        q = "  -  " if p.score is None else f"{p.score:5.1f}"
        print(f"  {'/'.join(cell):52s} q={q} ep{p.episode:06d} "
              f"{len(ms):2d}m {p.task[:34]}")
        if not args.dry_run:
            manifest.append(write_pick(p, args.out, rank[cell], args))

    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_index(manifest, args.out / "INDEX.md")

    def rel(x: Path) -> str:
        try:
            return str(x.relative_to(REPO))
        except ValueError:
            return str(x)

    print(f"\n[ok] {len(manifest)} comparisons -> {rel(args.out)}")
    print(f"[ok] {rel(args.out / 'manifest.json')}")
    print(f"[ok] {rel(args.out / 'INDEX.md')}")


if __name__ == "__main__":
    main()
