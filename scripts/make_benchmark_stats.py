#!/usr/bin/env python3
"""Measure the REAL benchmark statistics from `datasets/` (no hand-typed numbers).

Counts episodes / views / prompts / skills / scenes per
(embodiment x view-track x markovian-split) cell, straight off what is actually
on disk, and prints both a console table and a LaTeX-ready block.

Each LeRobot unit = a dir containing `meta/episodes.jsonl`. bimanual and
humanoid-multiview are partitioned into per-task units that REUSE episode
indices, so everything is keyed by (unit, episode_index) — keying by index
alone silently collapses them.

    python3 scripts/make_benchmark_stats.py
    python3 scripts/make_benchmark_stats.py --json assets/figures/benchmark_stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets"

EMBODIMENTS = ["single_arm", "bimanual", "humanoid"]
TRACKS = ["singleview", "multiview"]
SPLITS = ["makovian", "non_makovian"]

# source dataset per (embodiment, track) — traced from the prepare/download
# scripts and the upstream dataset cards kept in-tree (see docs/PROVENANCE)
SOURCE = {
    ("single_arm", "multiview"): "DROID (cadene/droid_1.0.1)",
    ("single_arm", "singleview"): "RoboChallenge (Traly/RoboChallenge-lerobot)",
    ("bimanual", "multiview"): "AIST Bimanual Manipulation (ALOHA)",
    # singleview is NOT a separate source: it is cam_high of the multiview
    # episodes (scripts/export_bimanual_singleview_cam_high.py symlinks them)
    ("bimanual", "singleview"): "AIST Bimanual Manipulation (ALOHA)",
    ("humanoid", "singleview"): "NVIDIA PhysicalAI-Robotics-GR00T-Teleop-GR1",
    ("humanoid", "multiview"): "RoboCOIN (AIRBOT MMK2)",
}

WAIST = re.compile(r"^(un)?locked waist:\s*", re.I)
EPNUM = re.compile(r"episode_(\d+)")
# label noise in the upstream GR1 / DROID annotations — these are NOT tasks
JUNK = re.compile(r"^(invalid( task)?|no action|fail to .*|)$", re.I)
# GR1 scene labels (`grid_Hotel`) are SCENES, not instructions
SCENE_LABEL = re.compile(r"^grid_", re.I)
# "Use the eraser to wipe ..." / "brush screws into ..." — verb is not first
TOOL_USE = re.compile(r"^use (the |a )?[\w\s]+? to (?P<verb>[a-z]+)", re.I)

# manipulation-primitive lexicon: verb (or verb phrase) -> canonical skill.
# Only used to GROUP the measured prompts; every prompt's leading verb that is
# not here is reported as `unmapped` so the taxonomy stays auditable.
SKILL = {
    "pick": "pick & place", "place": "pick & place", "put": "pick & place",
    "take": "pick & place", "get": "pick & place", "lift": "pick & place",
    "grab": "pick & place", "drop": "pick & place", "throw": "pick & place",
    "storage": "pick & place", "store": "pick & place",
    "move": "push / slide", "push": "push / slide", "slide": "push / slide",
    "shift": "push / slide", "pull": "push / slide", "drag": "push / slide",
    "open": "open / close", "close": "open / close",
    "remove": "remove / extract", "extract": "remove / extract",
    "unstack": "remove / extract", "unhang": "remove / extract",
    "uncover": "remove / extract", "empty": "remove / extract",
    "fold": "fold / unfold", "unfold": "fold / unfold",
    "flip": "fold / unfold", "turn": "fold / unfold",
    "straighten": "fold / unfold", "lay": "fold / unfold",
    "stack": "stack", "pour": "pour", "stir": "stir / mix", "mix": "stir / mix",
    "wipe": "wipe / clean", "clean": "wipe / clean", "sweep": "wipe / clean",
    "insert": "insert / assemble", "assemble": "insert / assemble",
    "press": "press / actuate", "switch": "press / actuate",
    "rotate": "press / actuate", "twist": "press / actuate",
    "hang": "hang / cover", "cover": "hang / cover",
    "handover": "handover", "transfer": "pick & place", "separate": "sort",
    "sort": "sort", "route": "route", "draw": "draw / trace",
    "doodle": "draw / trace", "trace": "draw / trace",
    "brush": "wipe / clean", "spell": "draw / trace", "write": "draw / trace",
    "spoon": "pour", "scoop": "pour", "fetch": "pick & place",
    "build": "stack", "fit": "insert / assemble", "hook": "insert / assemble",
    "tape": "insert / assemble", "hit": "press / actuate",
    "click": "press / actuate", "roll": "push / slide", "make": "push / slide",
}

# source-annotation typos, normalised before verb lookup
TYPO = [(re.compile(r"^clover\b", re.I), "cover")]

# verb+particle overrides — "turn ON a switch" is actuation, "turn OVER a towel"
# is a fold/flip. Checked before the bare-verb table above.
PHRASE = [
    (re.compile(r"\bturn (on|off)\b", re.I), "press / actuate"),
    (re.compile(r"\bturn (over|upside)\b", re.I), "fold / unfold"),
    (re.compile(r"\bhand (it |the .+ )?over\b|\bhandover\b", re.I), "handover"),
    (re.compile(r"\bspoon (of|to transfer)\b", re.I), "pour"),
    (re.compile(r"\bpick (it |the .+ )?up\b.*\bput\b", re.I), "pick & place"),
]

# A prompt may describe several primitives ("remove the pen from the box AND put
# it on the table"). Split on clause boundaries and label each clause by its own
# leading verb, so multi-step (non-Markovian) prompts are not collapsed to one
# skill. Splitting is what keeps nouns that happen to be verbs ("put the SPOON
# in the cup") from being counted as a second skill.
CLAUSE = re.compile(r",|\band then\b|\bthen\b|\band\b", re.I)


def units(emb: str, track: str, split: str) -> list:
    root = DS / emb / track / split
    if not root.exists():
        return []
    return sorted(p.parent.parent for p in root.rglob("meta/episodes.jsonl"))


def present(unit: Path) -> set:
    v = unit / "videos"
    return {int(EPNUM.search(p.name).group(1))
            for p in v.rglob("episode_*.mp4")} if v.exists() else set()


def views(unit: Path) -> list:
    f = unit / "meta" / "info.json"
    if not f.exists():
        return []
    i = json.loads(f.read_text())
    return sorted(k for k, v in i.get("features", {}).items()
                  if v.get("dtype") == "video")


def instruction(o: dict) -> str:
    ts = [t for t in (o.get("tasks") or [])
          if t and t.strip().lower() not in {"valid", "invalid"}]
    t = ts[0].strip() if ts else (o.get("source_task") or "").replace("_", " ").strip()
    body = WAIST.sub("", t).strip()
    if JUNK.match(body) or SCENE_LABEL.match(body):
        return ""                  # annotation noise / scene label, not an instruction
    return t


def _clause_skill(c: str) -> str:
    """Canonical skill for ONE clause, or '' if it names no primitive."""
    for rx, skill in PHRASE:       # verb+particle wins over the bare verb
        if rx.search(c):
            return skill
    m = TOOL_USE.match(c)          # "use the eraser to WIPE the markings"
    if m and m.group("verb") in SKILL:
        return SKILL[m.group("verb")]
    for w in re.findall(r"[a-z]+", c)[:2]:   # leading verb (or word after article)
        if w in SKILL:
            return SKILL[w]
    return ""


def skills_of(prompt: str) -> list:
    """All primitives a prompt asks for, in order. [] => unmapped."""
    p = WAIST.sub("", prompt).lower().strip()
    for rx, fix in TYPO:
        p = rx.sub(fix, p)
    out = []
    for c in CLAUSE.split(p):
        s = _clause_skill(c.strip())
        if s and s not in out:
            out.append(s)
    if not out:                    # no clause parsed -> try the whole string
        s = _clause_skill(p)
        if s:
            out.append(s)
    return out


def skill_of(prompt: str) -> str:
    """Primary (first) primitive — used for single-label reporting."""
    s = skills_of(prompt)
    return s[0] if s else "unmapped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cells, allp, allsk = [], set(), collections.Counter()
    for emb in EMBODIMENTS:
        for tr in TRACKS:
            for sp in SPLITS:
                us = units(emb, tr, sp)
                if not us:
                    continue
                eps = 0
                vids = 0
                pr = collections.Counter()
                vset = set()
                for u in us:
                    keep = present(u)
                    eps += len(keep)
                    vids += len(list((u / "videos").rglob("episode_*.mp4"))) \
                        if (u / "videos").exists() else 0
                    vset.add(len(views(u)))
                    for line in (u / "meta" / "episodes.jsonl").read_text().splitlines():
                        if not line.strip():
                            continue
                        o = json.loads(line)
                        if int(o["episode_index"]) not in keep:
                            continue
                        t = instruction(o)
                        if t:
                            pr[t] += 1
                sk = collections.Counter(skill_of(t) for t in pr)
                allp |= set(pr)
                allsk.update(sk)
                cells.append(dict(
                    embodiment=emb, track=tr, split=sp, units=len(us),
                    episodes=eps, videos=vids,
                    views=sorted(vset)[-1] if vset else 0,
                    prompts=len(pr), skills=len([k for k in sk if k != "unmapped"]),
                    source=SOURCE.get((emb, tr), "?")))

    w = "{:<10} {:<10} {:<13} {:>5} {:>6} {:>6} {:>5} {:>8} {:>7}"
    print(w.format("embodiment", "track", "split", "tasks", "eps", "vids",
                   "views", "prompts", "skills"))
    print("-" * 82)
    for c in cells:
        print(w.format(c["embodiment"], c["track"], c["split"], c["units"],
                       c["episodes"], c["videos"], c["views"], c["prompts"],
                       c["skills"]))
    print("-" * 82)
    tot_eps = sum(c["episodes"] for c in cells)
    tot_vid = sum(c["videos"] for c in cells)
    print(w.format("TOTAL", "", "", sum(c["units"] for c in cells), tot_eps,
                   tot_vid, 0, len(allp), len([k for k in allsk if k != "unmapped"])))

    print("\n== skills (canonical manipulation primitives) ==")
    prim = collections.Counter(skill_of(t) for t in allp)      # 1 label / prompt
    multi = collections.Counter(s for t in allp for s in skills_of(t))
    nmulti = sum(1 for t in allp if len(skills_of(t)) > 1)
    print(f"   {'skill':<20} {'primary':>8} {'mentions':>9}")
    for s, n in prim.most_common():
        print(f"   {s:<20} {n:>8} {multi[s]:>9}")
    print(f"   {'-'*20} {'-'*8} {'-'*9}")
    print(f"   {'distinct skills':<20} "
          f"{len([k for k in prim if k != 'unmapped']):>8} "
          f"{len([k for k in multi if k != 'unmapped']):>9}")
    print(f"   multi-primitive prompts: {nmulti} / {len(allp)} "
          f"({100 * nmulti / max(len(allp), 1):.0f}%)")
    # taxonomy-free control: how many distinct leading verbs are actually there?
    lead = collections.Counter()
    for t in allp:
        wds = re.findall(r"[a-z]+", WAIST.sub("", t).lower())
        if wds:
            lead[wds[0]] += 1
    print(f"   raw distinct leading verbs (no taxonomy): {len(lead)} "
          f"-> grouped into {len([k for k in prim if k != 'unmapped'])} primitives")
    if prim.get("unmapped"):
        print("   UNMAPPED (fix the lexicon):")
        for t in sorted(allp):
            if skill_of(t) == "unmapped":
                print(f"      {t}")

    # scenes: GR1 ships an explicit scene label per task_index
    f = DS / "humanoid/singleview/makovian/meta/tasks.jsonl"
    scenes = sorted({json.loads(l)["task"].split("grid_", 1)[1]
                     for l in f.read_text().splitlines() if "grid_" in l})
    print(f"\n== scenes (GR1 humanoid, explicit `grid_*` labels): {len(scenes)} ==")
    print("   " + ", ".join(scenes[:12]) + ", ...")

    out = dict(cells=cells, total_episodes=tot_eps, total_videos=tot_vid,
               distinct_prompts=len(allp),
               skills_primary={k: v for k, v in prim.most_common()},
               skills_mentions={k: v for k, v in multi.most_common()},
               multi_primitive_prompts=nmulti,
               prompt_skills={t: skills_of(t) for t in sorted(allp)},
               gr1_scenes=scenes,
               source_datasets=sorted(set(SOURCE.values())))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        try:
            shown = args.json.relative_to(REPO)
        except ValueError:
            shown = args.json
        print(f"\n[ok] {shown}")


if __name__ == "__main__":
    main()
