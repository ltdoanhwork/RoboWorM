"""Generate `augment_manifest.json` — the plan for the augment benchmark track.

Everything is DISCOVERED from the real datasets / Ctrl-World annotation trees
(repo rule: never hardcode episode lists). For each config we pick episodes,
derive the object to recolor from that episode's own instruction, and emit the
edit instruction + the prompt that inference should actually use.

Scope (fixed by the augment plan, see .cursor/rules/augment-track-layout.mdc):

  DreamGen — 8 configs = {single_arm, bimanual} x {singleview, multiview}
             x {makovian, non_makovian}; humanoid EXCLUDED.
             15 videos per config, in 3 branches of 5:
               edit_only   : recolor the object on the conditioning first frame,
                             ORIGINAL prompt kept.
               prompt_only : frame untouched, prompt REPLACED by a harder
                             counterfactual + DreamGen prompt refiner on.
               combo       : recolor + prompt rewritten to match the new color,
                             refiner on.

  Ctrl-World — 6 configs = {single_arm, bimanual, humanoid} x multiview
               x {makovian, non_makovian}. 10 videos per config, RECOLOR only
               (states/joints/actions untouched). Where a split has fewer than
               10 real episodes (bimanual/humanoid) episodes are reused with a
               different recolor, tagged by `variant`, so every output is distinct.

Usage:
  python scripts/augment/gen_manifest.py            # write augment_manifest.json
  python scripts/augment/gen_manifest.py --check    # validate only, no write
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "augment_manifest.json"
DATASETS = ROOT / "datasets"
CW_EXAMPLES = ROOT / "models" / "Ctrl-World" / "dataset_example"

SPLITS = ("makovian", "non_makovian")
BRANCHES = ("edit_only", "prompt_only", "combo")
# dreamgen_bimanual_singleview ONLY gets a 4th branch: the conditioning frame
# stays the episode's own (untouched), but the prompt names a DIFFERENT real
# task/object borrowed from elsewhere in the dataset — tests whether the model
# actually follows the text or just continues whatever it saw in the frame.
BIMANUAL_SV_BRANCHES = BRANCHES + ("task_swap",)
PER_BRANCH = 5
CW_PER_CONFIG = 10

# Distinct, unambiguous recolor targets (avoid near-duplicates like "grey"/"gray").
COLORS = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "cyan", "white", "black"]

# Longest-match-first vocabulary of manipulable objects seen in these datasets.
OBJECT_VOCAB = [
    "cardboard box", "zip bag", "bottle part", "light switch", "small gear", "large gear",
    "medium gear", "lan cable", "8pin cable", "bath towel", "big towel", "blue towel",
    "green towel", "orange towel", "yellow towel", "light blue towel", "chinese spoon",
    "white case", "dust pan", "trash bin", "coffee pod", "highlighter", "screwdriver",
    "coffeemaker", "cardboard", "toolbox", "dustpan", "hammer", "washer", "sponge",
    "tissue", "marker", "bottle", "drawer", "switch", "spoon", "shaft", "screw",
    "brush", "towel", "block", "cable", "gear", "bowl", "plate", "tray", "cart",
    "cube", "cup", "mug", "pen", "lid", "box", "pot", "pan", "can", "jar", "pod",
    "tape", "case", "reel", "cloth", "rod", "board", "buttons", "button", "car",
    "mango", "pomegranate", "fruit", "apple", "duvet", "hub", "coaster",
    "paper", "tube", "bag", "cloth", "sponge", "knife", "fork", "bread", "banana",
]
# Color adjectives that may sit in front of an object in the source instruction.
COLOR_WORDS = COLORS + ["grey", "gray", "brown", "light blue", "metallic", "clear", "silver", "golden"]
# Non-color adjectives/materials that also bind tightly to the noun. A recolor
# prompt must place the new color BEFORE this chain ("red rubber shaft", not
# "rubber red shaft").
MODIFIER_WORDS = [
    "rubber", "square", "cylindrical", "building", "large", "small", "medium", "big",
    "long", "short", "wide", "narrow", "open", "closed", "upright", "plastic", "wooden",
    "metal", "paper", "coffee", "lan", "light", "bath", "empty", "full",
]
# Prepositions that mark the end of the direct object and the start of the target
# location. "place the cup on the coaster" -> the object is the CUP, not the coaster.
# NOTE: "out"/"off" are deliberately absent — they are verb particles ("pull out a
# tissue"), and treating them as prepositions would cut the object away.
PREPOSITIONS = [
    "on", "onto", "in", "into", "inside", "to", "from", "with", "at", "over", "under",
    "next", "toward", "towards", "against", "near", "beside", "behind",
]
# Verbs that introduce the direct object (used when the vocabulary misses the noun).
ACTION_VERBS = [
    "pick up", "put", "place", "move", "push", "pull", "remove", "open", "close", "lift",
    "insert", "hook", "fold", "wipe", "turn on", "turn off", "turn", "press", "grab",
    "take out", "take", "slide", "stack", "pour", "hit", "tape", "scoop", "fit", "find",
    "hang", "throw", "handover", "brush", "screw", "unfold", "discard", "drop",
]
_STOPWORDS = {"the", "a", "an", "it", "them", "this", "that", "and", "then", "your", "his", "her",
              "of", "some", "any", "one", "two", "both", "all", "up", "down", "out", "off", "back"}
# Adjective-ish words that must not be returned as a standalone "object".
_NOT_A_NOUN = set(COLORS) | {"grey", "gray", "brown", "plain", "clear", "metallic", "silver",
                             "golden", "left", "right", "top", "bottom", "front", "rear"}

# Harder / counterfactual prompts for the prompt_only branch. `{obj}` is the
# episode's own object so the new prompt stays physically plausible, while the
# described behaviour is deliberately NOT what the recorded actions do.
PROMPT_PERTURBATIONS = [
    "{agent} lifts the {obj} high above the table and holds it in the air.",
    "{agent} hovers over the {obj} without ever touching it.",
    "{agent} slides the {obj} to the left edge of the table without lifting it.",
    "{agent} knocks the {obj} off the table onto the floor.",
    "{agent} moves the {obj} back and forth twice, then leaves it where it started.",
]


def _agent(embodiment: str) -> str:
    return "A robot arm" if embodiment == "single_arm" else "The bimanual robot"


def recolor(obj: str, color: str) -> str:
    return f"change the color of the {obj} to {color}, keep its shape and position"


def _bimanual_task_caption(task_name: str) -> str:
    """Mirror augment_prompt.bimanual_task_caption / the dense
    bimanual_task_to_caption template — keep in sync if that template changes."""
    name = task_name.replace("_", " ")
    name = name[0].upper() + name[1:] if name else name
    return f"A bimanual robot performing the task: {name}."


def _direct_object_segment(text: str) -> str:
    """Text up to the first preposition — where the DIRECT object lives.

    "place the cup on the coaster" -> "place the cup". Without this the vocabulary
    search happily returns the target location ("coaster") instead of the object
    the robot actually manipulates.
    """
    m = re.search(rf"\b(?:{'|'.join(PREPOSITIONS)})\b", text)
    return text[:m.start()] if m else text


def _vocab_match(text: str) -> str | None:
    """Longest vocabulary noun in `text`, with any leading color adjective."""
    for obj in OBJECT_VOCAB:  # OBJECT_VOCAB is longest-first
        m = re.search(rf"\b(?:({'|'.join(COLOR_WORDS)})\s+)?({re.escape(obj)})\b", text)
        if m:
            return f"{m.group(1)} {m.group(2)}" if m.group(1) else m.group(2)
    return None


def _verb_object_match(text: str) -> str | None:
    """Noun phrase right after an action verb, for objects outside the vocabulary."""
    m = re.search(
        rf"\b(?:{'|'.join(ACTION_VERBS)})\b\s+(?:the|a|an)?\s*"
        r"([a-z0-9]+(?:\s+[a-z0-9]+)?)", text)
    if not m:
        return None
    words = [w for w in m.group(1).split() if w not in _STOPWORDS and w not in PREPOSITIONS]
    # A bare adjective ("white", "plain") is not a usable recolor target.
    while words and words[-1] in _NOT_A_NOUN:
        words.pop()
    return " ".join(words) or None


def extract_object(instruction: str, task: str = "") -> str:
    """Best-effort object noun (plus any color adjective) from an instruction.

    Order matters: the direct-object segment of the real instruction wins, then
    the whole instruction, then the task name, then a verb-based guess. Falls back
    to the generic "object" — a recolor instruction on "the object" still works
    for Qwen-Image-Edit, it just gives the editor less to latch onto.
    """
    instr = (instruction or "").lower()
    task_text = task.replace("__", " ").replace("_", " ").lower()
    for text in (_direct_object_segment(instr), instr,
                 _direct_object_segment(task_text), task_text):
        if not text.strip():
            continue
        hit = _vocab_match(text)
        if hit:
            return hit
    for text in (instr, task_text):
        hit = _verb_object_match(text) if text.strip() else None
        if hit:
            return hit
    return "object"


def _strip_color(obj: str) -> str:
    """Drop a leading color adjective so we can re-prefix the NEW color."""
    for c in sorted(COLOR_WORDS, key=len, reverse=True):
        if obj.startswith(f"{c} "):
            return obj[len(c) + 1:]
    return obj


def recolored_prompt(instruction: str, obj: str, color: str, agent: str) -> str:
    """Original instruction with the object renamed to its new color.

    The color is inserted BEFORE any modifier chain bound to the noun, so we get
    "Hook the red rubber shaft", not "Hook the rubber red shaft".
    """
    base = (instruction or "").strip().rstrip(".")
    bare = _strip_color(obj)
    if base:
        # Widen the match left over color/modifier adjectives already present.
        mods = "|".join(COLOR_WORDS + MODIFIER_WORDS)
        pattern = rf"\b((?:(?:{mods})\s+)*){re.escape(bare)}\b"
        m = re.search(pattern, base, flags=re.IGNORECASE)
        if m:
            existing_mods = re.sub(rf"\b(?:{'|'.join(COLOR_WORDS)})\s+", "", m.group(1),
                                   flags=re.IGNORECASE)
            replacement = f"{color} {existing_mods}{bare}".replace("  ", " ")
            out = base[:m.start()] + replacement + base[m.end():]
            return f"{out[0].upper()}{out[1:]}."
    return f"{agent} interacts with the {color} {bare}."


def _read_episodes_jsonl(meta_dir: Path) -> dict[int, str]:
    """episode_index -> first task string (the human instruction)."""
    f = meta_dir / "episodes.jsonl"
    if not f.exists():
        return {}
    out: dict[int, str] = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks = [t for t in (row.get("tasks") or []) if t and t.lower() != "valid"]
        out[int(row["episode_index"])] = tasks[0] if tasks else ""
    return out


def _episode_ids(task_root: Path, cam: str) -> list[int]:
    ids = []
    for chunk in sorted(task_root.glob("videos/chunk-*")):
        for mp4 in sorted((chunk / cam).glob("episode_*.mp4")):
            ids.append(int(mp4.stem.split("_")[1]))
    return sorted(ids)


def _episode_ids_all_cams(task_root: Path, cams: list[str]) -> list[int]:
    """Episodes that have an mp4 for EVERY camera.

    The multiview preps build a grid from all cameras and silently skip episodes
    where one view is missing, so the manifest must apply the same filter or it
    would reference an episode that has no dense first frame (and cannot get one).
    """
    sets = [set(_episode_ids(task_root, c)) for c in cams]
    return sorted(set.intersection(*sets)) if sets and all(sets) else []


# ===================================================================== DreamGen


def _spread(items: list, n: int) -> list:
    """Take n items spread evenly across the list (diversity over adjacency)."""
    if not items:
        return []
    if len(items) <= n:
        return [items[i % len(items)] for i in range(n)]
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def dreamgen_single_arm(view: str, split: str) -> dict:
    """15 entries (5 per branch) for single_arm singleview/multiview."""
    root = DATASETS / "single_arm" / view / split
    if view == "singleview":
        ids = _episode_ids(root, "observation.images.global")
    else:
        ids = _episode_ids_all_cams(root, [
            "observation.images.exterior_1_left",
            "observation.images.exterior_2_left",
            "observation.images.wrist_left",
        ])
    instr = _read_episodes_jsonl(root / "meta")
    # An empty or "No action" instruction gives the editor nothing to work with;
    # prefer episodes that actually describe a manipulation.
    usable = [i for i in ids
              if instr.get(i, "").strip()
              and not instr[i].strip().lower().startswith("no action")]
    ids = usable or ids
    picked = _spread(ids, PER_BRANCH * len(BRANCHES))
    agent = _agent("single_arm")

    out: dict[str, list] = {b: [] for b in BRANCHES}
    for k, ep in enumerate(picked):
        branch = BRANCHES[k // PER_BRANCH]
        i_in_branch = k % PER_BRANCH
        instruction = instr.get(ep, "").strip()
        obj = extract_object(instruction)
        color = COLORS[k % len(COLORS)]
        entry = {
            "episode": f"{ep:06d}",
            "source_instruction": instruction,
            "object": obj,
        }
        if branch == "edit_only":
            entry["edit_instruction"] = recolor(obj, color)
            entry["prompt"] = (instruction or f"{agent} interacts with the {obj}.")
        elif branch == "prompt_only":
            entry["prompt"] = PROMPT_PERTURBATIONS[i_in_branch].format(agent=agent, obj=obj)
        else:
            entry["edit_instruction"] = recolor(obj, color)
            entry["prompt"] = recolored_prompt(instruction, obj, color, agent)
        out[branch].append(entry)
    return out


def dreamgen_bimanual(view: str, split: str) -> dict:
    """15 (or 20 for singleview) entries for bimanual singleview/multiview.

    Episodes are spread across DIFFERENT tasks so the videos cover different
    scenes rather than N clips of one task. singleview additionally gets a 4th
    `task_swap` branch (5 more): frame kept as-is, but the prompt names a
    DIFFERENT real task borrowed from elsewhere in the dataset — this tests
    whether the model follows the text description or just continues whatever
    object it saw in the (unedited) conditioning frame. multiview does not get
    this branch (out of scope, see augment-track-layout.mdc).
    """
    branches = BIMANUAL_SV_BRANCHES if view == "singleview" else BRANCHES
    base = DATASETS / "bimanual" / view / split
    cams = (["observation.images.cam_high"] if view == "singleview" else [
        "observation.images.cam_high", "observation.images.cam_low",
        "observation.images.cam_left_wrist", "observation.images.cam_right_wrist",
    ])
    pool: list[tuple[str, int, str]] = []  # (task, episode, instruction)
    tasks = sorted(d.name for d in base.iterdir() if d.is_dir()) if base.exists() else []
    per_task: dict[str, list[tuple[int, str]]] = {}
    for task in tasks:
        instr = _read_episodes_jsonl(base / task / "meta")
        eps = _episode_ids_all_cams(base / task, cams)
        if eps:
            per_task[task] = [(e, instr.get(e, "").replace("_", " ").strip()) for e in eps]
    # Round-robin over tasks: first episode of each task, then the second, ...
    depth = 0
    while len(pool) < PER_BRANCH * len(branches) and per_task:
        added = False
        for task in list(per_task):
            eps = per_task[task]
            if depth < len(eps):
                ep, instruction = eps[depth]
                pool.append((task, ep, instruction))
                added = True
            if len(pool) >= PER_BRANCH * len(branches):
                break
        if not added:
            break
        depth += 1

    agent = _agent("bimanual")
    out: dict[str, list] = {b: [] for b in branches}
    core_n = PER_BRANCH * len(BRANCHES)
    for k, (task, ep, instruction) in enumerate(pool[:PER_BRANCH * len(branches)]):
        i_in_branch = k % PER_BRANCH
        if k < core_n:
            branch = BRANCHES[k // PER_BRANCH]
        else:
            branch = "task_swap"
        obj = extract_object(instruction, task)
        color = COLORS[k % len(COLORS)]
        entry = {
            "task": task,
            "episode": f"{ep:06d}",
            "source_instruction": instruction,
            "object": obj,
        }
        if branch == "edit_only":
            entry["edit_instruction"] = recolor(obj, color)
            entry["prompt"] = instruction or f"{agent} interacts with the {obj}."
        elif branch == "prompt_only":
            entry["prompt"] = PROMPT_PERTURBATIONS[i_in_branch].format(agent=agent, obj=obj)
        elif branch == "task_swap":
            # Borrow a REAL task's instruction from a DIFFERENT task, jumping
            # ~halfway across the pool for max scene/object diversity (never
            # borrows its own task, guarded below); frame stays untouched.
            donor_idx = (k + len(pool) // 2) % len(pool)
            donor_task, _, donor_instruction = pool[donor_idx]
            tries = 0
            while donor_task == task and tries < len(pool):
                donor_idx = (donor_idx + 1) % len(pool)
                donor_task, _, donor_instruction = pool[donor_idx]
                tries += 1
            entry["swap_task"] = donor_task
            entry["swap_instruction"] = donor_instruction
            entry["prompt"] = _bimanual_task_caption(donor_task)
        else:
            entry["edit_instruction"] = recolor(obj, color)
            entry["prompt"] = recolored_prompt(instruction, obj, color, agent)
        out[branch].append(entry)
    return out


# =================================================================== Ctrl-World


def _cw_subset_dir(embodiment: str, split: str) -> Path:
    """Ctrl-World prepared trees: single_arm is split per subset dir, the others
    keep both splits in one dir and encode the split in the annotation stem."""
    if embodiment == "single_arm":
        return CW_EXAMPLES / f"single_arm_multiview_{split}"
    return CW_EXAMPLES / f"{embodiment}_multiview"


def _cw_annotations(embodiment: str, split: str) -> list[tuple[str, str]]:
    """[(annotation_stem, instruction)] for one embodiment+split."""
    d = _cw_subset_dir(embodiment, split) / "annotation" / "val"
    if not d.exists():
        return []
    out = []
    for f in sorted(d.glob("*.json")):
        stem = f.stem
        if embodiment != "single_arm":
            # Both splits live together; keep only this split's stems. Check
            # non_makovian first — "makovian_" is a prefix of neither, but
            # `startswith("makovian")` would wrongly reject nothing here.
            if split == "makovian" and not stem.startswith("makovian_"):
                continue
            if split == "non_makovian" and not stem.startswith("non_makovian_"):
                continue
        try:
            anno = json.loads(f.read_text())
            texts = anno.get("texts") or [""]
            instruction = (texts[0] or "").strip()
        except (json.JSONDecodeError, OSError):
            continue
        # "No action" / empty episodes give the editor no object to recolor.
        if not instruction or instruction.lower().startswith("no action"):
            continue
        out.append((stem, instruction))
    return out


def ctrlworld_config(embodiment: str, split: str) -> list[dict]:
    """CW_PER_CONFIG recolor entries; episodes are reused (with a different
    color + `variant` tag) when the split has fewer real episodes than that."""
    anns = _cw_annotations(embodiment, split)
    if not anns:
        return []
    picked = _spread(anns, CW_PER_CONFIG) if len(anns) >= CW_PER_CONFIG else \
        [anns[i % len(anns)] for i in range(CW_PER_CONFIG)]
    agent = _agent("single_arm" if embodiment == "single_arm" else "bimanual")
    out = []
    for i, (stem, instruction) in enumerate(picked):
        color = COLORS[i % len(COLORS)]
        obj = extract_object(instruction, stem)
        out.append({
            "episode": stem,
            "variant": f"v{i:02d}_{color}",
            "source_instruction": instruction,
            "object": obj,
            "edit_instruction": recolor(obj, color),
            "new_instruction": recolored_prompt(instruction, obj, color, agent),
        })
    return out


# ======================================================================== build


def build() -> dict:
    manifest: dict = {
        "_readme": (
            "Augment plan. Generated by scripts/augment/gen_manifest.py — edit the "
            "generator, never this file. DreamGen configs hold 3 branches x 5 entries "
            "(edit_only / prompt_only / combo); Ctrl-World configs hold 10 recolor "
            "entries keyed by 'variant'. Episodes and instructions are discovered from "
            "datasets/ and models/Ctrl-World/dataset_example/. Output paths come from "
            "scripts/augment/augment_paths.py; see .cursor/rules/augment-track-layout.mdc."
        ),
    }

    for embodiment in ("single_arm", "bimanual"):
        for view in ("singleview", "multiview"):
            key = f"dreamgen_{embodiment}_{view}"
            branches_here = (BIMANUAL_SV_BRANCHES
                              if key == "dreamgen_bimanual_singleview" else BRANCHES)
            note = (
                f"DreamGen {embodiment} {view}. {len(branches_here)} branches x "
                f"{PER_BRANCH}: edit_only (recolor object on the conditioning first "
                "frame, original prompt), prompt_only (frame kept, prompt replaced + "
                "refiner on), combo (recolor + prompt rewritten to the new color + "
                "refiner on)"
            )
            if "task_swap" in branches_here:
                note += (", task_swap (frame kept, prompt is a DIFFERENT real task's "
                          "caption borrowed from elsewhere in the dataset + refiner on)")
            section = {"_note": note + "."}
            for split in SPLITS:
                fn = dreamgen_single_arm if embodiment == "single_arm" else dreamgen_bimanual
                section[split] = fn(view, split)
            manifest[key] = section

    for embodiment in ("single_arm", "bimanual", "humanoid"):
        key = f"ctrlworld_{embodiment}_multiview"
        section = {
            "_note": (
                f"Ctrl-World {embodiment} multiview. RECOLOR only, first-frame-only edit, "
                f"no-gt inference. {CW_PER_CONFIG} entries/split; when the split has fewer "
                "real episodes, episodes repeat with a different color and a distinct "
                "'variant' tag so every output is unique."
            ),
        }
        for split in SPLITS:
            section[split] = ctrlworld_config(embodiment, split)
        manifest[key] = section

    # 4-VIEW GRID variants (bimanual/humanoid only — single_arm has no 4-view
    # grid finetune). Dense keeps ctrlworld_4view_grid as a SEPARATE method dir
    # from ctrlworld (different checkpoint/config, same episode ids), so augment
    # mirrors that: reuse the EXACT SAME entries (episode/color/variant) as the
    # 3-view section instead of re-deriving them, so the two checkpoints are
    # benchmarked on identical inputs.
    for embodiment in ("bimanual", "humanoid"):
        src_key = f"ctrlworld_{embodiment}_multiview"
        key = f"{src_key}_4view_grid"
        manifest[key] = {
            "_note": (
                f"Ctrl-World {embodiment} multiview 4-VIEW GRID (checkpoint_4view_grid). "
                f"Same entries as '{src_key}' (identical episode ids exist in the "
                "*_4view_grid dataset_example tree) so the 3-view and 4-view grid "
                "checkpoints are benchmarked on identical recolors."
            ),
            **{split: manifest[src_key][split] for split in SPLITS},
        }

    return manifest


def summarize(manifest: dict) -> bool:
    ok = True
    for key, section in manifest.items():
        if key.startswith("_"):
            continue
        for split in SPLITS:
            entries = section.get(split)
            if key.startswith("dreamgen"):
                branches_here = (BIMANUAL_SV_BRANCHES
                                  if key == "dreamgen_bimanual_singleview" else BRANCHES)
                counts = {b: len(entries.get(b, [])) for b in branches_here}
                total = sum(counts.values())
                bad = "" if total == PER_BRANCH * len(branches_here) else "  <-- WRONG COUNT"
                ok = ok and not bad
                print(f"  {key}/{split}: {counts} total={total}{bad}")
            else:
                n = len(entries or [])
                uniq = len({e["episode"] for e in (entries or [])})
                bad = "" if n == CW_PER_CONFIG else "  <-- WRONG COUNT"
                ok = ok and not bad
                print(f"  {key}/{split}: {n} entries ({uniq} distinct episodes){bad}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="Validate without writing.")
    a = ap.parse_args()

    manifest = build()
    ok = summarize(manifest)
    if a.check:
        print("CHECK OK" if ok else "CHECK FAILED")
        raise SystemExit(0 if ok else 1)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"WROTE {OUT}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
