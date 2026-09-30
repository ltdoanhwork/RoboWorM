#!/usr/bin/env python3
"""Assert every numeric claim in appendix/1_dataset.tex against measured evidence.

This is the Evidence gate for the dataset appendix: it fails loudly if the prose
drifts from ``metrics/reports/dataset_appendix_evidence.json`` (per-cell counts)
or ``metrics/reports/benchmark_stats.json`` (prompts, skills, scenes, videos).

    python3 assets/scripts/check_dataset_appendix_claims.py

Regenerate the evidence first:

    python3 assets/scripts/make_dataset_appendix_tables.py
    python3 assets/scripts/make_benchmark_stats.py --json metrics/reports/benchmark_stats.json
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(REPO, "assets/paper/appendix/1_dataset.tex")
METHOD_TEX = os.path.join(REPO, "assets/paper/sections/3_method.tex")
MAIN_TEX = os.path.join(REPO, "assets/paper/iclr2027_conference.tex")
EV = os.path.join(REPO, "metrics/reports/dataset_appendix_evidence.json")
STATS = os.path.join(REPO, "metrics/reports/benchmark_stats.json")
AUDIT = os.path.join(REPO, "metrics/reports/sampling_audit.json")

# Which acceleration family each audited method belongs to. The method section
# quotes per-family rollout subtotals, so the gate has to aggregate the audit the
# same way the figures do.
FAMILY = {
    "dense": "dense",
    "sito": "token", "itm": "token",
    "worldcache": "cache", "dicache": "cache", "fastercache": "cache",
    "pisa": "sparse", "radial": "sparse", "svg1": "sparse", "svg2": "sparse",
}


def norm(text: str) -> str:
    """Strip LaTeX so numeric claims are greppable as plain digits."""
    text = re.sub(r"\{,\}", "", text)          # 1{,}608 -> 1608
    text = re.sub(r"\\[a-zA-Z]+\s*", " ", text)  # drop macros
    text = re.sub(r"[{}$~]", "", text)
    return text


def main() -> int:
    tex = norm(open(TEX).read())
    ev = json.load(open(EV))
    st = json.load(open(STATS))

    cells = {(c["embodiment"], c["view"], c["split"]): c for c in ev["cells"]}

    def cell(emb, view, split, key):
        return cells[(emb, view, split)][key]

    checks: list[tuple[str, object]] = [
        ("total evaluation conditions", ev["totals"]["episode_instances"]),
        ("distinct robot trajectories", ev["totals"]["distinct_source_episodes"]),
        ("view-videos", st["total_videos"]),
        ("total frames", sum(c["frames_selected"] for c in ev["cells"])),
        ("distinct instructions", st["distinct_prompts"]),
        ("multi-primitive prompts", st["multi_primitive_prompts"]),
        ("multi-label mentions", sum(st["skills_mentions"].values())),
        ("distinct skills", len(st["skills_primary"])),
        ("GR1 scenes", len(st["gr1_scenes"])),
        ("DROID pool", cell("single_arm", "multiview", "makovian", "episodes_available")),
        ("humanoid mv nonmk pool", cell("humanoid", "multiview", "non_makovian", "episodes_available")),
        ("bimanual mk instructions", cell("bimanual", "multiview", "makovian", "instructions_distinct")),
        ("humanoid mv mk instructions", cell("humanoid", "multiview", "makovian", "instructions_distinct")),
        ("DROID mk instructions", cell("single_arm", "multiview", "makovian", "instructions_distinct")),
        ("GR1 mk instructions", cell("humanoid", "singleview", "makovian", "instructions_distinct")),
        ("shortest episode", min(c["length_min"] for c in ev["cells"])),
        ("longest episode", max(c["length_max"] for c in ev["cells"])),
    ]
    # Primitive counts: the prose names the head of the distribution explicitly and
    # collapses every singleton tail primitive into "one instruction each".
    for name, count in st["skills_primary"].items():
        if count > 1:
            checks.append((f"skill count: {name}", count))
    singletons = sorted(n for n, c in st["skills_primary"].items() if c == 1)

    failures = []
    for label, value in checks:
        if not re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", tex):
            failures.append(f"{label}: {value} not stated in the appendix prose")

    # Sanity: the taxonomy must be exhaustive and the prose must say so.
    if sum(st["skills_primary"].values()) != st["distinct_prompts"]:
        failures.append(
            "skills_primary does not sum to distinct_prompts "
            f"({sum(st['skills_primary'].values())} vs {st['distinct_prompts']})"
        )
    if st["skills_primary"].get("unmapped"):
        failures.append(
            f"{st['skills_primary']['unmapped']} instructions are unmapped; "
            f"the appendix claims none of {st['distinct_prompts']} are"
        )
    # Every singleton tail primitive must still be named in the prose.
    for name in singletons:
        head = name.split("/")[0].split("&")[0].strip()
        if head and head.lower() not in tex.lower():
            failures.append(f"tail primitive '{name}' not named in the prose")

    pct = round(100 * st["multi_primitive_prompts"] / st["distinct_prompts"])
    if not re.search(rf"{pct}\s*%|{pct}\\%", tex):
        failures.append(f"compositional share {pct}% not stated")

    # Style gate: a paper must not cite repository paths, script names or on-disk
    # directory names. These belong in the code, not in the prose. The main-paper
    # dataset section is held to the same standard as the appendix.
    for path, where in ((TEX, "appendix"), (METHOD_TEX, "method")):
        raw = open(path).read()
        for pat, why in (
            (r"\\path\{", "\\path{...} repository path"),
            (r"scripts/", "script path"),
            (r"metrics/reports/", "report path"),
            (r"\bmeta/episodes\.jsonl\b", "metadata filename"),
            (r"episode\\_\*\.parquet", "on-disk filename"),
            (r"\bmakovian\b", "misspelled on-disk split name"),
            (r"\.py\b", "script filename"),
            (r"\bcam\\_[a-z]+", "raw camera key"),
            (r"\bgrid\\_\*", "raw scene identifier"),
        ):
            for m in re.finditer(pat, raw):
                line = raw[: m.start()].count("\n") + 1
                failures.append(
                    f"{where} line {line}: prose cites a {why} (\"{m.group(0)}\")"
                )

    # Every number the main-paper dataset section states must also be measured.
    method = norm(open(METHOD_TEX).read())
    audit = json.load(open(AUDIT))
    if not audit.get("scope_matches_design", False):
        failures.append("sampling audit scope does not match the design scope; "
                        "rerun audit_sampling_distribution.py --json")
    fam = collections.Counter()
    for c in audit["method_cells"]:
        fam[FAMILY[c["method"]]] += c["total"]

    method_must_hold = [
        ("evaluation conditions", ev["totals"]["episode_instances"]),
        ("distinct trajectories", ev["totals"]["distinct_source_episodes"]),
        ("view-videos", st["total_videos"]),
        ("distinct instructions", st["distinct_prompts"]),
        ("primitives", len(st["skills_primary"])),
        ("cells", ev["totals"]["cells"]),
        # Rollout counts come from the audit, never from a hardcoded literal --
        # an inflated directory scan once put 31,483 here instead of 29,343.
        ("total rollouts", audit["grand_total"]),
    ]
    # Measured values the section MAY quote (the per-family subtotals are deferred
    # to the appendix in the current draft). Listing them here means that if the
    # prose does cite one it is checked against the audit rather than flagged as
    # untraced -- required numbers and optional ones need different treatment.
    method_allowed = [
        ("dense rollouts", fam["dense"]),
        ("caching rollouts", fam["cache"]),
        ("sparse-attention rollouts", fam["sparse"]),
        ("token-reduction rollouts", fam["token"]),
        ("GR1 scenes", len(st["gr1_scenes"])),
        ("total frames", sum(c["frames_selected"] for c in ev["cells"])),
    ]
    for label, value in method_must_hold:
        if not re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", method):
            failures.append(f"method section omits measured {label}: {value}")

    # Any other multi-digit number in the method section must be traceable.
    # Small integers that describe the design itself (3 models, 9 methods, 12
    # cells, 6 settings, 5 corpora, 14 primitives, 2B/14B sizes) are structural,
    # not measured.
    known = {str(v) for _, v in method_must_hold} | {
        str(v) for _, v in method_allowed
    } | {
        str(v) for _, v in checks
    } | {"2", "3", "5", "6", "9", "12", "14"}
    # A closed interval such as `$[0,100]$` is a metric's range, not a count, and
    # the comma inside it made the scanner read it as the number "0,100". Strip
    # bracketed intervals before scanning so a score range cannot be reported as
    # an untraceable measurement.
    scanned = re.sub(r"\[\s*\d+\s*,\s*\d+\s*\]", "[range]", method)
    for m in re.finditer(r"(?<![\w.])\d[\d,]*(?![\w])", scanned):
        tok = m.group(0)
        if tok not in known and tok.replace(",", "") not in known:
            line = scanned[: m.start()].count("\n") + 1
            failures.append(
                f"method line {line}: untraced number '{tok}'"
            )

    # The ABSTRACT is where an inflated number does the most damage and where it
    # survives longest: it once claimed "1,628 episodes across 5 distinct
    # embodiments" while the measured corpus is 1,608 instances drawn from 5
    # CORPORA over 3 morphologies. Held to the same standard as the body.
    abstract = ""
    main_raw = open(MAIN_TEX).read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_raw,
                  re.S)
    if m is None:
        failures.append("could not locate the abstract to verify its numbers")
    else:
        # Commented-out placeholder lines are drafting scaffolding, not claims.
        abstract = norm("\n".join(
            l for l in m.group(1).splitlines()
            if not l.lstrip().startswith("%")))
        n_eps = ev["totals"]["episode_instances"]
        if not re.search(rf"(?<!\d){n_eps}(?!\d)", abstract):
            failures.append(
                f"abstract omits the measured episode count: {n_eps}")
        # "N distinct embodiments" is the specific wording that was wrong: the
        # five are source corpora, and the three are morphologies.
        bad = re.search(r"\d+\s*\$?\s*distinct embodiments", abstract)
        if bad:
            failures.append(
                f"abstract says \"{bad.group(0)}\"; the five are source "
                "corpora, not embodiments (3 morphologies)")
        abstract_known = {str(n_eps), "3", "5", "9", "4", "11", "0.43"}
        for mm in re.finditer(r"(?<![\w.$])\d[\d,.]*(?![\w])", abstract):
            tok = mm.group(0).rstrip(".")
            if tok not in abstract_known and tok.replace(",", "") not in (
                    abstract_known | {str(v) for _, v in checks}):
                failures.append(f"abstract: untraced number '{tok}'")

    if failures:
        print("Dataset appendix claim check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"Dataset appendix claim check passed ({len(checks)} numeric claims "
          f"traced to measured evidence).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
