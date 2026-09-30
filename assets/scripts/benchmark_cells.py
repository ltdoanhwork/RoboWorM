#!/usr/bin/env python3
"""The single source of truth for which benchmark cells the paper reports.

WHY THIS EXISTS
---------------
The benchmark generated more cells than the paper reports, and the decision of
which ones to report was previously implicit: every table/figure generator
carried its own hand-written cell list, so excluding a cell meant editing ten
scripts and hoping none was missed. A cell that survived in one generator but
not another produced a paper whose counts disagreed with each other -- the exact
failure the appendix claim-checkers exist to catch.

This module holds the exclusion in ONE auditable place. Generators import
``is_excluded`` / ``filter_rows`` and are then automatically consistent, and
reversing a decision is a one-line edit here rather than an archaeology exercise.

THE CURRENT EXCLUSION
---------------------
Ctrl-World's 3-view latent strip is dropped for **bimanual** and **humanoid**,
because for those two embodiments the same episodes were regenerated on the
4-view 2x2 latent canvas, which supersedes the strip: it carries one extra
camera, a different checkpoint, and materially different cache behaviour
(WorldCache 1.27x -> 1.81x on bimanual). Reporting both would double-count the
same episodes under two layouts of the same model and invite the reader to
average across them.

**single_arm keeps its 3-view strip**, because that embodiment has no 4-view
grid variant -- the strip is its only Ctrl-World multi-view cell, so dropping it
would remove Ctrl-World from single-arm entirely.

The excluded rollouts remain on disk and their run scripts remain runnable; this
is a reporting decision, not a deletion. Nothing here deletes data.

USAGE
-----
    from benchmark_cells import is_excluded, filter_rows, EXCLUDED_CELLS

    if is_excluded(model="ctrlworld", embodiment="humanoid", view="multiview"):
        continue

Ctrl-World layout is expressed two different ways across the repo -- sometimes
as the ``model`` field (``ctrlworld`` vs ``ctrlworld_4view_grid``), sometimes as
a separate ``layout`` field next to ``model="ctrlworld"``. Both spellings are
accepted, so callers do not have to normalise first.
"""

from __future__ import annotations

# (embodiment, view) pairs whose Ctrl-World *3-view strip* cell is not reported.
# Kept as data rather than an `if` so the reason can travel with the decision.
EXCLUDED_CELLS = (
    {
        "model": "ctrlworld",
        "embodiment": "bimanual",
        "view": "multiview",
        "layout": "3x1 latent strip",
        "superseded_by": "ctrlworld_4view_grid",
        "reason": "superseded by the 4-view 2x2 latent canvas on the same episodes",
    },
    {
        "model": "ctrlworld",
        "embodiment": "humanoid",
        "view": "multiview",
        "layout": "3x1 latent strip",
        "superseded_by": "ctrlworld_4view_grid",
        "reason": "superseded by the 4-view 2x2 latent canvas on the same episodes",
    },
)

# Embodiments that keep the Ctrl-World 3-view strip, and why. Stated explicitly
# so a future reader does not "finish the job" by removing single_arm too.
RETAINED_STRIP = {
    "single_arm": "no 4-view grid variant exists; the strip is its only "
                  "Ctrl-World multi-view cell",
}

# Generators that intentionally do NOT call `is_excluded`, and why. Listed so a
# future audit does not "fix" them by adding the filter.
EXEMPT_GENERATORS = {
    "make_checkpoint_finetune_comparison.py":
        "base-vs-finetuned ablation, not a benchmark cell. Only ever run on the "
        "3-view strip checkpoints; no grid ablation exists, so filtering would "
        "delete the only finetuning evidence for Ctrl-World. The layout mismatch "
        "is disclosed in the appendix prose instead.",
    "make_finetune_qualitative_figure.py":
        "same ablation, figure form (sources the same strip episodes).",
}

_EXCLUDED_KEYS = frozenset(
    (c["embodiment"], c["view"]) for c in EXCLUDED_CELLS
)


def _is_grid(model: str, layout: str | None) -> bool:
    """True if this row describes the 4-view grid rather than the 3-view strip.

    The grid is identified from either field, because the two generators that
    emit these rows disagree on where it lives: one puts it in `model`, the
    other keeps `model="ctrlworld"` and distinguishes by `layout`.
    """
    blob = "%s %s" % (model or "", layout or "")
    return "4view" in blob or "4-view" in blob or "2x2" in blob


def is_excluded(model: str | None, embodiment: str | None,
                view: str | None = None, layout: str | None = None) -> bool:
    """True if this cell is deliberately not reported in the paper.

    Only the Ctrl-World 3-view strip of an excluded embodiment matches: the
    4-view grid of the same embodiment is reported, and so is every non-
    Ctrl-World cell.
    """
    model = (model or "").strip()
    embodiment = (embodiment or "").strip()
    view = (view or "").strip() or "multiview"
    if not model.startswith("ctrlworld"):
        return False
    if _is_grid(model, layout):
        return False
    return (embodiment, view) in _EXCLUDED_KEYS


def cell_is_excluded(cell: str) -> bool:
    """Same decision from a slash-joined cell id, e.g. ``bimanual/multiview/ctrlworld``.

    Order of the three fields varies across reports, so the id is matched on
    content rather than position.
    """
    parts = [p for p in (cell or "").replace("\\", "/").split("/") if p]
    if not any(p.startswith("ctrlworld") for p in parts):
        return False
    if _is_grid(cell, None):
        return False
    for emb, view in _EXCLUDED_KEYS:
        if emb in parts and view in parts:
            return True
    return False


def filter_rows(rows, model="model", embodiment="embodiment", view="view",
                layout="layout"):
    """Drop excluded cells from a list of dict rows, preserving order.

    Field names are parameters because the reports use different column names
    for the same quantity (``track`` vs ``view``, ``family`` vs ``model``).
    """
    out = []
    for r in rows:
        if is_excluded(r.get(model), r.get(embodiment), r.get(view),
                       r.get(layout)):
            continue
        out.append(r)
    return out


def describe() -> str:
    """One-line summary for generator logs, so each run states what it dropped."""
    return ("excluding %d cell(s): %s (single_arm strip retained: %s)" % (
        len(EXCLUDED_CELLS),
        ", ".join("%s/%s/ctrlworld 3-view" % (c["embodiment"], c["view"])
                  for c in EXCLUDED_CELLS),
        "; ".join(RETAINED_STRIP)))


if __name__ == "__main__":
    print(__doc__.strip().splitlines()[0])
    print()
    print(describe())
    print()
    # Self-test: the retained and excluded cases must not collapse into each
    # other. This runs on `python3 assets/scripts/benchmark_cells.py`.
    cases = [
        (("ctrlworld", "bimanual", "multiview", None), True),
        (("ctrlworld", "humanoid", "multiview", None), True),
        (("ctrlworld", "single_arm", "multiview", None), False),
        (("ctrlworld_4view_grid", "bimanual", "multiview", None), False),
        (("ctrlworld_4view_grid", "humanoid", "multiview", None), False),
        (("ctrlworld", "humanoid", "multiview", "2x2 latent grid"), False),
        (("ctrlworld", "bimanual", "multiview", "ctrlworld_4view_grid"), False),
        (("dreamgen", "bimanual", "multiview", None), False),
        (("dreamdojo", "humanoid", "singleview", None), False),
    ]
    bad = 0
    for args, want in cases:
        got = is_excluded(*args)
        flag = "ok  " if got == want else "FAIL"
        bad += got != want
        print("  %s is_excluded%-58s -> %-5s (want %s)" % (flag, args, got, want))
    cell_cases = [
        ("bimanual/multiview/ctrlworld", True),
        ("humanoid/multiview/ctrlworld", True),
        ("single_arm/multiview/ctrlworld", False),
        ("bimanual/multiview/ctrlworld_4view_grid", False),
        ("humanoid/singleview/dreamdojo", False),
    ]
    for cell, want in cell_cases:
        got = cell_is_excluded(cell)
        flag = "ok  " if got == want else "FAIL"
        bad += got != want
        print("  %s cell_is_excluded(%-42s) -> %-5s (want %s)" % (
            flag, repr(cell), got, want))
    print()
    raise SystemExit("%d self-test failure(s)" % bad if bad else 0)
