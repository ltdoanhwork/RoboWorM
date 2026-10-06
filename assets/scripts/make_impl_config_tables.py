#!/usr/bin/env python3
"""Emit the appendix acceleration-configuration tables from the run scripts.

Nothing here is hand-typed. The shipped configuration of every acceleration
method is read out of the canonical run scripts
(``scripts/run/{model}/run_{embodiment}_{view}_{model}[_4view_grid]_{method}.sh``), where each
knob is declared as ``VAR="${VAR:-default}"``, and joined against
``results/reports/sampling_audit.json`` so that a row is only printed for a
(method, configuration) pair that actually produced rollouts.

Three facts about the sources force the design:

1. **A runnable script is not coverage.** SiTo/ITM have runnable DreamDojo
   scripts and FasterCache has runnable Ctrl-World scripts, but the audit
   records ``0`` rollouts for those cells. The audit wins; the script defaults
   are reported as ``not run`` so a reader can tell "not run" from "ran and
   lost" (acceleration-benchmark policy).
2. **The same knob has different env names across backbones.** Ctrl-World's
   wrappers say ``PISA_START_LAYER`` / ``RADIAL_DECAY`` /
   ``{SVG,RADIAL}_FIRST_LAYERS_FP`` where the DiT wrappers say
   ``PISA_START_LAYER_IDX`` / ``RADIAL_DECAY_FACTOR`` / ``*_START_LAYER_IDX``.
   ``CANON`` maps both spellings onto one canonical parameter so a column means
   the same thing down its whole length. "First N layers dense" and "start
   sparsifying at layer N" are the same quantity.
3. **A method can be instantiated as two different variants.** SiTo and ITM run
   as spatio-temporal prune-once-hold on the Ctrl-World U-Net (keep-ratio +
   downsample-ratio) and as per-attention pruning on the DiTs (prune/compress
   ratio + start layer). The variant is printed as its own column instead of
   silently mixing two parameterisations in one cell.

    python3 assets/scripts/make_impl_config_tables.py

Writes:
    results/reports/impl_accel_configs.json
    assets/paper/appendix/tables/accel_hyperparams_token.tex
    assets/paper/appendix/tables/accel_hyperparams_cache.tex
    assets/paper/appendix/tables/accel_hyperparams_sparse.tex
"""

from __future__ import annotations

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "scripts")
# Per-cell launchers live in scripts/run/{ctrlworld,dreamgen,dreamdojo}/ since
# the 2026-10-05 reorg; index them by basename.
RUN_DIRS = [os.path.join(SCRIPTS, "run", d) for d in ("ctrlworld", "dreamgen", "dreamdojo")]


def _launcher_index():
    idx = {}
    for d in RUN_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                idx.setdefault(f, os.path.join(d, f))
    return idx


LAUNCHERS = _launcher_index()
AUDIT = os.path.join(REPO, "results/reports/sampling_audit.json")
OUT_JSON = os.path.join(REPO, "results/reports/impl_accel_configs.json")
TABLE_DIR = os.path.join(REPO, "assets/paper/appendix/tables")

# Which cells the paper reports is decided in one shared place, so these tables,
# the dense-config table, the speedup CSV and the figures cannot disagree about
# the benchmark's size.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_cells import EXCLUDED_CELLS  # noqa: E402
from benchmark_cells import describe as describe_exclusions  # noqa: E402
from benchmark_cells import is_excluded  # noqa: E402

METHODS = [
    "sito", "itm",
    "worldcache", "dicache", "fastercache",
    "pisa", "svg1", "svg2", "radial",
]

FAMILY = {
    "sito": "token", "itm": "token",
    "worldcache": "cache", "dicache": "cache", "fastercache": "cache",
    "pisa": "sparse", "svg1": "sparse", "svg2": "sparse", "radial": "sparse",
}

METHOD_DISPLAY = {
    "sito": "SiTo", "itm": "ITM",
    "worldcache": "WorldCache", "dicache": "DiCache", "fastercache": "FasterCache",
    "pisa": "PISA", "svg1": "SVG", "svg2": "SVG2", "radial": "Radial",
}

MODEL_DISPLAY = {
    "ctrlworld": "Ctrl-World",
    "ctrlworld_4view_grid": "Ctrl-World",
    "dreamgen": "DreamGen",
    "dreamdojo": "DreamDojo",
}

EMB_DISPLAY = {"single_arm": "single-arm", "bimanual": "bimanual", "humanoid": "humanoid"}
VIEW_DISPLAY = {"multiview": "MV", "singleview": "SV"}

# Raw env var -> canonical parameter name. Both spellings of a knob map to one
# name; see docstring note 2.
CANON = {
    # --- SiTo ---
    "SITO_ST_KEEP_RATIO": "keep_ratio",
    "SITO_PRUNE_RATIO": "prune_ratio",
    "SITO_MAX_DOWNSAMPLE_RATIO": "downsample",
    "SITO_START_LAYER_IDX": "start_layer",
    "SITO_KEEP_LAST_N_DENSE": "keep_last_dense",
    "SITO_GROUP_MODE": "group_mode",
    "SITO_NOISE_ALPHA": "noise_alpha",
    # --- ITM ---
    "ITM_ST_KEEP_RATIO": "keep_ratio",
    "ITM_COMPRESS_RATIO": "compress_ratio",
    "ITM_MAX_DOWNSAMPLE_RATIO": "downsample",
    "ITM_START_LAYER_IDX": "start_layer",
    "ITM_KEEP_LAST_N_DENSE": "keep_last_dense",
    "ITM_MERGE_FROM_STEP": "merge_from_step",
    "ITM_PRUNE_FROM_STEP": "prune_from_step",
    # --- caches ---
    "WORLDCACHE_REL_L1_THRESH": "rel_l1",
    "WORLDCACHE_RET_RATIO": "ret_ratio",
    "WORLDCACHE_PROBE_DEPTH": "probe",
    "WORLDCACHE_MOTION_SENSITIVITY": "motion_sens",
    "WORLDCACHE_DYNAMIC_DECAY": "dyn_decay",
    "WORLDCACHE_NUM_STEPS": "num_steps",
    "DICACHE_REL_L1_THRESH": "rel_l1",
    "DICACHE_RET_RATIO": "ret_ratio",
    "DICACHE_PROBE_DEPTH": "probe",
    "DICACHE_NUM_STEPS": "num_steps",
    "FASTERCACHE_MODEL_INTERVAL": "model_interval",
    "FASTERCACHE_BLOCK_INTERVAL": "block_interval",
    "FASTERCACHE_START_STEP": "start_step",
    # --- PISA ---
    "PISA_DENSITY": "density",
    "PISA_BLOCK_SIZE": "block",
    "PISA_START_LAYER_IDX": "start_layer",
    "PISA_START_LAYER": "start_layer",
    "PISA_USE_BIAS": "use_bias",
    # --- SVG (both generations share the SVG_ prefix) ---
    "SVG_SPARSITY": "sparsity",
    "SVG_NUM_SAMPLED_ROWS": "rows",
    "SVG_DENSE_STEP_FRAC": "dense_frac",
    "SVG_PROFILE_RECOMPUTE_EVERY": "profile_every",
    "SVG_SAMPLE_MSE_MAX_ROW": "mse_max_row",
    "SVG_NUM_Q_CENTROIDS": "q_centroids",
    "SVG_NUM_K_CENTROIDS": "k_centroids",
    "SVG_TOP_P_KMEANS": "top_p",
    "SVG_MIN_KC_RATIO": "min_kc",
    "SVG_KMEANS_ITER_INIT": "kmeans_init",
    "SVG_KMEANS_ITER_STEP": "kmeans_step",
    "SVG_START_LAYER_IDX": "start_layer",
    "SVG_FIRST_LAYERS_FP": "start_layer",
    # --- Radial ---
    "RADIAL_DECAY_FACTOR": "decay",
    "RADIAL_DECAY": "decay",
    "RADIAL_MODEL_TYPE": "mask",
    "RADIAL_BLOCK_SIZE": "block",
    "RADIAL_START_LAYER_IDX": "start_layer",
    "RADIAL_FIRST_LAYERS_FP": "start_layer",
}

# Canonical parameter -> printed column header, per method. Order is the column
# order of that method's block in the table.
COLUMNS = {
    "sito": [
        ("variant", "variant"),
        ("keep_ratio", "keep"),
        ("prune_ratio", "prune"),
        ("downsample", "down"),
        ("start_layer", "start"),
        ("keep_last_dense", "last dense"),
        ("group_mode", "grouping"),
    ],
    "itm": [
        ("variant", "variant"),
        ("keep_ratio", "keep"),
        ("compress_ratio", "compress"),
        ("downsample", "down"),
        ("start_layer", "start"),
        ("keep_last_dense", "last dense"),
        ("merge_from_step", "merge from"),
    ],
    "worldcache": [
        ("rel_l1", "$\\delta$"),
        ("ret_ratio", "warm-up"),
        ("probe", "probe"),
        ("motion_sens", "motion"),
    ],
    "dicache": [
        ("rel_l1", "$\\delta$"),
        ("ret_ratio", "warm-up"),
        ("probe", "probe"),
    ],
    "fastercache": [
        ("model_interval", "model int."),
        ("block_interval", "block int."),
        ("start_step", "start step"),
    ],
    "pisa": [
        ("density", "density"),
        ("block", "block"),
        ("start_layer", "start"),
    ],
    "svg1": [
        ("sparsity", "sparsity"),
        ("rows", "rows"),
        ("dense_frac", "dense frac."),
        ("start_layer", "start"),
        ("profile_every", "profile cache"),
    ],
    "svg2": [
        ("q_centroids", "$q$ cent."),
        ("k_centroids", "$k$ cent."),
        ("top_p", "top-$p$"),
        ("min_kc", "min $k$ frac."),
        ("dense_frac", "dense frac."),
        ("start_layer", "start"),
    ],
    "radial": [
        ("decay", "decay"),
        ("mask", "mask"),
        ("block", "block"),
        ("start_layer", "start"),
    ],
}

SCRIPT_RE = re.compile(
    r"^run_(single_arm|bimanual|humanoid)_(singleview|multiview)_"
    r"(ctrlworld_4view_grid|ctrlworld|dreamgen|dreamdojo)_(" + "|".join(METHODS) + r")\.sh$"
)
# VAR="${VAR:-default}" / VAR=${VAR:-default}
DEFAULT_RE = re.compile(r'^([A-Z][A-Z0-9_]*)="?\$\{\1:-([^}"]*)\}"?', re.M)
# A wrapper can delegate to a shared base script (the Ctrl-World 4-view-grid
# wrappers do: they only re-point ckpt/dataset/output and then call the base).
# The method flags then live in the base, so follow the call one level.
DELEGATE_RE = re.compile(r"^\s*bash\s+scripts/(?:run/[a-z]+/)?(run_[a-z0-9_]+\.sh)\s*$", re.M)


def _read_with_delegate(fname):
    """Return (own_text, effective_text). Effective text appends the delegate."""
    path = LAUNCHERS[fname]
    own = open(path).read()
    text = own
    for target in DELEGATE_RE.findall(own):
        if target == fname:
            continue
        tpath = LAUNCHERS.get(target)
        if tpath and os.path.exists(tpath):
            text = text + "\n" + open(tpath).read()
    return own, text


# A tuned default carries its sweep in a comment block directly above the knob,
# e.g. "# PISA -- TUNED (see ...). d0.2/s0 is CORRUPT here (PSNR 17.6) ...".
# This is how a divergent value is told apart from a stale copy-paste: the
# comment names the measurement that chose it.
PROVENANCE_HINT = re.compile(
    r"TUNED|tuned|knee|PSNR|SSIM|LPIPS|speedup|[0-9]\.[0-9]+x|CORRUPT|BREAK|insights")


def extract_provenance(text, method):
    """The comment block immediately above this method's first knob assignment."""
    prefix = {"svg1": "SVG", "svg2": "SVG"}.get(method, method.upper())
    m = re.search(r"^%s_[A-Z0-9_]*=" % prefix, text, re.M)
    if not m:
        return None
    lines = text[:m.start()].rstrip("\n").split("\n")
    block = []
    for line in reversed(lines):
        if line.startswith("#"):
            block.insert(0, line.lstrip("# ").rstrip())
        elif not line.strip():
            if block:
                break
        else:
            break
    if not block:
        return None
    return " ".join(block)


def find_tuning_note(text):
    """Any comment line in the script that records a measurement.

    Some scripts put the sweep in the header rather than directly above the
    knob, so the block-above heuristic alone would call them undocumented.
    """
    notes = [ln.lstrip("# ").rstrip() for ln in text.split("\n")
             if ln.startswith("#") and PROVENANCE_HINT.search(ln)]
    return " ".join(notes) if notes else None


RULES_DIR = os.path.join(REPO, ".cursor/rules")
RULE_FAMILY = {"caching": "cache", "pruning": "token", "sparse-attention": "sparse"}


def load_insight_rules():
    """Index the per-cell tuning-insight notes: {(layout, emb, view, family): file}.

    A divergent value can be justified either by a comment in its run script or
    by the sweep recorded in the corresponding insights rule. Checking both is
    what separates "tuned elsewhere and written down" from "never measured".
    """
    index = {}
    if not os.path.isdir(RULES_DIR):
        return index
    for fname in sorted(os.listdir(RULES_DIR)):
        if not fname.endswith("-insights.mdc"):
            continue
        stem = fname[: -len("-insights.mdc")]
        fam = None
        for suffix, f in RULE_FAMILY.items():
            if stem.endswith("-" + suffix):
                fam, stem = f, stem[: -len(suffix) - 1]
                break
        if fam is None:
            continue
        grid = "-4view-grid" in stem
        stem = stem.replace("-4view-grid", "")
        parts = stem.split("-")
        model = parts[0]
        view = "multiview" if "multiview" in stem else ("singleview" if "singleview" in stem else None)
        # Embodiment is whatever sits between model and view, with dashes
        # removed: the corpus spells it both `single-arm` and `singlearm`.
        mid = [p for p in parts[1:] if p not in ("multiview", "singleview")]
        emb = "".join(mid)
        emb = {"singlearm": "single_arm", "bimanual": "bimanual", "humanoid": "humanoid"}.get(emb)
        layout = "ctrlworld_4view_grid" if (model == "ctrlworld" and grid) else model
        # Two rules are deliberately view- or embodiment-agnostic
        # (`ctrlworld-pruning-insights`, `dreamgen-bimanual-pruning-insights`);
        # they are indexed with a wildcard so they still count as documentation
        # for the cells they cover.
        index[(layout, emb, view, fam)] = fname
    return index


def rule_for(index, layout, emb, view, fam):
    """Look up an insight rule, honouring the view/embodiment wildcards."""
    for key in ((layout, emb, view, fam), (layout, emb, None, fam),
                (layout, None, view, fam), (layout, None, None, fam)):
        if key in index:
            return index[key]
    return None


def parse_scripts():
    """Return {(model, embodiment, view, method): {canonical: value}} plus raw names."""
    out = {}
    for fname in sorted(LAUNCHERS):
        m = SCRIPT_RE.match(fname)
        if not m:
            continue
        emb, view, model, method = m.groups()
        own, text = _read_with_delegate(fname)
        params, raw = {}, {}
        # The wrapper's own defaults win over the delegate's, so parse the
        # delegate first and let the wrapper overwrite.
        for source in (text, own):
            for var, val in DEFAULT_RE.findall(source):
                if var not in CANON:
                    continue
                params[CANON[var]] = val.strip()
                raw[var] = val.strip()
        # The spatio-temporal prune-once-hold variant is selected by a flag, not
        # a value, so read it off the flag list (of the effective script, which
        # includes any delegate) rather than guessing from the knobs.
        if method in ("sito", "itm"):
            hold = ("--%s_spatiotemporal_hold" % method) in text
            params["variant"] = "ST-hold" if hold else "per-attn."
        prov = extract_provenance(own, method) or extract_provenance(text, method)
        note = prov if (prov and PROVENANCE_HINT.search(prov)) else find_tuning_note(own)
        out[(model, emb, view, method)] = {
            "script": os.path.relpath(LAUNCHERS[fname], REPO),
            "params": params,
            "raw_env": raw,
            "provenance": prov,
            "tuning_note": note,
            "provenance_is_tuning_note": bool(note),
        }
    return out


def load_coverage():
    """{(model_family, embodiment, view, method): rollouts} from the audit."""
    audit = json.load(open(AUDIT))
    cov = {}
    for cell in audit["method_cells"]:
        view = "multiview" if cell["view"] == "MV" else "singleview"
        cov[(cell["model"], cell["embodiment"], view, cell["method"])] = cell["total"]
    return cov, audit


def family_of(model):
    return "ctrlworld" if model.startswith("ctrlworld") else model


LAYOUT_DISPLAY = {"ctrlworld": "3-view", "ctrlworld_4view_grid": "4-view grid"}

# Values that are booleans-as-integers or sentinels, rendered as words.
WORD = {
    ("profile_every", "999999"): "cached",
    ("use_bias", "0"): "off",
    ("use_bias", "1"): "on",
    ("dyn_decay", "0"): "off",
    ("dyn_decay", "1"): "on",
}


def fmt_value(key, val):
    if (key, val) in WORD:
        return r"\textsc{%s}" % WORD[(key, val)]
    if key in ("group_mode", "mask", "variant"):
        return r"\textsc{%s}" % val.replace("_", " ")
    # Numeric -> math mode, canonically formatted. Scripts spell the same value
    # as both `0.2` and `0.20`; printing them verbatim would read as two
    # different settings, so trailing zeros are normalised away.
    try:
        f = float(val)
    except ValueError:
        return val.replace("_", r"\_")
    if f == int(f) and abs(f) < 1e6:
        return "$%d$" % int(f)
    return "$%s$" % ("%.10f" % f).rstrip("0")


def hyperparam_cell(method, params):
    """Render one configuration's knobs as a compact `name value` list."""
    bits = []
    for key, label in COLUMNS[method]:
        if key not in params:
            continue
        bits.append("%s~%s" % (label, fmt_value(key, params[key])))
    return ", ".join(bits)


def build_rows(parsed, coverage):
    """One row per (method, configuration), carrying coverage and layout notes."""
    rows = []
    for method in METHODS:
        # Group the Ctrl-World layouts that share one audit cell.
        by_cell = {}
        for (model, emb, view, meth), rec in parsed.items():
            if meth != method:
                continue
            # Cells the paper does not report are dropped here, before variants
            # and `layouts_agree` are computed, so an excluded layout cannot
            # influence how the surviving one is labelled.
            if is_excluded(model, emb, view):
                continue
            by_cell.setdefault((family_of(model), emb, view), []).append((model, rec))
        for (fam, emb, view), entries in sorted(by_cell.items()):
            n = coverage.get((fam, emb, view, method))
            cells = hyperparam_cell(method, dict(entries[0][1]["params"]))
            variants = sorted(set(hyperparam_cell(method, r["params"]) for _, r in entries))
            for model, rec in sorted(entries):
                rows.append({
                    "method": method,
                    "family": FAMILY[method],
                    "model": fam,
                    "layout": model,
                    "embodiment": emb,
                    "view": view,
                    "rollouts": n,
                    "ran": bool(n),
                    "hyperparams": hyperparam_cell(method, rec["params"]),
                    "params": rec["params"],
                    "script": rec["script"],
                    "raw_env": rec["raw_env"],
                    "provenance": rec["provenance"],
                    "tuning_note": rec["tuning_note"],
                    "provenance_is_tuning_note": rec["provenance_is_tuning_note"],
                    "layouts_agree": len(variants) == 1,
                    # How many layouts of this cell survive exclusion. The table
                    # collapses duplicates only when there is genuinely more than
                    # one layout to collapse; without this a cell whose 3-view
                    # strip was excluded would print nothing at all.
                    "n_layouts": len(entries),
                })
            del cells
    return rows


def _norm(val):
    """Canonical spelling of a knob value, for comparing across scripts."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return val
    return str(int(f)) if f == int(f) else ("%.10f" % f).rstrip("0")


def audit_divergences(rows, rules):
    """Per-cell divergence from the method's modal configuration.

    A method's value differing across cells is expected -- each cell was swept
    separately -- but it must be *justified*. For every knob we take the method's
    modal value as the reference and flag each cell that departs from it,
    recording whether the choice is documented, either by a tuning note in the
    run script or by the sweep recorded in the matching insights rule. A
    divergence with neither was never measured and is reported as such.
    """
    findings = []
    for method in METHODS:
        mrows = [r for r in rows if r["method"] == method and r["ran"]]
        if len(mrows) < 2:
            continue
        keys = set()
        for r in mrows:
            keys |= set(r["params"])
        for key in sorted(keys):
            vals = {}
            for r in mrows:
                if key in r["params"]:
                    # Compare numerically: `0.2` and `0.20` are the same setting
                    # spelled two ways and must not register as a divergence.
                    vals.setdefault(_norm(r["params"][key]), []).append(r)
            if len(vals) < 2:
                continue
            modal = max(vals, key=lambda v: len(vals[v]))
            for val, rs in vals.items():
                if val == modal:
                    continue
                for r in rs:
                    rule = rule_for(rules, r["layout"], r["embodiment"],
                                    r["view"], r["family"])
                    findings.append({
                        "method": method,
                        "cell": "%s/%s/%s" % (r["model"], r["embodiment"], r["view"]),
                        "layout": r["layout"],
                        "parameter": key,
                        "value": val,
                        "modal_value": modal,
                        "n_cells_at_modal": len(vals[modal]),
                        "documented_in_script": r["provenance_is_tuning_note"],
                        "documented_in_rule": rule,
                        "justified": bool(r["provenance_is_tuning_note"] or rule),
                        "note": (r["tuning_note"] or r["provenance"] or "")[:400],
                        "script": r["script"],
                    })
    return findings


def emit_table(family, rows, path):
    """One table per family; a full-width header row opens each method block.

    No \\multirow: these can break across a page and a spanning cell silently
    mis-aligns (same reasoning as make_accel_config_table.py).

    These use ``longtable`` rather than ``table``+``tabular``: the sparse-attention
    family lists 9 methods x up to 14 cells and does not fit on one page, and a
    float that overfull-warns is silently clipped at the page bottom. longtable
    breaks across pages and repeats the column header, so no row is lost.
    """
    fam_name = {"token": "token reduction", "cache": "step caching",
                "sparse": "sparse attention"}[family]
    fam_label = {"token": "token", "cache": "cache", "sparse": "sparse"}[family]
    colspec = r"@{}lllp{0.44\linewidth}r@{}"
    header = r"Model & Track & Embodiment & Shipped configuration & $n$ \\"
    out = [
        "% Generated by assets/scripts/make_impl_config_tables.py -- do not edit by hand.",
        "% Source: the shipped defaults of scripts/run/*/run_*_{method}.sh, joined against",
        "%         results/reports/sampling_audit.json for actual rollout coverage.",
        r"{\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{%s}" % colspec,
        r"\caption{\textbf{Shipped %s configurations.}" % fam_name,
        r"Every value is read out of the canonical run script of that configuration,"
        r" not restated by hand; each row is the setting that generated the released"
        r" rollouts of its cell. Warm-up knobs are printed explicitly, since a"
        r" method's speedup is meaningless without them: \emph{warm-up} is the"
        r" leading fraction of denoising steps computed densely, \emph{dense frac.}"
        r" the same quantity for sparse attention, and \emph{start} the first"
        r" transformer block the method is allowed to touch. \texttt{--} marks a"
        r" configuration whose script exists and runs but which is outside this"
        r" method's released coverage, so it is reported rather than dropped."
        r" Ctrl-World multi-view is reported on one latent canvas per embodiment ---"
        r" the $4$-view $2{\times}2$ canvas for bimanual and humanoid, the $3$-view"
        r" strip for single-arm, which has no grid variant --- so each row is that"
        r" embodiment's only Ctrl-World multi-view configuration. $n$ is the released"
        r" rollout count of the cell.}",
        r"\label{tab:supp_accel_cfg_%s}\\" % fam_label,
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{@{}l}{\emph{\tablename~\thetable\ (continued)}} \\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r@{}}{\emph{continued on the next page}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    first = True
    for method in [m for m in METHODS if FAMILY[m] == family]:
        mrows = [r for r in rows if r["method"] == method]
        if not mrows:
            continue
        if not first:
            out.append(r"\addlinespace[2pt]")
        first = False
        out.append(r"\multicolumn{5}{@{}l}{\textbf{%s}} \\*" % METHOD_DISPLAY[method])
        out.append(r"\addlinespace[1pt]")
        prev_model = None
        for r in sorted(mrows, key=lambda x: (x["model"], x["view"], x["embodiment"], x["layout"])):
            if r["model"] == "ctrlworld" and r["n_layouts"] > 1 and not r["layouts_agree"]:
                track = "%s, %s" % (VIEW_DISPLAY[r["view"]], LAYOUT_DISPLAY[r["layout"]])
            elif (r["model"] == "ctrlworld" and r["n_layouts"] > 1
                    and r["layout"] != "ctrlworld"):
                continue  # identical to the 3-view layout already printed
            else:
                # A single surviving layout is NOT labelled. After exclusion each
                # embodiment reports exactly one Ctrl-World multi-view canvas, so a
                # "3-view"/"4-view grid" tag distinguishes nothing within the table
                # and instead invites the reader to look for the other layout. Which
                # canvas each embodiment uses is stated once in the caption.
                track = VIEW_DISPLAY[r["view"]]
            # Print the model name only on its first surviving row; computing
            # this after the `continue` above is what keeps it attached to the
            # row actually emitted.
            model_cell = MODEL_DISPLAY[r["model"]] if r["model"] != prev_model else ""
            n_cell = "$%s$" % "{,}".join(_grp(r["rollouts"])) if r["ran"] else r"\texttt{--}"
            cfg = r["hyperparams"] if r["ran"] else r"\emph{not in released coverage}"
            out.append("%s & %s & %s & %s & %s \\\\" % (
                model_cell, track, EMB_DISPLAY[r["embodiment"]], cfg, n_cell))
            prev_model = r["model"]
    out += [r"\end{longtable}", r"}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))
    return path


def _grp(n):
    s = str(int(n))
    parts = []
    while len(s) > 3:
        parts.insert(0, s[-3:])
        s = s[:-3]
    parts.insert(0, s)
    return parts


def main():
    parsed = parse_scripts()
    coverage, audit = load_coverage()
    rows = build_rows(parsed, coverage)

    os.makedirs(TABLE_DIR, exist_ok=True)
    written = []
    for family in ("token", "cache", "sparse"):
        path = os.path.join(TABLE_DIR, "accel_hyperparams_%s.tex" % family)
        written.append(emit_table(family, [r for r in rows if r["family"] == family], path))

    # Divergences worth a human decision, surfaced rather than smoothed over.
    name_variants = {}
    for rec in parsed.values():
        for var, _ in rec["raw_env"].items():
            name_variants.setdefault(CANON[var], set()).add(var)
    aliased = {k: sorted(v) for k, v in name_variants.items() if len(v) > 1}

    ran_but_no_script, script_but_not_run = [], []
    for key, n in coverage.items():
        fam, emb, view, method = key
        if method not in METHODS:
            continue  # `dense` is the baseline, it has no method wrapper
        has = any(family_of(m) == fam and (e, v, me) == (emb, view, method)
                  and not is_excluded(m, e, v)
                  for (m, e, v, me) in parsed)
        if n and not has:
            ran_but_no_script.append(list(key))
        if not n and has:
            script_but_not_run.append(list(key))

    rules = load_insight_rules()
    divergences = audit_divergences(rows, rules)
    unjustified = [d for d in divergences if not d["justified"]]
    # The prose claims every undocumented divergence sits in a cell that has no
    # insights rule for that family. Verify it here so the claim cannot drift.
    for d in unjustified:
        assert rule_for(rules, d["layout"], d["cell"].split("/")[1],
                        d["cell"].split("/")[2], FAMILY[d["method"]]) is None, d

    payload = {
        "generated_by": "assets/scripts/make_impl_config_tables.py",
        "purpose": "Shipped acceleration-method configurations per benchmark cell, "
                   "parsed from the canonical run scripts and joined against the "
                   "rollout audit.",
        "n_scripts_parsed": len(parsed),
        "n_rows": len(rows),
        # The paper does not report every cell that exists on disk. Record the
        # exclusion in the artefact itself, so a reader of this JSON can tell a
        # cell that was never built from one that was deliberately not reported.
        "excluded_cells": [
            {"model": c["model"], "embodiment": c["embodiment"],
             "view": c["view"], "layout": c["layout"],
             "superseded_by": c["superseded_by"], "reason": c["reason"]}
            for c in EXCLUDED_CELLS
        ],
        "excluded_cells_note": (
            "Ctrl-World's 3-view latent strip is not reported for bimanual or "
            "humanoid: the same episodes were regenerated on the 4-view 2x2 "
            "latent canvas, which supersedes it. single_arm keeps its 3-view "
            "strip because it has no 4-view grid variant. The rollouts and run "
            "scripts of the excluded cells still exist; this is a reporting "
            "decision, not a deletion."),
        "n_scripts_excluded": sum(
            1 for (m, e, v, _me) in parsed if is_excluded(m, e, v)),
        # `n_scripts_parsed` counts every run script on disk; the paper's
        # coverage claim must use the REPORTED count instead, which is what the
        # tables actually print.
        "n_scripts_reported": sum(
            1 for (m, e, v, _me) in parsed if not is_excluded(m, e, v)),
        "audit_method_totals": audit["method_totals"],
        "aliased_env_names": aliased,
        "coverage_without_script": sorted(ran_but_no_script),
        "script_without_coverage": sorted(script_but_not_run),
        "insight_rules_indexed": {"/".join(str(x) for x in k): v
                                  for k, v in sorted(rules.items(), key=str)},
        "divergences": divergences,
        "n_divergences": len(divergences),
        "n_divergences_undocumented": len(unjustified),
        "reconciliation": {
            "policy": (
                "Each cell reports the configuration that generated its released "
                "rollouts, not one global default per method: every acceleration "
                "method was swept per (backbone, embodiment, track) because the "
                "attainable operating point depends on that cell's attention share "
                "and sequence length. A value departing from the method's modal "
                "value is therefore expected, and is accepted when the sweep that "
                "chose it is recorded either in the run script or in the matching "
                "insights rule."),
            "divergent_but_documented": (
                "%d of %d divergences carry a recorded sweep and are reported as "
                "deliberate per-cell operating points." % (
                    len(divergences) - len(unjustified), len(divergences))),
            "divergent_and_undocumented": (
                "%d divergences have no recorded sweep. All of them fall in "
                "(cell, method-family) combinations for which no insights rule "
                "exists: %s. Their values were therefore inherited rather than "
                "measured on that cell, which the appendix discloses instead of "
                "presenting them as tuned optima." % (
                    len(unjustified),
                    "; ".join(sorted({"%s/%s" % (d["cell"], FAMILY[d["method"]])
                                      for d in unjustified})))),
            "naming": (
                "The same knob is spelled differently across backbones (for "
                "example PISA_START_LAYER on the Ctrl-World wrappers versus "
                "PISA_START_LAYER_IDX on the DiT wrappers, and *_FIRST_LAYERS_FP "
                "versus *_START_LAYER_IDX, which express the identical quantity). "
                "All spellings are mapped onto one canonical parameter so that a "
                "column means the same thing over its whole length; the raw "
                "environment names are retained per row for reproducibility."),
            "runnable_but_not_released": (
                "%d cells have a working run script but zero released rollouts "
                "(SiTo and ITM on DreamDojo, FasterCache on Ctrl-World). They are "
                "printed as not-run rather than omitted, so that an absent result "
                "is never mistaken for a negative one." % len(script_but_not_run)),
        },
        "rows": rows,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)

    print("parsed %d method scripts -> %d rows" % (len(parsed), len(rows)))
    for p in written:
        print("  wrote", os.path.relpath(p, REPO))
    print("  wrote", os.path.relpath(OUT_JSON, REPO))
    print("\naliased env names (canonicalised):")
    for k, v in sorted(aliased.items()):
        print("  %-16s <- %s" % (k, ", ".join(v)))
    print("\nscript exists but zero released rollouts: %d cells" % len(script_but_not_run))
    for c in sorted(script_but_not_run):
        print("  ", "/".join(c))
    print("released rollouts with no matching script: %d cells" % len(ran_but_no_script))
    for c in sorted(ran_but_no_script):
        print("  ", "/".join(c))
    print("\nper-cell divergences from the modal configuration: %d" % len(divergences))
    print("  documented (script note or insights rule): %d"
          % (len(divergences) - len(unjustified)))
    print("  UNDOCUMENTED (no recorded sweep): %d" % len(unjustified))
    for d in unjustified:
        print("   %-12s %-36s %-14s %s (modal %s)" % (
            d["method"], d["cell"], d["parameter"], d["value"], d["modal_value"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
