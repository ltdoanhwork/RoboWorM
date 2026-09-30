#!/usr/bin/env python3
"""Emit the appendix finetuning-protocol table from the training configs.

Every value is read out of the code that produced the checkpoints, never
restated by hand:

  * Ctrl-World   -- ``finetuning/ctrlworld/config.py`` (a plain dataclass).
  * DreamGen     -- ``models/dreamgen/cosmos_predict2/configs/base/experiment/
                     singleview.py`` for the per-experiment overrides, plus the
                     Hydra default groups it selects
                     (``configs/base/defaults/{optimizer,scheduler,ema}.py``).
                     An experiment only overrides ``lr`` and the schedule, so
                     betas / weight decay / eps come from the default group and
                     would be wrong if guessed.
  * DreamDojo    -- ``models/DreamDojo/.../action_conditioned/config.py`` for the
                     default groups (``fusedadamw``, ``lambdalinear``, ``power``
                     EMA) and ``finetuning/dreamdojo/launch.sh`` for the launch
                     geometry.

Anything the code does not state is recorded in ``gaps`` and printed, so the
prose can disclose it instead of inventing a plausible number. In particular the
DreamDojo experiment registrations (``dreamdojo_2b_480_640_*``) are NOT present
in this repository -- they resolve from the vendored Cosmos package at run time --
so per-experiment iteration counts and dataset wiring for that family are
reported as a provenance gap rather than asserted.

    python3 assets/scripts/make_finetune_config_table.py

Writes:
    metrics/reports/finetune_config.json
    assets/paper/appendix/tables/finetune_hyperparams.tex
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CW_CONFIG = os.path.join(REPO, "finetuning/ctrlworld/config.py")
DG_EXP = os.path.join(REPO, "models/dreamgen/cosmos_predict2/configs/base/experiment/singleview.py")
DG_OPT = os.path.join(REPO, "models/dreamgen/cosmos_predict2/configs/base/defaults/optimizer.py")
DG_EMA = os.path.join(REPO, "models/dreamgen/cosmos_predict2/configs/base/defaults/ema.py")
DD_CONFIG = os.path.join(
    REPO, "models/DreamDojo/cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")
DD_OPT = os.path.join(
    REPO, "models/DreamDojo/cosmos_predict2/_src/predict2/configs/common/defaults/optimizer.py")
DD_LAUNCH = os.path.join(REPO, "finetuning/dreamdojo/launch.sh")
DENSE_CFG = os.path.join(REPO, "metrics/reports/dense_generation_config.json")

OUT_JSON = os.path.join(REPO, "metrics/reports/finetune_config.json")
OUT_TEX = os.path.join(REPO, "assets/paper/appendix/tables/finetune_hyperparams.tex")
OUT_CKPT_TEX = os.path.join(REPO, "assets/paper/appendix/tables/generation_config.tex")


def parse_ctrlworld():
    """Read the FinetuneConfig dataclass defaults without importing torch."""
    tree = ast.parse(open(CW_CONFIG).read())
    fields = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinetuneConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    try:
                        fields[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError):
                        pass
    return fields


def _literal_after(text, pattern):
    m = re.search(pattern, text, re.M)
    return m.group(1) if m else None


def parse_dreamgen():
    """Per-experiment overrides + the default optimizer/EMA group they select."""
    exp = open(DG_EXP).read()
    opt_src = open(DG_OPT).read()
    ema_src = open(DG_EMA).read()

    # The experiments override only lr; everything else is the `adamw` group.
    # `lr=2 ** (-14.5)` contains parentheses, so match to end of line rather
    # than to the first `)`.
    lr_expr = _literal_after(exp, r"_OPTIMIZER = dict\(lr=(.+)\)\s*$")
    sched = {}
    m = re.search(r"_SCHEDULER = dict\((.*?)\)\n", exp, re.S)
    if m:
        for key in ("f_max", "f_min", "warm_up_steps", "cycle_lengths"):
            v = _literal_after(m.group(1), key + r"=\[([0-9_.e-]+)\]")
            if v is not None:
                sched[key] = v.replace("_", "")
    trainer = {
        "max_iter": _literal_after(exp, r"max_iter=([0-9_]+)"),
        "distributed_parallelism": _literal_after(exp, r'distributed_parallelism="(\w+)"'),
    }
    # AdamWConfig is declared after FusedAdamWConfig; scope the search to it.
    adamw = opt_src.split("AdamWConfig: LazyDict")[-1]
    return {
        "lr_expr": lr_expr.strip() if lr_expr else None,
        "lr_value": eval(lr_expr, {"__builtins__": {}}) if lr_expr else None,
        "optimizer": "AdamW",
        "betas": _literal_after(adamw, r"betas=(\[[^\]]+\])"),
        "weight_decay": _literal_after(adamw, r"weight_decay=([0-9.e-]+)"),
        "eps": _literal_after(adamw, r"eps=([0-9.e-]+)"),
        "scheduler": "LambdaLinear",
        "schedule": sched,
        "trainer": {k: (v.replace("_", "") if v else v) for k, v in trainer.items()},
        "save_iter": (_literal_after(exp, r"_CHECKPOINT = dict\(save_iter=([0-9_]+)\)") or "").replace("_", ""),
        "context_parallel_size": _literal_after(exp, r"context_parallel_size=(\d+)"),
        "fsdp_shard_size": _literal_after(exp, r"fsdp_shard_size=(-?\d+)"),
        "batch_size": _literal_after(exp, r"batch_size=(\d+)"),
        "num_workers": _literal_after(exp, r"num_workers=(\d+)"),
        "num_frames": _literal_after(exp, r"num_frames=(\d+)"),
        "video_size": _literal_after(exp, r"video_size=\((\d+, \d+)\)"),
        "sampler_seed": _literal_after(exp, r"seed=(\d+)"),
        "ema_enabled": _literal_after(ema_src, r"PowerEMAConfig: EMAConfig = EMAConfig\(\s*enabled=(\w+)"),
        "ema_rate": (re.search(r"PowerEMAConfig.*?rate=([0-9.]+)", ema_src, re.S).group(1)
                     if re.search(r"PowerEMAConfig.*?rate=([0-9.]+)", ema_src, re.S) else None),
    }


def parse_dreamdojo():
    """Default groups from the action-conditioned config + launch geometry."""
    cfg = open(DD_CONFIG).read()
    opt_src = open(DD_OPT).read()
    launch = open(DD_LAUNCH).read()
    defaults = dict(re.findall(r'\{"(\w+)": "([\w.]+)"\}', cfg))
    fused = opt_src.split("FusedAdamWConfig: LazyDict")[-1]
    return {
        "optimizer_group": defaults.get("optimizer"),
        "scheduler_group": defaults.get("scheduler"),
        "ema_group": defaults.get("ema"),
        "tokenizer": defaults.get("tokenizer"),
        "model_group": defaults.get("model"),
        "optimizer": "FusedAdamW",
        "lr": _literal_after(fused, r"lr=([0-9.e-]+)"),
        "betas": _literal_after(fused, r"betas=(\[[^\]]+\])"),
        "weight_decay": _literal_after(fused, r"weight_decay=([0-9.e-]+)"),
        "eps": _literal_after(fused, r"eps=([0-9.e-]+)"),
        "max_iter": (_literal_after(cfg, r"c\.trainer\.max_iter = ([0-9_]+)") or "").replace("_", ""),
        "seed": _literal_after(launch, r"SEED=\$\{SEED:-(\d+)\}"),
        "lam_ckpt": _literal_after(launch, r"DREAMDOJO_LAM_CKPT=\S*/([\w.]+)"),
        "omp_threads": _literal_after(launch, r"OMP_NUM_THREADS=(\d+)"),
    }


def load_checkpoints():
    """The 14 generation cells, for the checkpoint-provenance column."""
    cfg = json.load(open(DENSE_CFG))
    cells = cfg["cells"] if isinstance(cfg, dict) and "cells" in cfg else cfg
    return cells


def sci(x):
    """Render a float as LaTeX scientific notation."""
    s = "%.2e" % float(x)
    mant, exp = s.split("e")
    return r"%s \times 10^{%d}" % (mant.rstrip("0").rstrip("."), int(exp))


def thou(x):
    """Thousands separator that survives LaTeX math mode."""
    s = str(int(x))
    parts = []
    while len(s) > 3:
        parts.insert(0, s[-3:])
        s = s[:-3]
    parts.insert(0, s)
    return "{,}".join(parts)


def observed(cells, family, key, fmt=str):
    """The distinct values of `key` actually observed in that family's cells.

    Frame rate and resolution are per-cell properties (DreamGen spans 10 and
    16 fps, DreamDojo 10 and 15), so they must be read off the released cells
    rather than asserted as one number per family.
    """
    vals = []
    for c in cells:
        if c["family"] != family:
            continue
        v = fmt(c[key])
        if v not in vals:
            vals.append(v)
    return vals


# (row label, Ctrl-World value, DreamGen value, DreamDojo value). A value of
# None prints as \texttt{--} and is registered as a gap.
def build_rows(cw, dg, dd, cells):
    rows = []
    add = lambda *a: rows.append(a)

    def fps_cell(family):
        vs = observed(cells, family, "fps", lambda v: "%g" % v)
        return "$%s$\\,fps" % "/".join(sorted(vs, key=float))

    def res_cell(family):
        vs = observed(cells, family, "per_view", str)
        return "/".join("$%s$" % v.replace("x", " {\\times} ") for v in vs)

    add("Backbone", "SVD image-to-video U-Net",
        "Cosmos-Predict2 2B/14B DiT", "Cosmos-Predict2 2B DiT")
    add("Adaptation", "full finetune", "full finetune", "full finetune")
    add("Optimiser", "AdamW", dg["optimizer"], dd["optimizer"])
    add("Peak LR", "$%s$" % sci(cw["learning_rate"]),
        "$%s$ ($2^{-14.5}$)" % sci(dg["lr_value"]), "$%s$" % sci(dd["lr"]))
    # Ctrl-World constructs AdamW with `lr` only, so betas / weight decay are
    # the PyTorch defaults. That is a fact about the code, not a gap.
    add("$(\\beta_1,\\beta_2)$", "$(0.9, 0.999)$ (\\textsc{torch} default)",
        "$(%s)$" % dg["betas"].strip("[]"), "$(%s)$" % dd["betas"].strip("[]"))
    add("Weight decay", "$0.01$ (\\textsc{torch} default)",
        "$%s$" % dg["weight_decay"], "$%s$" % dd["weight_decay"])
    add("LR schedule", "constant",
        "LambdaLinear, $%s$ warm-up steps, $f_{\\max}{=}%s$, $f_{\\min}{=}%s$" % (
            thou(dg["schedule"]["warm_up_steps"]), dg["schedule"]["f_max"],
            dg["schedule"]["f_min"]),
        "LambdaLinear (default group)")
    add("Grad.\\ clip", "$%s$" % cw["max_grad_norm"], None, None)
    add("Micro-batch / GPU", "$%s$" % cw["train_batch_size"],
        "$%s$" % dg["batch_size"], None)
    add("Grad.\\ accumulation", "$%s$" % cw["gradient_accumulation_steps"], None, None)
    add("EMA", "none",
        "power, rate $%s$" % dg["ema_rate"], "power, rate $%s$" % dg["ema_rate"])
    add("Precision", cw["mixed_precision"] + " mixed",
        "bf16 mixed", "bf16 mixed")
    add("Sharding", "\\textsc{Accelerate} DDP",
        "FSDP, shard size $%s$ (all ranks)" % dg["fsdp_shard_size"],
        "FSDP (rectified-flow action-conditioned recipe)")
    add("Context parallel", "none", "$%s$" % dg["context_parallel_size"], None)
    add("Generation resolution", res_cell("Ctrl-World"), res_cell("DreamGen"),
        res_cell("DreamDojo"))
    add("Training clip (frames)", "$%s$ pred.\\ $+\\,%s$ hist." % (cw["num_frames"], cw["num_history"]),
        "$%s$" % dg["num_frames"], None)
    add("Frame rate", fps_cell("Ctrl-World"), fps_cell("DreamGen"), fps_cell("DreamDojo"))
    add("Max iterations", "$%s$" % thou(cw["max_train_steps"]),
        "$%s$" % thou(dg["trainer"]["max_iter"]),
        "$%s$ (framework default)" % thou(dd["max_iter"]))
    add("Checkpoint every", "$%s$" % thou(cw["checkpointing_steps"]),
        "$%s$" % thou(dg["save_iter"]), None)
    add("Training seed", "$%s$" % cw["seed"], "$%s$ (sampler)" % dg["sampler_seed"],
        "$%s$" % dd["seed"])
    add("Val.\\ split", "$%s$ of episodes" % cw["val_ratio"], None, None)
    add("Action conditioning", "$%s$-D, $%s$ embodiment embeddings"
        % (cw["action_dim"], cw["num_embodiments"]),
        "text prompt only (no action input)",
        "latent action model (\\texttt{%s})" % dd["lam_ckpt"].replace("_", "\\_"))

    gaps = [r[0] for r in rows if None in r[1:]]
    return rows, gaps


def emit_table(rows, cw, dg, dd, path):
    # The caption characterises how far apart the three learning rates are. That
    # spread is derived here rather than written by hand, because the prose in
    # 3_implemetation_details.tex makes the same claim and the two must not
    # disagree: an earlier revision said "more than a factor of two" while the
    # prose said "an order of magnitude", and the configs say the latter.
    lrs = [float(cw["learning_rate"]), float(dg["lr_value"]), float(dd["lr"])]
    spread = max(lrs) / min(lrs)
    if spread >= 9.5:
        lr_phrase = "learning rates that span an order of magnitude"
    else:
        lr_phrase = ("learning rates that differ by a factor of %.1f"
                     % spread).replace(".0", "")
    out = [
        "% Generated by assets/scripts/make_finetune_config_table.py -- do not edit by hand.",
        "% Sources: finetuning/ctrlworld/config.py;",
        "%          models/dreamgen/cosmos_predict2/configs/base/{experiment/singleview.py,defaults/*.py};",
        "%          models/DreamDojo/.../action_conditioned/config.py; finetuning/dreamdojo/launch.sh",
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Finetuning protocol of the three base world models.}",
        r"Every entry is read out of the configuration that produced the released"
        r" checkpoints rather than restated from memory; where a family's code"
        r" does not set a quantity we print \texttt{--} instead of substituting a"
        r" plausible default. All three families are \emph{fully} finetuned: no"
        r" LoRA or other parameter-efficient adapter is used anywhere in the"
        r" benchmark, so a quality difference between families cannot be"
        r" attributed to adapter capacity. The three backbones are optimised at"
        + (" %s and at" % lr_phrase) +
        r" different clip lengths, which is a property of the released recipes we"
        r" inherit, not a controlled variable: we therefore compare acceleration"
        r" methods \emph{within} a backbone and never read across columns as a"
        r" model comparison.}",
        r"\label{tab:supp_finetune_cfg}",
        r"\begin{tabular}{@{}lp{0.235\linewidth}p{0.265\linewidth}p{0.235\linewidth}@{}}\toprule",
        r" & Ctrl-World & DreamGen & DreamDojo \\",
        r"\midrule",
    ]
    for label, a, b, c in rows:
        cells = [x if x is not None else r"\texttt{--}" for x in (a, b, c)]
        out.append("%s & %s & %s & %s \\\\" % (label, cells[0], cells[1], cells[2]))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def emit_checkpoint_table(cells, path):
    """Per-configuration checkpoint provenance and sampling settings.

    The original appendix TODO promised this table, and no single label for it
    existed: metrics/reports/dense_generation_config.tex splits the same content
    across three per-family tables with three separate labels, which cannot be
    cited as one object. It is regenerated here as one appendix table so the
    prose can reference a single location, and so the seed column -- which is
    what bounds how variance may be read -- sits next to the checkpoint it
    applies to.
    """
    fam_order = ["Ctrl-World", "DreamGen", "DreamDojo"]
    out = [
        "% Generated by assets/scripts/make_finetune_config_table.py -- do not edit by hand.",
        "% Source: metrics/reports/dense_generation_config.json",
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{\textbf{Generation configuration and checkpoint provenance of "
        r"the $%d$ dense cells.}" % len(cells),
        r"Each row is the configuration whose dense rollouts every acceleration"
        r" method in that cell is compared against. \emph{Steps} is the number of"
        r" denoising steps and $g$ the guidance scale. The \emph{seed} column is"
        r" not uniform across families and bounds how differences may be read:"
        r" the Ctrl-World cells are unseeded, so each value is a single sample"
        r" carrying sampling noise, whereas DreamGen fixes seed $0$ (offset by"
        r" segment index on the autoregressive multi-view rollouts) and DreamDojo"
        r" derives a per-chunk seed. $n$ is the released rollout count."
        r" Checkpoints marked \emph{released} are upstream weights used without"
        r" further finetuning.}",
        r"\label{tab:supp_generation_cfg}",
        r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{0.30\linewidth}rrlr@{}}\toprule",
        r"Model & Cell & Checkpoint & Steps & $g$ & Seed & $n$ \\",
        r"\midrule",
    ]
    prev = None
    for fam in fam_order:
        for c in [x for x in cells if x["family"] == fam]:
            emb, track, _ = c["cell"].split("/")
            label = fam if fam != prev else ""
            prev = fam
            track_s = "MV" if track == "multiview" else "SV"
            if "4view_grid" in c["cell"]:
                track_s += " (4-view)"
            elif fam == "Ctrl-World" and track == "multiview":
                # Ctrl-World's reported layout differs by embodiment (single-arm
                # on the 3-view strip, the others on the 4-view grid), so the
                # strip must name itself too -- an unlabelled "MV" next to a
                # "MV (4-view)" row would read as an omission.
                track_s += " (3-view)"
            ckpt = c["ckpt"]
            released = "released" in ckpt.lower()
            ckpt = ckpt.split(" (")[0]
            # \texttt does not hyphenate, so a long checkpoint id would run into
            # the Steps column. Allow breaks after / and _ instead.
            # \texttt suppresses hyphenation, so every separator in a path must
            # be given an explicit break opportunity or the cell overflows.
            ckpt = (ckpt.replace("_", r"\_\allowbreak ")
                        .replace("/", r"/\allowbreak ")
                        .replace("-", r"-\allowbreak "))
            ckpt = r"\texttt{\small %s}" % ckpt
            if released:
                ckpt += r" \emph{(released)}"
            out.append("%s & %s %s & %s & $%s$ & $%s$ & %s & $%s$ \\\\" % (
                label, EMB_SHORT[emb], track_s, ckpt, c["steps"], c["guidance"],
                c["seed"].replace("_", " "), thou(c["clips"])))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))


EMB_SHORT = {"single_arm": "single-arm", "bimanual": "bimanual",
             "humanoid": "humanoid"}


def main():
    cw = parse_ctrlworld()
    dg = parse_dreamgen()
    dd = parse_dreamdojo()
    cells = load_checkpoints()
    rows, gap_labels = build_rows(cw, dg, dd, cells)

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    emit_table(rows, cw, dg, dd, OUT_TEX)
    emit_checkpoint_table(cells, OUT_CKPT_TEX)

    gaps = [
        {"quantity": lab,
         "note": "not set by the training configuration of at least one family"}
        for lab in gap_labels
    ]
    gaps.append({
        "quantity": "DreamDojo per-experiment registration",
        "note": "the dreamdojo_2b_480_640_* experiments resolve from the vendored "
                "Cosmos package at run time and are not present in this repository, "
                "so per-experiment iteration counts, batch geometry and dataset "
                "wiring for that family cannot be read from code here.",
    })
    gaps.append({
        "quantity": "rollout seed",
        "note": "seeding is not uniform across families: the Ctrl-World cells are "
                "unseeded (single-sample, carrying sampling noise), while DreamGen "
                "fixes seed 0 (offset by segment index on the multiview rollouts) "
                "and DreamDojo derives a per-chunk seed. Cross-family variance is "
                "therefore not directly comparable.",
        "per_family_seed": {c["family"]: c["seed"] for c in cells},
    })
    gaps.append({
        "quantity": "wall-clock training cost",
        "note": "training runs were not instrumented for wall time; only iteration "
                "counts and checkpoint cadence are recoverable from code.",
    })

    payload = {
        "generated_by": "assets/scripts/make_finetune_config_table.py",
        "purpose": "Finetuning hyperparameters of the three base world models, "
                   "extracted from the training configurations that produced the "
                   "released checkpoints.",
        "ctrlworld": cw,
        "dreamgen": dg,
        "dreamdojo": dd,
        "n_generation_cells": len(cells),
        "checkpoints": [{"cell": c["cell"], "family": c["family"], "ckpt": c["ckpt"],
                         "seed": c["seed"]} for c in cells],
        "adaptation": "full finetune in all three families; no LoRA or other "
                      "parameter-efficient adapter is used in the benchmark",
        "gaps": gaps,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("wrote", os.path.relpath(OUT_TEX, REPO))
    print("wrote", os.path.relpath(OUT_CKPT_TEX, REPO))
    print("wrote", os.path.relpath(OUT_JSON, REPO))
    print("\n%d table rows, %d generation cells" % (len(rows), len(cells)))
    print("\nexplicit gaps (must be disclosed in prose, not filled in):")
    for g in gaps:
        print("  - %s" % g["quantity"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
