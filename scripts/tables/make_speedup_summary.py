#!/usr/bin/env python3
"""Aggregate dense-vs-acceleration-method results across the whole
sampling_dataset tree into one master table for the paper's "Main Results"
section (speedup techniques, C.i).

Design notes (read before editing)
-----------------------------------
sampling_dataset/ is NOT uniform:

- Ctrl-World and DreamDojo write ONE `all_summary.json` per
  {embodiment}/output/{view}/{model}/{split}[/iter_XXXXXX] directory, with
  `mean_psnr/mean_ssim/mean_lpips/num_episodes[_processed]`. Only a small
  subset of these (measured: 20 of 130 files, all Ctrl-World multiview) also
  carry `mean_wall_time_s` -- DreamDojo's `all_summary.json` never does.
- DreamGen MULTIVIEW writes per-episode `metrics.json` (no `all_summary.json`
  at all) -- real full-scale PSNR/SSIM/LPIPS exist here for all 9 methods.
- DreamGen SINGLEVIEW writes only `.mp4` + `.txt` sidecars -- there is no
  PSNR/SSIM/LPIPS anywhere in sampling_dataset for these cells.

So per-cell timing (for "speedup") is persisted for only a handful of
Ctrl-World multiview cells. Everywhere else we fall back to the exact numbers
already measured and written down during method tuning, in this repo's own
`.cursor/rules/*-sparse-attention-insights.mdc` /
`*-caching-insights.mdc` / `*-pruning-insights.mdc` files (see
SPEEDUP_LOOKUP below, each entry cites which rule file it came from). This
is a deliberate mixed-provenance design: prefer measured/persisted numbers,
fall back to the documented tuning-run numbers, and ALWAYS record which one
was used in the `timing_source` / `quality_source` columns -- never silently
blend or hide provenance (per this repo's acceleration-benchmark rule).

Usage
-----
    python scripts/tables/make_speedup_summary.py \
        --sampling-root sampling_dataset \
        --out-dir metrics/reports

Emits:
    metrics/reports/speedup_summary_full.csv       (one row per cell)
    metrics/reports/speedup_summary_by_method.csv  (one row per method, rolled up)
    metrics/reports/speedup_summary_appendix.tex   (LaTeX longtable, appendix-ready)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional

# Which cells the paper reports is decided in one shared place, so this CSV, the
# appendix tables and the figures cannot disagree about the benchmark's size.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "assets" / "scripts"))
from benchmark_cells import describe as describe_exclusions  # noqa: E402
from benchmark_cells import is_excluded  # noqa: E402

METHODS = [
    "dense", "dicache", "worldcache", "fastercache",
    "sito", "itm", "pisa", "radial", "svg1", "svg2",
]
ACCEL_METHODS = [m for m in METHODS if m != "dense"]

METHOD_DISPLAY = {
    "dense": "Dense", "dicache": "DiCache", "worldcache": "WorldCache",
    "fastercache": "FasterCache", "sito": "SiTo", "itm": "ITM",
    "pisa": "PISA", "radial": "Radial", "svg1": "SVG1", "svg2": "SVG2",
}

MODEL_FAMILY = {
    "ctrlworld": "ctrlworld",
    "ctrlworld_4view_grid": "ctrlworld",
    "dreamgen": "dreamgen",
    "dreamdojo": "dreamdojo",
}
FAMILY_DISPLAY = {"ctrlworld": "Ctrl-World", "dreamgen": "DreamGen", "dreamdojo": "DreamDojo"}

# Regex over a path relative to sampling_dataset/<method>/, e.g.
#   bimanual/output/multiview/dreamgen/non_makovian/iter_000120000/fold_green_towel/episode_000090/metrics.json
#   humanoid/output/multiview/ctrlworld/makovian/all_summary.json
CELL_RE = re.compile(
    r"^(?P<embodiment>single_arm|bimanual|humanoid)/output/"
    r"(?P<view>singleview|multiview)/(?P<model>[a-zA-Z0-9_]+)/"
    r"(?P<split>[a-zA-Z0-9_]+)"
    r"(?:/(?P<extra>iter_\d+[a-zA-Z0-9_]*))?"
)

# ---------------------------------------------------------------------------
# SPEEDUP_LOOKUP: (method, model_family, embodiment, view) -> dict(speedup=..,
# note=.., source=<rule file stem>). Populated only for cells where
# sampling_dataset has NO persisted mean_wall_time_s. Numbers are transcribed
# verbatim from this repo's own tuning-insight rules (DEFAULT / winning
# config per rule, i.e. the config actually used by the canonical
# run_*_<method>.sh script for that cell).
# ---------------------------------------------------------------------------
SPEEDUP_LOOKUP: dict[tuple[str, str, str, str], dict] = {}


def _add(method, family, embodiment, view, speedup, source, note=""):
    """Register a documented speedup.

    `family` here is the **model key**, i.e. the 4th path segment under
    sampling_dataset/<method>/<emb>/output/<view>/ — so `ctrlworld` and
    `ctrlworld_4view_grid` are DISTINCT keys. They MUST be: the 4-view 2x2 grid
    tolerates caching far better than the 3-view strip (WorldCache 1.81x/1.60x
    vs 1.27x/1.16x), so keying on the collapsed display family silently
    stamped 3-view numbers onto every 4-view-grid row.
    """
    SPEEDUP_LOOKUP[(method, family, embodiment, view)] = {
        "speedup": speedup, "source": source, "note": note,
    }


# --- DreamGen humanoid singleview (2B, static16fps) ---
# dreamgen-humanoid-singleview-sparse-attention-insights.mdc, dense=115.3s/chunk
_add("pisa", "dreamgen", "humanoid", "singleview", 1.45, "dreamgen-humanoid-singleview-sparse-attention-insights", "density0.2")
_add("radial", "dreamgen", "humanoid", "singleview", 1.36, "dreamgen-humanoid-singleview-sparse-attention-insights", "start4/hunyuan")
_add("svg1", "dreamgen", "humanoid", "singleview", 1.28, "dreamgen-humanoid-singleview-sparse-attention-insights", "sparsity0.25/start4")
_add("svg2", "dreamgen", "humanoid", "singleview", 1.15, "dreamgen-humanoid-singleview-sparse-attention-insights", "q32/k64/start1")
# dreamgen-humanoid-singleview-caching-insights.mdc, dense=259.9s/2-chunk ep
_add("dicache", "dreamgen", "humanoid", "singleview", 1.95, "dreamgen-humanoid-singleview-caching-insights", "thresh0.15")
_add("worldcache", "dreamgen", "humanoid", "singleview", 1.89, "dreamgen-humanoid-singleview-caching-insights", "thresh0.15/ret0.3")
_add("fastercache", "dreamgen", "humanoid", "singleview", 1.15, "dreamgen-humanoid-singleview-caching-insights", "model_interval5/block_interval3")
# no SiTo/ITM singleview-humanoid dreamgen rule was measured; leave absent

# --- DreamGen single_arm singleview (2B, static16fps) ---
# dreamgen-single-arm-singleview-sparse-attention-insights.mdc, dense=225.6s/2chunks
_add("pisa", "dreamgen", "single_arm", "singleview", 1.37, "dreamgen-single-arm-singleview-sparse-attention-insights", "density0.2")
_add("radial", "dreamgen", "single_arm", "singleview", 1.34, "dreamgen-single-arm-singleview-sparse-attention-insights", "start4/hunyuan")
_add("svg1", "dreamgen", "single_arm", "singleview", 1.39, "dreamgen-single-arm-singleview-sparse-attention-insights", "sparsity0.15 (DEFAULT)")
_add("svg2", "dreamgen", "single_arm", "singleview", 1.26, "dreamgen-single-arm-singleview-sparse-attention-insights", "q32/k64/top_p0.7")
# dreamgen-singlearm-singleview-pruning-insights.mdc, dense=227.5s/2-chunk ep
_add("sito", "dreamgen", "single_arm", "singleview", 1.24, "dreamgen-singlearm-singleview-pruning-insights", "prune_ratio0.30")
_add("itm", "dreamgen", "single_arm", "singleview", 0.91, "dreamgen-singlearm-singleview-pruning-insights", "start11/keep4/compress0.20 (net-negative)")

# --- DreamGen single_arm multiview (14B, droid) ---
# dreamgen-single-arm-multiview-sparse-attention-insights.mdc, dense=491.9s/ep
_add("pisa", "dreamgen", "single_arm", "multiview", 1.32, "dreamgen-single-arm-multiview-sparse-attention-insights", "density0.2")
_add("radial", "dreamgen", "single_arm", "multiview", 1.29, "dreamgen-single-arm-multiview-sparse-attention-insights", "start4/hunyuan")
_add("svg1", "dreamgen", "single_arm", "multiview", 1.22, "dreamgen-single-arm-multiview-sparse-attention-insights", "sparsity0.25/start4")
_add("svg2", "dreamgen", "single_arm", "multiview", 1.02, "dreamgen-single-arm-multiview-sparse-attention-insights", "q32/k64/start1")

# --- DreamGen humanoid multiview (2B, 2x2 grid) ---
# dreamgen-humanoid-multiview-sparse-attention-insights.mdc, dense=115.1s/seg
_add("pisa", "dreamgen", "humanoid", "multiview", 1.44, "dreamgen-humanoid-multiview-sparse-attention-insights", "density0.2")
_add("svg1", "dreamgen", "humanoid", "multiview", 1.29, "dreamgen-humanoid-multiview-sparse-attention-insights", "sparsity0.25/start4")
_add("svg2", "dreamgen", "humanoid", "multiview", 1.19, "dreamgen-humanoid-multiview-sparse-attention-insights", "q32/k64/start1")
_add("radial", "dreamgen", "humanoid", "multiview", 1.36, "dreamgen-humanoid-multiview-sparse-attention-insights", "start4/hunyuan")
# dreamgen-humanoid-multiview-caching-insights.mdc, dense=116.1s/seg
_add("dicache", "dreamgen", "humanoid", "multiview", 2.28, "dreamgen-humanoid-multiview-caching-insights", "thresh0.20/ret0.2/probe2")
_add("worldcache", "dreamgen", "humanoid", "multiview", 2.08, "dreamgen-humanoid-multiview-caching-insights", "thresh0.20/ret0.2/probe2")
_add("fastercache", "dreamgen", "humanoid", "multiview", 1.16, "dreamgen-humanoid-multiview-caching-insights", "m5/b5")

# --- DreamGen bimanual multiview (2B, 2x2 grid) ---
# dreamgen-bimanual-multiview-sparse-attention-insights.mdc, dense=118.4s/seg
_add("pisa", "dreamgen", "bimanual", "multiview", 1.46, "dreamgen-bimanual-multiview-sparse-attention-insights", "density0.2/start2")
_add("svg1", "dreamgen", "bimanual", "multiview", 1.45, "dreamgen-bimanual-multiview-sparse-attention-insights", "sparsity0.25")
_add("svg2", "dreamgen", "bimanual", "multiview", 1.40, "dreamgen-bimanual-multiview-sparse-attention-insights", "q32/k64/top_p0.6")
_add("radial", "dreamgen", "bimanual", "multiview", 1.41, "dreamgen-bimanual-multiview-sparse-attention-insights", "start8/wan")
_add("sito", "dreamgen", "bimanual", "multiview", 1.30, "dreamgen-bimanual-multiview-sparse-attention-insights", "prune0.30 (winner among pruning)")
_add("itm", "dreamgen", "bimanual", "multiview", 0.96, "dreamgen-bimanual-multiview-sparse-attention-insights", "net-negative")

# --- DreamDojo humanoid singleview (2B GR1) ---
# MEASURED 2026-09-22 by scripts/tools/tune/_bench_uniform_timing.sh (1 makovian + 1
# non_makovian episode, dense CONTROL re-run in the same batch on an idle GPU).
# These supersede the single-episode tuning-note numbers for this cell: same
# protocol for dense and every method, so the ratios are comparable.
_add("dicache", "dreamdojo", "humanoid", "singleview", 1.56, "measured:_bench_uniform_timing 2026-09-22", "thresh0.15/ret0.2/probe2 (PSNR == dense)")
_add("worldcache", "dreamdojo", "humanoid", "singleview", 1.08, "measured:_bench_uniform_timing 2026-09-22", "thresh0.15/ret0.3/probe2")
_add("fastercache", "dreamdojo", "humanoid", "singleview", 0.91, "measured:_bench_uniform_timing 2026-09-22", "m5/b5 (net-negative, quality == dense)")
_add("radial", "dreamdojo", "humanoid", "singleview", 1.00, "measured:_bench_uniform_timing 2026-09-22", "start2/hunyuan/decay1.0")
_add("pisa", "dreamdojo", "humanoid", "singleview", 1.01, "measured:_bench_uniform_timing 2026-09-22", "density0.2/block64/start2")
_add("svg1", "dreamdojo", "humanoid", "singleview", 0.96, "measured:_bench_uniform_timing 2026-09-22", "sparsity0.25 (profile cached)")
_add("svg2", "dreamdojo", "humanoid", "singleview", 0.70, "measured:_bench_uniform_timing 2026-09-22", "q32/k64/top_p0.6")
# SiTo / ITM on this cell FAIL to run at all: their run scripts point at
# datasets/humanoid/singleview/PhysicalAI-Robo... which does not exist, and they
# call `examples.action_conditioned` rather than the canonical infer script. Left
# absent (not 0, not estimated) until those scripts are repaired.

# --- DreamDojo bimanual singleview (2B ALOHA) ---
# dreamdojo-bimanual-singleview-sparse-attention-insights.mdc, dense=390.08s
_add("radial", "dreamdojo", "bimanual", "singleview", 1.02, "dreamdojo-bimanual-singleview-sparse-attention-insights", "start2/hunyuan/decay0.5")
_add("pisa", "dreamdojo", "bimanual", "singleview", 1.03, "dreamdojo-bimanual-singleview-sparse-attention-insights", "density0.2/block64/start2 (already lossy)")
_add("svg1", "dreamdojo", "bimanual", "singleview", 0.97, "dreamdojo-bimanual-singleview-sparse-attention-insights", "sparsity0.25 (SLOWER)")
_add("svg2", "dreamdojo", "bimanual", "singleview", 0.69, "dreamdojo-bimanual-singleview-sparse-attention-insights", "q32/k64/top_p0.6")

# --- DreamDojo single_arm singleview (2B DROID) ---
# dreamdojo-single-arm-singleview-sparse-attention-insights.mdc, dense=144.80s
_add("pisa", "dreamdojo", "single_arm", "singleview", 1.02, "dreamdojo-single-arm-singleview-sparse-attention-insights", "density0.2/block64/start2")
_add("radial", "dreamdojo", "single_arm", "singleview", 1.01, "dreamdojo-single-arm-singleview-sparse-attention-insights", "start2/hunyuan/decay0.5")
_add("svg1", "dreamdojo", "single_arm", "singleview", 0.91, "dreamdojo-single-arm-singleview-sparse-attention-insights", "sparsity0.25/rows16 (SLOWER)")
_add("svg2", "dreamdojo", "single_arm", "singleview", 0.69, "dreamdojo-single-arm-singleview-sparse-attention-insights", "q32/k64/top_p0.6")

# --- Ctrl-World humanoid multiview 3-VIEW STRIP (SVD UNet) ---
# ctrlworld-humanoid-multiview-caching-insights.mdc, dense=117.5s total gen (2 val eps)
_add("worldcache", "ctrlworld", "humanoid", "multiview", 1.16, "ctrlworld-humanoid-multiview-caching-insights", "thresh0.04/ret0.4")
_add("dicache", "ctrlworld", "humanoid", "multiview", 1.11, "ctrlworld-humanoid-multiview-caching-insights", "thresh0.03/ret0.4")
# ctrlworld-humanoid-multiview-pruning-insights.mdc
_add("sito", "ctrlworld", "humanoid", "multiview", 1.05, "ctrlworld-humanoid-multiview-pruning-insights", "ST-hold k0.8/d4")
_add("itm", "ctrlworld", "humanoid", "multiview", 1.25, "ctrlworld-humanoid-multiview-pruning-insights", "ST-hold k0.8/d4")

# --- Ctrl-World humanoid multiview 4-VIEW 2x2 GRID (separate ckpt, ~1.6x cache) ---
# ctrlworld-humanoid-multiview-4view-grid-caching-insights.mdc, dense=130.5s (2 val eps)
_add("worldcache", "ctrlworld_4view_grid", "humanoid", "multiview", 1.60, "ctrlworld-humanoid-multiview-4view-grid-caching-insights", "thresh0.03/ret0.0/probe1")
_add("dicache", "ctrlworld_4view_grid", "humanoid", "multiview", 1.56, "ctrlworld-humanoid-multiview-4view-grid-caching-insights", "thresh0.03/ret0.0/probe1")
# ctrlworld-humanoid-multiview-4view-grid-pruning-insights.mdc, dense=139.0s
_add("sito", "ctrlworld_4view_grid", "humanoid", "multiview", 1.11, "ctrlworld-humanoid-multiview-4view-grid-pruning-insights", "ST-hold k0.8/d4")
_add("itm", "ctrlworld_4view_grid", "humanoid", "multiview", 1.11, "ctrlworld-humanoid-multiview-4view-grid-pruning-insights", "ST-hold k0.85/d4")
# ctrlworld-humanoid-multiview-4view-grid-sparse-attention-insights.mdc, dense=82.5s
# All NET-NEGATIVE (kernel-bound at T<=3840) — valid negative benchmark points.
_add("pisa", "ctrlworld_4view_grid", "humanoid", "multiview", 0.96, "ctrlworld-humanoid-multiview-4view-grid-sparse-attention-insights", "density0.2/block64/start0")
_add("radial", "ctrlworld_4view_grid", "humanoid", "multiview", 0.92, "ctrlworld-humanoid-multiview-4view-grid-sparse-attention-insights", "fp2/hunyuan (mask held)")
_add("svg1", "ctrlworld_4view_grid", "humanoid", "multiview", 0.85, "ctrlworld-humanoid-multiview-4view-grid-sparse-attention-insights", "sparsity0.25/fp2 (masked SDPA)")
_add("svg2", "ctrlworld_4view_grid", "humanoid", "multiview", 0.66, "ctrlworld-humanoid-multiview-4view-grid-sparse-attention-insights", "q32/k64/top_p0.5/fp2")

# --- Ctrl-World bimanual multiview 3-VIEW (5fps ckpt-280000) ---
# ctrlworld-bimanual-multiview-caching-insights.mdc, dense=250.9s (117fr val ep)
_add("worldcache", "ctrlworld", "bimanual", "multiview", 1.27, "ctrlworld-bimanual-multiview-caching-insights", "thresh0.04/ret0.4")
_add("dicache", "ctrlworld", "bimanual", "multiview", 1.23, "ctrlworld-bimanual-multiview-caching-insights", "thresh0.04/ret0.4")
_add("fastercache", "ctrlworld", "bimanual", "multiview", 1.00, "ctrlworld-bimanual-multiview-caching-insights", "m5/b3 (no-op: 0 attn_processors, CFG off)")

# --- Ctrl-World bimanual multiview 4-VIEW 2x2 GRID (ckpt-110000) ---
# ctrlworld-bimanual-multiview-4view-grid-caching-insights.mdc, dense=270.3s
_add("worldcache", "ctrlworld_4view_grid", "bimanual", "multiview", 1.81, "ctrlworld-bimanual-multiview-4view-grid-caching-insights", "thresh0.04/ret0.0/probe1")
_add("dicache", "ctrlworld_4view_grid", "bimanual", "multiview", 1.72, "ctrlworld-bimanual-multiview-4view-grid-caching-insights", "thresh0.03/ret0.0/probe1")
_add("fastercache", "ctrlworld_4view_grid", "bimanual", "multiview", 1.02, "ctrlworld-bimanual-multiview-4view-grid-caching-insights", "m5/b3 (no-op: 0 attn layers)")
# ctrlworld-bimanual-multiview-4view-grid-pruning-insights.mdc, dense=225.7s/ep
_add("sito", "ctrlworld_4view_grid", "bimanual", "multiview", 1.08, "ctrlworld-bimanual-multiview-4view-grid-pruning-insights", "ST-hold k0.8/d4 (lossy)")
_add("itm", "ctrlworld_4view_grid", "bimanual", "multiview", 1.04, "ctrlworld-bimanual-multiview-4view-grid-pruning-insights", "ST-hold k0.85/d4")
# ctrlworld-bimanual-multiview-4view-grid-sparse-attention-insights.mdc, dense=225.7s/ep
_add("pisa", "ctrlworld_4view_grid", "bimanual", "multiview", 0.98, "ctrlworld-bimanual-multiview-4view-grid-sparse-attention-insights", "density0.2/start0")
_add("radial", "ctrlworld_4view_grid", "bimanual", "multiview", 0.94, "ctrlworld-bimanual-multiview-4view-grid-sparse-attention-insights", "fp2/hunyuan")
_add("svg1", "ctrlworld_4view_grid", "bimanual", "multiview", 0.88, "ctrlworld-bimanual-multiview-4view-grid-sparse-attention-insights", "sparsity0.25/fp2")
_add("svg2", "ctrlworld_4view_grid", "bimanual", "multiview", 0.74, "ctrlworld-bimanual-multiview-4view-grid-sparse-attention-insights", "q32/k64/top_p0.5/fp2")

# --- Ctrl-World single_arm multiview 3-VIEW STRIP (pretrained ckpt-10000) ---
# ctrlworld-single-arm-multiview-sparse-attention-insights.mdc, dense=164.6s/ep
# All NET-NEGATIVE; time is FLAT vs the quality knob (launch/kernel bound).
_add("pisa", "ctrlworld", "single_arm", "multiview", 0.95, "ctrlworld-single-arm-multiview-sparse-attention-insights", "density0.5/block64/start2")
_add("radial", "ctrlworld", "single_arm", "multiview", 0.93, "ctrlworld-single-arm-multiview-sparse-attention-insights", "fp4/hunyuan/decay1.0")
_add("svg1", "ctrlworld", "single_arm", "multiview", 0.88, "ctrlworld-single-arm-multiview-sparse-attention-insights", "sparsity0.25/rows32/fp2")
_add("svg2", "ctrlworld", "single_arm", "multiview", 0.65, "ctrlworld-single-arm-multiview-sparse-attention-insights", "q32/k64/top_p0.7/fp2")
_add("sito", "ctrlworld", "single_arm", "multiview", 1.00, "ctrlworld-pruning-insights", "ST-hold k0.9/d1 (quality-leaning, ~no speedup)")
# NOT-YET-MEASURED on this cell (scripts exist, timing never recorded):
#   dicache / worldcache / fastercache / itm -> run scripts/_timing_ctrlworld_sa_mv.sh

# --- DreamGen bimanual multiview (2B seg16) ---
# dreamgen-bimanual-multiview-caching-insights.mdc, dense=338s/ep (2 segments)
_add("dicache", "dreamgen", "bimanual", "multiview", 1.66, "dreamgen-bimanual-multiview-caching-insights", "thresh0.20/ret0.2")
_add("worldcache", "dreamgen", "bimanual", "multiview", 1.56, "dreamgen-bimanual-multiview-caching-insights", "thresh0.20/ret0.2/probe2")
_add("fastercache", "dreamgen", "bimanual", "multiview", 0.94, "dreamgen-bimanual-multiview-caching-insights", "m2/b2 (clean but slower than dense)")

# --- DreamGen bimanual singleview (2B, iter_000110000) ---
# dreamgen-bimanual-singleview-caching-insights.mdc, dense~263s/2-chunk ep
_add("dicache", "dreamgen", "bimanual", "singleview", 1.32, "dreamgen-bimanual-singleview-caching-insights", "thresh0.20/ret0.2/probe2")
_add("worldcache", "dreamgen", "bimanual", "singleview", 1.31, "dreamgen-bimanual-singleview-caching-insights", "thresh0.20/ret0.2/probe2")
_add("fastercache", "dreamgen", "bimanual", "singleview", 0.76, "dreamgen-bimanual-singleview-caching-insights", "m2/b2 (clean, net-negative)")
# dreamgen-bimanual-singleview-sparse-attention-insights.mdc, dense~263s
_add("pisa", "dreamgen", "bimanual", "singleview", 1.14, "dreamgen-bimanual-singleview-sparse-attention-insights", "density0.1/start0")
_add("radial", "dreamgen", "bimanual", "singleview", 1.10, "dreamgen-bimanual-singleview-sparse-attention-insights", "decay0.2/start1")
_add("svg1", "dreamgen", "bimanual", "singleview", 0.77, "dreamgen-bimanual-singleview-sparse-attention-insights", "sparsity0.25 (net-negative)")
_add("svg2", "dreamgen", "bimanual", "singleview", 0.84, "dreamgen-bimanual-singleview-sparse-attention-insights", "kmeans warm cache (net-negative)")
# dreamgen-bimanual-pruning-insights.mdc, dense 119.0s/seg (MV) & 118.9s/chunk (SV)
_add("sito", "dreamgen", "bimanual", "singleview", 1.29, "dreamgen-bimanual-pruning-insights", "prune0.30")
_add("itm", "dreamgen", "bimanual", "singleview", 0.95, "dreamgen-bimanual-pruning-insights", "mid start11/keep4/compress0.20 (net-negative)")

# --- DreamGen humanoid multiview pruning ---
# dreamgen-humanoid-multiview-pruning-insights.mdc, dense=119.9s/seg
_add("sito", "dreamgen", "humanoid", "multiview", 1.20, "dreamgen-humanoid-multiview-pruning-insights", "prune0.20 (knee)")
_add("itm", "dreamgen", "humanoid", "multiview", 0.96, "dreamgen-humanoid-multiview-pruning-insights", "mid (net-negative)")

# =============================================================================
# STILL NOT MEASURED ANYWHERE (no rule, no persisted timing) — these stay blank
# in the CSV on purpose (never estimated). Runner helpers exist for each:
#   Ctrl-World single_arm MV : dicache, worldcache, fastercache, itm
#       -> bash scripts/_timing_ctrlworld_sa_mv.sh
#   Ctrl-World humanoid MV 3-view : pisa, radial, svg1, svg2
#       -> scripts/run_humanoid_multiview_ctrlworld_{pisa,radial,svg1,svg2}.sh
#   DreamGen single_arm SV   : dicache, worldcache, fastercache
#   DreamGen single_arm MV   : dicache, worldcache, fastercache, sito, itm
#   DreamGen humanoid SV     : sito, itm
#       -> bash scripts/_timing_dreamgen_gaps.sh {sa_sv,sa_mv,hum_sv}
#   DreamDojo {humanoid,single_arm,bimanual} SV : dicache, worldcache, fastercache
#   DreamDojo {humanoid,single_arm} SV          : sito, itm
#       -> bash scripts/_timing_dreamdojo_sv_cache.sh <emb> <gpu>
# NO script at all (real holes in the benchmark grid):
#   FasterCache on Ctrl-World humanoid MV (3-view and 4-view grid)
#   SiTo / ITM on DreamDojo bimanual SV
# =============================================================================


@dataclass
class EpisodeRecord:
    method: str
    embodiment: str
    view: str
    model: str
    split: str
    extra: str
    pair_key: str
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]


@dataclass
class SummaryRecord:
    """A category-level all_summary.json record, used only for its
    (mean_wall_time_s, num_episodes) fields -- quality is re-derived from
    per-episode metrics.json wherever possible for consistency."""
    method: str
    embodiment: str
    view: str
    model: str
    split: str
    extra: str
    mean_wall_time_s: Optional[float]
    num_episodes: Optional[int]
    mean_psnr: Optional[float]
    mean_ssim: Optional[float]
    mean_lpips: Optional[float]


def parse_cell(relpath: str) -> Optional[dict]:
    m = CELL_RE.match(relpath)
    if not m:
        return None
    d = m.groupdict()
    d["extra"] = d["extra"] or ""
    return d


def walk_episode_metrics(method_root: Path, method: str) -> list[EpisodeRecord]:
    out = []
    for mp in method_root.rglob("metrics.json"):
        rel = mp.parent.relative_to(method_root).as_posix()
        cell = parse_cell(rel)
        if cell is None:
            continue
        try:
            data = json.loads(mp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append(EpisodeRecord(
            method=method, embodiment=cell["embodiment"], view=cell["view"],
            model=cell["model"], split=cell["split"], extra=cell["extra"],
            pair_key=rel, psnr=data.get("psnr"), ssim=data.get("ssim"),
            lpips=data.get("lpips"),
        ))
    return out


def walk_all_summary(method_root: Path, method: str) -> list[SummaryRecord]:
    out = []
    for sp in method_root.rglob("all_summary.json"):
        rel = sp.parent.relative_to(method_root).as_posix()
        cell = parse_cell(rel)
        if cell is None:
            continue
        try:
            data = json.loads(sp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "per_view" in data and len(rel.split("/")) <= 4:
            # This is a rolled-up top-level {model}/all_summary.json that
            # pools makovian+non_makovian -- skip it, we already get the
            # same info split-by-split from the per-split files.
            continue
        out.append(SummaryRecord(
            method=method, embodiment=cell["embodiment"], view=cell["view"],
            model=cell["model"], split=cell["split"], extra=cell["extra"],
            mean_wall_time_s=data.get("mean_wall_time_s"),
            num_episodes=data.get("num_episodes") or data.get("num_episodes_processed"),
            mean_psnr=data.get("mean_psnr"), mean_ssim=data.get("mean_ssim"),
            mean_lpips=data.get("mean_lpips"),
        ))
    return out


def _grain_key(rec) -> tuple:
    return (rec.embodiment, rec.view, rec.model, rec.split, rec.extra)


def _family_grain_key(rec) -> tuple:
    """Coarser key used only to consult SPEEDUP_LOOKUP, which was recorded
    per (method, family, embodiment, view) -- not per split/extra."""
    return (MODEL_FAMILY.get(rec.model, rec.model), rec.embodiment, rec.view)


def build_cells(sampling_root: Path) -> list[dict]:
    """Returns one dict per (method, model, embodiment, view, split, extra)
    cell that has at least a quality OR a timing number, joined against its
    dense counterpart."""
    episodes_by_method: dict[str, list[EpisodeRecord]] = {}
    summaries_by_method: dict[str, list[SummaryRecord]] = {}
    for method in METHODS:
        root = sampling_root / method
        if not root.is_dir():
            episodes_by_method[method] = []
            summaries_by_method[method] = []
            continue
        episodes_by_method[method] = walk_episode_metrics(root, method)
        summaries_by_method[method] = walk_all_summary(root, method)

    # dense pair_key -> (psnr, ssim, lpips) for exact episode-level joins.
    dense_by_pairkey = {
        r.pair_key: r for r in episodes_by_method.get("dense", [])
    }
    # dense grain -> SummaryRecord, for timing / quality when no per-episode
    # metrics.json exists (Ctrl-World / DreamDojo all_summary.json path).
    dense_summary_by_grain: dict[tuple, SummaryRecord] = {}
    for r in summaries_by_method.get("dense", []):
        dense_summary_by_grain[_grain_key(r)] = r

    rows = []
    for method in ACCEL_METHODS:
        # --- quality from per-episode metrics.json, grouped by grain ---
        by_grain: dict[tuple, list[EpisodeRecord]] = defaultdict(list)
        for r in episodes_by_method[method]:
            by_grain[_grain_key(r)].append(r)

        for grain, recs in by_grain.items():
            embodiment, view, model, split, extra = grain
            family = MODEL_FAMILY.get(model, model)
            psnrs = [r.psnr for r in recs if r.psnr is not None]
            ssims = [r.ssim for r in recs if r.ssim is not None]
            lpips_ = [r.lpips for r in recs if r.lpips is not None]
            # paired dense values, episode-matched where possible
            d_psnrs, d_ssims, d_lpips = [], [], []
            n_paired = 0
            for r in recs:
                dr = dense_by_pairkey.get(r.pair_key)
                if dr is not None:
                    n_paired += 1
                    if dr.psnr is not None:
                        d_psnrs.append(dr.psnr)
                    if dr.ssim is not None:
                        d_ssims.append(dr.ssim)
                    if dr.lpips is not None:
                        d_lpips.append(dr.lpips)
            row = _make_row(
                method=method, family=family, model=model, embodiment=embodiment,
                view=view, split=split, extra=extra,
                mean_psnr=mean(psnrs) if psnrs else None,
                mean_ssim=mean(ssims) if ssims else None,
                mean_lpips=mean(lpips_) if lpips_ else None,
                num_episodes=len(recs), quality_source="metrics.json (per-episode)",
                dense_mean_psnr=mean(d_psnrs) if d_psnrs else None,
                dense_mean_ssim=mean(d_ssims) if d_ssims else None,
                dense_mean_lpips=mean(d_lpips) if d_lpips else None,
                dense_num_episodes=n_paired,
            )
            rows.append(row)

        # --- category-level all_summary.json cells (Ctrl-World/DreamDojo) ---
        for r in summaries_by_method[method]:
            grain = _grain_key(r)
            if grain in by_grain:
                # already covered by the per-episode pass above; just merge
                # in the persisted num_episodes/timing onto that row below.
                continue
            dr = dense_summary_by_grain.get(grain)
            family = MODEL_FAMILY.get(r.model, r.model)
            row = _make_row(
                method=method, family=family, model=r.model, embodiment=r.embodiment,
                view=r.view, split=r.split, extra=r.extra,
                mean_psnr=r.mean_psnr, mean_ssim=r.mean_ssim, mean_lpips=r.mean_lpips,
                num_episodes=r.num_episodes, quality_source="all_summary.json",
                dense_mean_psnr=dr.mean_psnr if dr else None,
                dense_mean_ssim=dr.mean_ssim if dr else None,
                dense_mean_lpips=dr.mean_lpips if dr else None,
                dense_num_episodes=dr.num_episodes if dr else None,
            )
            rows.append(row)

        # --- attach persisted timing (mean_wall_time_s) onto matching rows,
        # falling back to SPEEDUP_LOOKUP where nothing is persisted ---
        method_time_by_grain = {
            _grain_key(r): r.mean_wall_time_s
            for r in summaries_by_method[method] if r.mean_wall_time_s is not None
        }
        dense_time_by_grain = {
            _grain_key(r): r.mean_wall_time_s
            for r in summaries_by_method.get("dense", []) if r.mean_wall_time_s is not None
        }
        for row in rows:
            if row["method"] != method:
                continue
            grain = (row["embodiment"], row["view"], row["model"], row["split"], row["extra"])
            mt = method_time_by_grain.get(grain)
            dt = dense_time_by_grain.get(grain)
            if mt is not None and dt is not None and mt > 0:
                row["speedup"] = round(dt / mt, 3)
                row["timing_source"] = "persisted (mean_wall_time_s)"
                row["timing_provenance"] = "persisted"
                row["method_wall_time_s"] = round(mt, 2)
                row["dense_wall_time_s"] = round(dt, 2)
            else:
                # Key on the RAW model (ctrlworld vs ctrlworld_4view_grid are
                # different checkpoints/configs with very different speedups),
                # not on the collapsed display family.
                fam_grain = (row["model"], row["embodiment"], row["view"])
                lut = SPEEDUP_LOOKUP.get((method, *fam_grain))
                if lut is not None:
                    # Two provenances live in this LUT and must NOT be conflated.
                    # `measured:_bench_uniform_timing` rows come from the uniform
                    # pass: dense CONTROL re-run in the same batch, same episodes,
                    # same pinned config, idle GPU. Everything else is a per-cell
                    # tuning run -- same script and config, but its own episode
                    # count and its own dense reference. Tagging the row here is
                    # what lets the table mark them differently instead of
                    # printing one undifferentiated number.
                    uniform = lut["source"].startswith("measured:_bench_uniform_timing")
                    row["speedup"] = lut["speedup"]
                    row["timing_provenance"] = ("uniform_pass" if uniform
                                                else "tuning_notes")
                    row["timing_source"] = (
                        f"{lut['source']} ({lut['note']})" if uniform
                        else f"tuning_notes:{lut['source']} ({lut['note']})")
                else:
                    row["speedup"] = None
                    row["timing_source"] = "unavailable"
                    row["timing_provenance"] = "unavailable"

    # Cells the paper does not report are dropped last, so that the dense
    # pairing and timing joins above are unaffected by the exclusion (a dropped
    # row must not change any surviving row's numbers). See
    # assets/scripts/benchmark_cells.py for the decision and its rationale.
    kept = [r for r in rows
            if not is_excluded(r["model"], r["embodiment"], r["view"])]
    dropped = len(rows) - len(kept)
    if dropped:
        print("benchmark_cells: %s" % describe_exclusions())
        print("  dropped %d of %d rows as unreported cells" % (dropped, len(rows)))
    return kept


def _make_row(method, family, model, embodiment, view, split, extra,
              mean_psnr, mean_ssim, mean_lpips, num_episodes, quality_source,
              dense_mean_psnr, dense_mean_ssim, dense_mean_lpips, dense_num_episodes):
    def _ratio(a, b, invert=False):
        if a is None or b is None or b == 0:
            return None
        r = (b / a) if invert else (a / b)
        return round(r, 4)

    psnr_retention = _ratio(mean_psnr, dense_mean_psnr)
    ssim_retention = _ratio(mean_ssim, dense_mean_ssim)
    lpips_retention = _ratio(mean_lpips, dense_mean_lpips, invert=True)  # lower lpips is better
    comps = [v for v in (psnr_retention, ssim_retention, lpips_retention) if v is not None]
    quality_retention = round(mean(comps), 4) if comps else None

    return {
        "method": method, "method_display": METHOD_DISPLAY.get(method, method),
        "model_family": family, "family_display": FAMILY_DISPLAY.get(family, family),
        "model": model, "embodiment": embodiment, "view": view, "split": split,
        "extra": extra, "num_episodes": num_episodes,
        "mean_psnr": round(mean_psnr, 4) if mean_psnr is not None else None,
        "mean_ssim": round(mean_ssim, 4) if mean_ssim is not None else None,
        "mean_lpips": round(mean_lpips, 4) if mean_lpips is not None else None,
        "quality_source": quality_source,
        "dense_mean_psnr": round(dense_mean_psnr, 4) if dense_mean_psnr is not None else None,
        "dense_mean_ssim": round(dense_mean_ssim, 4) if dense_mean_ssim is not None else None,
        "dense_mean_lpips": round(dense_mean_lpips, 4) if dense_mean_lpips is not None else None,
        "dense_num_episodes": dense_num_episodes,
        "psnr_retention": psnr_retention, "ssim_retention": ssim_retention,
        "lpips_retention": lpips_retention, "quality_retention": quality_retention,
        "speedup": None, "timing_source": "unavailable",
        "timing_provenance": "unavailable",
        "method_wall_time_s": None, "dense_wall_time_s": None,
    }


def roll_up_by_method(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = []
    for method, recs in by_method.items():
        speedups = [r["speedup"] for r in recs if r["speedup"] is not None]
        qualities = [r["quality_retention"] for r in recs if r["quality_retention"] is not None]
        families = sorted({r["model_family"] for r in recs})
        n_cells = len(recs)
        n_episodes = sum(r["num_episodes"] or 0 for r in recs)
        # Provenance travels with the roll-up. A mean speedup over cells of mixed
        # provenance is still useful, but a reader must be able to see how much of
        # it rests on the uniform pass -- otherwise the pooled number looks like a
        # single calibrated measurement, which it is not.
        prov = defaultdict(int)
        for r in recs:
            if r["speedup"] is not None:
                prov[r.get("timing_provenance", "unknown")] += 1
        out.append({
            "method": method, "method_display": METHOD_DISPLAY.get(method, method),
            "model_families_covered": ",".join(families),
            "n_cells": n_cells, "n_cells_with_speedup": len(speedups),
            "n_cells_with_quality": len(qualities),
            "n_cells_uniform_pass": prov.get("uniform_pass", 0),
            "n_cells_tuning_notes": prov.get("tuning_notes", 0),
            "n_cells_persisted": prov.get("persisted", 0),
            "total_episodes": n_episodes,
            "mean_speedup": round(mean(speedups), 3) if speedups else None,
            "mean_quality_retention": round(mean(qualities), 4) if qualities else None,
        })
    out.sort(key=lambda r: METHODS.index(r["method"]) if r["method"] in METHODS else 99)
    return out


FULL_FIELDS = [
    "method_display", "family_display", "model", "embodiment", "view", "split", "extra",
    "num_episodes", "mean_psnr", "mean_ssim", "mean_lpips", "quality_source",
    "dense_mean_psnr", "dense_mean_ssim", "dense_mean_lpips", "dense_num_episodes",
    "psnr_retention", "ssim_retention", "lpips_retention", "quality_retention",
    "speedup", "timing_source", "timing_provenance",
    "method_wall_time_s", "dense_wall_time_s",
]

BY_METHOD_FIELDS = [
    "method_display", "model_families_covered", "n_cells", "n_cells_with_speedup",
    "n_cells_with_quality", "n_cells_uniform_pass", "n_cells_tuning_notes",
    "n_cells_persisted", "total_episodes", "mean_speedup",
    "mean_quality_retention",
]


def write_csv(rows: list[dict], fields: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _tex_escape(s) -> str:
    if s is None:
        return "--"
    s = str(s)
    return s.replace("_", r"\_").replace("%", r"\%")


def write_appendix_tex(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (
        r["family_display"], r["embodiment"], r["view"],
        METHODS.index(r["method"]) if r["method"] in METHODS else 99,
    ))
    lines = []
    lines.append(r"% Auto-generated by scripts/tables/make_speedup_summary.py -- do not hand-edit.")
    lines.append(r"\begin{longtable}{l l l l r r r r r}")
    lines.append(r"\caption{Full per-cell speed--fidelity results for all acceleration methods "
                 r"(supplement to the star-chart summary in the main text). "
                 r"Speedups carry their provenance: $^{\ddagger}$ marks the uniform timing "
                 r"pass (dense control re-run in the same batch, same episodes, same pinned "
                 r"configuration, idle GPU), $^{\dagger}$ marks a per-cell tuning run (same "
                 r"script and configuration, but its own episode count and its own dense "
                 r"reference), and \texttt{--} marks a cell whose timing was never "
                 r"instrumented --- never a value interpolated from a neighbouring cell. "
                 r"See \texttt{metrics/reports/speedup\_summary\_full.csv} column "
                 r"\texttt{timing\_provenance} for the exact source per row.}")
    lines.append(r"\label{tab:speedup_full_appendix}\\")
    lines.append(r"\toprule")
    lines.append(r"Family & Embodiment & View & Method & PSNR & SSIM & LPIPS & Speedup & N \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Family & Embodiment & View & Method & PSNR & SSIM & LPIPS & Speedup & N \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    for r in rows:
        psnr = f"{r['mean_psnr']:.2f}" if r["mean_psnr"] is not None else "--"
        ssim = f"{r['mean_ssim']:.3f}" if r["mean_ssim"] is not None else "--"
        lpips = f"{r['mean_lpips']:.3f}" if r["mean_lpips"] is not None else "--"
        speedup = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "--"
        # Two provenances, two marks. Collapsing them into one dagger would hide
        # that exactly one cell has been measured under the uniform protocol.
        if r.get("timing_provenance") == "uniform_pass":
            speedup += r"$^{\ddagger}$"
        elif r["timing_source"].startswith("tuning_notes"):
            speedup += r"$^{\dagger}$"
        n = r["num_episodes"] if r["num_episodes"] is not None else "--"
        lines.append(
            f"{_tex_escape(r['family_display'])} & {_tex_escape(r['embodiment'])} & "
            f"{_tex_escape(r['view'])} & {_tex_escape(r['method_display'])} & "
            f"{psnr} & {ssim} & {lpips} & {speedup} & {n} \\\\"
        )
    lines.append(r"\multicolumn{9}{l}{\footnotesize $\ddagger$ uniform timing pass "
                 r"(\texttt{scripts/\_bench\_uniform\_timing.sh}): dense control re-run in "
                 r"the same batch on an idle GPU. \quad $\dagger$ per-cell tuning run: same "
                 r"script and configuration, own episode count and own dense reference.} \\")
    lines.append(r"\end{longtable}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sampling-root", type=Path, default=Path("sampling_dataset"))
    ap.add_argument("--out-dir", type=Path, default=Path("metrics/reports"))
    args = ap.parse_args()

    rows = build_cells(args.sampling_root)
    by_method = roll_up_by_method(rows)

    write_csv(rows, FULL_FIELDS, args.out_dir / "speedup_summary_full.csv")
    write_csv(by_method, BY_METHOD_FIELDS, args.out_dir / "speedup_summary_by_method.csv")
    write_appendix_tex(rows, args.out_dir / "speedup_summary_appendix.tex")

    print(f"Wrote {len(rows)} cell rows -> {args.out_dir / 'speedup_summary_full.csv'}")
    print(f"Wrote {len(by_method)} method rows -> {args.out_dir / 'speedup_summary_by_method.csv'}")
    print(f"Wrote appendix table -> {args.out_dir / 'speedup_summary_appendix.tex'}")
    print()
    print(f"{'method':<12}{'families':<28}{'cells':>6}{'speedup':>10}{'quality':>10}")
    for r in by_method:
        sp = f"{r['mean_speedup']:.2f}x" if r["mean_speedup"] else "--"
        q = f"{r['mean_quality_retention']:.3f}" if r["mean_quality_retention"] else "--"
        print(f"{r['method_display']:<12}{r['model_families_covered']:<28}{r['n_cells']:>6}{sp:>10}{q:>10}")


if __name__ == "__main__":
    main()
