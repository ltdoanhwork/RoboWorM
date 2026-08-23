#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline predictor diagnostics for DreamGen GR00T LVG.

Runs ONE dense LVG rollout with PredictorProbe hooked on pipe.dit, scoring -- per
block, per net call -- the relative reuse error of each candidate RFE predictor
WITHOUT changing the real output (read-only forward hooks). This tells us *why*
the online linear predictor underperforms `copy` on DreamGen:

  copy        : O_a
  linear      : O_a + s (I - I_a)
  band_linear : O_a + sum_b s_b band_b(I - I_a)
  snr_shrink  : O_a + g s (I - I_a)
  clamp{c}    : O_a + clip(s, 1/c, c) (I - I_a)        (two-sided clamp sweep)
  step_extrap : O_{t-1} + (O_{t-1} - O_{t-2})          (WorldCache-style temporal)

CFG note: DreamGen calls dit twice per step (cond, uncond). We route the two
branches to independent probe anchors via set_branch(call_parity % 2).

Outputs (under --out-dir):
  records.json   : raw per-(block, call) errors
  summary.json   : mean error per predictor, per-block, per-step-bin, ratio_s stats
  *.png          : error charts

Usage:
  python -m examples.debug_dreamgen_predictor \
     --model_size 14B --gr00t_variant droid \
     --batch_input_json .../batch_input.json --num_chunks 2 \
     --out-dir .../outputs/dreamgen_predictor_debug
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch

from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
from examples.video2world_gr00t_lvg import setup_pipeline

_AM_DIT_SRC = os.environ.get(
    "AM_DIT_SRC", "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/AM_DiT/src")
if _AM_DIT_SRC not in sys.path:
    sys.path.insert(0, _AM_DIT_SRC)
# Probe lives under src/snr/experiments
_EXP = os.path.join(_AM_DIT_SRC, "snr", "experiments")
if _EXP not in sys.path:
    sys.path.insert(0, _EXP)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", choices=["2B", "14B"], default="14B")
    p.add_argument("--dit_path", type=str, default="")
    p.add_argument("--load_ema", action="store_true")
    p.add_argument("--gr00t_variant", type=str, required=True, choices=["gr1", "droid"])
    p.add_argument("--batch_input_json", type=str, required=True)
    p.add_argument("--num_chunks", type=int, default=2)
    p.add_argument("--num_conditional_frames", type=int, default=1, choices=[1, 5])
    p.add_argument("--aspect_ratio", default="16:9", type=str)
    p.add_argument("--guidance", type=float, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_gpus", type=int, default=1)
    p.add_argument("--disable_guardrail", action="store_true")
    p.add_argument("--prompt_prefix", type=str, default="")
    p.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    p.add_argument("--enable_prompt_refiner_per_chunk", action="store_true")
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--snr-shrink-scale", type=float, default=5.0)
    p.add_argument("--downsample", type=int, default=2,
                   help="Spatial avg-pool factor for stored anchors (memory). "
                        "2 = 4x smaller; relative-error ordering is preserved.")
    p.add_argument("--no-bands", action="store_true",
                   help="Disable the band-linear candidate (saves the most memory).")
    p.add_argument("--out-dir", type=str, required=True)
    # Sparse-backend args are referenced by setup_pipeline via getattr; supply none.
    from examples.video2world_gr00t_lvg import _import_sparse_backends
    (add_pisa, add_svg, add_radial, add_sito, add_itm, add_cache, _b) = _import_sparse_backends()
    add_pisa(p); add_svg(p); add_radial(p); add_sito(p); add_itm(p); add_cache(p)
    return p.parse_args()


def _get_dit(pipe):
    return pipe.dit


def attach_probe(pipe, args):
    from _predictor_probe import PredictorProbe

    dit = _get_dit(pipe)
    probe = PredictorProbe(dit, warmup=args.warmup, snr_shrink_scale=args.snr_shrink_scale,
                           s_clamps=(1.0, 1.5, 2.0, 3.0),
                           downsample=args.downsample, use_bands=not args.no_bands)
    original_forward = dit.forward
    parity = {"n": 0}

    def wrapped(*fa, **fk):
        branch = parity["n"] % 2
        probe.set_branch(branch)
        # advance the global call counter once per full step (after uncond branch)
        if branch == 0:
            probe.call_idx += 1
        parity["n"] += 1
        return original_forward(*fa, **fk)

    dit.forward = wrapped

    def restore():
        dit.forward = original_forward
        probe.cleanup()

    return probe, restore, parity


def run_one(pipe, input_path, prompt, args, probe, parity):
    if not validate_input_file(input_path, args.num_conditional_frames):
        return False
    full_prompt = args.prompt_prefix + prompt
    current = input_path
    with tempfile.TemporaryDirectory() as tmp:
        for chunk_id in range(args.num_chunks):
            parity["n"] = 0  # reset cond/uncond parity each chunk (fresh denoise)
            video, _ = pipe(
                prompt=full_prompt, negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio, input_path=current,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance, seed=args.seed + chunk_id, return_prompt=True,
            )
            last = video[:, :, -args.num_conditional_frames:, :, :]
            ext = "png" if args.num_conditional_frames == 1 else "mp4"
            p = os.path.join(tmp, f"c{chunk_id}.{ext}")
            save_image_or_video(last, p, fps=16)
            current = p
    return True


def summarize_and_plot(records, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "records.json").write_text(json.dumps(records[:5000], indent=2))

    keys = [k for k in records[0].keys() if k.startswith("err_")] if records else []

    def col(name):
        return np.array([r[name] for r in records if r.get(name) is not None], dtype=float)

    means = {k: (float(np.mean(col(k))) if len(col(k)) else None) for k in keys}
    med = {k: (float(np.median(col(k))) if len(col(k)) else None) for k in keys}
    s_vals = np.array([r["ratio_s"] for r in records if r.get("ratio_s") is not None], dtype=float)

    summary = {
        "n_records": len(records),
        "mean_err": means,
        "median_err": med,
        "ratio_s": {
            "mean": float(np.mean(s_vals)) if len(s_vals) else None,
            "median": float(np.median(s_vals)) if len(s_vals) else None,
            "p90": float(np.percentile(s_vals, 90)) if len(s_vals) else None,
            "max": float(np.max(s_vals)) if len(s_vals) else None,
            "frac_gt_1": float(np.mean(s_vals > 1.0)) if len(s_vals) else None,
        },
    }

    # Per-block mean error for copy vs linear vs best.
    blocks = sorted({r["block"] for r in records})
    per_block = {}
    for b in blocks:
        rb = [r for r in records if r["block"] == b]
        per_block[b] = {k: (float(np.mean([r[k] for r in rb if r.get(k) is not None]))
                            if any(r.get(k) is not None for r in rb) else None)
                        for k in keys}
    summary["per_block"] = per_block
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Charts ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available; skipping charts")
        return summary

    # 1) mean error per predictor (bar)
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [k.replace("err_", "") for k in keys]
    vals = [means[k] if means[k] is not None else 0 for k in keys]
    order = np.argsort(vals)
    ax.bar([names[i] for i in order], [vals[i] for i in order], color="steelblue")
    ax.set_ylabel("mean relative reuse error")
    ax.set_title("Predictor reuse error (lower=better) — DreamGen")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(out_dir / "err_by_predictor.png", dpi=120); plt.close(fig)

    # 2) per-block: copy vs linear vs step_extrap
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, c in [("err_copy", "gray"), ("err_linear", "crimson"),
                 ("err_step_extrap", "green"), ("err_clamp1.5", "orange")]:
        if k in keys:
            ax.plot(blocks, [per_block[b][k] for b in blocks], label=k.replace("err_", ""), color=c, marker=".")
    ax.set_xlabel("DiT block index"); ax.set_ylabel("mean reuse error")
    ax.set_title("Per-block reuse error"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "err_by_block.png", dpi=120); plt.close(fig)

    # 3) error vs denoise step (call_idx bins)
    calls = sorted({r["call_idx"] for r in records})
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, c in [("err_copy", "gray"), ("err_linear", "crimson"), ("err_step_extrap", "green")]:
        if k in keys:
            ys = []
            for ci in calls:
                rc = [r[k] for r in records if r["call_idx"] == ci and r.get(k) is not None]
                ys.append(np.mean(rc) if rc else np.nan)
            ax.plot(calls, ys, label=k.replace("err_", ""), color=c, marker=".")
    ax.set_xlabel("denoise step (call_idx)"); ax.set_ylabel("mean reuse error")
    ax.set_title("Reuse error vs denoise step"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "err_by_step.png", dpi=120); plt.close(fig)

    # 4) ratio_s histogram
    if len(s_vals):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(s_vals, 0, 10), bins=60, color="purple", alpha=0.8)
        ax.axvline(1.0, color="k", ls="--", label="s=1")
        ax.set_xlabel("fitted ratio_s = ||dO||/||dI||"); ax.set_ylabel("count")
        ax.set_title(f"ratio_s distribution (median={summary['ratio_s']['median']:.2f}, "
                     f"p90={summary['ratio_s']['p90']:.2f})")
        ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "ratio_s_hist.png", dpi=120); plt.close(fig)

    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    pipe = setup_pipeline(args)
    probe, restore, parity = attach_probe(pipe, args)

    with open(args.batch_input_json) as f:
        batch = json.load(f)[: args.num_episodes]

    all_records = []
    try:
        for idx, item in enumerate(batch):
            iv, pr = item.get("input_video", ""), item.get("prompt", "")
            if not iv or not pr:
                continue
            probe.reset_episode()
            log.info(f"[debug] episode {idx}: {iv}")
            run_one(pipe, iv, pr, args, probe, parity)
            for r in probe.records:
                r["episode"] = idx
            all_records.extend(probe.records)
    finally:
        restore()

    if not all_records:
        log.error("No probe records collected.")
        return
    summary = summarize_and_plot(all_records, out_dir)

    print("\n=== Predictor reuse error (mean, lower=better) ===")
    for k, v in sorted(summary["mean_err"].items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
        print(f"  {k.replace('err_',''):<14}: {v:.4f}" if v is not None else f"  {k}: n/a")
    rs = summary["ratio_s"]
    print(f"\nratio_s: median={rs['median']:.3f}  p90={rs['p90']:.3f}  "
          f"max={rs['max']:.3f}  frac>1={rs['frac_gt_1']:.2f}")
    print(f"\nCharts + JSON in: {out_dir}")


if __name__ == "__main__":
    main()
