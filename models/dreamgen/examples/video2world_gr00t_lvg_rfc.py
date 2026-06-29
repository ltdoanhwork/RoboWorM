#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DreamGen GR00T LVG rollout with the block-level RFC runtime (your algorithm).

This is the DreamGen counterpart of the DreamDojo ``run_snr_exp3_online.py``
harness: it reuses DreamGen's own LVG pipeline (``setup_pipeline`` +
``process_single_generation`` from ``examples.video2world_gr00t_lvg``) so the
model/dataset/long-rollout logic is *identical* to the reference WorldCache run,
but optionally attaches the block-level RFC cache (``src/rfc/core.RFCRuntime``)
to ``pipe.dit`` for the ``copy`` / ``linear_clamp`` / ``snr_shrink`` variants.

Cache backends compared on the SAME harness / same episodes / same chunks:
  * ``--rfc-mode none``         : dense (no cache)            -- or use WorldCache
                                  via DreamGen's own ``--worldcache-enabled``.
  * ``--rfc-mode copy``         : naive reuse.
  * ``--rfc-mode linear_clamp`` : your clamp-regularized RFE.
  * ``--rfc-mode snr_shrink``   : your SNR-shrink-regularized RFE.

The RFC drift gate is content-based and needs no SNR signal, so ``snr_shrink``
falls back to a gain of 1 (== ``linear_clamp``) unless a per-step dlogsnr is
threaded in; on DreamGen we run the drift gate only (no SNR tap), which still
exercises the clamp/shrink regularization at full-compute time.

Usage (single-GPU; cache backends are single-GPU only)::

    python -m examples.video2world_gr00t_lvg_rfc \
        --model_size 14B --gr00t_variant droid \
        --batch_input_json .../batch_input.json \
        --num_chunks 6 --rfc-mode linear_clamp --s-clamp 3.0 \
        --output-root .../rfc_compare/linear_clamp
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from tqdm import tqdm

from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video, save_text_prompts
from examples.video2world import _DEFAULT_NEGATIVE_PROMPT, validate_input_file
from examples.video2world_gr00t_lvg import setup_pipeline, parse_args as _lvg_parse_args  # noqa: F401


# --- Make the AM_DiT RFC runtime importable (block-level cache = "our" method) ---
_AM_DIT_SRC = os.environ.get(
    "AM_DIT_SRC",
    "/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/AM_DiT/src",
)
if _AM_DIT_SRC not in sys.path:
    sys.path.insert(0, _AM_DIT_SRC)


def _add_rfc_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("RFC (block-level cache)")
    g.add_argument("--rfc-mode", type=str, default="none",
                   choices=["none", "copy", "linear", "linear_clamp", "snr_shrink"],
                   help="Block-level RFC predictor. 'none' = dense / use WorldCache "
                        "via DreamGen's own --worldcache-enabled flag.")
    g.add_argument("--rcs-threshold", type=float, default=0.1)
    g.add_argument("--min-full-interval", type=int, default=3)
    g.add_argument("--warmup-steps", type=int, default=3)
    g.add_argument("--s-clamp", type=float, default=3.0)
    g.add_argument("--snr-shrink-scale", type=float, default=5.0)
    g.add_argument("--output-root", type=str, default=None,
                   help="If set, batch outputs go under this dir (one mp4 per item) "
                        "and a per-episode metrics JSON is appended.")


def _get_dit(pipe):
    return pipe.dit


def attach_rfc(pipe, args):
    """Attach the block-level RFC runtime to pipe.dit and wrap dit.forward so
    each net call advances the per-step gate.

    DreamGen uses classifier-free guidance: each denoising step calls
    ``dit.forward`` TWICE (conditional then unconditional), strictly alternating.
    A single set of per-block anchors would let the two branches pollute each
    other's drift gate (cond input vs uncond anchor). We therefore keep TWO
    independent ``block_states`` dicts and swap them in based on call parity
    before each forward -- the same cond/uncond split WorldCache does with
    ``cnt % 2``. This needs no change to the core RFCRuntime.

    Returns (runtime, restore_fn, stats_fn).
    """
    from rfc.core import RFCRuntime, RFCBlockState

    dit = _get_dit(pipe)
    predictor = "copy" if args.rfc_mode == "copy" else args.rfc_mode
    runtime = RFCRuntime(
        dit,
        rcs_threshold=args.rcs_threshold,
        rfe_order=1,
        predictor_mode=predictor,
        warmup_steps=args.warmup_steps,
        min_full_interval=args.min_full_interval,
        gate_mode="drift",
        s_clamp=(args.s_clamp if args.rfc_mode in ("linear_clamp", "snr_shrink") else None),
        snr_shrink_scale=args.snr_shrink_scale,
    )

    # Two independent anchor sets (CFG cond / uncond branches).
    branch_states = [
        {idx: RFCBlockState() for idx in runtime.state.block_states},
        {idx: RFCBlockState() for idx in runtime.state.block_states},
    ]
    branch_steps = [0, 0]
    call_parity = {"n": 0}

    original_forward = dit.forward

    def wrapped_forward(*fwd_args, **fwd_kwargs):
        branch = call_parity["n"] % 2
        call_parity["n"] += 1
        # Swap in this branch's anchors + step counter.
        runtime.state.block_states = branch_states[branch]
        runtime.state.current_step = branch_steps[branch]
        ctx = fwd_kwargs.get("attention_runtime_context", None)
        step_frac = 0.0
        if isinstance(ctx, dict) and ctx.get("num_sampling_steps"):
            n = ctx["num_sampling_steps"]
            step_frac = (ctx.get("diffusion_step_idx", 0) or 0) / max(n, 1)
        runtime.reset_step(step_frac=step_frac)
        out = original_forward(*fwd_args, **fwd_kwargs)
        branch_steps[branch] = runtime.state.current_step
        return out

    dit.forward = wrapped_forward

    def reset_episode():
        call_parity["n"] = 0
        for b in (0, 1):
            branch_steps[b] = 0
            for bs in branch_states[b].values():
                bs.anchor_input = None
                bs.anchor_output = None
                bs.prev_anchor_input = None
                bs.prev_anchor_output = None
                bs.ratio_s = 1.0
                bs.steps_since_full = 0

    def stats():
        skips = computes = 0
        for b in (0, 1):
            for bs in branch_states[b].values():
                skips += bs.total_skips
                computes += bs.total_computes
        total = skips + computes
        return {"skip_ratio": skips / total if total else 0.0,
                "total_skips": skips, "total_computes": computes}

    def restore():
        dit.forward = original_forward
        runtime.detach()

    return runtime, restore, reset_episode, stats


def main():
    parser = argparse.ArgumentParser(description="DreamGen GR00T LVG + block-level RFC")
    # Reuse the full LVG arg surface (model/checkpoint/cache/sparse flags).
    # We rebuild it here rather than import parse_args() to add RFC args cleanly.
    from examples.video2world_gr00t_lvg import _import_sparse_backends
    (add_pisa_args, add_svg_args, add_radial_args, add_sito_args,
     add_itm_args, add_cache_args, _build_cache) = _import_sparse_backends()

    parser.add_argument("--model_size", choices=["2B", "14B"], default="14B")
    parser.add_argument("--dit_path", type=str, default="")
    parser.add_argument("--load_ema", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--input_path", type=str, default="assets/video2world/input0.jpg")
    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--aspect_ratio", default="16:9", type=str)
    parser.add_argument("--num_conditional_frames", type=int, default=1, choices=[1, 5])
    parser.add_argument("--num_chunks", type=int, default=6)
    parser.add_argument("--batch_input_json", type=str, default=None)
    parser.add_argument("--no_skip_existing", action="store_true")
    parser.add_argument("--guidance", type=float, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="output/generated_video.mp4")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--disable_guardrail", action="store_true")
    parser.add_argument("--gr00t_variant", type=str, required=True, choices=["gr1", "droid"])
    parser.add_argument("--prompt_prefix", type=str, default="The robot arm is performing a task. ")
    parser.add_argument("--enable_prompt_refiner_per_chunk", action="store_true")
    add_pisa_args(parser); add_svg_args(parser); add_radial_args(parser)
    add_sito_args(parser); add_itm_args(parser); add_cache_args(parser)
    _add_rfc_args(parser)
    args = parser.parse_args()

    if args.num_gpus != 1 and args.rfc_mode != "none":
        raise SystemExit("[RFC] block-level RFC is single-GPU only; set --num_gpus 1.")

    pipe = setup_pipeline(args)

    runtime, restore, reset_episode, stats = (None, None, None, None)
    if args.rfc_mode != "none":
        runtime, restore, reset_episode, stats = attach_rfc(pipe, args)
        log.info(f"[RFC] attached mode={args.rfc_mode} tau={args.rcs_threshold} "
                 f"s_clamp={args.s_clamp} shrink={args.snr_shrink_scale}")

    try:
        _run_batch(pipe, args, runtime, reset_episode, stats)
    finally:
        if restore is not None:
            restore()


def _run_batch(pipe, args, runtime, reset_episode, stats):
    if args.batch_input_json is None:
        raise SystemExit("This RFC harness expects --batch_input_json (use the "
                         "prepared single_arm multiview batch).")

    with open(args.batch_input_json) as f:
        batch_inputs = json.load(f)

    out_root = Path(args.output_root) if args.output_root else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
    metrics = []

    for idx, item in enumerate(tqdm(batch_inputs, desc=f"RFC[{args.rfc_mode}]")):
        input_video = item.get("input_video", "")
        prompt = item.get("prompt", "")
        if not input_video or not prompt:
            log.warning(f"Skipping item {idx}: missing input_video/prompt")
            continue
        if out_root:
            output_path = str(out_root / f"episode_{idx:04d}.mp4")
        else:
            output_path = item.get("output_video", f"output_{idx}.mp4")

        if runtime is not None:
            reset_episode()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = _generate_one(pipe, input_video, prompt, output_path, args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        rec = {"episode": idx, "time_s": elapsed, "output": output_path, "ok": ok}
        if runtime is not None:
            rec["skip_ratio"] = stats()["skip_ratio"]
        metrics.append(rec)
        log.info(f"  ep{idx}: {elapsed:.1f}s"
                 + (f", skip={rec.get('skip_ratio', 0)*100:.1f}%" if runtime else ""))

    if out_root:
        summary = {
            "rfc_mode": args.rfc_mode,
            "num_chunks": args.num_chunks,
            "guidance": args.guidance,
            "rcs_threshold": args.rcs_threshold,
            "s_clamp": args.s_clamp,
            "snr_shrink_scale": args.snr_shrink_scale,
            "episodes": metrics,
            "mean_time_s": (sum(m["time_s"] for m in metrics) / len(metrics)) if metrics else None,
            "mean_skip_ratio": (
                sum(m.get("skip_ratio", 0.0) for m in metrics) / len(metrics)
                if (metrics and runtime is not None) else None
            ),
        }
        with open(out_root / "rfc_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        log.success(f"[RFC] wrote {out_root/'rfc_results.json'}")


def _generate_one(pipe, input_path, prompt, output_path, args):
    """Single LVG rollout (reuses the reference chunk-chaining logic)."""
    import tempfile

    if not validate_input_file(input_path, args.num_conditional_frames):
        log.warning(f"Input validation failed: {input_path}")
        return False

    full_prompt = args.prompt_prefix + prompt
    all_chunks = []
    current_input_path = input_path
    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk_id in tqdm(range(args.num_chunks), leave=False):
            video, _ = pipe(
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                aspect_ratio=args.aspect_ratio,
                input_path=current_input_path,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=args.seed + chunk_id,
                return_prompt=True,
            )
            chunk = video if chunk_id == 0 else video[:, :, args.num_conditional_frames:, :, :]
            all_chunks.append(chunk.cpu())
            last_frames = video[:, :, -args.num_conditional_frames:, :, :]
            ext = "png" if args.num_conditional_frames == 1 else "mp4"
            last_frames_path = os.path.join(tmpdir, f"chunk{chunk_id}_overlap.{ext}")
            save_image_or_video(last_frames, last_frames_path, fps=16)
            current_input_path = last_frames_path

    if not all_chunks:
        return False
    video = torch.cat(all_chunks, dim=2)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_image_or_video(video, output_path, fps=16)
    save_text_prompts({"prompt": prompt, "full_prompt": full_prompt}, os.path.splitext(output_path)[0] + ".txt")
    return True


if __name__ == "__main__":
    main()
