"""Run-level cache accounting for the DreamGen baselines (WorldCache/DiCache/FasterCache).

WHY THIS FILE EXISTS
--------------------
The DreamGen rollout harness (``scripts/infer/infer_bimanual_multiview_rollout.py``) emits
PSNR/SSIM/LPIPS and **nothing about cache behaviour** for any backend. So on this
backbone "the run produced a plausible video" and "the gate never fired and the cache
did nothing" are the same observable. BSC worked around this with its own
``$BSC_STATS_JSON`` dump; the three baselines had no equivalent, which means they could
not be *ranked* at all -- and an untunable baseline is how a method ends up compared
against a strawman (``AM_DiT/docs/INSIGHTS.md`` sections 41, 55, 57).

Two things here are deliberate, and each is a bug this project has already paid for:

1. **The accumulators are NOT reset per denoise loop.**
   ``WorldCache``'s in-tree ``model.worldcache_step_skipped_count`` is zeroed by
   ``initialize_worldcache_state``, which the runtime calls itself at the end of every
   denoise loop (``WorldCache/runtime.py``, the ``self.cnt >= self.worldcache_num_steps``
   branch), and which ``reset_cache_runtime`` calls again per chunk. A run over many
   chunks therefore reports the LAST chunk's skips while reading like a total. That is
   accounting bug 1 of ``AM_DiT/docs/INSIGHTS.md`` section 55, which silently published
   50 decisions as 1200. ``_RUN`` below is process-global and survives both resets.

2. **The dump is written on every decision, not at ``atexit``.**
   ``atexit`` does not run under SIGTERM, and a reclaimed Determined shell ends a job
   exactly that way -- which cost a previous session 466 GPU-seconds of accounting
   (``AM_DiT/docs/INSIGHTS.md`` section 58). The file is ~1 KB and the thing it sits
   next to is a 2B-parameter DiT forward, so writing it every time is free.

THE COST UNIT, STATED EXACTLY
-----------------------------
``skip_pct`` is the percentage of **DiT block invocations elided**, i.e.

    skip_pct = 100 * (1 - blocks_executed / (net_forwards * num_blocks))

This is the same unit BSC reports on this backbone, which is what makes the rows
comparable. It is NOT FLOPs: ``AM_DiT/analysis/flops_profile.py``'s stored profile is
Ctrl-World's SVD U-Net and must not be applied to a 28-block DiT.

It can go **negative**, and that is the point rather than a defect. A step cache decides
*inside* ``forward`` after running ``probe_depth`` blocks, so a step it decides NOT to
skip can cost ``num_blocks + probe_depth`` block invocations -- the probe is wasted work.
A method whose threshold is too tight genuinely does more block work than dense, and the
number says so instead of flooring at zero.

``attn_reuses`` is tracked separately and is deliberately **not** folded into
``skip_pct``: FasterCache's block-level mechanism reuses a block's self-attention while
still running its cross-attention and MLP, so counting it as a skipped block would
overstate it, and counting it as nothing understates it. Both numbers are reported.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

#: Env var naming where to write the accounting. Unset = no dump (and no file I/O).
STATS_ENV = "CACHE_STATS_JSON"

_LOCK = threading.Lock()

_RUN: dict = {
    "method": None,
    "num_blocks": 0,
    "config": {},
    # --- block-level work, the ranking axis -------------------------------------
    "net_forwards": 0,        # model forward calls (2 per denoise step: cond, uncond)
    "blocks_executed": 0,     # actual block invocations, probe overhead INCLUDED
    "blocks_nominal": 0,      # net_forwards * num_blocks, i.e. what dense would run
    # --- step-level decisions ---------------------------------------------------
    "step_skips": 0,          # forwards whose deep stack was elided
    "gate_decisions": 0,      # forwards where the gate was armed (past ret warmup)
    # --- FasterCache block mechanism, reported but NOT in skip_pct --------------
    "attn_calls": 0,
    "attn_reuses": 0,
    # --- reset accounting: proves the cache is cleared per chunk -----------------
    "resets": 0,
}


def configure(method: str, num_blocks: int, config: dict | None = None) -> None:
    """Start a fresh run record. Called from each ``apply_*``."""
    with _LOCK:
        _RUN.update(
            method=method,
            num_blocks=int(num_blocks),
            config=dict(config or {}),
            net_forwards=0,
            blocks_executed=0,
            blocks_nominal=0,
            step_skips=0,
            gate_decisions=0,
            attn_calls=0,
            attn_reuses=0,
            resets=0,
        )


def note_reset() -> None:
    """One per ``reset_cache_runtime`` call, i.e. one per chunk.

    Published so the insight document can state whether the cache was actually cleared
    between chunks rather than asserting it from a code path. The DeepCache baseline in
    ``AM_DiT/docs/INSIGHTS.md`` section 47 had to be retracted twice because its cache
    was never cleared, which flattered compute and corrupted quality at once.
    """
    with _LOCK:
        if _RUN["method"] is not None:
            _RUN["resets"] += 1


def record_forward(*, blocks_executed: int, skipped: bool, gated: bool) -> None:
    """One model forward: how many block invocations it really cost."""
    if _RUN["method"] is None:
        return
    with _LOCK:
        _RUN["net_forwards"] += 1
        _RUN["blocks_executed"] += int(blocks_executed)
        _RUN["blocks_nominal"] += _RUN["num_blocks"]
        if skipped:
            _RUN["step_skips"] += 1
        if gated:
            _RUN["gate_decisions"] += 1
    dump()


def record_attn(*, reused: bool) -> None:
    if _RUN["method"] is None:
        return
    with _LOCK:
        _RUN["attn_calls"] += 1
        if reused:
            _RUN["attn_reuses"] += 1


def snapshot() -> dict:
    with _LOCK:
        out = dict(_RUN)
    nominal = out["blocks_nominal"]
    out["skip_ratio"] = (1.0 - out["blocks_executed"] / nominal) if nominal else 0.0
    out["skip_pct"] = 100.0 * out["skip_ratio"]
    out["step_skip_pct"] = (
        100.0 * out["step_skips"] / out["net_forwards"] if out["net_forwards"] else 0.0
    )
    out["attn_reuse_pct"] = (
        100.0 * out["attn_reuses"] / out["attn_calls"] if out["attn_calls"] else 0.0
    )
    out["_unit"] = (
        "skip_pct = 100*(1 - blocks_executed/(net_forwards*num_blocks)); block "
        "invocations elided, NOT FLOPs. Negative means the probe cost more block work "
        "than the skips saved. attn_reuses is NOT included in skip_pct."
    )
    return out


def dump(path: str | None = None) -> dict | None:
    """Write the accounting to ``path`` (default ``$CACHE_STATS_JSON``). No-op if unset."""
    target = path or os.environ.get(STATS_ENV)
    if not target:
        return None
    stats = snapshot()
    p = Path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats, indent=2) + "\n")
    return stats
