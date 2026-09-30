#!/usr/bin/env python3
"""Prove the BSC-on-DreamGen wiring works without a GPU or a checkpoint.

Run from video_gen_physics with AM_DiT/src on PYTHONPATH:

    PYTHONPATH=$VGP:$AM_DIT/src python3 methods/cache_strategy/verify_bsc_wiring.py

The DreamGen DiT itself needs gated 2B/14B weights, but every seam this script checks is
weight-independent: argparse -> BSCConfig -> apply_bsc -> reset_cache_runtime. It
deliberately imports in the pipeline's real order (`methods.cache_strategy.common` first,
then the `bsc` shim), because that order is what used to break the shim -- and is why the
ScalingCache branch next to BSC's had never run.
"""
from __future__ import annotations

import argparse
import sys

# ---- import in the PIPELINE'S order: host first, shim second. ----
# video2world.py imports methods.cache_strategy.common at module scope, which pins
# sys.modules['methods'] to this repo's namespace package before any shim runs.
from methods.cache_strategy.common import (  # noqa: E402
    BSCConfig,
    get_cache_backend_name,
    reset_cache_runtime,
)
from methods.cache_strategy.config import (  # noqa: E402
    add_dreamgen_cache_args,
    build_cache_runtime_config,
    enabled_cache_backends,
)

import torch  # noqa: E402

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        _FAILURES.append(label)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--num-gpus", type=int, default=1)
    add_dreamgen_cache_args(p)
    return p


class _Block(torch.nn.Module):
    def __init__(self, dim: int, idx: int):
        super().__init__()
        g = torch.Generator().manual_seed(1234 + idx)
        self.w = torch.nn.Parameter(torch.eye(dim) + 0.05 * torch.randn(dim, dim, generator=g))

    def forward(self, x, *a, **kw):
        return x @ self.w.T


class _FakeDiT(torch.nn.Module):
    """`MinimalV1LVGDiT`'s shape as far as BSC is concerned: a flat `.blocks` ModuleList
    and a forward taking `timesteps_B_T` / `attention_runtime_context` by keyword."""

    def __init__(self, n_blocks: int = 8, dim: int = 16):
        super().__init__()
        self.blocks = torch.nn.ModuleList(_Block(dim, i) for i in range(n_blocks))

    def forward(self, x_B_C_T_H_W, timesteps_B_T=None, attention_runtime_context=None, **kw):
        h = x_B_C_T_H_W
        for b in self.blocks:
            h = b(h)
        return h


def main() -> int:
    print("\n[1] CLI -> BSCConfig through build_cache_runtime_config")
    args = _parser().parse_args([
        "--bsc-enabled", "--bsc-gate-mode", "group",
        "--bsc-rcs-threshold", "0.02", "--bsc-min-full-interval", "6",
        "--bsc-group-size", "4", "--bsc-group-predict", "io",
        "--bsc-warmup-steps", "2", "--bsc-skip-eligibility",
    ])
    config = build_cache_runtime_config(args)
    check("returns a BSCConfig", isinstance(config, BSCConfig), type(config).__name__)
    check("backend name is 'bsc'", get_cache_backend_name(config) == "bsc")
    check("flags reached the config",
          (config.rcs_threshold, config.min_full_interval, config.group_size,
           config.skip_eligibility) == (0.02, 6, 4, True),
          f"tau={config.rcs_threshold} mfi={config.min_full_interval} "
          f"gsize={config.group_size} elig={config.skip_eligibility}")
    check("defaults survive round-trip", config.predictor_mode == "linear_s1",
          config.predictor_mode)
    check("no cache selected -> None", build_cache_runtime_config(_parser().parse_args([])) is None)

    print("\n[2] mutual exclusion")
    both = _parser().parse_args(["--bsc-enabled", "--worldcache-enabled"])
    check("two backends listed", sorted(enabled_cache_backends(both)) == ["bsc", "worldcache"],
          str(sorted(enabled_cache_backends(both))))
    try:
        build_cache_runtime_config(both)
        check("enabling two backends raises", False, "it returned instead")
    except ValueError as e:
        check("enabling two backends raises", "mutually exclusive" in str(e), str(e)[:70])

    print("\n[3] apply_bsc attaches to a DreamGen-shaped net (no checkpoint)")
    from bsc.backend import RUNTIME_ATTR, apply_bsc, detach_bsc  # the pipeline's import

    net = _FakeDiT()
    original = [b.forward for b in net.blocks]

    def drive(module, n=12, total=6):
        torch.manual_seed(0)
        x = torch.randn(1, 4, 16)
        outs = []
        for i in range(n):
            x = x + 0.01 * torch.randn(1, 4, 16, generator=torch.Generator().manual_seed(i))
            outs.append(module(x, timesteps_B_T=torch.full((1, 4), 1.0 - (i // 2) / total),
                               attention_runtime_context={"diffusion_step_idx": i // 2,
                                                          "num_sampling_steps": total}))
        return outs

    dense = drive(net)
    check("apply_bsc returns the net", apply_bsc(net, config) is net)
    check("all blocks patched", all(b.forward is not o for b, o in zip(net.blocks, original)))
    drive(net)
    runtime = getattr(net, RUNTIME_ATTR)
    stats = runtime.get_stats()
    check("the gate actually skipped", stats["total_skips"] > 0,
          f"skips={stats['total_skips']} computes={stats['total_computes']} "
          f"ratio={stats['skip_ratio']:.2f}")
    check("the gate is not stuck open", stats["total_computes"] > 0)
    check("steps advanced (2 per denoise step)", runtime.state.current_step == 12,
          str(runtime.state.current_step))
    check("physical noise level reached the gate", runtime.state.current_noise_t > 0,
          f"t={runtime.state.current_noise_t:.3f}")

    print("\n[4] reset_cache_runtime clears BSC between episodes (INSIGHTS section 47)")
    cached = sum(1 for s in runtime.state.block_states.values() if s.anchor_output is not None)
    check("anchors exist before the reset", cached > 0, f"{cached} blocks cached")
    returned = reset_cache_runtime(net, config, num_steps=6)
    check("reset dispatched to BSC", returned == 12, f"returned {returned}")
    check("every anchor cleared",
          all(s.anchor_input is None and s.anchor_output is None
              for s in runtime.state.block_states.values()))
    check("step counter cleared", runtime.state.current_step == 0)
    try:
        reset_cache_runtime(_FakeDiT(), config, num_steps=6)
        check("reset without a runtime raises", False, "it returned quietly -- the DeepCache bug")
    except RuntimeError as e:
        check("reset without a runtime raises", "no runtime found" in str(e), str(e)[:60])

    print("\n[5] detach_bsc restores the net exactly")
    detach_bsc(net)
    check("block forwards restored", all(b.forward == o for b, o in zip(net.blocks, original)))
    check("net.forward restored", "forward" not in net.__dict__)
    check("dense output reproduced bit-identically",
          all(torch.equal(a, b) for a, b in zip(dense, drive(net))))

    print("\n[6] the shim resolved to AM_DiT, and the host kept its own modules")
    import methods.cache_strategy.BSC.dreamgen as impl
    import methods.cache_strategy.common as host
    check("BSC impl comes from AM_DiT", "AM_DiT" in impl.__file__, impl.__file__)
    check("host common.py is still the host's", "video_gen_physics" in host.__file__,
          host.__file__)

    print()
    if _FAILURES:
        print(f"*** {len(_FAILURES)} FAILED: {_FAILURES}")
        return 1
    print("ALL CHECKS PASSED -- BSC is selectable on DreamGen; only a GPU run remains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
