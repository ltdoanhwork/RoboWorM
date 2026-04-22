from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

einops = types.ModuleType("einops")
einops.rearrange = lambda tensor, pattern, **kwargs: tensor
sys.modules.setdefault("einops", einops)

from methods.cache_strategy.DiCache.config import add_dicache_args
from methods.cache_strategy.DiCache.ctrl_world_runtime import CtrlWorldDiCacheRuntime
from methods.cache_strategy.WorldCache.config import add_worldcache_args
from methods.cache_strategy.WorldCache.ctrl_world_runtime import CtrlWorldWorldCacheRuntime
from methods.cache_strategy.common import DiCacheConfig, WorldCacheConfig
from methods.cache_strategy.ctrl_world_utils import validate_ctrl_world_backend_args


def _make_backend_args(**kwargs):
    defaults = {
        "use_pisa": False,
        "use_svg": False,
        "use_radial": False,
        "use_sito": False,
        "use_itm": False,
        "use_fastercache": False,
        "use_dicache": False,
        "use_worldcache": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_ctrl_world_cache_cli_helpers_parse_values():
    parser = argparse.ArgumentParser()
    add_dicache_args(parser)
    add_worldcache_args(parser)

    args = parser.parse_args(
        [
            "--use_dicache",
            "--dicache_rel_l1_thresh",
            "0.11",
            "--dicache_ret_ratio",
            "0.3",
            "--dicache_probe_depth",
            "4",
            "--use_worldcache",
            "--worldcache_rel_l1_thresh",
            "0.05",
            "--worldcache_ret_ratio",
            "0.6",
            "--worldcache_probe_depth",
            "5",
            "--worldcache_motion_sensitivity",
            "7.0",
            "--worldcache_hf_enabled",
            "--worldcache_saliency_enabled",
            "--worldcache_osi_enabled",
            "--worldcache_dynamic_decay",
        ]
    )

    assert args.use_dicache is True
    assert args.dicache_rel_l1_thresh == 0.11
    assert args.dicache_ret_ratio == 0.3
    assert args.dicache_probe_depth == 4
    assert args.use_worldcache is True
    assert args.worldcache_probe_depth == 5
    assert args.worldcache_motion_sensitivity == 7.0
    assert args.worldcache_hf_enabled is True
    assert args.worldcache_saliency_enabled is True
    assert args.worldcache_osi_enabled is True
    assert args.worldcache_dynamic_decay is True


def test_ctrl_world_backend_validator_rejects_multiple_backends():
    args = _make_backend_args(use_dicache=True, use_worldcache=True)

    try:
        validate_ctrl_world_backend_args(args)
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected mutual-exclusion validation to raise.")


def test_ctrl_world_dicache_reset_and_skip_use_denoising_steps():
    runtime = CtrlWorldDiCacheRuntime(
        config=DiCacheConfig(num_steps=50, rel_l1_thresh=0.08, ret_ratio=0.0, probe_depth=2),
        total_steps=50,
    )
    runtime.reset(12)

    assert runtime.total_steps == 12
    assert runtime.current_step_idx == -1
    assert runtime.output_window == []
    assert runtime.previous_probe_state is None

    sample = torch.zeros(1, 2, 4, 3, 3)
    probe = torch.ones(2, 8, 3, 3)
    out = torch.full((1, 2, 4, 3, 3), 2.0)

    runtime.begin_step(sample, step_idx=0, do_classifier_free_guidance=False)
    assert runtime.observe_probe(probe) is None
    assert runtime.observe_probe(probe) is None
    runtime.finish_step(out)

    runtime.begin_step(sample, step_idx=1, do_classifier_free_guidance=False)
    assert runtime.observe_probe(probe) is None
    skipped = runtime.observe_probe(probe)

    assert skipped is not None
    assert skipped.shape == out.shape
    assert torch.allclose(skipped, out)


def test_ctrl_world_dicache_requires_both_cfg_halves_to_pass():
    runtime = CtrlWorldDiCacheRuntime(
        config=DiCacheConfig(num_steps=10, rel_l1_thresh=0.05, ret_ratio=0.0, probe_depth=1),
        total_steps=10,
    )

    sample = torch.zeros(2, 2, 4, 2, 2)
    prev_probe = torch.zeros(4, 6, 2, 2)
    runtime.begin_step(sample, step_idx=0, do_classifier_free_guidance=True)
    assert runtime.observe_probe(prev_probe) is None
    runtime.finish_step(torch.zeros(2, 2, 4, 2, 2))

    current_probe = prev_probe.clone()
    current_probe[2:] = 10.0
    runtime.begin_step(sample, step_idx=1, do_classifier_free_guidance=True)
    skipped = runtime.observe_probe(current_probe)

    assert skipped is None


def test_ctrl_world_worldcache_dynamic_threshold_can_block_skip():
    runtime = CtrlWorldWorldCacheRuntime(
        config=WorldCacheConfig(
            num_steps=10,
            rel_l1_thresh=0.2,
            ret_ratio=0.0,
            probe_depth=1,
            motion_sensitivity=5.0,
        ),
        total_steps=10,
    )

    sample = torch.ones(1, 2, 4, 2, 2)
    probe = torch.ones(2, 6, 2, 2)
    runtime.begin_step(sample, step_idx=0, do_classifier_free_guidance=False)
    assert runtime.observe_probe(probe) is None
    runtime.finish_step(torch.ones(1, 2, 4, 2, 2))

    runtime.begin_step(sample * 50.0, step_idx=1, do_classifier_free_guidance=False)
    skipped = runtime.observe_probe(probe * 1.02)

    assert skipped is None


def test_ctrl_world_worldcache_hf_guard_blocks_skip():
    runtime = CtrlWorldWorldCacheRuntime(
        config=WorldCacheConfig(
            num_steps=10,
            rel_l1_thresh=1.0,
            ret_ratio=0.0,
            probe_depth=1,
            hf_enabled=True,
            hf_thresh=0.001,
        ),
        total_steps=10,
    )

    sample = torch.zeros(1, 2, 4, 4, 4)
    probe = torch.zeros(2, 6, 4, 4)
    runtime.begin_step(sample, step_idx=0, do_classifier_free_guidance=False)
    assert runtime.observe_probe(probe) is None
    runtime.finish_step(torch.zeros(1, 2, 4, 4, 4))

    sample_next = sample.clone()
    sample_next[..., 1:3, 1:3] = 1.0
    runtime.begin_step(sample_next, step_idx=1, do_classifier_free_guidance=False)
    skipped = runtime.observe_probe(probe)

    assert skipped is None


def test_ctrl_world_worldcache_osi_changes_output_extrapolation():
    runtime = CtrlWorldWorldCacheRuntime(
        config=WorldCacheConfig(
            num_steps=10,
            rel_l1_thresh=1.0,
            ret_ratio=0.0,
            probe_depth=1,
            osi_enabled=True,
        ),
        total_steps=10,
    )
    runtime.output_cache = torch.ones(1, 2, 4, 2, 2)
    runtime.output_window = [
        torch.zeros(1, 2, 4, 2, 2),
        torch.ones(1, 2, 4, 2, 2),
    ]
    runtime.full_probe_window = [
        torch.zeros(2, 6, 2, 2),
        torch.ones(2, 6, 2, 2),
    ]
    runtime.probe_state_for_step = torch.full((2, 6, 2, 2), 3.0)

    predicted = runtime._predict_output()

    assert predicted.shape == runtime.output_cache.shape
    assert torch.all(predicted > runtime.output_cache)
