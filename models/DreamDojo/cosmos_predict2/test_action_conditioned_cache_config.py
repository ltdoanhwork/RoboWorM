from __future__ import annotations

from pathlib import Path
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cosmos_cuda = types.ModuleType("cosmos_cuda")
cosmos_cuda.__version__ = "1.4.1"
sys.modules.setdefault("cosmos_cuda", cosmos_cuda)

from methods.cache_strategy.common import FasterCacheConfig, WorldCacheConfig
from cosmos_predict2.cache_runtime import (
    build_cache_runtime_config,
    validate_runtime_backend_args,
)


def _make_setup(**kwargs):
    defaults = dict(
        output_dir=Path("/tmp/dreamdojo_cache_test"),
        experiment="dummy_experiment",
        context_parallel_size=1,
        use_svg=False,
        use_sito=False,
        use_itm=False,
        worldcache_enabled=False,
        worldcache_num_steps=35,
        worldcache_rel_l1_thresh=0.03,
        worldcache_ret_ratio=0.4,
        worldcache_probe_depth=4,
        worldcache_motion_sensitivity=5.0,
        worldcache_flow_enabled=False,
        worldcache_flow_scale=0.5,
        worldcache_hf_enabled=False,
        worldcache_hf_thresh=0.01,
        worldcache_saliency_enabled=False,
        worldcache_saliency_weight=5.0,
        worldcache_osi_enabled=False,
        worldcache_dynamic_decay=False,
        worldcache_aduc_enabled=False,
        worldcache_aduc_start=0.5,
        worldcache_parallel_cfg=False,
        dicache_enabled=False,
        dicache_num_steps=35,
        dicache_rel_l1_thresh=0.08,
        dicache_ret_ratio=0.2,
        dicache_probe_depth=2,
        fastercache_enabled=False,
        fastercache_start_step=0,
        fastercache_model_interval=5,
        fastercache_block_interval=3,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _make_inference(**kwargs):
    defaults = dict(guidance=0)
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_build_worldcache_runtime_config_uses_prefixed_fields():
    config = build_cache_runtime_config(
        _make_setup(
            worldcache_enabled=True,
            worldcache_rel_l1_thresh=0.07,
            worldcache_probe_depth=6,
        )
    )

    assert isinstance(config, WorldCacheConfig)
    assert config.rel_l1_thresh == 0.07
    assert config.probe_depth == 6


def test_build_fastercache_runtime_config_uses_prefixed_fields():
    config = build_cache_runtime_config(
        _make_setup(
            fastercache_enabled=True,
            fastercache_start_step=0,
            fastercache_model_interval=7,
            fastercache_block_interval=4,
        )
    )

    assert isinstance(config, FasterCacheConfig)
    assert config.start_step == 0
    assert config.model_interval == 7
    assert config.block_interval == 4


def test_multiple_cache_backends_raise():
    setup_args = _make_setup(worldcache_enabled=True, dicache_enabled=True)
    inference_args = _make_inference()

    try:
        validate_runtime_backend_args(setup_args, inference_args)
    except ValueError as exc:
        assert "cache backends are mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when more than one cache backend is enabled.")


def test_cache_and_sparse_backends_raise():
    setup_args = _make_setup(worldcache_enabled=True, use_sito=True)
    inference_args = _make_inference(guidance=1)

    try:
        validate_runtime_backend_args(setup_args, inference_args)
    except ValueError as exc:
        assert "mutually exclusive with sparse/token backends" in str(exc)
    else:
        raise AssertionError("Expected ValueError when cache and sparse backends are combined.")


def test_cache_with_context_parallel_raises():
    setup_args = _make_setup(dicache_enabled=True, context_parallel_size=2)
    inference_args = _make_inference()

    try:
        validate_runtime_backend_args(setup_args, inference_args)
    except ValueError as exc:
        assert "single-GPU only" in str(exc)
    else:
        raise AssertionError("Expected ValueError when cache backend is used with context parallelism.")


def test_no_legacy_world_cache_dependency_in_primary_runtime_files():
    repo_root = Path(__file__).resolve().parents[3]
    files = [
        repo_root / "models/DreamDojo/cosmos_predict2/action_conditioned.py",
        repo_root / "models/DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py",
        repo_root / "methods/cache_strategy/WorldCache/runtime.py",
        repo_root / "methods/cache_strategy/DiCache/runtime.py",
    ]

    forbidden_snippets = [
        "world_cache/VLA-Humanoid/Models/DreamDojo",
        "predict2.inference.worldcache_utils",
        "predict2.inference.dicache_utils",
    ]
    required_snippets = [
        "methods.cache_strategy",
    ]

    for path in files:
        text = path.read_text()
        for snippet in forbidden_snippets:
            assert snippet not in text, f"Unexpected legacy dependency {snippet!r} in {path}"

    runtime_text = (repo_root / "models/DreamDojo/cosmos_predict2/_src/predict2/inference/video2world.py").read_text()
    for snippet in required_snippets:
        assert snippet in runtime_text
