from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from methods.prunning.importance_token_merge.runtime import (  # noqa: E402
    ITMTokenMerger,
    build_itm_attention_context,
    compute_itm_heat_map,
    infer_ctrl_world_heat_map_shape,
)


def test_itm_per_frame_merge_roundtrip_preserves_shape():
    merger = ITMTokenMerger(group_mode="per_frame", compress_ratio=0.25)
    hidden_states = torch.randn(2, 3 * 4 * 4, 8)
    heat_map = torch.rand(2, 3, 4, 4)

    plan = merger.prepare(hidden_states, heat_map=heat_map, video_size=SimpleNamespace(T=3, H=4, W=4))

    assert plan is not None
    merged = merger.merge(hidden_states, plan, mode="mean")
    restored = merger.unmerge(merged, plan, mode="mean")

    assert merged.shape[1] < hidden_states.shape[1]
    assert restored.shape == hidden_states.shape


def test_itm_full_2d_framewise_roundtrip_preserves_shape():
    merger = ITMTokenMerger(group_mode="full_2d_framewise", compress_ratio=0.25)
    hidden_states = torch.randn(2, 18 * 10, 16)
    heat_map = torch.rand(2, 18, 10)

    plan = merger.prepare(hidden_states, heat_map=heat_map)

    assert plan is not None
    merged = merger.merge(hidden_states, plan, mode="none")
    restored = merger.unmerge(merged, plan, mode="none")

    assert restored.shape == hidden_states.shape


def test_compute_itm_heat_map_downsamples_to_patch_grid():
    cond = torch.randn(2, 4, 3, 8, 12)
    uncond = torch.randn(2, 4, 3, 8, 12)

    heat_map = compute_itm_heat_map(cond, uncond, patch_spatial=2)

    assert heat_map.shape == (2, 3, 4, 6)


def test_build_itm_attention_context_attaches_itm_state():
    context = build_itm_attention_context(
        runtime_context={"sigma": 1.0},
        itm_state={"phase": "prune", "merge_mode": "none", "heat_map": torch.ones(1, 1, 2, 2)},
        diffusion_step_idx=3,
    )

    assert context["sigma"] == 1.0
    assert context["itm"]["phase"] == "prune"
    assert context["itm"]["merge_mode"] == "none"
    assert context["itm"]["diffusion_step_idx"] == 3


def test_invalid_compress_ratio_raises():
    with pytest.raises(ValueError, match="compress_ratio"):
        ITMTokenMerger(compress_ratio=0.8)


def test_ctrl_world_geometry_inference_maps_expected_lengths():
    assert (infer_ctrl_world_heat_map_shape(2880).H, infer_ctrl_world_heat_map_shape(2880).W) == (72, 40)
    assert (infer_ctrl_world_heat_map_shape(720).H, infer_ctrl_world_heat_map_shape(720).W) == (36, 20)
    assert (infer_ctrl_world_heat_map_shape(180).H, infer_ctrl_world_heat_map_shape(180).W) == (18, 10)
    assert infer_ctrl_world_heat_map_shape(45) is None
