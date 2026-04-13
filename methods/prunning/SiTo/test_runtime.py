from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from methods.prunning.SiTo.runtime import SiToTokenPruner


def test_sito_per_frame_roundtrip_preserves_shape_and_kept_tokens():
    pruner = SiToTokenPruner(group_mode="per_frame", prune_ratio=0.20, patch_h=2, patch_w=2)
    hidden_states = torch.randn(2, 3 * 4 * 4, 8)
    plan = pruner.prepare(hidden_states, video_size=SimpleNamespace(T=3, H=4, W=4))

    assert plan is not None
    pruned = pruner.prune(hidden_states, plan)
    recovered = pruner.recover(pruned, plan)

    assert recovered.shape == hidden_states.shape
    assert torch.allclose(recovered[:, plan.keep_indices], hidden_states[:, plan.keep_indices])


def test_sito_full_2d_roundtrip_supports_rectangular_grids():
    pruner = SiToTokenPruner(group_mode="full_2d", prune_ratio=0.35, patch_h=2, patch_w=2)
    hidden_states = torch.randn(2, 18 * 10, 16)
    plan = pruner.prepare(hidden_states, video_size=SimpleNamespace(T=1, H=18, W=10))

    assert plan is not None
    pruned = pruner.prune(hidden_states, plan)
    recovered = pruner.recover(pruned, plan)

    assert pruned.shape[1] < hidden_states.shape[1]
    assert recovered.shape == hidden_states.shape


def test_sito_invalid_prune_ratio_raises():
    with pytest.raises(ValueError, match="prune_ratio"):
        SiToTokenPruner(group_mode="per_frame", prune_ratio=0.80, patch_h=2, patch_w=2)
