from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cosmos_predict2.configs.base.config_sito import build_sito_parameters


def test_build_sito_parameters_defaults_for_14b_depth():
    params = build_sito_parameters(num_blocks=36)
    assert len(params) == 36
    assert params[5] is None
    assert params[6]["layer_idx"] == 6
    assert params[-2] is None
    assert params[-1] is None


def test_build_sito_parameters_defaults_for_2b_depth():
    params = build_sito_parameters(num_blocks=28)
    assert len(params) == 28
    assert params[3] is None
    assert params[4]["layer_idx"] == 4
    assert params[-2] is None
    assert params[-1] is None


def test_build_sito_parameters_rejects_invalid_prune_ratio():
    with pytest.raises(ValueError, match="prune_ratio"):
        build_sito_parameters(num_blocks=28, prune_ratio=0.8, patch_h=2, patch_w=2)
