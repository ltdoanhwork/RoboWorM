from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from cosmos_predict2.configs.base.config_itm import build_itm_parameters


def test_build_itm_parameters_defaults_for_14b_depth():
    params = build_itm_parameters(num_blocks=36)
    assert len(params) == 36
    assert params[5] is None
    assert params[6]["layer_idx"] == 6
    assert params[-2] is None
    assert params[-1] is None


def test_build_itm_parameters_defaults_for_2b_depth():
    params = build_itm_parameters(num_blocks=28)
    assert len(params) == 28
    assert params[3] is None
    assert params[4]["layer_idx"] == 4
    assert params[-2] is None
    assert params[-1] is None


def test_build_itm_parameters_rejects_invalid_compress_ratio():
    with pytest.raises(ValueError, match="compress_ratio"):
        build_itm_parameters(num_blocks=28, compress_ratio=0.8)
