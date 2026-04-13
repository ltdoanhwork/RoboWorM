from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from methods.prunning.SiTo.runtime import build_sito_parameters
from cosmos_predict2._src.predict2.networks import minimal_v4_dit as dit_mod


def _build_tiny_model(**kwargs):
    return dit_mod.MiniTrainDIT(
        max_img_h=32,
        max_img_w=32,
        max_frames=8,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=128,
        num_blocks=4,
        num_heads=4,
        concat_padding_mask=False,
        pos_emb_cls="rope3d",
        pos_emb_learnable=False,
        atten_backend="minimal_a2a",
        sac_config=dit_mod.SACConfig(mode="none"),
        **kwargs,
    )


def test_build_sito_parameters_defaults_for_2b_depth():
    params = build_sito_parameters(num_blocks=28)
    assert len(params) == 28
    assert params[3] is None
    assert params[4]["layer_idx"] == 4
    assert params[-2] is None
    assert params[-1] is None


def test_sito_and_svg_are_mutually_exclusive():
    try:
        _build_tiny_model(
            use_sito=True,
            svg_parameters=[{"variant": "svg1"} for _ in range(4)],
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when SiTo and SVG are both enabled.")


def test_sito_routes_through_dense_self_attention_backend():
    model = _build_tiny_model(use_sito=True, sito_start_layer_idx=1, sito_keep_last_n_dense=1)

    assert model.blocks[0].self_attn.backend == "minimal_a2a"
    assert model.blocks[1].self_attn.backend == "minimal_a2a"
    assert model.blocks[1].sito_runtime is not None
    assert model.blocks[0].cross_attn.backend == "minimal_a2a"
