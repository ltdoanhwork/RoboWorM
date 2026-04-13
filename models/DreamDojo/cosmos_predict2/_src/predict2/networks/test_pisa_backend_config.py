from __future__ import annotations

import torch

from methods.sparse_attention.pisa.config import build_dreamdojo_pisa_parameters
from methods.sparse_attention.svg.config import build_dreamdojo_svg_parameters

from cosmos_predict2._src.predict2.networks import minimal_v4_dit as dit_mod


class _DummyPISAOp(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def forward(self, q, k, v, *args, **kwargs):
        del k, v, args, kwargs
        return torch.zeros_like(q)

    def set_context_parallel_group(self, *args, **kwargs):
        del args, kwargs
        return None


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


def test_build_dreamdojo_pisa_parameters_2b_defaults():
    params = build_dreamdojo_pisa_parameters(num_blocks=28)
    assert len(params) == 28
    assert params[9] is None
    assert params[10]["density"] == 0.45
    assert params[10]["block_size"] == 80
    assert params[10]["use_bias"] is False


def test_build_dreamdojo_pisa_parameters_14b_defaults():
    params = build_dreamdojo_pisa_parameters(num_blocks=36)
    assert len(params) == 36
    assert params[11] is None
    assert params[12]["density"] == 0.40
    assert params[12]["block_size"] == 80
    assert params[12]["use_bias"] is False


def test_pisa_and_svg_are_mutually_exclusive():
    pisa_parameters = build_dreamdojo_pisa_parameters(num_blocks=4, start_layer_idx=1)
    svg_parameters = build_dreamdojo_svg_parameters(num_blocks=4, variant="svg1", start_layer_idx=1)
    try:
        _build_tiny_model(
            self_attention_backend="pisa",
            cross_attention_backend="minimal_a2a",
            pisa_parameters=pisa_parameters,
            svg_parameters=svg_parameters,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when PISA and SVG are both enabled.")


def test_pisa_only_routes_self_attention(monkeypatch):
    monkeypatch.setattr(dit_mod, "PISAA2AAttnOp", _DummyPISAOp)
    pisa_parameters = build_dreamdojo_pisa_parameters(num_blocks=4, start_layer_idx=1)
    model = _build_tiny_model(
        self_attention_backend="pisa",
        cross_attention_backend="minimal_a2a",
        pisa_parameters=pisa_parameters,
    )

    assert model.blocks[0].self_attn.backend == "minimal_a2a"
    assert model.blocks[1].self_attn.backend == "pisa"
    assert model.blocks[0].cross_attn.backend == "minimal_a2a"


def test_pisa_cross_attention_is_rejected():
    try:
        _build_tiny_model(
            self_attention_backend="minimal_a2a",
            cross_attention_backend="pisa",
        )
    except ValueError as exc:
        assert "Cross-attention is not supported" in str(exc)
    else:
        raise AssertionError("Expected ValueError when configuring PISA for cross-attention.")
