from __future__ import annotations

import torch

from methods.sparse_attention.svg.config import build_dreamdojo_svg_parameters

from cosmos_predict2._src.predict2.networks import minimal_v4_dit as dit_mod


class _DummySVGOp(torch.nn.Module):
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


def test_build_dreamdojo_svg_parameters_2b_defaults():
    params = build_dreamdojo_svg_parameters(num_blocks=28, variant="svg1")
    assert len(params) == 28
    assert params[8] is None
    assert params[9]["layer_idx"] == 9
    assert params[9]["dense_step_frac"] == 0.20
    assert params[9]["sparsity"] == 0.40


def test_build_dreamdojo_svg_parameters_14b_svg2_defaults():
    params = build_dreamdojo_svg_parameters(num_blocks=36, variant="svg2")
    assert len(params) == 36
    assert params[12] is None
    assert params[13]["layer_idx"] == 13
    assert params[13]["dense_step_frac"] == 0.25
    assert params[13]["num_q_centroids"] == 24
    assert params[13]["num_k_centroids"] == 96


def test_svg_and_natten_are_mutually_exclusive():
    svg_parameters = build_dreamdojo_svg_parameters(num_blocks=4, variant="svg1", start_layer_idx=1)
    natten_parameters = [{"window_size": (3, 3, 3)} for _ in range(4)]
    try:
        _build_tiny_model(
            self_attention_backend="svg",
            cross_attention_backend="minimal_a2a",
            svg_parameters=svg_parameters,
            natten_parameters=natten_parameters,
        )
    except ValueError as exc:
        assert "mutually exclusive" in str(exc)
    else:
        raise AssertionError("Expected ValueError when SVG and NATTEN are both enabled.")


def test_svg_only_routes_self_attention(monkeypatch):
    monkeypatch.setattr(dit_mod, "SVGA2AAttnOp", _DummySVGOp)
    svg_parameters = build_dreamdojo_svg_parameters(num_blocks=4, variant="svg1", start_layer_idx=1)
    model = _build_tiny_model(
        self_attention_backend="svg",
        cross_attention_backend="minimal_a2a",
        svg_parameters=svg_parameters,
    )

    assert model.blocks[0].self_attn.backend == "minimal_a2a"
    assert model.blocks[1].self_attn.backend == "svg"
    assert model.blocks[0].cross_attn.backend == "minimal_a2a"
