from __future__ import annotations

import pytest
import torch

from methods.sparse_attention.radial.config import build_dreamdojo_radial_parameters

from cosmos_predict2._src.predict2.networks import a2a_cp as a2a_cp_mod
from cosmos_predict2._src.predict2.networks import minimal_v4_dit as dit_mod
from cosmos_predict2._src.predict2.networks.a2a_cp import RadialA2AAttnOp


class _DummyRadialOp(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.last_video_size = None

    def forward(self, q, k, v, *args, **kwargs):
        del k, v, args
        self.last_video_size = kwargs.get("video_size")
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2] * q.shape[3], dtype=q.dtype, device=q.device)

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


def test_build_dreamdojo_radial_parameters_2b_defaults():
    params = build_dreamdojo_radial_parameters(num_blocks=28)
    assert len(params) == 28
    assert params[5] is None
    assert params[6]["layer_idx"] == 6
    assert params[6]["decay_factor"] == 1.0
    assert params[6]["block_size"] == 64
    assert params[6]["model_type"] == "hunyuan"


def test_build_dreamdojo_radial_parameters_14b_defaults():
    params = build_dreamdojo_radial_parameters(num_blocks=36)
    assert len(params) == 36
    assert params[7] is None
    assert params[8]["layer_idx"] == 8
    assert params[8]["decay_factor"] == 1.0
    assert params[8]["block_size"] == 64


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"decay_factor": 0.0}, "positive"),
        ({"block_size": 32}, "64 or 128"),
        ({"start_layer_idx": -1}, "non-negative"),
        ({"model_type": "bad"}, "'hunyuan' or 'wan'"),
    ],
)
def test_build_dreamdojo_radial_parameters_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_dreamdojo_radial_parameters(num_blocks=4, **kwargs)


def test_radial_and_svg_are_mutually_exclusive():
    radial_parameters = build_dreamdojo_radial_parameters(num_blocks=4, start_layer_idx=1)
    svg_parameters = [{"variant": "svg1", "layer_idx": i, "dense_step_frac": 0.2, "sparsity": 0.4, "num_sampled_rows": 32, "sample_mse_max_row": 64} for i in range(4)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_tiny_model(
            self_attention_backend="radial",
            cross_attention_backend="minimal_a2a",
            radial_parameters=radial_parameters,
            svg_parameters=svg_parameters,
        )


def test_radial_and_natten_are_mutually_exclusive():
    radial_parameters = build_dreamdojo_radial_parameters(num_blocks=4, start_layer_idx=1)
    natten_parameters = [{"window_size": (3, 3, 3)} for _ in range(4)]
    with pytest.raises(ValueError, match="mutually exclusive"):
        _build_tiny_model(
            self_attention_backend="radial",
            cross_attention_backend="minimal_a2a",
            radial_parameters=radial_parameters,
            natten_parameters=natten_parameters,
        )


def test_radial_only_routes_self_attention(monkeypatch):
    monkeypatch.setattr(dit_mod, "RadialA2AAttnOp", _DummyRadialOp)
    radial_parameters = build_dreamdojo_radial_parameters(num_blocks=4, start_layer_idx=1)
    model = _build_tiny_model(
        self_attention_backend="radial",
        cross_attention_backend="minimal_a2a",
        radial_parameters=radial_parameters,
    )

    assert model.blocks[0].self_attn.backend == "minimal_a2a"
    assert model.blocks[1].self_attn.backend == "radial"
    assert model.blocks[0].cross_attn.backend == "minimal_a2a"


def test_radial_attention_smoke_passes_video_size(monkeypatch):
    monkeypatch.setattr(dit_mod, "RadialA2AAttnOp", _DummyRadialOp)
    radial_parameters = build_dreamdojo_radial_parameters(num_blocks=4, start_layer_idx=1)
    model = _build_tiny_model(
        self_attention_backend="radial",
        cross_attention_backend="minimal_a2a",
        radial_parameters=radial_parameters,
    )

    x = torch.randn(1, 8, 128)
    video_size = dit_mod.VideoSize(T=2, H=2, W=2)
    out = model.blocks[1].self_attn(x, video_size=video_size)

    assert out.shape == x.shape
    assert model.blocks[1].self_attn.attn_op.last_video_size == video_size


def test_radial_a2a_rejects_context_parallel():
    op = RadialA2AAttnOp(block_size=64, decay_factor=1.0, model_type="hunyuan")
    with pytest.raises(RuntimeError, match="single-GPU|multi-GPU|Context parallel"):
        op.set_context_parallel_group(process_group=None, ranks=[0, 1], stream=None)


def test_radial_local_attn_op_strips_layer_idx(monkeypatch):
    captured = {}

    class _DummyRuntime:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(a2a_cp_mod, "_load_radial_runtime_class", lambda: _DummyRuntime)
    a2a_cp_mod.RadialLocalAttnOp(
        layer_idx=6,
        block_size=64,
        decay_factor=1.0,
        model_type="hunyuan",
    )

    assert "layer_idx" not in captured
    assert captured == {
        "block_size": 64,
        "decay_factor": 1.0,
        "model_type": "hunyuan",
    }


def test_action_conditioned_radial_guard():
    pytest.importorskip("piq")
    from cosmos_predict2 import action_conditioned as action_mod
    from cosmos_predict2.action_conditioned_config import (
        ActionConditionedInferenceArguments,
        ActionConditionedSetupArguments,
    )

    setup_args = ActionConditionedSetupArguments(use_radial=True, context_parallel_size=2)
    inference_args = ActionConditionedInferenceArguments()

    with pytest.raises(ValueError, match="single-GPU"):
        action_mod.inference(setup_args, inference_args, checkpoint_path="unused.pt")
