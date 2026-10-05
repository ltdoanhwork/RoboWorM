"""Tests for the BaseMetric framework.  Run:

    models/dreamgen/.venv/bin/python -m pytest metrics/test_framework.py -q

CPU-only and synthetic except ``test_parity_with_inference_scripts``, which
uses one real episode if the sampling_dataset is present (skipped otherwise).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from metrics.base import BaseMetric, FrameMetric, Sample, safe_id, staged_names
from metrics.evaluate import discover_samples
from metrics.registry import available, build_metric, get_metric_class
from metrics.video_io import load_frames, read_prompt

REPO = Path(__file__).resolve().parents[1]
REAL_DIR = REPO / ("sampling_dataset/dense/single_arm/output/singleview/dreamgen/"
                   "makovian/iter_000090000_static16fps/global")


def _video(T=6, H=64, W=80, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (T, H, W, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
class _Const(BaseMetric):
    name = "const"
    higher_is_better = {"v": True}

    def compute(self, sample):
        if sample.id == "boom":
            raise RuntimeError("nope")
        return {"v": float("nan") if sample.id == "nan" else float(sample.meta["v"])}


def test_base_aggregate_nan_and_errors():
    s = [Sample("a", None, meta={"v": 1}), Sample("b", None, meta={"v": 3}),
         Sample("nan", None), Sample("boom", None)]
    r = _Const().evaluate(s, verbose=False)
    assert r.summary["v"]["mean"] == 2.0 and r.summary["v"]["n"] == 2
    assert r.summary["v"]["n_missing"] == 1          # the NaN row
    assert set(r.errors) == {"boom"}
    d = r.to_dict()                                   # strict-JSON safe
    assert d["per_sample"]["nan"]["v"] is None


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        _Const().evaluate([Sample("a", None, meta={"v": 1})] * 2, verbose=False)


def test_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        BaseMetric()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        FrameMetric()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
def test_psnr_ssim_lpips_identity_and_order():
    a = _video()
    noisy = np.clip(a.astype(int) + np.random.default_rng(1).integers(-20, 21, a.shape), 0, 255).astype(np.uint8)
    f = build_metric("fidelity", device="cpu")
    same = f(Sample("s", a, a))
    diff = f(Sample("d", a, noisy))
    assert same["ssim"] > 0.999 and same["lpips"] < 1e-4 and same["psnr"] > 70
    assert diff["psnr"] < same["psnr"] and diff["ssim"] < same["ssim"] and diff["lpips"] > same["lpips"]
    # the single-metric classes agree with the combined one
    for name in ("psnr", "ssim", "lpips"):
        assert math.isclose(build_metric(name, device="cpu")(Sample("d", a, noisy))[name],
                            diff[name], rel_tol=1e-5)


def test_chunking_does_not_change_result():
    a, b = _video(T=11, seed=2), _video(T=11, seed=3)
    r1 = build_metric("fidelity", device="cpu", chunk=4)(Sample("x", a, b))
    r2 = build_metric("fidelity", device="cpu", chunk=64)(Sample("x", a, b))
    for k in ("psnr", "ssim", "lpips"):
        assert math.isclose(r1[k], r2[k], rel_tol=1e-5), k


def test_frame_alignment_policy():
    a, b = _video(T=8), _video(T=5, seed=4)
    r = build_metric("psnr", device="cpu")(Sample("x", a, b))
    assert (r["n_frames_pred"], r["n_frames_ref"], r["n_frames"]) == (8, 5, 5)
    with pytest.raises(ValueError, match="frame count"):
        build_metric("psnr", device="cpu", frame_align="error")(Sample("x", a, b))
    with pytest.raises(ValueError, match="resolution"):
        build_metric("psnr", device="cpu")(Sample("x", a, _video(W=96)))


def test_requires_ref_and_text():
    with pytest.raises(ValueError, match="reference"):
        build_metric("psnr", device="cpu")(Sample("x", _video()))
    with pytest.raises(ValueError, match="text"):
        get_metric_class("clip_score")(device="cpu").validate(Sample("x", _video()))


def test_reference_free_metric_ignores_ref():
    """A shorter GT must not truncate a metric that does not use it."""
    class _Len(FrameMetric):
        name = "len"
        higher_is_better = {"T": True}

        def score_frames(self, pred, ref, text):
            assert ref is None
            return {"T": float(len(pred))}

    r = _Len(device="cpu")(Sample("x", _video(T=9), ref=_video(T=3, W=96)))
    assert r["T"] == 9 and "n_frames_ref" not in r


# --------------------------------------------------------------------------- #
def test_discovery_pairs_pred_gt_prompt(tmp_path):
    import imageio.v3 as iio

    d = tmp_path / "cell" / "global"
    d.mkdir(parents=True)
    for ep in ("episode_000000", "episode_000001"):
        iio.imwrite(d / f"{ep}.mp4", _video(T=4, H=64, W=64), fps=4)
        (d / f"{ep}.txt").write_text("[Prompt]\nA robot arm places the cup.\n[Negative Prompt]\nugly\n")
    iio.imwrite(d / "episode_000000_gt.mp4", _video(T=4, H=64, W=64), fps=4)
    s = discover_samples(tmp_path)
    assert [x.id for x in s] == ["cell/global/episode_000000", "cell/global/episode_000001"]
    assert s[0].ref is not None and s[0].ref.name == "episode_000000_gt.mp4"
    assert s[1].ref is None                                    # missing gt -> None, not a crash
    assert s[0].text == "A robot arm places the cup."
    assert load_frames(s[0].pred).shape == (4, 64, 64, 3)


def test_read_prompt_without_headers(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("just a caption\n")
    assert read_prompt(p) == "just a caption"
    assert read_prompt(tmp_path / "missing.txt") is None


def test_safe_id_and_collisions():
    assert safe_id("cell/global/ep 1") == "cell__global__ep_1"
    with pytest.raises(ValueError, match="collide"):
        staged_names([Sample("a b", None), Sample("a_b", None)])


def test_registry_lazy_and_complete():
    assert {"psnr", "ssim", "lpips", "fidelity", "clip_score",
            "multiview_consistency", "hallucination"} <= set(available())
    assert get_metric_class("hal") is get_metric_class("hallucination")
    h = build_metric("hallucination", embodiment="bimanual", score_ref=True)
    assert h.requires_ref and "rigidity_rate_excess" in h.higher_is_better
    assert all(v is False for v in h.higher_is_better.values())   # all lower-is-better
    with pytest.raises(ValueError):
        build_metric("hallucination", embodiment="quadruped")
    m = build_metric("multiview_consistency")
    assert m.launcher_path().is_file()
    assert build_metric("hallucination").launcher_path().is_file()


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REAL_DIR / "episode_000000_gt.mp4").is_file(), reason="no sampling_dataset")
def test_parity_with_inference_scripts():
    """Same numbers as the piq call in scripts/infer_single_arm_singleview_*.py."""
    import piq
    import torch

    pred = load_frames(REAL_DIR / "episode_000000.mp4")[:48]
    gt = load_frames(REAL_DIR / "episode_000000_gt.mp4")[:40]
    n = min(len(pred), len(gt))                                # their min_len truncation
    x = torch.clamp(torch.from_numpy(pred[:n].copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y = torch.clamp(torch.from_numpy(gt[:n].copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    want = {"psnr": piq.psnr(x, y).mean().item(), "ssim": piq.ssim(x, y).mean().item(),
            "lpips": piq.LPIPS()(x, y).mean().item()}
    got = build_metric("fidelity", device="cpu", chunk=16)(Sample("ep0", pred, gt))
    for k, v in want.items():
        assert math.isclose(got[k], v, rel_tol=1e-4, abs_tol=1e-5), (k, got[k], v)
