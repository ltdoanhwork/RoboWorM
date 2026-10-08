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
        get_metric_class("text_alignment")(device="cpu").validate(Sample("x", _video()))


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
    assert {"kiva", "rigis", "visual_integrity", "text_alignment", "fvd", "psnr", "ssim",
            "lpips", "fidelity", "motion_smoothness", "motion_amplitude"} == set(available())
    # pre-refactor names stay valid aliases
    assert get_metric_class("hallucination") is get_metric_class("kiva")
    assert get_metric_class("hal") is get_metric_class("kiva")
    assert get_metric_class("multiview_consistency") is get_metric_class("rigis")
    assert get_metric_class("clip_score") is get_metric_class("text_alignment")
    from metrics.text_alignment import CLIPScore, TextAlignment
    assert CLIPScore is TextAlignment
    h = build_metric("hallucination", embodiment="bimanual", score_ref=True)
    assert h.name == "kiva"
    assert h.requires_ref and "rigidity_rate_excess" in h.higher_is_better
    assert all(v is False for v in h.higher_is_better.values())   # all lower-is-better
    with pytest.raises(ValueError):
        build_metric("kiva", embodiment="quadruped")
    m = build_metric("rigis")
    assert m.name == "rigis" and m.launcher_path().is_file()
    assert build_metric("kiva").launcher_path().is_file()
    assert build_metric("fvd").requires_ref
    assert not build_metric("fvd", ref_pool="pool.npy").requires_ref


def test_fvd_frechet_and_guards():
    from metrics import fvd

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 8))
    assert abs(fvd.frechet(x, x)) < 1e-6                       # identical sets -> 0
    y = x + 3.0
    assert math.isclose(fvd.frechet(x, y), 9.0 * 8, rel_tol=1e-6)  # pure mean shift
    mx, cx = fvd.moments(x)
    my, cy = fvd.moments(y)
    assert math.isclose(fvd.frechet_from_moments(mx, cx, my, cy), fvd.frechet(x, y), rel_tol=1e-9)
    assert fvd.fvd_or_none(x[:10], y[:10]) is None             # < FVD_MIN_CLIPS windows
    v = np.zeros((40, 4, 4, 3), np.uint8)
    assert len(fvd.resample_fps(v, 16.0)) == 16                # max(round(40*5/16)=12, 16)
    assert len(fvd.resample_fps(v, 5.0)) == 40                 # never upsampled / touched
    tiles = fvd.split_mosaic(np.zeros((2, 432, 768, 3), np.uint8))
    assert len(tiles) == 4 and tiles[0].shape == (2, 216, 384, 3)
    assert fvd.ep_key("a/makovian/episode_close_toolbox__000002/gt_all_views.mp4") == ("close_toolbox", 2)


def test_fvd_window_stream_matches_batch():
    """I3DWindowStream (DR pass) and i3d_features (pool pass) cut identical windows."""
    import torch
    from metrics import fvd

    calls = []

    def fake_i3d(w, rescale, resize, return_features):
        calls.append(w.shape)
        return w.mean(dim=(2, 3, 4))                           # (1, 3)

    vid = _video(T=40, H=32, W=48)
    batch = fvd.i3d_features(vid, fake_i3d, "cpu")
    st = fvd.I3DWindowStream(fake_i3d)
    t = torch.from_numpy(vid).permute(0, 3, 1, 2).float() / 255
    for s in range(0, len(t), 16):
        st.push(t[s:s + 16])
    assert len(batch) == len(st.feats) == 2                    # trailing 8 frames dropped
    np.testing.assert_allclose(np.asarray(batch), np.asarray(st.feats), rtol=1e-6)
    assert calls[0] == (1, 3, 16, fvd.I3D_SIZE, fvd.I3D_SIZE)


def test_fvd_shared_pool_dedup_and_floor():
    from metrics.fvd import SharedRealPool, budgeted, ref_identity
    from metrics.fvd.pool import mean_over_common_cameras

    a = "/r/sampling_dataset/dense/bimanual/output/x/gt_all_views.mp4"
    b = "/r/sampling_dataset/pisa/bimanual/output/x/gt_all_views.mp4"
    assert ref_identity(a) == ref_identity(b) == "bimanual/output/x/gt_all_views.mp4"
    rng = np.random.default_rng(0)
    pool = SharedRealPool(min_windows=8)
    key = ("bimanual", "cam0", "makovian")
    assert pool.add_real(key, ref_identity(a), rng.normal(size=(40, 4)))
    assert not pool.add_real(key, ref_identity(b), rng.normal(size=(40, 4)))   # same video
    for i in range(6):
        pool.add_real(key, f"ep{i}", rng.normal(size=(40, 4)))
    pool.add_pred(("cell", "dense", "cam0"), rng.normal(size=(80, 4)) + 2.0)
    v = pool.fvd(("cell", "dense", "cam0"), key)
    assert v is not None and v > 10                                       # ~16 = 4 * 2^2
    assert pool.floor(key) is not None and pool.floor(key) < v
    x = np.arange(100)[:, None]
    np.testing.assert_array_equal(budgeted(x, 10, "s"), budgeted(x, 10, "s"))   # deterministic
    agg = mean_over_common_cameras({("e", "A", "c0"): 1.0, ("e", "A", "c1"): 3.0,
                                    ("e", "B", "c0"): 5.0})
    assert agg == {("e", "A"): 1.0, ("e", "B"): 5.0}                      # only c0 is common


# --------------------------------------------------------------------------- #
def test_task_clause_and_text_conditioning():
    from metrics.text_alignment import task_clause, text_conditioned

    p = ("A 2x2 grid video of a robot performing the task: put the cup on the coaster. "
         "The video shows four camera views.")
    assert task_clause(p) == "put the cup on the coaster"
    assert task_clause("pick up the apple.") == "pick up the apple"
    assert task_clause(None) == ""
    assert text_conditioned("dreamgen") and text_conditioned("gt_ctrlworld")
    assert not text_conditioned("dreamdojo") and not text_conditioned("gt_dreamdojo")


def test_visual_integrity_is_p10_x100():
    from metrics.visual_integrity import visual_integrity

    s = np.linspace(0.2, 0.8, 11)                       # p10 = 0.26
    assert math.isclose(visual_integrity(s), 26.0, rel_tol=1e-9)
    assert visual_integrity([1.3, 1.4]) == 100.0       # clamped to [0, 1] before x100


def test_reference_free_protocol_selects_frames(tmp_path):
    """paper protocol: drop cond frame, 5 fps, 256x192, every 5th frame, JPEG round trip."""
    import cv2

    from metrics.base import ReferenceFreeFrameMetric
    from metrics.video_io import nr_analysis_frames

    p = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (640, 192))
    for i in range(41):
        w.write(np.full((192, 640, 3), i * 5, np.uint8))
    w.release()
    frames, info = nr_analysis_frames(p, layout="strip", view_index=1, n_views=2)
    assert info["source_interval"] == 2 and info["sampled_source_frames"][:3] == [1, 3, 5]
    assert frames.shape == (20, 192, 256, 3)

    class _Count(ReferenceFreeFrameMetric):
        name = "count"
        higher_is_better = {"n": True}

        def score_frames(self, pred, ref, text):
            ims = self.pil_frames(pred)
            return {"n": float(len(ims)), "w": float(ims[0].size[0])}

    r = _Count(device="cpu")(Sample("x", p, meta={"layout": "strip", "view_index": 1, "n_views": 2}))
    assert r["n"] == 4 and r["w"] == 256 and r["n_frames"] == 20          # 20 frames, stride 5
    raw = _Count(device="cpu", protocol="raw")(Sample("x", p))
    assert raw["n"] == 41 and raw["w"] == 640


def test_mas_summary_gates():
    from metrics.motion_amplitude import camera_mount, summarize

    assert camera_mount("humanoid", "cam_head_rgb") == "head"
    assert camera_mount("bimanual", "cam_left_wrist") == "wrist"
    assert camera_mount("single_arm", "exterior_1_left") == "static"
    pf = [dict(pair=i, fg_flow_p95=2.0 * i, fg_flow_med=0.1, frame_flow_med=1.0,
               bg_cov_pct=50.0, fg_flow_ego_p95=1.0, fg_flow_ego_med=0.5) for i in range(4)]
    f, motion, n_active, ok = summarize({"per_frame": pf, "n_arm_dets": 1, "seed_frame": 0},
                                        "static", 2, 4, 10.0)
    assert ok and f["actor_motion_raw"] == 1.5 and n_active == 3     # p95 / interval = 0,1,2,3
    assert f["actor_motion_active_rate"] == 0.75 and f["actor_motion_px_per_second"] == 15.0
    f, *_ , ok = summarize({"per_frame": pf, "n_arm_dets": 0}, "static", 2, 4, 10.0)
    assert not ok and f["actor_motion_raw"] is None                       # no robot -> N/A


# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (REAL_DIR / "episode_000000_gt.mp4").is_file(), reason="no sampling_dataset")
def test_parity_with_inference_scripts():
    """PSNR/LPIPS = the piq call in scripts/infer/*.py; SSIM = compute_dr_metrics.py."""
    import piq
    import torch
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim_f

    pred = load_frames(REAL_DIR / "episode_000000.mp4")[:48]
    gt = load_frames(REAL_DIR / "episode_000000_gt.mp4")[:40]
    n = min(len(pred), len(gt))                                # their min_len truncation
    x = torch.clamp(torch.from_numpy(pred[:n].copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    y = torch.clamp(torch.from_numpy(gt[:n].copy()) / 255.0, 0, 1).permute(0, 3, 1, 2)
    want = {"psnr": piq.psnr(x, y).mean().item(),
            "ssim": ssim_f(x, y, data_range=1.0).item(),
            "lpips": piq.LPIPS()(x, y).mean().item()}
    got = build_metric("fidelity", device="cpu", chunk=16)(Sample("ep0", pred, gt))
    for k, v in want.items():
        assert math.isclose(got[k], v, rel_tol=1e-4, abs_tol=1e-5), (k, got[k], v)
