"""Third-party backends borrowed from the vendored WorldScore checkout.

``metrics/WorldScore/`` (Duan et al., ICCV 2025) is kept in place as a dependency:
it ships the model code the reference-free metrics need, under
``worldscore/benchmark/metrics/third_party/``:

  ============== ===================================== =============================
  backend        used by                                checkpoint (checkpoints/metrics/worldscore/)
  ============== ===================================== =============================
  SEA-RAFT       motion_amplitude (optical flow)        Tartan-C-T-TSKH-spring540x960-M.pth
  VFIMamba       motion_smoothness (interpolation)      VFIMamba.pkl
  GroundingDINO  motion_amplitude (subject boxes)       groundingdino_swint_ogc.pth
  SAM 2.1        motion_amplitude (subject masks)       sam2.1_hiera_base_plus.pt
  ============== ===================================== =============================

CLIP-IQA+ (``pyiqa``) and CLIPScore (``torchmetrics``) are pip packages of the
dreamgen venv and do not need this module.

Nothing here imports the ``worldscore`` package itself (its ``__init__`` pulls
pyiqa and the whole evaluator); the third-party sub-repos are put on
``sys.path`` directly, exactly like ``scripts/metrics/run_dense_nr.sh`` does.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORLDSCORE_ROOT = REPO_ROOT / "metrics" / "WorldScore"
THIRD_PARTY = WORLDSCORE_ROOT / "worldscore" / "benchmark" / "metrics" / "third_party"
CKPT_DIR = Path(os.environ.get("WORLDSCORE_CKPT_DIR", REPO_ROOT / "checkpoints" / "metrics" / "worldscore"))

SEARAFT_CFG = THIRD_PARTY / "SEA-RAFT" / "config" / "eval" / "spring-M.json"
SEARAFT_CKPT = CKPT_DIR / "Tartan-C-T-TSKH-spring540x960-M.pth"
GDINO_CFG = THIRD_PARTY / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = CKPT_DIR / "groundingdino_swint_ogc.pth"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"
SAM2_CKPT = CKPT_DIR / "sam2.1_hiera_base_plus.pt"
VFIMAMBA_CKPT = CKPT_DIR / "VFIMamba.pkl"

_PATHS = [THIRD_PARTY, THIRD_PARTY / "SEA-RAFT" / "core", THIRD_PARTY / "SEA-RAFT",
          THIRD_PARTY / "VFIMamba", THIRD_PARTY / "groundingdino", THIRD_PARTY / "sam2"]


def add_paths() -> None:
    """Put the WorldScore third-party sub-repos on ``sys.path`` (idempotent)."""
    if not THIRD_PARTY.is_dir():
        raise FileNotFoundError(f"WorldScore third_party not found: {THIRD_PARTY}")
    for p in reversed(_PATHS):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


@contextlib.contextmanager
def worldscore_cwd():
    """VFIMamba loads ``worldscore/benchmark/metrics/checkpoints/<name>.pkl`` relative
    to the cwd, so model construction must run from ``metrics/WorldScore``."""
    old = os.getcwd()
    os.chdir(WORLDSCORE_ROOT)
    try:
        yield
    finally:
        os.chdir(old)


class SeaRaftFlow:
    """SEA-RAFT dense optical flow on two image files (WorldScore ``OpticalFlowMetric``).

    ``compute(f1, f2)`` -> (H, W, 2) flow in px. ``scale`` is set from the config
    (``spring-M`` = 1); the dense-NR wave overrides ``flow.a.scale = 0``.
    """

    def __init__(self, cfg: str | os.PathLike | None = None,
                 ckpt: str | os.PathLike | None = None, device: str = "cuda") -> None:
        add_paths()
        from core.parser import parse_args
        from core.raft import RAFT
        from core.utils.utils import load_ckpt

        cfg, ckpt = str(cfg or SEARAFT_CFG), str(ckpt or SEARAFT_CKPT)
        args = parse_args(argparse.Namespace(cfg=cfg, path=ckpt))
        m = RAFT(args)
        load_ckpt(m, ckpt)
        m.to(device).eval()
        self.m, self.a, self.dev = m, args, device

    def _img(self, f):
        import cv2
        import torch

        im = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        return torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None].to(self.dev)

    def compute(self, f1, f2):
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            i1, i2 = self._img(f1), self._img(f2)
            s = self.a.scale
            i1 = F.interpolate(i1, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            i2 = F.interpolate(i2, scale_factor=2 ** s, mode="bilinear", align_corners=False)
            with torch.amp.autocast(device_type="cuda"):
                out = self.m(i1, i2, iters=self.a.iters, test_mode=True)
            fl = out["flow"][-1]
            fl = F.interpolate(fl, scale_factor=0.5 ** s, mode="bilinear", align_corners=False) * (0.5 ** s)
            return fl.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

    def median_mag(self, f1, f2) -> float:
        import numpy as np

        fl = self.compute(f1, f2)
        return float(np.median(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2)))


def write_frames(frames, dest: Path, stride: int = 1) -> list[str]:
    """uint8 RGB (T,H,W,3) -> ``dest/00000.jpg ...`` (cv2 JPEG, quality 95).

    The WorldScore backends read frames from disk, so every in-process metric that
    uses them goes through this exact JPEG round trip (as the dense-NR wave did).
    """
    import cv2

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, fr in enumerate(frames):
        if i % stride:
            continue
        p = dest / f"{len(paths):05d}.jpg"
        cv2.imwrite(str(p), fr[:, :, ::-1])  # RGB -> BGR
        paths.append(str(p))
    return paths
