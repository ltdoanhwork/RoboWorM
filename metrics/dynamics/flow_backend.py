"""SEA-RAFT optical-flow backend for physics metrics.

Self-contained loader that reuses the vendored SEA-RAFT model code under
``metrics/WorldScore/.../third_party/SEA-RAFT`` but avoids the WorldScore
package import chain (which pulls in heavy, uninstalled deps like sam2 /
mamba_ssm). The pretrained weights are loaded straight from the HuggingFace
hub via ``PyTorchModelHubMixin`` (no manual checkpoint file needed).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SEA_RAFT_DIR = (
    _REPO_ROOT
    / "metrics/WorldScore/worldscore/benchmark/metrics/third_party/SEA-RAFT"
)
_DEFAULT_CFG = _SEA_RAFT_DIR / "config/eval/spring-M.json"
_DEFAULT_HF_REPO = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"


def _load_cfg(cfg_path: Path) -> argparse.Namespace:
    with open(cfg_path, "r") as f:
        data = json.load(f)
    args = argparse.Namespace()
    for k, v in data.items():
        setattr(args, k, v)
    return args


class SeaRaftFlow:
    """Thin wrapper around the SEA-RAFT model producing dense flow fields."""

    def __init__(
        self,
        device: str = "cuda",
        cfg_path: Path = _DEFAULT_CFG,
        hf_repo: str = _DEFAULT_HF_REPO,
        iters: int | None = None,
        scale: int | None = 0,
    ) -> None:
        self.device = device
        args = _load_cfg(cfg_path)
        # spring-M's cfg uses scale=-1 (built for 540x960); our robot views are
        # small (192x320) so the -1 downsample underflows the 4-level corr
        # pyramid. Default to scale=0 (full-res) for these inputs.
        if scale is not None:
            args.scale = scale
        if iters is not None:
            args.iters = iters
        self.args = args

        # Import the vendored SEA-RAFT model. It does `from core.raft import RAFT`,
        # so the SEA-RAFT dir must be on sys.path as the top-level package root.
        if str(_SEA_RAFT_DIR) not in sys.path:
            sys.path.insert(0, str(_SEA_RAFT_DIR))
        from core.raft import RAFT  # type: ignore

        model = RAFT.from_pretrained(hf_repo, args=args)
        model = model.to(device)
        model.eval()
        self.model = model

    @staticmethod
    def _to_tensor(frame: np.ndarray) -> torch.Tensor:
        """(H,W,3) uint8 RGB -> (1,3,H,W) float tensor in [0,255]."""
        t = torch.from_numpy(np.ascontiguousarray(frame)).float()
        return t.permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Return dense flow (H, W, 2) mapping frame1 -> frame2, in pixels."""
        img1 = self._to_tensor(frame1).to(self.device)
        img2 = self._to_tensor(frame2).to(self.device)

        scale = self.args.scale
        img1s = F.interpolate(img1, scale_factor=2 ** scale, mode="bilinear", align_corners=False)
        img2s = F.interpolate(img2, scale_factor=2 ** scale, mode="bilinear", align_corners=False)

        with torch.amp.autocast(device_type="cuda", enabled=(self.device != "cpu")):
            out = self.model(img1s, img2s, iters=self.args.iters, test_mode=True)
        flow = out["flow"][-1]
        flow = F.interpolate(flow, scale_factor=0.5 ** scale, mode="bilinear", align_corners=False) * (0.5 ** scale)
        flow = flow[0].permute(1, 2, 0).float().cpu().numpy()
        return flow
