from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from finetuning.config import FinetuneConfig, add_config_arguments, config_from_args


class RoboCtrlWorldDataset(Dataset):
    def __init__(self, cfg: FinetuneConfig, mode: str = "train"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.processed_root = Path(cfg.processed_root)
        sample_path = self.processed_root / "meta" / f"{mode}_sample.json"
        stat_path = self.processed_root / "meta" / "stat.json"
        if not sample_path.exists():
            raise FileNotFoundError(f"Missing sample file: {sample_path}. Run finetuning/data/prepare_latents.py first.")
        if not stat_path.exists():
            raise FileNotFoundError(f"Missing stat file: {stat_path}. Run finetuning/data/prepare_latents.py first.")
        self.samples: list[dict[str, Any]] = json.loads(sample_path.read_text(encoding="utf-8"))
        self.stats = json.loads(stat_path.read_text(encoding="utf-8"))
        self.action_min = np.asarray(self.stats["action_01"], dtype=np.float32)[None, :]
        self.action_max = np.asarray(self.stats["action_99"], dtype=np.float32)[None, :]

    def __len__(self) -> int:
        return len(self.samples)

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def annotation_path(self, sample: dict[str, Any]) -> Path:
        return (
            self.processed_root
            / "datasets"
            / sample["dataset_name"]
            / "annotation"
            / sample["split"]
            / f"{sample['episode_id']}.json"
        )

    def load_latent_window(self, dataset_name: str, latent_rel_path: str, start: int, end: int) -> torch.Tensor:
        latent_path = self.processed_root / "datasets" / dataset_name / latent_rel_path
        latent = torch.load(latent_path, map_location="cpu")
        window = latent[start:end]
        if window.shape[0] != self.cfg.sequence_length:
            raise ValueError(f"{latent_path}: expected {self.cfg.sequence_length} frames, got {window.shape[0]}")
        return window.float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        ann = json.loads(self.annotation_path(sample).read_text(encoding="utf-8"))
        start = int(sample["start_idx"])
        end = start + self.cfg.sequence_length

        view_latents = []
        view_mask = []
        latent_entries = ann["latent_videos"][: self.cfg.n_views]
        if len(latent_entries) < self.cfg.n_views:
            raise ValueError(f"{self.annotation_path(sample)} has {len(latent_entries)} views, expected {self.cfg.n_views}")
        for entry in latent_entries:
            rel = entry["latent_video_path"]
            latent = self.load_latent_window(sample["dataset_name"], rel, start, end)
            view_latents.append(latent)
            view_mask.append(float(latent.abs().sum().item() > 0))

        latent_stacked = torch.cat(view_latents, dim=2)  # [T, C, V*H, W]
        expected = (self.cfg.sequence_length, 4, self.cfg.latent_height, self.cfg.latent_slot_width)
        if tuple(latent_stacked.shape) != expected:
            raise ValueError(f"latent shape {tuple(latent_stacked.shape)} != {expected}")

        actions = np.asarray(ann["actions"][start:end], dtype=np.float32)
        if actions.shape != (self.cfg.sequence_length, self.cfg.action_dim):
            raise ValueError(f"action shape {actions.shape} != {(self.cfg.sequence_length, self.cfg.action_dim)}")
        actions = self.normalize_bound(actions, self.action_min, self.action_max)

        return {
            "latent": latent_stacked,
            "action": torch.from_numpy(actions).float(),
            "text": ann["texts"][0] if ann.get("texts") else "",
            "embodiment_id": torch.tensor(int(ann.get("embodiment_id", sample.get("embodiment_id", 0))), dtype=torch.long),
            "view_mask": torch.tensor(view_mask, dtype=torch.float32),
            "dataset_name": sample["dataset_name"],
            "episode_id": sample["episode_id"],
            "start_idx": torch.tensor(start, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the processed Robo Ctrl-World dataset.")
    add_config_arguments(parser)
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cfg = config_from_args(args)
    ds = RoboCtrlWorldDataset(cfg, mode=args.mode)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print(f"samples={len(ds)}")
    print(f"latent={tuple(batch['latent'].shape)}")
    print(f"action={tuple(batch['action'].shape)}")
    print(f"embodiment_id={batch['embodiment_id'].tolist()}")
    print(f"text={batch['text'][0]}")


if __name__ == "__main__":
    main()

