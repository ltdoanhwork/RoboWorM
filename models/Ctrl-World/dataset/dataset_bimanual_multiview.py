"""
Ctrl-World training dataset for datasets/bimanual/multiview (ALOHA).

Adapted from dataset_droid_exp33.Dataset_mix. Key differences:
  - Action/state condition is the 14-D qpos (observation.state.qpos), not DROID's
    7-D cartesian+gripper.
  - State arrays in the annotation are already frame-aligned with the latent
    frames (prepared by prepare_ctrlworld_bimanual_multiview.py), so we index the
    state with the SAME frame ids as the latents (down_sample handled at prep time;
    here down_sample defaults to 1 -> state_id == rgb_id).
  - View stacking is config-driven. By default (no grid config on args) it keeps
    the original 3-view VERTICAL-STRIP layout (height 72 = 3*24, width 40). If
    `args.grid_rows`/`args.grid_cols` are set (e.g. 2x2 for the 4-view grid
    variant), views are placed ROW-MAJOR into a (grid_rows*24, grid_cols*40)
    latent canvas instead.
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

# per-view SVD-VAE latent tile size (192px/8 x 320px/8)
_VIEW_LAT_H = 24
_VIEW_LAT_W = 40


class BimanualMultiviewDataset(Dataset):
    def __init__(self, args, mode="train"):
        super().__init__()
        self.args = args
        self.mode = mode

        # View layout. Default = original 3-view vertical strip (3 rows x 1 col).
        # A grid variant sets grid_rows/grid_cols (+ num_views) on the config.
        self.num_views = int(getattr(args, "num_views", 3))
        self.grid_rows = int(getattr(args, "grid_rows", self.num_views))
        self.grid_cols = int(getattr(args, "grid_cols", 1))
        assert self.grid_rows * self.grid_cols >= self.num_views, (
            f"grid {self.grid_rows}x{self.grid_cols} too small for "
            f"{self.num_views} views"
        )
        self.canvas_h = self.grid_rows * _VIEW_LAT_H
        self.canvas_w = self.grid_cols * _VIEW_LAT_W

        self.dataset_path_all = []
        self.samples_all = []
        self.samples_len = []
        self.norm_all = []

        dataset_root_path = args.dataset_root_path
        dataset_names = args.dataset_names.split("+")
        dataset_meta_info_path = args.dataset_meta_info_path
        dataset_cfgs = args.dataset_cfgs.split("+")
        self.prob = args.prob

        for dataset_name, dataset_cfg in zip(dataset_names, dataset_cfgs):
            data_json_path = f"{dataset_meta_info_path}/{dataset_cfg}/{mode}_sample.json"
            with open(data_json_path, "r") as f:
                samples = json.load(f)
            dataset_path = [os.path.join(dataset_root_path, dataset_name) for _ in samples]
            print(f"[BimanualMultiviewDataset] {dataset_name}/{mode}: {len(samples)} samples")
            self.dataset_path_all.append(dataset_path)
            self.samples_all.append(samples)
            self.samples_len.append(len(samples))

            with open(f"{dataset_meta_info_path}/{dataset_name}/stat.json", "r") as f:
                data_stat = json.load(f)
                state_p01 = np.array(data_stat["state_01"])[None, :]
                state_p99 = np.array(data_stat["state_99"])[None, :]
                self.norm_all.append((state_p01, state_p99))

        self.max_id = max(self.samples_len)
        print("[BimanualMultiviewDataset] samples_len:", self.samples_len, "max_id:", self.max_id)

    def __len__(self):
        return self.max_id

    def _load_latent_video(self, video_path, frame_ids):
        with open(video_path, "rb") as file:
            video_tensor = torch.load(file)
            video_tensor.requires_grad = False
        max_frames = video_tensor.size()[0]
        frame_ids = [int(fid) if fid < max_frames else max_frames - 1 for fid in frame_ids]
        return video_tensor[frame_ids]

    def _get_frames(self, label, frame_ids, cam_id, video_dir):
        video_path = label["latent_videos"][cam_id]["latent_video_path"]
        video_path = os.path.join(video_dir, video_path)
        return self._load_latent_video(video_path, frame_ids)

    def normalize_bound(self, data, data_min, data_max, clip_min=-1, clip_max=1, eps=1e-8):
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def __getitem__(self, index):
        dataset_id = np.random.choice(len(self.samples_all), p=self.prob)
        samples = self.samples_all[dataset_id]
        dataset_path = self.dataset_path_all[dataset_id]
        state_p01, state_p99 = self.norm_all[dataset_id]
        index = index % len(samples)
        sample = samples[index]
        dataset_dir = dataset_path[index]

        frame_ids = sample["frame_ids"]
        ann_file = f'{dataset_dir}/{self.args.annotation_name}/{self.mode}/{sample["episode_id"]}.json'
        with open(ann_file, "r") as f:
            label = json.load(f)

        # State arrays are already frame-aligned with latent frames. frame_len is the
        # number of available latent/state frames.
        qpos = np.array(label["observation.state.qpos"])
        frame_len = qpos.shape[0] - 1
        if frame_len < 1:
            frame_len = 1

        skip = random.randint(1, 2)
        skip_his = int(skip * 4)
        p = random.random()
        if p < 0.15:
            skip_his = 0

        frame_now = frame_ids[0]
        rgb_id = []
        for i in range(self.args.num_history, 0, -1):
            rgb_id.append(int(frame_now - i * skip_his))
        rgb_id.append(frame_now)
        for i in range(1, self.args.num_frames):
            rgb_id.append(int(frame_now + i * skip))
        rgb_id = np.array(rgb_id)
        rgb_id = np.clip(rgb_id, 0, frame_len).tolist()
        rgb_id = [int(fid) for fid in rgb_id]
        # State is frame-aligned with latents (down_sample handled at prep time).
        state_id = np.array(rgb_id) * self.args.down_sample
        state_id = np.clip(state_id, 0, qpos.shape[0] - 1)

        data = dict()
        data["text"] = label["texts"][0]

        # Load each view's latent and place it ROW-MAJOR into the canvas.
        # 3-view default -> vertical strip (3x1); 4-view grid -> 2x2.
        latent = torch.zeros(
            (self.args.num_frames + self.args.num_history, 4, self.canvas_h, self.canvas_w),
            dtype=torch.float32,
        )
        for view_idx in range(self.num_views):
            view_latent = self._get_frames(label, rgb_id, view_idx, video_dir=dataset_dir)
            r = view_idx // self.grid_cols
            c = view_idx % self.grid_cols
            h0, w0 = r * _VIEW_LAT_H, c * _VIEW_LAT_W
            latent[:, :, h0:h0 + _VIEW_LAT_H, w0:w0 + _VIEW_LAT_W] = view_latent
        data["latent"] = latent.float()

        action = qpos[state_id]  # (num_history+num_frames, 14)
        action = self.normalize_bound(action, state_p01, state_p99)
        data["action"] = torch.tensor(action).float()

        return data


# Alias so train_wm.py can `from dataset.dataset_bimanual_multiview import Dataset_mix`
Dataset_mix = BimanualMultiviewDataset


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import wm_args
    args = wm_args()
    ds = BimanualMultiviewDataset(args, mode="val")
    print("len:", len(ds))
    sample = ds[0]
    print("latent:", sample["latent"].shape, "action:", sample["action"].shape, "text:", sample["text"])
