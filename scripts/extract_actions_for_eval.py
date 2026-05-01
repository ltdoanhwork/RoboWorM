"""Extract action JSONs for the action-conditioned video2world model.

Auto-detects the dataset's action layout from `meta/modality.json` and writes
JSONs in the schema expected by `video2world_action.get_action_sequence`:

    {
      "action": (T, 6) -- per-frame 6D delta, pre-divided by 20.0
                          (inference re-multiplies by 20)
      "continuous_gripper_state": (T+1,) -- gripper open/close ratio in [0, 1]
    }

Two source formats are supported:

  (a) GR00T 1027-D layout (e.g. In-lab_Eval) -- has `delta_right_wrist` block
      of shape (T, 144) = 24 future-frames * 6D. We take the first 6D as the
      next-frame delta. This matches training distribution.

  (b) GR00T 52-D layout (e.g. EgoDex_Eval, DreamDojo-HV_Eval) -- only has
      absolute joint angles. We compute joint-space delta via `np.diff` on
      right_arm[:, :6]. This is best-effort: the model was trained on
      EE-space deltas, so generation quality will be lower than (a).
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

INFERENCE_SCALE = 20.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to a leaf LeRobot dataset (containing meta/, data/, videos/)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="Where to save extracted action JSONs")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to the batch_input.json")
    parser.add_argument("--output_video_dir", type=str,
                        default="./output/action_gen",
                        help="Directory used in the batch_input.json's output_video field")
    parser.add_argument("--video_subdir", type=str,
                        default="observation.images.ego_view_freq20",
                        help="Video subdirectory under videos/chunk-*/")
    return parser.parse_args()


def load_modality(dataset_path):
    path = os.path.join(dataset_path, "meta", "modality.json")
    with open(path) as f:
        return json.load(f)


def extract_action_ee(raw_actions, modality):
    """Return (T, 6) array of per-frame deltas, pre-divided by INFERENCE_SCALE."""
    action_def = modality["action"]

    if "delta_right_wrist" in action_def:
        s = action_def["delta_right_wrist"]["start"]
        # First 6D of the 144-D delta block = next-frame EE wrist delta.
        delta = raw_actions[:, s:s + 6]
        return delta / INFERENCE_SCALE, "delta_right_wrist"

    if "right_arm" in action_def:
        s = action_def["right_arm"]["start"]
        # Take first 6 joints of right_arm as proxy; compute frame-to-frame delta.
        arm = raw_actions[:, s:s + 6]
        delta = np.diff(arm, axis=0, prepend=arm[:1])  # (T, 6)
        return delta / INFERENCE_SCALE, "right_arm_diff (best-effort)"

    raise ValueError("No usable action channel found in modality.json")


def extract_gripper(raw_actions, modality):
    """Return (T,) gripper state in [0, 1]."""
    action_def = modality["action"]
    if "right_hand" not in action_def:
        # Fall back to zeros if no hand channel.
        return np.zeros(raw_actions.shape[0], dtype=np.float32)

    s = action_def["right_hand"]["start"]
    g = raw_actions[:, s].astype(np.float32)
    g_min, g_max = float(g.min()), float(g.max())
    if g_max - g_min > 1e-6:
        g = (g - g_min) / (g_max - g_min)
    else:
        g = np.zeros_like(g)
    return g


def main():
    args = parse_args()
    modality = load_modality(args.dataset_path)

    video_files = sorted(glob.glob(
        os.path.join(args.dataset_path, "videos", "**", "episode_*.mp4"),
        recursive=True,
    ))
    # Optionally filter by camera subdir.
    if args.video_subdir:
        video_files = [v for v in video_files if args.video_subdir in v]

    os.makedirs(args.save_path, exist_ok=True)
    batch_input = []

    print(f"[extract] dataset={args.dataset_path}")
    print(f"[extract] found {len(video_files)} videos")

    fmt_used = None
    for vid_path in tqdm(video_files):
        match = re.search(r"episode_(\d+)", os.path.basename(vid_path))
        if not match:
            continue
        ep_idx = int(match.group(1))

        parquets = glob.glob(
            os.path.join(args.dataset_path, "data", "**", f"episode_{ep_idx:06d}.parquet"),
            recursive=True,
        )
        if not parquets:
            print(f"[warn] parquet for episode {ep_idx} not found")
            continue

        df = pd.read_parquet(parquets[0])
        raw_actions = np.stack(df["action"].values)  # (T, D)

        action_ee, fmt = extract_action_ee(raw_actions, modality)
        gripper = extract_gripper(raw_actions, modality)
        fmt_used = fmt

        # video2world_action.get_action_sequence does gripper[1:], so we pad to T+1.
        gripper_extended = np.append(gripper, gripper[-1])

        anno = {
            "action": action_ee.tolist(),
            "continuous_gripper_state": gripper_extended.tolist(),
        }
        anno_path = os.path.join(args.save_path, f"episode_{ep_idx:06d}_action.json")
        with open(anno_path, "w") as f:
            json.dump(anno, f)

        batch_input.append({
            "input_video": vid_path,
            "input_annotation": anno_path,
            "output_video": os.path.join(args.output_video_dir, f"gen_{os.path.basename(vid_path)}"),
        })

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(batch_input, f, indent=4)

    print(f"[extract] action format: {fmt_used}")
    print(f"[extract] wrote {len(batch_input)} entries -> {args.output_json}")


if __name__ == "__main__":
    main()
