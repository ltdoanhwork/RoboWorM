"""
Generate Ctrl-World meta-info for the prepared bimanual_multiview dataset.

Produces:
  dataset_meta_info/{name}/train_sample.json
  dataset_meta_info/{name}/val_sample.json
      list of {"episode_id": <id>, "frame_ids": [start]} windows
  dataset_meta_info/{name}/stat.json
      {"state_01": [...14...], "state_99": [...14...]} percentiles over qpos

Run from models/Ctrl-World:
  python dataset_meta_info/create_meta_info_bimanual.py \
    --dataset-root dataset_example/bimanual_multiview \
    --dataset-name bimanual_multiview
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm


def load_ann(data_root, split, ann_name):
    """Return list of (episode_id, video_length, qpos_array) for a split."""
    ann_dir = os.path.join(data_root, "annotation", split)
    out = []
    if not os.path.isdir(ann_dir):
        return out
    files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
    for fn in files:
        try:
            with open(os.path.join(ann_dir, fn)) as f:
                ann = json.load(f)
        except Exception:
            print(f"skip {fn}")
            continue
        out.append(ann)
    return out


def build_samples(anns, sequence_length, start_interval):
    """Sliding-window start frames per episode (mirrors create_meta_info.py)."""
    samples = []
    for ann in anns:
        n_frames = ann["video_length"]
        traj_len = int(sequence_length)
        end_idx = n_frames - int(traj_len * 0.5)
        if end_idx < 1:
            end_idx = 1
        for start_frame in range(0, end_idx, start_interval):
            samples.append({"episode_id": ann["episode_id"], "frame_ids": [start_frame]})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="dataset_example/bimanual_multiview")
    parser.add_argument("--dataset-name", type=str, default="bimanual_multiview")
    parser.add_argument("--annotation-name", type=str, default="annotation")
    parser.add_argument("--sequence-length", type=int, default=11,
                        help="num_history(6)+num_frames(5) window size.")
    parser.add_argument("--start-interval", type=int, default=1)
    parser.add_argument("--meta-out", type=str, default="dataset_meta_info")
    args = parser.parse_args()

    out_dir = os.path.join(args.meta_out, args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- percentiles over ALL qpos (train + val) ----
    all_anns = {}
    qpos_all = []
    for split in ["train", "val"]:
        anns = load_ann(args.dataset_root, split, args.annotation_name)
        all_anns[split] = anns
        for ann in anns:
            qpos_all.append(np.array(ann["observation.state.qpos"], dtype=np.float64))
    if qpos_all:
        qpos_cat = np.concatenate(qpos_all, axis=0)  # (sum_frames, 14)
        state_01 = np.percentile(qpos_cat, 1, axis=0).tolist()
        state_99 = np.percentile(qpos_cat, 99, axis=0).tolist()
    else:
        raise RuntimeError(f"No annotations found under {args.dataset_root}")

    with open(os.path.join(out_dir, "stat.json"), "w") as f:
        json.dump({"state_01": state_01, "state_99": state_99}, f)
    print(f"stat.json: dim={len(state_01)}")

    # ---- window samples per split ----
    import random
    for split in ["train", "val"]:
        anns = all_anns[split]
        samples = build_samples(anns, args.sequence_length, args.start_interval)
        random.shuffle(samples)
        with open(os.path.join(out_dir, f"{split}_sample.json"), "w") as f:
            json.dump(samples, f, indent=2)
        print(f"{split}: {len(anns)} episodes -> {len(samples)} windows")

    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
