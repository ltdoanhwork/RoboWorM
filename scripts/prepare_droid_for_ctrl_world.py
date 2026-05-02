"""
Convert LeRobot DROID dataset (v3.0) into the Ctrl-World annotation format.
LOCAL ONLY: Uses HF_HUB_OFFLINE=1 to prevent any downloads.
"""

import argparse
import json
import os
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot DROID to Ctrl-World format")
    parser.add_argument("--lerobot_root", type=str, default="./datasets/droid_20p")
    parser.add_argument("--output_root", type=str, default="./datasets/droid_20p_ctrl_world")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    args = parser.parse_args()

    # Block ALL HuggingFace network access - use local cache only
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = os.path.abspath(args.lerobot_root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading LeRobot DROID dataset from {args.lerobot_root} (LOCAL ONLY)...")
    episodes_list = None
    if args.max_episodes:
        episodes_list = list(range(args.max_episodes))

    try:
        ds = LeRobotDataset(
            "lerobot/droid_1.0.1",
            episodes=episodes_list,
            root=args.lerobot_root,
            download_videos=False,  # Don't download missing video chunks
        )
    except Exception as e:
        print(f"\n[!] Error loading local dataset: {e}")
        print("Ensure you have finished downloading the dataset first.")
        return

    # Prepare output dirs
    ann_dir = os.path.join(args.output_root, "annotation", args.split)
    vid_dir = os.path.join(args.output_root, "videos", args.split)
    meta_dir = os.path.join(args.output_root, "meta_info")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    ep_table = ds.meta.episodes
    num_episodes = len(ep_table)
    if args.max_episodes:
        num_episodes = min(num_episodes, args.max_episodes)

    print(f"Processing {num_episodes} episodes...")

    cam_keys = [
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
        "observation.images.wrist_left",
    ]

    all_states = []
    success_count = 0
    val_samples = []

    for ep_idx in tqdm(range(num_episodes), desc="Converting episodes"):
        try:
            # SKIP IF ALREADY CONVERTED
            ann_path = os.path.join(ann_dir, f"{ep_idx}.json")
            if os.path.exists(ann_path):
                # Briefly check if stats need the states from this episode
                # For simplicity, we assume if JSON exists, it's done.
                success_count += 1
                val_samples.append({"episode_id": ep_idx, "frame_ids": [0]})
                continue

            row = ep_table[ep_idx]
            # In LeRobot v0.4.4, the episode table uses 'dataset_from_index' and 'dataset_to_index'
            global_start = int(row["dataset_from_index"])
            global_end_exclusive = int(row["dataset_to_index"])
            ep_len = int(row["length"])
            global_end = global_end_exclusive - 1


            states_ep = []
            joints_ep = []
            cam_frames = {k: [] for k in cam_keys}

            for global_frame_idx in range(global_start, global_end + 1):
                frame = ds[global_frame_idx]

                # State extraction
                cart = frame.get("observation.state.cartesian_position")
                grip = frame.get("observation.state.gripper_position")
                cart_list = cart.numpy().flatten().tolist() if cart is not None else [0.0]*6
                grip_list = grip.numpy().flatten().tolist() if grip is not None else [0.0]
                states_ep.append((cart_list + grip_list + [0.0]*7)[:7])

                # Joints extraction
                jp = frame.get("observation.state.joint_position")
                jp_list = jp.numpy().flatten().tolist() if jp is not None else [0.0]*7
                joints_ep.append((jp_list + grip_list + [0.0]*8)[:8])

                # Images
                for k in cam_keys:
                    if k in frame:
                        img_tensor = frame[k]
                        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                        cam_frames[k].append(img_np)

            # Save videos
            ep_vid_dir = os.path.join(vid_dir, str(ep_idx))
            os.makedirs(ep_vid_dir, exist_ok=True)
            video_entries = []
            for cam_i, cam_key in enumerate(cam_keys):
                frames = cam_frames[cam_key]
                if not frames: continue
                v_path = f"videos/{args.split}/{ep_idx}/{cam_i}.mp4"
                import imageio
                imageio.mimsave(os.path.join(args.output_root, v_path), frames, fps=15, codec='libx264')
                video_entries.append({"video_path": v_path})

            # Instruction
            first_frame = ds[global_start]
            instruction = str(first_frame.get("language_instruction", "")).strip()

            annotation = {
                "texts": [instruction],
                "episode_id": ep_idx,
                "video_length": ep_len,
                "videos": video_entries,
                "states": states_ep,
                "joints": joints_ep,
            }
            with open(os.path.join(ann_dir, f"{ep_idx}.json"), "w") as f:
                json.dump(annotation, f)

            all_states.extend(states_ep)
            val_samples.append({"episode_id": ep_idx, "frame_ids": [0]})
            success_count += 1

        except Exception as e:
            print(f"Skipping episode {ep_idx}: {e}")
            continue

    if all_states:
        all_states_np = np.array(all_states)
        stat = {
            "state_01": np.percentile(all_states_np, 1, axis=0).tolist(),
            "state_99": np.percentile(all_states_np, 99, axis=0).tolist()
        }
        with open(os.path.join(meta_dir, "stat.json"), "w") as f:
            json.dump(stat, f)

    with open(os.path.join(meta_dir, f"{args.split}_sample.json"), "w") as f:
        json.dump(val_samples, f, indent=4)

    print(f"Done! Converted {success_count} episodes.")


if __name__ == "__main__":
    main()
