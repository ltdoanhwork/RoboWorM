import pandas as pd
import json
import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the LeRobot dataset")
    parser.add_argument("--save_path", type=str, required=True, help="Where to save extracted action JSONs")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the new batch_input.json")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Scan for videos
    video_files = glob.glob(os.path.join(args.dataset_path, "videos", "**", "episode_*.mp4"), recursive=True)
    
    os.makedirs(args.save_path, exist_ok=True)
    batch_input = []
    
    print(f"Found {len(video_files)} videos. Extracting actions...")
    
    for vid_path in tqdm(video_files):
        # Extract episode index
        match = re.search(r'episode_(\d+)', os.path.basename(vid_path))
        if not match: continue
        ep_idx = int(match.group(1))
        
        # Find corresponding parquet file
        # Format usually: data/chunk-000/episode_000000.parquet
        # We look for it relative to the dataset root
        dataset_root = args.dataset_path
        parquet_pattern = os.path.join(dataset_root, "data", "**", f"episode_{ep_idx:06d}.parquet")
        parquets = glob.glob(parquet_pattern, recursive=True)
        
        if not parquets:
            print(f"Warning: Parquet for episode {ep_idx} not found.")
            continue
            
        parquet_path = parquets[0]
        df = pd.read_parquet(parquet_path)
        
        # Extract actions (52D vector)
        # Based on modality.json: right_arm is [30:37], right_hand (gripper) is [37:43]
        # We take 7D as expected by standard action-conditioned models
        raw_actions = np.stack(df['action'].values)  # (T, 52)

        # Simple mapping: 6D from right_arm + 1D from right_hand
        action_ee = raw_actions[:, 30:36]  # (T, 6)
        gripper = raw_actions[:, 37]       # (T,) - 1D gripper state

        # video2world_action.py's get_action_sequence() does:
        #   gripper = np.array(data["continuous_gripper_state"])[1:, None]
        # It expects T+1 gripper states (treating gripper as "next state").
        # We append a copy of the last frame so that after [1:] we still have T frames.
        gripper_extended = np.append(gripper, gripper[-1])  # (T+1,)

        anno_data = {
            "action": action_ee.tolist(),
            "continuous_gripper_state": gripper_extended.tolist()  # T+1 values
        }
        
        anno_file_name = f"episode_{ep_idx:06d}_action.json"
        anno_path = os.path.join(args.save_path, anno_file_name)
        with open(anno_path, 'w') as f:
            json.dump(anno_data, f)
            
        batch_input.append({
            "input_video": vid_path,
            "input_annotation": anno_path,
            "output_video": os.path.join("./output/action_gen", f"gen_{os.path.basename(vid_path)}")
        })
        
    with open(args.output_json, 'w') as f:
        json.dump(batch_input, f, indent=4)
        
    print(f"Created batch input with {len(batch_input)} items at {args.output_json}")

if __name__ == "__main__":
    main()
