import argparse
import csv
import json
import os
import glob
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_type", type=str, default="csv", choices=["csv", "lerobot"], help="Type of dataset metadata format")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for video files")
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    metadata_path = args.metadata_path
    save_path = args.save_path
    output_path = args.output_path
    dataset_type = args.dataset_type
    recursive = args.recursive

    # Read metadata if csv
    metadata_csv = {}
    if dataset_type == "csv":
        if not metadata_path:
            raise ValueError("--metadata_path is required for csv dataset_type")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metadata_csv[row['file_name']] = row['text']

    if recursive or dataset_type == "lerobot":
        input_files = glob.glob(os.path.join(dataset_path, "**", "*.mp4"), recursive=True) + \
                      glob.glob(os.path.join(dataset_path, "**", "*.jpg"), recursive=True) + \
                      glob.glob(os.path.join(dataset_path, "**", "*.png"), recursive=True)
    else:
        input_files = glob.glob(os.path.join(dataset_path, "*.mp4")) + \
                      glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                      glob.glob(os.path.join(dataset_path, "*.png"))
    
    # Cache for lerobot metadata
    lerobot_metadata_cache = {}

    output_json = []
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        
        prompt = ""
        if dataset_type == "csv":
            if file_name not in metadata_csv:
                print(f"Warning: {file_name} not found in metadata.")
                continue
            prompt = metadata_csv[file_name]
            output_video_name = file_name
        elif dataset_type == "lerobot":
            # LeRobot format: dataset_dir/videos/.../episode_000000.mp4
            # Metadata: dataset_dir/meta/episodes.jsonl
            if '/videos/' not in input_file.replace('\\', '/'):
                continue
            
            base_dir = input_file.replace('\\', '/').split('/videos/')[0]
            jsonl_path = os.path.join(base_dir, 'meta', 'episodes.jsonl')
            
            if jsonl_path not in lerobot_metadata_cache:
                if not os.path.exists(jsonl_path):
                    print(f"Warning: Metadata file not found: {jsonl_path}")
                    lerobot_metadata_cache[jsonl_path] = {}
                else:
                    meta_dict = {}
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            obj = json.loads(line)
                            idx = obj.get("episode_index")
                            if idx is not None:
                                tasks = obj.get("tasks", [])
                                desc = obj.get("description", "")
                                text = tasks[0] if tasks else desc
                                meta_dict[idx] = text
                    lerobot_metadata_cache[jsonl_path] = meta_dict
            
            # Extract episode index from filename (e.g., episode_000000.mp4)
            match = re.search(r'episode_(\d+)', file_name)
            if not match:
                print(f"Warning: Could not parse episode index from {file_name}")
                continue
                
            episode_idx = int(match.group(1))
            if episode_idx not in lerobot_metadata_cache[jsonl_path]:
                print(f"Warning: Episode {episode_idx} not found in {jsonl_path}")
                continue
                
            prompt = lerobot_metadata_cache[jsonl_path][episode_idx]
            
            # create a unique output name
            # use parent folder name + file name to avoid collisions
            parent_folder = os.path.basename(base_dir)
            output_video_name = f"{parent_folder}_{file_name}"
            
        if not output_video_name.endswith('.mp4'):
            output_video_name = os.path.splitext(output_video_name)[0] + '.mp4'
            
        output_json.append(
            {
                "input_video": input_file,
                "prompt": prompt,
                "output_video": os.path.join(save_path, output_video_name),
            }
        )
        
    print(f"Saved {len(output_json)} items to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)

if __name__ == "__main__":
    main()
