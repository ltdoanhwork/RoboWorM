import argparse
import csv
import json
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    metadata_path = args.metadata_path
    save_path = args.save_path
    output_path = args.output_path

    # Read metadata
    metadata = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['file_name']] = row['text']

    input_files = glob.glob(os.path.join(dataset_path, "*.mp4")) + \
                  glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_path, "*.png"))
    
    output_json = []
    for input_file in input_files:
        file_name = os.path.basename(input_file)
        if file_name not in metadata:
            print(f"Warning: {file_name} not found in metadata.")
            continue
            
        prompt = metadata[file_name]
        
        output_video_name = file_name
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
