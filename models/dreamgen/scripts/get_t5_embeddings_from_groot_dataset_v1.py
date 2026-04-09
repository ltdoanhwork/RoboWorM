# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pickle
import glob

import numpy as np
from tqdm import tqdm

from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR

"""example command
python -m scripts.get_t5_embeddings_from_groot_dataset --dataset_path datasets/benchmark_train/gr1
"""


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for text prompts")
    parser.add_argument(
        "--dataset_path", type=str, default="datasets/benchmark_train/gr1", help="Root path to the dataset"
    )
    parser.add_argument(
        "--prompt_prefix", type=str, default="The robot arm is performing a task. ", help="Prefix of the prompt"
    )
    parser.add_argument("--max_length", type=int, help="Maximum length of the text embedding")
    parser.add_argument("--cache_dir", type=str, default=T5_MODEL_DIR, help="Directory to cache the T5 model")
    parser.add_argument(
        "--meta_csv", type=str, default=None, help="Metadata csv file (optional - will use metas/ folder if not provided)"
    )
    parser.add_argument(
        "--use_folder_structure", action="store_true", help="Use metas/ and videos/ folder structure instead of CSV"
    )
    return parser.parse_args()


def get_data_from_folders(dataset_path):
    """Extract video-text pairs from metas/ and videos/ folder structure"""
    metas_dir = os.path.join(dataset_path, "metas")
    videos_dir = os.path.join(dataset_path, "videos")
    
    if not os.path.exists(metas_dir):
        raise FileNotFoundError(f"Metas directory not found: {metas_dir}")
    if not os.path.exists(videos_dir):
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    data_pairs = []
    
    # Get all .txt files from metas directory
    txt_files = glob.glob(os.path.join(metas_dir, "*.txt"))
    
    print(f"Found {len(txt_files)} text files in metas directory")
    
    for txt_file in txt_files:
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        video_file = f"{base_name}.mp4"
        video_path = os.path.join(videos_dir, video_file)
        
        # Check if corresponding video exists
        if os.path.exists(video_path):
            # Read text content
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
                
                data_pairs.append((video_file, text_content))
            except Exception as e:
                print(f"Warning: Could not read {txt_file}: {e}")
        else:
            print(f"Warning: Video file not found for {txt_file}: {video_path}")
    
    print(f"Successfully matched {len(data_pairs)} video-text pairs")
    return data_pairs


def get_data_from_csv(meta_csv):
    """Extract video-text pairs from CSV file"""
    try:
        with open(meta_csv, 'r', encoding='utf-8') as f:
            meta_lines = f.readlines()[1:]  # Skip header
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata CSV file not found: {meta_csv}")
    
    data_pairs = []
    for meta_line in meta_lines:
        if ',' in meta_line:
            video_filename, prompt = meta_line.split(",", 1)
            prompt = prompt.strip("\n")
            if prompt.startswith('"') and prompt.endswith('"'):
                # Remove the quotes for robocasa dataset
                prompt = prompt[1:-1]
            data_pairs.append((video_filename, prompt))
    
    return data_pairs


def main(args) -> None:
    # Determine data source
    if args.meta_csv and os.path.exists(args.meta_csv):
        print(f"Using CSV file: {args.meta_csv}")
        data_pairs = get_data_from_csv(args.meta_csv)
    else:
        print("Using folder structure (metas/ and videos/ directories)")
        data_pairs = get_data_from_folders(args.dataset_path)
    
    if not data_pairs:
        raise ValueError("No valid video-text pairs found!")
    
    # Create output directories
    t5_xxl_dir = os.path.join(args.dataset_path, "t5_xxl")
    os.makedirs(t5_xxl_dir, exist_ok=True)
    meta_txt_dir = os.path.join(args.dataset_path, "metas")
    os.makedirs(meta_txt_dir, exist_ok=True)

    # Initialize T5
    print("Initializing T5 encoder...")
    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=args.cache_dir)
    encoder = CosmosT5TextEncoder(config=encoder_config)
    
    print(f"Processing {len(data_pairs)} video-text pairs...")

    for video_filename, prompt in tqdm(data_pairs, desc="Processing embeddings"):
        # Add prompt prefix
        full_prompt = args.prompt_prefix + prompt
        
        # Create meta txt file (ensure it exists)
        meta_txt_filename = os.path.join(meta_txt_dir, os.path.basename(video_filename).replace(".mp4", ".txt"))
        with open(meta_txt_filename, "w", encoding='utf-8') as fp:
            fp.write(full_prompt)

        # Create T5 embedding file
        t5_xxl_filename = os.path.join(t5_xxl_dir, os.path.basename(video_filename).replace(".mp4", ".pickle"))
        if os.path.exists(t5_xxl_filename):
            print(f"Skipping {t5_xxl_filename} because it already exists")
            continue

        print(f"Encoding prompt: {full_prompt}")

        try:
            # Compute T5 embeddings
            encoded_text, mask_bool = encoder.encode_prompts(full_prompt, max_length=args.max_length, return_mask=True)
            attn_mask = mask_bool.long()
            lengths = attn_mask.sum(dim=1).cpu()

            encoded_text = encoded_text.cpu().numpy().astype(np.float16)

            # Trim zeros to save space
            encoded_text = [encoded_text[batch_id][: lengths[batch_id]] for batch_id in range(encoded_text.shape[0])]

            # Save T5 embeddings as pickle file
            with open(t5_xxl_filename, "wb") as fp:
                pickle.dump(encoded_text, fp)  # list of np.ndarray in (len, 1024)
        
        except Exception as e:
            print(f"Error processing {video_filename}: {e}")
            continue

    print("T5 embeddings extraction completed!")


if __name__ == "__main__":
    args = parse_args()
    main(args)