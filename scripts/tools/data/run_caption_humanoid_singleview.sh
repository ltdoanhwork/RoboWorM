#!/usr/bin/env bash
# Caption makovian + non_makovian training videos with Qwen3-VL-30B.
# Passes existing task-text hints + video frames to the LLM to produce
# natural-language prompts, then writes datasets/humanoid_dreamgen/ flat
# structure ready for DreamGen training.
#
# Usage:
#   bash scripts/tools/data/run_caption_humanoid_singleview.sh
#
# Options (override via env vars):
#   SPLITS="makovian non_makovian"   which splits to process (default: both)
#   CUDA_VISIBLE_DEVICES=0,1         GPUs to use
#   TENSOR_PARALLEL_SIZE=2           tensor parallelism for Qwen3-VL
#   NUM_FRAMES=8                     frames sampled per video
#   BATCH_SIZE=8                     videos per LLM batch
#   OVERWRITE=0                      set to 1 to re-caption existing files
#   DRY_RUN=0                        set to 1 to preview without running LLM
#   PYTHON=python                    python with vllm>=0.8, transformers>=4.57
#
# Requires (separate from DreamDojo venv):
#   pip install vllm>=0.8 transformers>=4.57 imageio imageio-ffmpeg pillow

set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$ROOT"

PYTHON=${PYTHON:-python}
SPLITS=${SPLITS:-"makovian non_makovian"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
NUM_FRAMES=${NUM_FRAMES:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
OVERWRITE=${OVERWRITE:-0}
DRY_RUN=${DRY_RUN:-0}

export CUDA_VISIBLE_DEVICES

EXTRA_FLAGS=""
[[ "$OVERWRITE" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --overwrite"
[[ "$DRY_RUN"   == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --dry_run"

echo "======================================================="
echo "Caption humanoid singleview → humanoid_dreamgen/"
echo "  splits:  $SPLITS"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES (TP=$TENSOR_PARALLEL_SIZE)"
echo "  frames:  $NUM_FRAMES  batch: $BATCH_SIZE"
echo "  root:    $ROOT"
echo "======================================================="

$PYTHON scripts/tools/data/caption_humanoid_singleview.py \
    --splits $SPLITS \
    --num_video_frames "$NUM_FRAMES" \
    --batch_size "$BATCH_SIZE" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    $EXTRA_FLAGS

echo ""
echo "=== captioning done ==="
echo ""
echo "Next: compute T5 embeddings for DreamGen training"
echo "  DATASETS=humanoid bash finetuning/dreamgen/scripts/compute_t5_embeddings.sh"
