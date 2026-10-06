#!/usr/bin/env bash
# Launch Qwen3-VL-30B-A3B-Instruct-FP8 via vLLM on port 8000.
# Requires ~30 GB VRAM; tested on A100 80 GB.
# The model is downloaded from HuggingFace Hub on first run.

set -euo pipefail

MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
PORT=8000
GPU_UTIL=0.90
MAX_MODEL_LEN=8192

echo "[launch] Starting vLLM server for ${MODEL} on port ${PORT} ..."

vllm serve "${MODEL}" \
    --dtype auto \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --trust-remote-code \
    --port "${PORT}" \
    --host 0.0.0.0
