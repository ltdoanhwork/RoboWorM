#!/usr/bin/env bash
#
# Upload / update sampling_dataset to the Hugging Face Hub.
#
# Usage:
#   ./scripts/upload_sampling_to_hf.sh                 # upload default folder/repo
#   ./scripts/upload_sampling_to_hf.sh --no-bars       # quieter output
#   HF_TOKEN=hf_xxx ./scripts/upload_sampling_to_hf.sh # override token
#
# It uses `hf upload-large-folder`, which:
#   - uploads in parallel and is robust for large folders (tens of GB / thousands of files)
#   - resumes automatically: re-running picks up where it left off (already-committed
#     files are skipped, only new/changed files are uploaded)
#   - stores resume state under sampling_dataset/.cache/huggingface (do not delete while running)
#
# So to "update" the dataset later, just run this script again.

set -euo pipefail

# ---- Config (override via env vars) ----
REPO_ID="${HF_REPO_ID:-doanh25032004/video_gen_physics}"
REPO_TYPE="${HF_REPO_TYPE:-dataset}"
# NEVER hardcode a token here: this file is tracked by git, so a literal token
# leaks to anyone with repo access and has to be rotated. Supply it via the
# environment instead (or `hf auth login`, which caches it outside the repo):
#   HF_TOKEN=hf_xxx ./scripts/upload_sampling_to_hf.sh
HF_TOKEN="${HF_TOKEN:-}"

# ---- Xet cache location (IMPORTANT, do not move back to $HOME) ----
# `hf upload-large-folder` uploads through the Xet backend, whose MerkleDB shards
# rely on memory-mapped I/O + file locking. Both are unreliable on parallel /
# network filesystems (GPFS, NFS, Lustre). $HOME here is on GPFS, and leaving the
# cache there produced a constant stream of:
#   Failed to commit: Data processing error: MerkleDB Shard error: File I/O error
# Measured impact: 95 failures vs 21 successes (82% of commits failing). Each
# failure halves the commit batch (50 -> 20), cutting throughput to ~87 files/min
# (~19h for the remaining files).
# Fix: keep the cache on a LOCAL filesystem. /dev/shm is tmpfs, so mmap + locking
# work correctly. This is ONLY a cache -- losing it costs local re-hashing (CPU),
# never a re-upload, because the server already holds the uploaded chunks.
HF_XET_CACHE="${HF_XET_CACHE:-/dev/shm/hf_xet_cache}"
mkdir -p "${HF_XET_CACHE}"
export HF_XET_CACHE

# Resolve project root (parent of this script's directory) so it works from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_FOLDER="${HF_LOCAL_FOLDER:-${PROJECT_ROOT}/sampling_dataset}"

export HF_TOKEN

if [[ ! -d "${LOCAL_FOLDER}" ]]; then
  echo "ERROR: local folder not found: ${LOCAL_FOLDER}" >&2
  exit 1
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' CLI not found. Install with: pip install -U huggingface_hub" >&2
  exit 1
fi

if [[ -z "${HF_TOKEN}" ]] && [[ ! -f "${HOME}/.cache/huggingface/token" ]]; then
  echo "ERROR: no HF credentials. Either export HF_TOKEN=hf_xxx or run 'hf auth login'." >&2
  exit 1
fi

echo "==> Uploading to https://huggingface.co/${REPO_TYPE}s/${REPO_ID}"
echo "    local folder : ${LOCAL_FOLDER}"
echo "    repo type    : ${REPO_TYPE}"
echo "    xet cache    : ${HF_XET_CACHE} ($(df -T "${HF_XET_CACHE}" 2>/dev/null | awk 'NR==2{print $2}'))"
echo "    (re-running this script resumes / updates incrementally)"
echo

hf upload-large-folder "${REPO_ID}" "${LOCAL_FOLDER}" \
  --repo-type="${REPO_TYPE}" \
  "$@"

echo
echo "==> Done. Repo: https://huggingface.co/${REPO_TYPE}s/${REPO_ID}"
