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
HF_TOKEN="${HF_TOKEN:-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}"

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

echo "==> Uploading to https://huggingface.co/${REPO_TYPE}s/${REPO_ID}"
echo "    local folder : ${LOCAL_FOLDER}"
echo "    repo type    : ${REPO_TYPE}"
echo "    (re-running this script resumes / updates incrementally)"
echo

hf upload-large-folder "${REPO_ID}" "${LOCAL_FOLDER}" \
  --repo-type="${REPO_TYPE}" \
  "$@"

echo
echo "==> Done. Repo: https://huggingface.co/${REPO_TYPE}s/${REPO_ID}"
