#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${1:-/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/VLA-Humanoid/output/dream_gen_benchmark}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required but not installed." >&2
  exit 1
fi

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Root directory does not exist: $ROOT_DIR" >&2
  exit 1
fi

processed=0
skipped=0

while IFS= read -r -d '' video_path; do
  output_dir="${video_path%.mp4}"

  if [[ -d "$output_dir" ]] && find "$output_dir" -maxdepth 1 -type f | grep -q .; then
    skipped=$((skipped + 1))
    printf 'skip: %s\n' "$video_path"
    continue
  fi

  mkdir -p "$output_dir"

  ffmpeg \
    -hide_banner \
    -loglevel error \
    -nostdin \
    -i "$video_path" \
    -q:v 2 \
    -start_number 0 \
    "$output_dir/frame_%06d.jpg"

  processed=$((processed + 1))
  printf 'done: %s\n' "$video_path"
done < <(find "$ROOT_DIR" -type f -name '*.mp4' -print0 | sort -z)

printf 'processed=%d skipped=%d\n' "$processed" "$skipped"
