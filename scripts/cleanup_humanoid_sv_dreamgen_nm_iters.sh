#!/usr/bin/env bash
# Move dense humanoid SV DreamGen non_makovian short-90k preds into the canonical
# iter_000090000_static16fps folder, then delete leftover iters.
#
# Why: the 130 short clips (69×185 + 61×277 frames) already exist in
# iter_000090000 (no tag). iter_000090000_static16fps was empty. iter_000070000
# is the old LONG rollout. Empty *_static16fps probe dirs are junk.
#
# Idempotent. Does NOT touch makovian/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NM="${ROOT}/sampling_dataset/dense/humanoid/output/singleview/dreamgen/non_makovian"
KEEP="${NM}/iter_000090000_static16fps"
SRC="${NM}/iter_000090000"

mkdir -p "${KEEP}"

n_keep="$(find "${KEEP}" -maxdepth 1 -name 'episode_*.mp4' | wc -l)"
n_src="$(find "${SRC}" -maxdepth 1 -name 'episode_*.mp4' 2>/dev/null | wc -l || echo 0)"
echo "keep=${KEEP} mp4=${n_keep}"
echo "src=${SRC} mp4=${n_src}"

if [[ "${n_keep}" -eq 130 ]]; then
  echo "Canonical folder already has 130 mp4 — skip move."
elif [[ "${n_src}" -eq 130 ]]; then
  echo "Moving ${n_src} mp4 + txt from iter_000090000 -> iter_000090000_static16fps"
  # -n: do not overwrite if a file already landed in KEEP
  mv -n "${SRC}"/episode_*.mp4 "${KEEP}/"
  shopt -s nullglob
  txts=("${SRC}"/episode_*.txt)
  if [[ ${#txts[@]} -gt 0 ]]; then
    mv -n "${txts[@]}" "${KEEP}/"
  fi
  leftover="$(find "${SRC}" -maxdepth 1 -type f | wc -l)"
  if [[ "${leftover}" -eq 0 ]]; then
    rmdir "${SRC}"
    echo "removed empty ${SRC}"
  else
    echo "WARN: ${SRC} still has ${leftover} files" >&2
  fi
else
  echo "ERROR: neither KEEP (got ${n_keep}) nor SRC (got ${n_src}) has 130 mp4." >&2
  echo "Do NOT delete 70k. Run WAVE humanoid fill instead." >&2
  exit 1
fi

n_keep="$(find "${KEEP}" -maxdepth 1 -name 'episode_*.mp4' | wc -l)"
if [[ "${n_keep}" -ne 130 ]]; then
  echo "ERROR: KEEP has ${n_keep} mp4, expected 130" >&2
  exit 1
fi

echo "Deleting leftover iters under non_makovian/ (not KEEP)"
for d in \
  "${NM}/iter_000070000" \
  "${NM}/iter_000090000" \
  "${NM}/base_2b" \
  "${NM}/iter_000010000_static16fps" \
  "${NM}/iter_000050000_static16fps" \
  "${NM}/iter_000060000_static16fps" \
  "${NM}/iter_000080000_static16fps" \
  "${NM}/iter_000100000_static16fps" \
  "${NM}/iter_000120000_static16fps"
do
  if [[ -e "${d}" ]]; then
    echo "  rm -rf $(basename "${d}")"
    rm -rf "${d}"
  fi
done

echo "Remaining:"
ls -la "${NM}"
echo "DONE keep_mp4=${n_keep}"
