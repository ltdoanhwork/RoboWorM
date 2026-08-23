#!/usr/bin/env bash
# Collect timing + quality metrics from all dreamdojo bimanual singleview tuning logs.
# Usage: bash scripts/_tune_dreamdojo_bimanual_sv_report.sh [scratch_dir]
set -uo pipefail

SCRATCH="${1:-/tmp/dd_tune_bimanual}"
LOGS="${SCRATCH}/logs"

dense_t=$(grep -oP "Generated 1 episodes in \K[0-9.]+" "${LOGS}/dense.log" 2>/dev/null | head -1)

printf "%-24s %9s %8s %8s %8s %8s\n" TAG TIME_s SPEEDUP PSNR SSIM LPIPS
printf "%-24s %9s %8s %8s %8s %8s\n" ------------------------ --------- -------- -------- -------- --------

for f in "${LOGS}"/*.log; do
  tag="$(basename "${f%.log}")"
  case "$tag" in prof|shape) continue ;; esac

  t=$(grep -oP "Generated 1 episodes in \K[0-9.]+" "$f" | head -1)
  p=$(grep -oP "^PSNR: \K[0-9.]+" "$f" | head -1)
  s=$(grep -oP "^SSIM: \K[0-9.]+" "$f" | head -1)
  l=$(grep -oP "^LPIPS: \K[0-9.]+" "$f" | head -1)

  if [[ -z "$t" ]]; then
    err=$(grep -oE "[A-Za-z]*(Error|Exception)[^\"]*" "$f" | tail -1 | cut -c1-40)
    printf "%-24s %9s %8s %8s %8s %8s\n" "$tag" FAIL "${err:-crash}" - - -
    continue
  fi

  sp="-"
  if [[ -n "$dense_t" ]]; then
    sp=$(awk -v d="$dense_t" -v x="$t" 'BEGIN{printf "%.2fx", d/x}')
  fi
  printf "%-24s %9s %8s %8s %8s %8s\n" "$tag" "$t" "$sp" "${p:--}" "${s:--}" "${l:--}"
done
