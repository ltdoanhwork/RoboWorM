#!/usr/bin/env bash
# =============================================================================
# _dispatch_timing_cells.sh — hand every UNFINISHED timing cell to a free
# Determined shell, and keep watching for shells that come up later.
#
# Written because the VLA_doanh shells get terminated/re-requested in bulk: a
# one-shot dispatch loses whatever had not finished, and hand-relaunching 9 cells
# after every restart is both slow and easy to get wrong (stale harness copies,
# forgotten SCRATCH isolation, clobbered tmux names).
#
# What it guarantees:
#   * a cell is dispatched only if its results.jsonl has < 10 records;
#   * one cell per shell, one SCRATCH per cell -> no results.jsonl contention;
#   * the shell's GPU must be idle (< 2 GB) or it is skipped, so a co-tenant job
#     never turns into contended timing;
#   * jobs run in detached tmux (bare nohup does NOT survive `det shell open`
#     closing -- see the determined-shell-job-submit skill);
#   * the harness is copied to .bench_dist.sh ONCE per dispatch round, so a cell
#     never starts on a half-edited script.
#
# Usage:
#   bash scripts/tools/tune/_dispatch_timing_cells.sh            # one round, then exit
#   WATCH=1 bash scripts/tools/tune/_dispatch_timing_cells.sh    # keep polling every 60s
# =============================================================================
set -uo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$ROOT"

DIST="${ROOT}/logs/exp_logs/timing_pass/dist"
MAX_CHUNKS="${MAX_CHUNKS:-3}"
WATCH="${WATCH:-0}"
POLL="${POLL:-60}"

# Cells the local GPU is NOT already covering. Local runs --all, so only hand out
# what it has not reached yet; a cell running in two places at once would have two
# writers on one scratch dir.
CELLS="${CELLS:-dreamgen_humanoid_sv dreamgen_bimanual_sv dreamgen_single_arm_sv
                 dreamgen_humanoid_mv dreamgen_bimanual_mv dreamgen_single_arm_mv
                 dreamdojo_humanoid_sv dreamdojo_single_arm_sv dreamdojo_bimanual_sv}"

mkdir -p "$DIST"

# All 10 methods, split into groups so a cell can be spread over several shells.
# A cell's 10 methods run SERIALLY inside one shell (they must -- two on one GPU
# would measure contention), so with more shells than cells the only way to go
# faster is to give each shell a SUBSET of the methods. Each subset writes to its
# own SCRATCH, and `collect` merges them, because two processes appending to one
# results.jsonl would interleave partial lines.
ALL_METHODS="dense sito itm worldcache dicache fastercache pisa radial svg1 svg2"

done_count() {  # done_count <cell> -> records across every shard of that cell
  local n=0 f
  for f in "${DIST}/$1"/results.jsonl "${DIST}/$1"__*/results.jsonl; do
    [[ -f "$f" ]] && n=$(( n + $(wc -l < "$f" | tr -d ' ') ))
  done
  echo "$n"
}

methods_done() {  # methods_done <cell> -> names already recorded
  local f
  for f in "${DIST}/$1"/results.jsonl "${DIST}/$1"__*/results.jsonl; do
    [[ -f "$f" ]] && python3 -c "
import json,sys
for l in open(sys.argv[1]):
    try: print(json.loads(l)['method'])
    except Exception: pass" "$f"
  done | sort -u
}

methods_missing() {  # methods_missing <cell>
  local have
  have="$(methods_done "$1")"
  for m in ${ALL_METHODS}; do
    grep -qx "$m" <<<"$have" || echo "$m"
  done
}

running_cells() {
  # Cells that currently have a bench_* tmux somewhere. Queried once per round so
  # the dispatch decision cannot race with jobs it started moments earlier.
  local ids
  ids="$(det shell list 2>/dev/null \
         | awk -F'|' '$3 ~ /VLA_doanh/ && $4 ~ /RUNNING/ {gsub(/ /,"",$1); print $1}')"
  for id in ${ids}; do
    timeout 45 det shell open "$id" -- 'tmux ls 2>/dev/null | cut -d: -f1' 2>/dev/null \
      | tr -d '\r' | sed -n 's/^bench_//p'
  done | sed 's/__shard[0-9]*$//' | sort -u
}


free_shells() {
  # RUNNING VLA_doanh shells whose GPU is idle and which have no bench_* tmux.
  local ids
  ids="$(det shell list 2>/dev/null \
         | awk -F'|' '$3 ~ /VLA_doanh/ && $4 ~ /RUNNING/ {gsub(/ /,"",$1); print $1}')"
  for id in ${ids}; do
    local info
    info="$(timeout 60 det shell open "$id" -- \
            'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; tmux ls 2>/dev/null | grep -c bench_ || true' \
            2>/dev/null | grep -vE 'Waiting|To reconnect|^$')"
    # `det shell open` interleaves banner lines with the remote output, and the
    # remote lines can carry trailing CR/space, so anchored integer matching finds
    # nothing and a later numeric test aborts the script with "invalid arithmetic
    # operator". Strip CR, then take the first two bare integers.
    local nums mem busy
    nums="$(tr -d '\r' <<<"$info" | grep -oE '[0-9]+' | head -2)"
    mem="$(sed -n 1p <<<"$nums")"
    busy="$(sed -n 2p <<<"$nums")"
    [[ -z "$mem" ]] && continue
    if (( mem < 2000 && ${busy:-0} == 0 )); then
      echo "$id"
    fi
  done
}

dispatch_round() {
  # Skip cells that already have a live job: without this, a cell whose methods
  # are still running looks "pending" on the next poll and gets dispatched AGAIN.
  # Two shells then share one SCRATCH and both append to the same results.jsonl --
  # observed: dreamgen_bimanual_sv ended up with 19 records for 10 methods, every
  # method measured twice. A cell is busy if any shell has a bench_<cell>* tmux.
  local busy_cells
  busy_cells="$(running_cells)"
  [[ -n "$busy_cells" ]] && echo "[dispatch] already running: $(echo $busy_cells | tr '\n' ' ')"

  local -a work_cell=() work_meth=()
  local pend_any=0
  for c in ${CELLS}; do
    grep -qx "$c" <<<"$busy_cells" && continue
    local miss
    miss="$(methods_missing "$c" | tr '\n' ' ')"
    [[ -z "${miss// }" ]] && continue
    pend_any=1
    work_cell+=("$c"); work_meth+=("$miss")
  done
  if [[ $pend_any -eq 0 ]]; then
    if [[ -n "$busy_cells" ]]; then
      echo "[dispatch] nothing to hand out (remaining cells are in flight)"
      return 0
    fi
    echo "[dispatch] nothing pending -- every cell has all 10 methods"
    return 1
  fi

  local shells
  shells="$(free_shells)"
  if [[ -z "$shells" ]]; then
    echo "[dispatch] no free VLA_doanh shell right now"
    return 0
  fi
  local -a slist=($shells)
  echo "[dispatch] ${#work_cell[@]} cell(s) pending, ${#slist[@]} shell(s) free"

  # Snapshot the harness once per round so no cell starts mid-edit.
  cp "${ROOT}/scripts/tools/tune/_bench_uniform_timing.sh" "${ROOT}/scripts/tools/tune/.bench_dist.sh"

  # Round-robin shells over cells; if there are more shells than cells, a cell
  # gets a second shard and its remaining methods are split between them.
  local si=0 ci=0 round=0
  while [[ $si -lt ${#slist[@]} ]]; do
    [[ ${#work_cell[@]} -eq 0 ]] && break
    local cell="${work_cell[$ci]}"
    local -a miss=(${work_meth[$ci]})
    [[ ${#miss[@]} -eq 0 ]] && { ci=$(( (ci+1) % ${#work_cell[@]} )); continue; }

    # how many shells will share this cell in this round
    local shards_left=$(( (${#slist[@]} - si + ${#work_cell[@]} - 1 - ci) / ${#work_cell[@]} ))
    [[ $shards_left -lt 1 ]] && shards_left=1
    local take=$(( (${#miss[@]} + shards_left - 1) / shards_left ))
    [[ $take -lt 1 ]] && take=1

    local subset="${miss[*]:0:$take}"
    work_meth[$ci]="${miss[*]:$take}"

    local id="${slist[$si]}"; si=$((si+1))
    # Unique tag per (cell, shell) so two shards NEVER share a SCRATCH. Using the
    # bare cell name for the first shard is what let a re-dispatch append to an
    # existing results.jsonl and double every method.
    local tag="${cell}__${id:0:8}"
    local scratch="${DIST}/${tag}"

    echo "[dispatch] ${tag} -> ${id:0:8}  methods: ${subset}"
    rm -rf "${scratch}"
    timeout 90 det shell open "$id" -- "
tmux kill-session -t bench_${tag} 2>/dev/null
tmux new-session -d -s bench_${tag} '
  cd ${ROOT} &&
  SCRATCH=${scratch} MAX_CHUNKS=${MAX_CHUNKS} METHODS=\"${subset}\" \
    bash scripts/tools/tune/.bench_dist.sh --cell ${cell} --gpu 0 \
    > ${DIST}/${tag}.log 2>&1
  echo EXIT=\$? >> ${DIST}/${tag}.log
'" >/dev/null 2>&1

    ci=$(( (ci+1) % ${#work_cell[@]} ))
    [[ $ci -eq 0 ]] && round=$((round+1))
  done
  return 0
}

if [[ "$WATCH" == "1" ]]; then
  while :; do
    dispatch_round || break
    sleep "$POLL"
  done
else
  dispatch_round
fi

echo
echo "[status]"
for c in ${CELLS}; do
  printf '  %-26s %s/10\n' "$c" "$(done_count "$c")"
done
