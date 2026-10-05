#!/usr/bin/env bash
# Tune WorldCache on DreamGen humanoid MULTIVIEW over N distinct episodes, and
# stage the resulting videos + timing for the website's qualitative panel.
#
# WHY THIS EXISTS
# ---------------
# The shipped WorldCache point for this cell (thresh 0.20 / ret 0.2 / probe 2 ->
# "2.08x") was tuned on 2 episodes of ONE task. The qualitative exam needs
#   (a) more than one task, so a reader cannot dismiss the result as cherry-picked
#       (see EPISODES below -- one episode per task, never N of the same task),
#   (b) a speedup at or ABOVE the shipped optimum, and
#   (c) timing measured under the same protocol for dense and every config.
#
# Protocol, fixed for every run here:
#   * the SAME episodes (one per task, chosen by dense reference quality) for
#     dense and for every candidate config -- per-episode comparison only means
#     something when the episode is identical;
#   * strictly sequential on ONE idle GPU;
#   * generation time read from the pipeline's `Total inference time` log line
#     (CUDA-synchronised), NOT wall clock, which would include model load,
#     ffmpeg and metric computation and would dilute every speedup;
#   * segment 1 recorded separately from the rest, because the first segment pays
#     one-off warm-up and would flatter dense;
#   * all output to scratch; nothing is written into the canonical tree.
#
# USAGE
#   bash scripts/_tune_worldcache_humanoid_mv.sh --estimate
#   bash scripts/_tune_worldcache_humanoid_mv.sh --dense           # reference
#   bash scripts/_tune_worldcache_humanoid_mv.sh --sweep           # candidates
#   bash scripts/_tune_worldcache_humanoid_mv.sh --config t0.20_r0.2_p2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRATCH="${SCRATCH:-/tmp/wc_humanoid_mv}"
GPU="${GPU:-0}"
RESULTS="${SCRATCH}/results.jsonl"

# The episodes: one per task, so N episodes span N DIFFERENT scenes. Distinct
# tasks matter more than distinct ids -- 2 episodes of one task would show the
# same scene twice. Each is the highest-dense-PSNR 2-segment episode of its task,
# and the two tasks kept here are the ones with the strongest dense reference
# (boxs_storage 16.87 dB, mango_pomegranate 16.67 dB): a qualitative panel is
# only readable if the reference itself is clean.
#   split|task|episode_id
EPISODES=(
  "non_makovian|AIRBOT_MMK2_boxs_storage|46"
  "non_makovian|Airbot_MMK2_Airbot_MMK2_storage_mango_pomegranate|66"
)

# Candidate configs: name|thresh|ret_ratio|probe_depth
# The shipped point is t0.20/r0.2/p2. The sweep walks the threshold up (more
# skipping -> faster) and probes the two knobs that the tuning note identified as
# the real levers, so we can land at or above 2.08x while checking what quality
# does at each step.
CONFIGS=(
  "t0.20_r0.2_p2|0.20|0.2|2"   # shipped optimum -- the point to match or beat
  "t0.25_r0.2_p2|0.25|0.2|2"
  "t0.30_r0.2_p2|0.30|0.2|2"
  "t0.25_r0.1_p2|0.25|0.1|2"
  "t0.30_r0.1_p2|0.30|0.1|2"
  "t0.25_r0.2_p4|0.25|0.2|4"
)

ids_csv() {  # task-scoped, so an id cannot leak into another task
  local out=(); for e in "${EPISODES[@]}"; do
    IFS='|' read -r _sp _tk _id <<<"$e"; out+=("${_tk}:${_id}")
  done
  (IFS=,; echo "${out[*]}")
}

# ------------------------------------------------------------------ estimate --
if [[ "${1:-}" == "--estimate" ]]; then
  python3 - "${#EPISODES[@]}" "${#CONFIGS[@]}" <<'PY'
import sys
neps, ncfg = int(sys.argv[1]), int(sys.argv[2])
DENSE_SEG, SEGS, LOAD = 116.0, 2, 90.0
dense = neps * (DENSE_SEG * SEGS) + LOAD
# A faster config costs proportionally less GPU time; assume ~2x mean speedup
# for the candidates, which is conservative (slower configs would cost more).
cand = ncfg * (neps * (DENSE_SEG * SEGS / 2.0) + LOAD)
print("episodes: %d   configs: %d   segments/episode: %d" % (neps, ncfg, SEGS))
print("dense reference : %5.1f min" % (dense / 60))
print("candidate sweep : %5.1f min  (%d runs)" % (cand / 60, ncfg))
print("TOTAL           : %5.1f min  (~%.1f h)" % ((dense + cand) / 60, (dense + cand) / 3600))
PY
  echo ""
  echo "episodes:"; for e in "${EPISODES[@]}"; do echo "  $e"; done
  echo "episode ids: $(ids_csv)"
  exit 0
fi

# --------------------------------------------------------------------- guard --
util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | head -1)"
mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | head -1)"
if [[ "${util}" -ge 15 || "${mem}" -ge 8000 ]]; then
  echo "ERROR: GPU ${GPU} busy (util=${util}%, mem=${mem}MiB). Timing taken now" >&2
  echo "would measure contention, not method speed." >&2
  [[ "${FORCE_BUSY_GPU:-0}" == "1" ]] || exit 1
fi

mkdir -p "${SCRATCH}"
: > "${RESULTS}.part"
[[ -f "$RESULTS" ]] || : > "$RESULTS"

# Record one run: parse the per-segment generation times out of the log and
# attach the per-episode quality the run itself computed.
record() {  # record <tag> <thresh> <ret> <probe> <log> <out> <wall>
  python3 - "$@" >> "${RESULTS}.part" <<'PY'
import json, os, re, sys
tag, thresh, ret, probe, log, out, wall = sys.argv[1:8]
txt = open(log, errors="ignore").read() if os.path.exists(log) else ""
# Per-segment generation time, CUDA-synchronised inside the pipeline. Wall clock
# is deliberately not used: it includes model load, ffmpeg and metrics.
segs = [float(x) for x in re.findall(r"Total inference time:\s*([0-9.]+)", txt)]
eps = []
for dirpath, _, files in os.walk(out):
    if "metrics.json" in files:
        try:
            m = json.load(open(os.path.join(dirpath, "metrics.json")))
        except Exception:
            continue
        eps.append({
            "episode": os.path.basename(dirpath),
            "task": os.path.basename(os.path.dirname(dirpath)),
            "psnr": m.get("psnr"), "ssim": m.get("ssim"),
            "lpips": m.get("lpips"), "num_segments": m.get("num_segments"),
        })
rec = {
    "tag": tag, "thresh": None if thresh == "-" else float(thresh),
    "ret_ratio": None if ret == "-" else float(ret),
    "probe_depth": None if probe == "-" else int(probe),
    "wall_s": int(wall), "n_segments_timed": len(segs),
    "segment_times_s": segs,
    "generation_s": round(sum(segs), 3) if segs else None,
    "mean_seg_s": round(sum(segs) / len(segs), 3) if segs else None,
    # Segment 0 pays one-off warm-up; the steady-state mean excludes it so a
    # cache method is not credited (or penalised) for start-up cost.
    "first_seg_s": segs[0] if segs else None,
    "steady_seg_s": (round(sum(segs[1:]) / len(segs[1:]), 3)
                     if len(segs) > 1 else None),
    "episodes": sorted(eps, key=lambda e: (e["task"], e["episode"])),
    "n_episodes": len(eps),
}
for k in ("psnr", "ssim", "lpips"):
    v = [e[k] for e in eps if isinstance(e.get(k), (int, float))]
    rec["mean_" + k] = round(sum(v) / len(v), 4) if v else None
print(json.dumps(rec))
PY
  cat "${RESULTS}.part" >> "$RESULTS"; : > "${RESULTS}.part"
}

run_one() {  # run_one <tag> <script> <thresh> <ret> <probe>
  local tag="$1" script="$2" thresh="$3" ret="$4" probe="$5"
  if grep -q "\"tag\": \"${tag}\"" "$RESULTS" 2>/dev/null; then
    echo "  [have] ${tag}"; return 0
  fi
  local out="${SCRATCH}/${tag}/out" log="${SCRATCH}/${tag}/run.log"
  mkdir -p "$out" "$(dirname "$log")"
  echo "  [run ] ${tag}"
  local t0 t1; t0=$(date +%s)
  set +e
  if [[ "$tag" == "dense" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" EPISODE_IDS="$(ids_csv)" SAVE_ROOT="$out" \
      bash "$script" > "$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$GPU" EPISODE_IDS="$(ids_csv)" SAVE_ROOT="$out" \
      WORLDCACHE_REL_L1_THRESH="$thresh" WORLDCACHE_RET_RATIO="$ret" \
      WORLDCACHE_PROBE_DEPTH="$probe" \
      bash "$script" > "$log" 2>&1
  fi
  local rc=$?
  set -e
  t1=$(date +%s)
  [[ $rc -eq 0 ]] || echo "    (exit ${rc} -- see ${log})"
  record "$tag" "$thresh" "$ret" "$probe" "$log" "$out" "$((t1 - t0))"

  # Assert the selector actually bound. A count above the expected number means
  # the
  # episode filter was ignored and this run is NOT comparable to the others.
  local n; n=$(find "$out" -name metrics.json 2>/dev/null | wc -l)
  if [[ "$n" -ne "${#EPISODES[@]}" ]]; then
    echo "    WARNING: ${tag} produced ${n} episodes, expected ${#EPISODES[@]}" >&2
    echo "    (episode selection did not bind -- run is not comparable)" >&2
  fi
}

DENSE_SCRIPT="scripts/run_humanoid_multiview_dreamgen_full.sh"
WC_SCRIPT="scripts/run_humanoid_multiview_dreamgen_worldcache.sh"

# Fail fast if a run script does not forward EPISODE_IDS. Without this the script
# silently falls back to "all episodes of every task" (248 here) and the run looks
# fine for hours while producing the wrong episode set -- which is exactly what
# happened before this check existed.
for s in "$DENSE_SCRIPT" "$WC_SCRIPT"; do
  if ! grep -q 'episode-ids' "$ROOT/$s"; then
    echo "ERROR: $s does not forward EPISODE_IDS to --episode-ids." >&2
    echo "It would run every episode of every task instead of the selected ones." >&2
    exit 1
  fi
done

case "${1:-}" in
  --dense)  run_one dense "$DENSE_SCRIPT" - - - ;;
  --config)
    for c in "${CONFIGS[@]}"; do
      IFS='|' read -r tag th rt pb <<<"$c"
      [[ "$tag" == "${2:-}" ]] && run_one "$tag" "$WC_SCRIPT" "$th" "$rt" "$pb"
    done ;;
  --sweep)
    for c in "${CONFIGS[@]}"; do
      IFS='|' read -r tag th rt pb <<<"$c"
      run_one "$tag" "$WC_SCRIPT" "$th" "$rt" "$pb"
    done ;;
  --all)
    run_one dense "$DENSE_SCRIPT" - - -
    for c in "${CONFIGS[@]}"; do
      IFS='|' read -r tag th rt pb <<<"$c"
      run_one "$tag" "$WC_SCRIPT" "$th" "$rt" "$pb"
    done ;;
  *) echo "usage: $0 --estimate|--dense|--sweep|--all|--config <tag>" >&2; exit 2 ;;
esac

echo ""
echo "results: ${RESULTS}"
