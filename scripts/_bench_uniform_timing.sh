#!/usr/bin/env bash
# Uniform timing pass: one matched episode per (cell, method), same protocol.
#
# WHY THIS EXISTS
# ---------------
# The speedup numbers currently in metrics/reports/speedup_summary_full.csv are
# not comparable across cells: 69 of 127 rows have no timing at all, and every
# row that does have timing inherited it from a tuning note, i.e. from a sweep
# that used its own episode count, its own split and its own scratch layout.
# Comparing a 2-episode Ctrl-World sweep against a 1-episode DreamDojo sweep
# measures the episodes, not the methods.
#
# This harness fixes the protocol instead:
#   * ONE episode per cell, the SAME episode for dense and for every method;
#   * strictly SEQUENTIAL on a single GPU (no co-tenancy, no context parallel);
#   * the first segment/chunk is reported separately, because it pays one-off
#     profiling and k-means warm-up for several methods and would otherwise
#     flatter dense;
#   * output to scratch only -- never into the canonical sampling_dataset tree
#     (sampling-dataset-layout rule).
#
# Timing is read from the authoritative log line of each family rather than from
# wall clock, which would include model load, ffmpeg and metric computation:
#   Ctrl-World -> per-episode  wall_time_s  in metrics.json
#   DreamGen   -> "Total inference time" / "LVG Total inference time"
#   DreamDojo  -> "[Timing] Generated"
#
# USAGE
#   bash scripts/_bench_uniform_timing.sh --estimate          # cost only, no GPU
#   bash scripts/_bench_uniform_timing.sh --cell <id> [--gpu N]
#   bash scripts/_bench_uniform_timing.sh --all --gpu 0
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/tmp/uniform_timing}"
GPU="${GPU:-0}"
MODE="estimate"
ONLY_CELL=""
METHODS_FILTER="${METHODS:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --estimate) MODE="estimate"; shift ;;
    --all)      MODE="run"; shift ;;
    --cell)     MODE="run"; ONLY_CELL="$2"; shift 2 ;;
    --gpu)      GPU="$2"; shift 2 ;;
    --scratch)  SCRATCH="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --- The benchmark cells, with their measured dense cost per unit of work. ---
# dense_s is the documented dense time for ONE episode of that cell (from the
# per-cell tuning-insight rules, which is where every current timing number in
# the repo comes from). It is used ONLY to estimate GPU cost up front; the pass
# itself re-measures everything.
#
# dense_script is named explicitly rather than derived, because the dense
# entrypoint differs per family and a wrong guess would silently time the wrong
# flow: Ctrl-World uses run_{emb}_{view}_ctrlworld.sh, DreamGen multiview uses
# _full.sh, DreamGen singleview uses _static16fps.sh, and DreamDojo uses
# _full_episode.sh (the entrypoint the sparse/cache scripts are required to
# mirror).
#   cell_id|embodiment|view|model|dense_s|unit|dense_script
CELLS=(
  "ctrlworld_single_arm_mv|single_arm|multiview|ctrlworld|59|episode|run_single_arm_multiview_ctrlworld.sh"
  "ctrlworld_bimanual_mv|bimanual|multiview|ctrlworld|59|episode|run_bimanual_multiview_ctrlworld.sh"
  "ctrlworld_humanoid_mv|humanoid|multiview|ctrlworld|59|episode|run_humanoid_multiview_ctrlworld.sh"
  "ctrlworld4_bimanual_mv|bimanual|multiview|ctrlworld_4view_grid|75|episode|run_bimanual_multiview_ctrlworld_4view_grid.sh"
  "ctrlworld4_humanoid_mv|humanoid|multiview|ctrlworld_4view_grid|75|episode|run_humanoid_multiview_ctrlworld_4view_grid.sh"
  "dreamgen_single_arm_mv|single_arm|multiview|dreamgen|492|episode|run_single_arm_multiview_dreamgen_batch.sh"
  "dreamgen_bimanual_mv|bimanual|multiview|dreamgen|237|2 segments|run_bimanual_multiview_dreamgen_full.sh"
  "dreamgen_humanoid_mv|humanoid|multiview|dreamgen|230|2 segments|run_humanoid_multiview_dreamgen_full.sh"
  "dreamgen_single_arm_sv|single_arm|singleview|dreamgen|226|2 chunks|run_single_arm_singleview_dreamgen_static16fps.sh"
  "dreamgen_bimanual_sv|bimanual|singleview|dreamgen|263|2 chunks|run_bimanual_singleview_dreamgen_16fps.sh"
  "dreamgen_humanoid_sv|humanoid|singleview|dreamgen|260|2 chunks|run_humanoid_singleview_dreamgen_static16fps.sh"
  "dreamdojo_single_arm_sv|single_arm|singleview|dreamdojo|145|episode|run_single_arm_singleview_dreamdojo_full_episode.sh"
  "dreamdojo_bimanual_sv|bimanual|singleview|dreamdojo|390|episode|run_bimanual_singleview_dreamdojo_full_episode.sh"
  "dreamdojo_humanoid_sv|humanoid|singleview|dreamdojo|56|episode|run_humanoid_singleview_dreamdojo_full_episode.sh"
)

# Per-method multiplier on the dense cost, from the documented speedups. Used
# only for the estimate: a method at 0.7x costs MORE wall time than dense.


# ---------------------------------------------------------------- estimate ---
if [[ "$MODE" == "estimate" ]]; then
  python3 - "$ROOT" "${CELLS[@]}" <<'PY'
import os, sys

root = sys.argv[1]
cells = []
for line in sys.argv[2:]:
    cell, emb, view, model, dense, unit, dscript = line.split("|")
    cells.append((cell, emb, view, model, float(dense), unit, dscript))

# Multiplier on dense generation cost, from the documented per-method speedups.
# A method slower than dense (SVG2 at ~0.7x) costs MORE wall time, so the
# estimate must not assume acceleration means less GPU time.
COST = {"dense": 1.00, "sito": 0.95, "itm": 0.95, "worldcache": 0.70,
        "dicache": 0.70, "fastercache": 0.95, "pisa": 0.80, "svg1": 1.05,
        "svg2": 1.35, "radial": 1.00}
LOAD = 60.0  # per-run model load + VAE decode + metric computation

def script_for(emb, view, model, method, dense_script):
    rel = ("scripts/" + dense_script if method == "dense"
           else "scripts/run_%s_%s_%s_%s.sh" % (emb, view, model, method))
    return rel if os.path.exists(os.path.join(root, rel)) else None

rows, missing, total, runs = [], [], 0.0, 0
for cell, emb, view, model, dense, unit, dscript in cells:
    cell_total, cell_runs = 0.0, 0
    for m, c in COST.items():
        rel = script_for(emb, view, model, m, dscript)
        if rel is None:
            missing.append((cell, m))
            continue
        cell_total += dense * c + LOAD
        cell_runs += 1
    rows.append((cell, unit, dense, cell_runs, cell_total))
    total += cell_total
    runs += cell_runs

print("%-25s %-11s %8s %6s %9s" % ("CELL", "UNIT", "DENSE_S", "RUNS", "EST_MIN"))
print("-" * 64)
for cell, unit, dense, n, t in rows:
    print("%-25s %-11s %8.0f %6d %9.1f" % (cell, unit, dense, n, t / 60.0))
print("-" * 64)
print("%-25s %-11s %8s %6d %9.1f" % ("TOTAL", "", "", runs, total / 60.0))
print("")
print("estimated GPU time: %.1f h for %d runs (%d of %d cell x method pairs "
      "have a script)" % (total / 3600.0, runs, runs, len(cells) * len(COST)))
print("  generation: %.1f h   per-run load/decode overhead: %.1f h"
      % ((total - runs * LOAD) / 3600.0, runs * LOAD / 3600.0))
print("")
print("per-family share:")
for fam in ("ctrlworld", "dreamgen", "dreamdojo"):
    t = sum(r[4] for r, c in zip(rows, cells) if c[3].startswith(fam))
    print("  %-11s %5.1f h" % (fam, t / 3600.0))
print("")
print("no run script -> reported as not-run, not estimated (%d):" % len(missing))
for cell, m in missing:
    print("  %-25s %s" % (cell, m))
PY
  echo ""
  echo "Nothing has been run. Re-invoke with --all (or --cell <id>) to execute."
  exit 0
fi

# ------------------------------------------------------------------- guard ---
# Timing measured while another job owns the GPU is not a speedup measurement,
# it is a measurement of contention. Refuse to run unless the device is idle,
# rather than silently producing numbers that cannot be defended.
require_idle_gpu() {
  local util mem
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | head -1)"
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" 2>/dev/null | head -1)"
  if [[ -z "${util}" ]]; then
    echo "ERROR: cannot query GPU ${GPU} via nvidia-smi." >&2
    exit 1
  fi
  if [[ "${util}" -ge 15 || "${mem}" -ge 8000 ]]; then
    echo "ERROR: GPU ${GPU} is busy (util=${util}%, mem=${mem} MiB)." >&2
    echo "A co-tenant job is using the device; timing taken now would measure" >&2
    echo "contention, not method speed. Set FORCE_BUSY_GPU=1 to override" >&2
    echo "(the result must then be reported as contended, not as a speedup)." >&2
    [[ "${FORCE_BUSY_GPU:-0}" == "1" ]] || exit 1
    echo "WARNING: FORCE_BUSY_GPU=1 -- results will be marked contended." >&2
    CONTENDED=1
  fi
}

# Extract the authoritative generation time from a run log. Each family logs a
# different line; wall clock is deliberately NOT used because it includes model
# load, ffmpeg and metric computation, which would dilute every speedup.
extract_time() {  # extract_time <model> <log> <output_root>
  local model="$1" log="$2" out="$3"
  case "$model" in
    ctrlworld|ctrlworld_4view_grid)
      # per-episode wall_time_s inside metrics.json
      python3 - "$out" <<'PY'
import json, os, sys
tot = []
for dirpath, _, files in os.walk(sys.argv[1]):
    if "metrics.json" in files:
        try:
            d = json.load(open(os.path.join(dirpath, "metrics.json")))
        except Exception:
            continue
        if isinstance(d, dict) and d.get("wall_time_s"):
            tot.append(float(d["wall_time_s"]))
print("%.3f" % sum(tot) if tot else "")
PY
      ;;
    dreamgen)
      # "Total inference time" / "LVG Total inference time", one per segment.
      grep -oE "(LVG )?Total inference time[^0-9]*([0-9]+\.[0-9]+)" "$log" 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+" | awk '{s+=$1} END{if(NR)printf "%.3f", s}'
      ;;
    dreamdojo)
      grep -oE "\[Timing\] Generated[^0-9]*([0-9]+\.[0-9]+)" "$log" 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+" | awk '{s+=$1} END{if(NR)printf "%.3f", s}'
      ;;
  esac
}

# Per-segment / per-chunk times, so the first unit (which pays one-off
# profiling and k-means warm-up for SVG/PISA) can be reported separately.
extract_units() {  # extract_units <model> <log>
  local model="$1" log="$2"
  case "$model" in
    dreamgen)
      grep -oE "(LVG )?Total inference time[^0-9]*([0-9]+\.[0-9]+)" "$log" 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+" | paste -sd, - ;;
    dreamdojo)
      grep -oE "\[Timing\] Generated[^0-9]*([0-9]+\.[0-9]+)" "$log" 2>/dev/null \
        | grep -oE "[0-9]+\.[0-9]+" | paste -sd, - ;;
    *) echo "" ;;
  esac
}

# --------------------------------------------------------------------- run ---
CONTENDED=0
require_idle_gpu

RESULTS="${SCRATCH}/results.jsonl"
mkdir -p "${SCRATCH}"
: > "${RESULTS}.tmp"
[[ -f "$RESULTS" ]] || : > "$RESULTS"

run_one() {  # run_one <cell> <emb> <view> <model> <unit> <dense_script> <method>
  local cell="$1" emb="$2" view="$3" model="$4" unit="$5" dscript="$6" method="$7"
  local rel
  if [[ "$method" == "dense" ]]; then rel="scripts/${dscript}"
  else rel="scripts/run_${emb}_${view}_${model}_${method}.sh"; fi
  if [[ ! -f "${ROOT}/${rel}" ]]; then
    echo "  [skip] ${cell}/${method}: no run script"
    python3 -c "
import json,sys
print(json.dumps({'cell':sys.argv[1],'method':sys.argv[2],'status':'no_script',
                  'reason':'no run script exists for this cell x method'}))" \
      "$cell" "$method" >> "${RESULTS}.tmp"
    return 0
  fi

  # Already measured in a previous invocation -> keep it (resumable).
  if grep -q "\"cell\": \"${cell}\", \"method\": \"${method}\"" "$RESULTS" 2>/dev/null; then
    echo "  [have] ${cell}/${method}"
    return 0
  fi

  local out="${SCRATCH}/${cell}/${method}/output"
  local inp="${SCRATCH}/${cell}/${method}/input"
  local log="${SCRATCH}/${cell}/${method}/run.log"
  mkdir -p "$out" "$inp" "$(dirname "$log")"

  echo "  [run ] ${cell}/${method}"
  local t0 t1 rc=0
  t0=$(date +%s)
  # NUM_EPISODES / MAX_EPISODES: one matched episode. SPLIT/SPLITS: the makovian
  # split, which every family has. Scratch OUTPUT_ROOT/INPUT_ROOT keeps the
  # canonical tree untouched (sampling-dataset-layout rule).
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" \
  NUM_EPISODES=1 MAX_EPISODES=1 MAX_SEGMENTS=2 NUM_CHUNKS=2 \
  SPLIT=makovian SPLITS=makovian \
  OUTPUT_ROOT="$out" INPUT_ROOT="$inp" SAVE_ROOT="$out" \
  FORCE_REGENERATE=1 \
    bash "${ROOT}/${rel}" > "$log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)

  local gen units
  gen="$(extract_time "$model" "$log" "$out" || true)"
  units="$(extract_units "$model" "$log" || true)"

  python3 - "$cell" "$method" "$model" "$unit" "$rel" "$gen" "$units" \
           "$((t1 - t0))" "$rc" "$CONTENDED" "$log" >> "${RESULTS}.tmp" <<'PY'
import json, sys
(cell, method, model, unit, script, gen, units, wall, rc, contended, log) = sys.argv[1:12]
rec = {
    "cell": cell, "method": method, "model": model, "work_unit": unit,
    "script": script, "wall_s": int(wall), "exit_code": int(rc),
    "contended": bool(int(contended)), "log": log,
    "status": "ok" if (rc == "0" and gen) else ("no_timing" if rc == "0" else "failed"),
}
rec["generation_s"] = float(gen) if gen else None
if units:
    vals = [float(x) for x in units.split(",") if x]
    rec["unit_times_s"] = vals
    # The first unit pays one-off profiling / k-means warm-up for several
    # methods; reporting it separately keeps the steady-state honest.
    rec["first_unit_s"] = vals[0] if vals else None
    rec["steady_state_s"] = (sum(vals[1:]) / len(vals[1:])) if len(vals) > 1 else None
print(json.dumps(rec))
PY
  cat "${RESULTS}.tmp" >> "$RESULTS"
  : > "${RESULTS}.tmp"
}

for spec in "${CELLS[@]}"; do
  IFS='|' read -r cell emb view model dense unit dscript <<<"$spec"
  [[ -n "$ONLY_CELL" && "$ONLY_CELL" != "$cell" ]] && continue
  echo "=== ${cell} (${model}, ${emb}, ${view}) ==="
  for method in dense sito itm worldcache dicache fastercache pisa svg1 svg2 radial; do
    if [[ -n "$METHODS_FILTER" ]]; then
      [[ " $METHODS_FILTER " == *" $method "* ]] || continue
    fi
    run_one "$cell" "$emb" "$view" "$model" "$unit" "$dscript" "$method"
  done
done

echo ""
echo "results: ${RESULTS}"
echo "now run: python3 assets/scripts/make_timing_pass_table.py --results ${RESULTS}"
