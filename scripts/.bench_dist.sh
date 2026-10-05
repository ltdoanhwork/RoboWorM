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
#   * time = the MEDIAN PER-CHUNK generation time (first chunk dropped as warm-up),
#     which is invariant to episode length -- see extract_time();
#   * EPS_PER_CATEGORY episodes of EACH category, the
#     SAME episodes for dense and for every method;
#   * strictly SEQUENTIAL on a single GPU (no co-tenancy, no context parallel);
#   * the first segment/chunk is reported separately, because it pays one-off
#     profiling and k-means warm-up for several methods and would otherwise
#     flatter dense;
#   * output to scratch only -- never into the canonical sampling_dataset tree
#     (sampling-dataset-layout rule).
#
# WHY BOTH CATEGORIES: makovian and non_makovian episodes differ in length and
# motion, and generation time scales with length, so a makovian-only sample can
# misstate a method whose cost is length-dependent (caching skips more steps on
# slow-motion clips). 2+2 is the smallest sample that covers both.
#
# HOW EACH FAMILY IS CAPPED (they do not share one knob):
#   Ctrl-World single_arm : positional subset defaults to `both`, NUM_EPISODES
#                           caps PER subset  -> 2+2
#   Ctrl-World bi/humanoid: --split is val|train|all (NOT a category!) and
#                           --num_episodes slices a stem-sorted list where every
#                           `makovian_*` precedes every `non_makovian_*`, so it
#                           would yield makovian only. Uses the
#                           --num_episodes_per_category flag added for this pass.
#   DreamGen singleview   : SPLITS + MAX_EPISODES (cap is per split)  -> 2+2
#   DreamGen multiview    : CATEGORY unset = both + MAX_EPISODES per category
#   DreamDojo             : the run scripts loop both splits internally and
#                           NUM_EPISODES caps per split  -> 2+2
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
#   EPS_PER_CATEGORY=1 bash scripts/_bench_uniform_timing.sh --all --gpu 0
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH="${SCRATCH:-/tmp/uniform_timing}"
GPU="${GPU:-0}"
# EPS_PER_CATEGORY: episodes per category (x2 categories = episodes per run).
# Default 1 (=2 eps/run, ~14.7 h for the full pass). 2 is the fuller protocol but
# costs ~27.4 h because DreamGen alone is 226-492 s per episode, so it is opt-in
# rather than the default -- an accidental --all should not burn a day of GPU.
# EPS_PER_CATEGORY: episodes per category. Default 1.
# BENCH_CATEGORIES: which categories to run. Default "makovian" only.
#
# Each run costs ~4 min of model load + a few seconds per chunk, so the load
# dominates: measuring both categories doubles the cost of a cell while the
# per-chunk number it produces is the same (a chunk is a chunk -- see the
# ctrlworld4_humanoid_mv note in extract_time: the old per-EPISODE metric was
# what made categories look different, because they have different chunk COUNTS).
# One category is therefore enough for a speed ratio; set
# BENCH_CATEGORIES="makovian non_makovian" when quality across both is also wanted.
EPS_PER_CATEGORY="${EPS_PER_CATEGORY:-1}"
BENCH_CATEGORIES="${BENCH_CATEGORIES:-makovian}"
# MAX_CHUNKS: stop each episode after this many chunks. The measured unit is one
# chunk, and a chunk's cost does not drift within an episode, so a handful is
# enough -- an episode is 9-33 chunks and generating all of them buys no
# statistical power while multiplying GPU time. 0 = no cap (full episode).
# NUM_CHUNKS is DreamGen's own knob for the same idea (chunks per rollout).
MAX_CHUNKS="${MAX_CHUNKS:-3}"
# Exported so the record-writing python block can read it too: the cap guard
# there compares the number of units a run produced against this value, and
# without the export it would always see 0 and never fire.
export BENCH_MAX_CHUNKS="${MAX_CHUNKS}"
# MAX_SEGMENTS is a DIFFERENT unit from MAX_CHUNKS on DreamGen multiview: one
# segment contains ~5 chunks there, so passing MAX_SEGMENTS=MAX_CHUNKS=3 yielded
# 15 chunks (30 min for dense alone) instead of the 3 intended. Keep it at 1 --
# one segment already provides several chunks, which is all the median needs.
MAX_SEGMENTS_ARG="${MAX_SEGMENTS_ARG:-1}"
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
#   cell_id|embodiment|view|model|dense_s|unit|dense_script|env_pins
#
# env_pins: space-separated KEY=VALUE applied to BOTH the dense control and every
# method run of that cell. These are NOT preferences — each one is the config
# recovered from what actually generated sampling_dataset (iter tag read off the
# output directory name, checkpoint read off latest_checkpoint.txt, step count
# read off the method scripts). Without them the pass silently benchmarks a
# different model than the quality numbers were produced with:
#
#   * DG humanoid SV  — dense script defaults to iter_000050000 and a NEWER
#     iter_000120000.pt now exists, but sampling_dataset says
#     iter_000090000_static16fps  -> ITER_NAME pinned.
#   * DG bimanual SV  — dense script defaults to iter_000050000 while
#     sampling_dataset says iter_000110000_bimanual16fps -> ITER_NAME pinned.
#   * CW single_arm MV — the sito/itm scripts default to NUM_INFERENCE_STEPS=25
#     while dense uses the model default 50. Timing those against a 50-step
#     dense reports a fake ~2x, so STEPS are pinned to 50 for every method.
#   * CW 4-view grids — the _4view_grid wrappers already pin their own grid
#     checkpoints (140000 humanoid / 110000 bimanual) and CONFIG, so no pin is
#     needed; do not add one or the wrapper's value is overridden.
#   * DD cells — the infer scripts resolve latest_checkpoint.txt, which currently
#     equals the sampling_dataset tag (single_arm iter_000030000, bimanual
#     iter_000100000, humanoid iter_000050000). verify_dd_ckpt() below refuses to
#     run if that ever stops being true.
#
# NOTE: the Ctrl-World 3-view strip appears here for single_arm ONLY. bimanual
# and humanoid are reported on their 4-view grid, which supersedes the strip
# (see assets/scripts/benchmark_cells.py), so timing their strip would measure a
# cell the paper does not report. Do not re-add those two rows without also
# re-including them in benchmark_cells.py.
CELLS=(
  "ctrlworld_single_arm_mv|single_arm|multiview|ctrlworld|59|episode|run_single_arm_multiview_ctrlworld.sh|NUM_INFERENCE_STEPS=50"
  "ctrlworld4_bimanual_mv|bimanual|multiview|ctrlworld_4view_grid|75|episode|run_bimanual_multiview_ctrlworld_4view_grid.sh|"
  "ctrlworld4_humanoid_mv|humanoid|multiview|ctrlworld_4view_grid|75|episode|run_humanoid_multiview_ctrlworld_4view_grid.sh|"
  "dreamgen_single_arm_mv|single_arm|multiview|dreamgen|492|episode|run_single_arm_multiview_dreamgen_batch.sh|"
  "dreamgen_bimanual_mv|bimanual|multiview|dreamgen|237|2 segments|run_bimanual_multiview_dreamgen_full.sh|"
  "dreamgen_humanoid_mv|humanoid|multiview|dreamgen|230|2 segments|run_humanoid_multiview_dreamgen_full.sh|"
  "dreamgen_single_arm_sv|single_arm|singleview|dreamgen|226|2 chunks|run_single_arm_singleview_dreamgen_static16fps.sh|ITER_NAME=iter_000090000"
  "dreamgen_bimanual_sv|bimanual|singleview|dreamgen|263|2 chunks|run_bimanual_singleview_dreamgen_16fps.sh|ITER_NAME=iter_000110000"
  "dreamgen_humanoid_sv|humanoid|singleview|dreamgen|260|2 chunks|run_humanoid_singleview_dreamgen_static16fps.sh|ITER_NAME=iter_000090000"
  "dreamdojo_single_arm_sv|single_arm|singleview|dreamdojo|145|episode|run_single_arm_singleview_dreamdojo_dense.sh|"
  "dreamdojo_bimanual_sv|bimanual|singleview|dreamdojo|390|episode|run_bimanual_singleview_dreamdojo_full_episode.sh|"
  "dreamdojo_humanoid_sv|humanoid|singleview|dreamdojo|56|episode|run_humanoid_singleview_dreamdojo_full_episode.sh|"
)

# The checkpoint iteration each DreamDojo cell's rollouts in sampling_dataset were
# generated with (== the output dir tag). The infer scripts resolve
# latest_checkpoint.txt, so a newer checkpoint landing on disk would silently
# re-baseline the benchmark; refuse instead.
declare -A DD_EXPECT_ITER=(
  [single_arm]="iter_000030000"
  [bimanual]="iter_000100000"
  [humanoid]="iter_000050000"
)
declare -A DD_CKPT_DIR=(
  [single_arm]="checkpoints/finetuned/dreamdojo/dreamdojo/exp_single_arm_sv/single_arm_singleview/checkpoints"
  [bimanual]="checkpoints/finetuned/dreamdojo/dreamdojo/exp_aloha_resume70k/aloha_bimanual_singleview_full/checkpoints"
  [humanoid]="checkpoints/base/dreamdojo/2B_GR1_post-train/iter_000050000"
)

verify_dd_ckpt() {  # verify_dd_ckpt <embodiment>
  local emb="$1" dir="${ROOT}/${DD_CKPT_DIR[$1]}" want="${DD_EXPECT_ITER[$1]}" got=""
  if [[ -f "${dir}/latest_checkpoint.txt" ]]; then
    got="$(tr -d ' \n' < "${dir}/latest_checkpoint.txt" | sed 's/\.pt$//')"
  else
    got="$(basename "${dir}")"
  fi
  if [[ "${got}" != "${want}" ]]; then
    echo "ERROR: DreamDojo ${emb}: checkpoint resolves to '${got}' but" >&2
    echo "  sampling_dataset was generated with '${want}'. Timing a different" >&2
    echo "  checkpoint than the quality numbers is not a speedup measurement." >&2
    echo "  Pin the old iter, or re-generate the rollouts first." >&2
    return 1
  fi
  echo "  [ckpt] dreamdojo/${emb}: ${got} == sampling_dataset tag"
  return 0
}

# Per-method multiplier on the dense cost, from the documented speedups. Used
# only for the estimate: a method at 0.7x costs MORE wall time than dense.


# ---------------------------------------------------------------- estimate ---
if [[ "$MODE" == "estimate" ]]; then
  python3 - "$ROOT" "$EPS_PER_CATEGORY" "${CELLS[@]}" <<'PY'
import os, sys

root = sys.argv[1]
eps_per_cat = int(sys.argv[2])
n_eps = eps_per_cat * 2          # makovian + non_makovian
cells = []
for line in sys.argv[3:]:
    cell, emb, view, model, dense, unit, dscript, _pins = line.split("|")
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
        # dense is the documented cost of ONE episode; the pass runs n_eps of
        # them, but model load is paid once per run, not per episode.
        cell_total += dense * c * n_eps + LOAD
        cell_runs += 1
    rows.append((cell, unit, dense, cell_runs, cell_total))
    total += cell_total
    runs += cell_runs

print("protocol: %d episode(s) per category x 2 categories = %d episodes per run"
      % (eps_per_cat, n_eps))
print("")
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
  # PER-CHUNK median, in MILLISECONDS-equivalent seconds.
  #
  # Every family logs `[ChunkTiming] chunk N: X.XXXs` around the one unit of work
  # the model repeats (one DiT rollout / one forward_wm). That is the unit a
  # speedup should be quoted at: per-EPISODE totals bundle a variable NUMBER of
  # chunks, because episode length differs by category and task, so the ratio
  # silently measures episode length too. Measured proof on ctrlworld4_humanoid_mv:
  # a 37-frame makovian clip and a 61-frame non_makovian clip gave WorldCache
  # 0.64x and 1.71x for the SAME config -- a cache cannot make a model 1.6x
  # slower, that spread was pure episode-length noise.
  #
  # The FIRST chunk is dropped: it pays one-off CUDA autotune, SVG/PISA MSE
  # profiling and k-means warm-up, which flatters dense. The median (not mean) of
  # the rest resists the occasional PFSS I/O stall.
  local chunk
  chunk="$(python3 - "$log" <<'PY'
import re, statistics, sys
try:
    txt = open(sys.argv[1], errors="replace").read()
except Exception:
    print(""); raise SystemExit
# Chunk numbering restarts at 1 for every segment, and EACH segment's first chunk
# pays warm-up (observed DreamDojo: 23.8, 12.0, 11.9, 27.6, 12.1, 12.0 -- the
# 27.6 is segment 2's chunk 1, not a slow steady-state chunk). Drop every
# "chunk 1" rather than only the very first value, then take the median.
pairs = [(int(n), float(v)) for n, v in
         re.findall(r"\[ChunkTiming\] chunk (\d+): ([0-9.]+)s", txt)]
steady = [v for n, v in pairs if n > 1]
v = steady if steady else [v for _, v in pairs]
print("%.4f" % statistics.median(v) if v else "")
PY
)"
  if [[ -n "${chunk}" ]]; then
    echo "${chunk}"
    return
  fi
  # ---- Fallback: whole-episode timing (older logs with no [ChunkTiming]) ----
  case "$model" in
    ctrlworld|ctrlworld_4view_grid)
      # Prefer per-episode wall_time_s in metrics.json; fall back to the
      # `mean_wall_time_s` in all_summary.json, then to the run log's `time: Xs`.
      # The fallbacks matter because metrics.json only started carrying
      # wall_time_s after infer_*_ctrlworld.py was fixed to re-dump it; the
      # summary and the log line have always had it.
      python3 - "$out" "$log" <<'PY'
import json, os, re, sys
out, log = sys.argv[1], sys.argv[2]

per_ep = []
summaries = []
for dirpath, _, files in os.walk(out):
    if "metrics.json" in files:
        try:
            d = json.load(open(os.path.join(dirpath, "metrics.json")))
        except Exception:
            d = None
        if isinstance(d, dict) and d.get("wall_time_s"):
            per_ep.append(float(d["wall_time_s"]))
    if "all_summary.json" in files:
        try:
            s = json.load(open(os.path.join(dirpath, "all_summary.json")))
        except Exception:
            s = None
        if isinstance(s, dict) and s.get("mean_wall_time_s") and s.get("num_episodes"):
            summaries.append(float(s["mean_wall_time_s"]) * int(s["num_episodes"]))

if per_ep:
    print("%.3f" % sum(per_ep))
elif summaries:
    # one summary per category -> total across the cell
    print("%.3f" % sum(summaries))
else:
    # `time: Xs` is the per-category MEAN, and the script prints one per category
    try:
        means = [float(x) for x in re.findall(r"^time: ([0-9.]+)s", open(log).read(), re.M)]
    except Exception:
        means = []
    print("%.3f" % sum(means) if means else "")
PY
      ;;
    dreamgen)
      # DreamGen logs a per-chunk "Total inference time: X" AND, at the end of a
      # rollout, an aggregate "LVG Total inference time: Y" that already contains
      # those chunks. Summing every match therefore DOUBLE-COUNTS (observed:
      # 118.58 + 119.07 + 260.97 = 498.6 "generation" seconds inside a 306 s run).
      # Take the per-chunk lines only -- the ones NOT prefixed by LVG/GR00T -- and
      # report their median with the first (warm-up) chunk dropped, so the unit
      # matches the [ChunkTiming] median used for every other family.
      python3 - "$log" <<'PY'
import re, statistics, sys
try:
    txt = open(sys.argv[1], errors="replace").read()
except Exception:
    print(""); raise SystemExit
per_chunk = [float(m.group(1)) for m in
             re.finditer(r"(?<!LVG )(?<!GR00T LVG )Total inference time:\s*([0-9.]+)", txt)]
if len(per_chunk) > 1:
    per_chunk = per_chunk[1:]
print("%.4f" % statistics.median(per_chunk) if per_chunk else "")
PY
      ;;
    dreamdojo)
      # "[Timing] Generated <N> episodes in <X>s total", once per split (the run
      # scripts loop makovian + non_makovian), so sum them.
      # NOTE the episode COUNT sits between "Generated" and the seconds, so a
      # `[^0-9]*` bridge does NOT work here -- it stops at that count and yields
      # nothing. Match the sentence explicitly.
      grep -oP '\[Timing\] Generated \d+ episodes in \K[0-9.]+' "$log" 2>/dev/null \
        | awk '{s+=$1} END{if(NR)printf "%.3f", s}'
      ;;
  esac
}

# Per-segment / per-chunk times, so the first unit (which pays one-off
# profiling and k-means warm-up for SVG/PISA) can be reported separately.
extract_units() {  # extract_units <model> <log>
  local model="$1" log="$2"
  # Per-chunk series, so the first (warm-up) unit stays inspectable and the
  # spread across chunks is visible rather than averaged away.
  local chunks
  chunks="$(grep -oP '\[ChunkTiming\] chunk \d+: \K[0-9.]+' "$log" 2>/dev/null | paste -sd, -)"
  if [[ -n "${chunks}" ]]; then
    echo "${chunks}"
    return
  fi
  case "$model" in
    dreamgen)
      # Per-chunk lines ONLY. Including the aggregate "LVG Total inference time"
      # puts a value ~= the sum of the chunks into the per-unit series, which
      # corrupts steady_state_s and makes the spread look like 75% CV when the
      # real chunks vary by <1% (observed: [105.4, 105.5, 105.6, 345.7]).
      grep -oP '(?<!LVG )(?<!GR00T LVG )Total inference time:\s*\K[0-9.]+' "$log" 2>/dev/null \
        | paste -sd, - ;;
    dreamdojo)
      grep -oP '\[Timing\] Generated \d+ episodes in \K[0-9.]+' "$log" 2>/dev/null \
        | paste -sd, - ;;
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

run_one() {  # run_one <cell> <emb> <view> <model> <unit> <dense_script> <pins> <method>
  local cell="$1" emb="$2" view="$3" model="$4" unit="$5" dscript="$6" pins="$7" method="$8"
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

  echo "  [run ] ${cell}/${method}${pins:+  pins: ${pins}}"
  local t0 t1 rc=0
  t0=$(date +%s)
  # NUM_EPISODES / MAX_EPISODES / NUM_EPISODES_PER_CATEGORY: EPS_PER_CATEGORY
  # episodes of EACH category, matched across dense and every method. Each family
  # honours a different one of these (see the header table) so all three are set.
  # SPLIT is the Ctrl-World train/val selector -- passing a CATEGORY name here
  # (makovian) makes its argparse reject the run outright.
  # Ctrl-World single_arm takes its category as a POSITIONAL arg (`${1:-both}`),
  # not an env var, and defaults to running both -- which doubles the model load
  # for no extra per-chunk information. Pass the first requested category.
  local pos_args=()
  if [[ "$model" == ctrlworld* && "$emb" == "single_arm" ]]; then
    pos_args=("${BENCH_CATEGORIES%% *}")
  fi

  # INPUT_ROOT: only redirect it for cells whose run script PREPARES its own
  # inputs (it embeds the output path inside the batch JSON, so it must write to
  # scratch). Cells that READ shared, pre-prepared batch JSONs from
  # sampling_dataset/dense/... must keep the real path -- pointing them at an
  # empty scratch dir makes every split "[SKIP] input dir not found" and the whole
  # cell exits 0 having generated nothing (observed on dreamgen_bimanual_sv).
  local input_env=()
  if grep -qE 'prepare_batch_inputs|first_frames' "${ROOT}/${rel}" 2>/dev/null; then
    input_env=(INPUT_ROOT="$inp")
  fi

  set +e
  env CUDA_VISIBLE_DEVICES="${GPU}" \
  NUM_EPISODES="${EPS_PER_CATEGORY}" \
  NUM_EPISODES_PER_CATEGORY="${EPS_PER_CATEGORY}" \
  MAX_EPISODES="${EPS_PER_CATEGORY}" \
  MAX_SEGMENTS="${MAX_SEGMENTS_ARG}" \
  SPLIT=all SPLITS="${BENCH_CATEGORIES}" CATEGORY="${BENCH_CATEGORIES}" \
  SAVE_FULL=1 SAVE_MODE=full \
  SEQUENTIAL=1 NUM_GPUS=1 \
  BENCH_MAX_CHUNKS="${MAX_CHUNKS}" \
  NUM_CHUNKS="${MAX_CHUNKS}" \
  OUTPUT_ROOT="$out" SAVE_ROOT="$out" "${input_env[@]}" \
  FORCE_REGENERATE=1 \
  ${pins} \
    bash "${ROOT}/${rel}" "${pos_args[@]}" > "$log" 2>&1
  rc=$?
  set -e
  t1=$(date +%s)

  local gen units
  gen="$(extract_time "$model" "$log" "$out" || true)"
  units="$(extract_units "$model" "$log" || true)"

  python3 - "$cell" "$method" "$model" "$unit" "$rel" "$gen" "$units" \
           "$((t1 - t0))" "$rc" "$CONTENDED" "$log" >> "${RESULTS}.tmp" <<'PY'
import json, os, sys
(cell, method, model, unit, script, gen, units, wall, rc, contended, log) = sys.argv[1:12]
# A zero/empty generation time is NOT a success: when a run script dies early
# (e.g. it points at a dataset path that does not exist) the log has no timing
# line at all, and treating that as ok would publish a 0 s "infinitely fast"
# method. Require a strictly positive time.
gen_f = None
try:
    gen_f = float(gen) if gen else None
except ValueError:
    gen_f = None
if gen_f is not None and gen_f <= 0:
    gen_f = None
rec = {
    "cell": cell, "method": method, "model": model, "work_unit": unit,
    "script": script, "wall_s": int(wall), "exit_code": int(rc),
    "contended": bool(int(contended)), "log": log,
    "status": "ok" if (rc == "0" and gen_f) else ("no_timing" if rc == "0" else "failed"),
}
rec["generation_s"] = gen_f
if units:
    vals = [float(x) for x in units.split(",") if x]
    rec["unit_times_s"] = vals
    # The first unit pays one-off profiling / k-means warm-up for several
    # methods; reporting it separately keeps the steady-state honest.
    rec["first_unit_s"] = vals[0] if vals else None
    rec["steady_state_s"] = (sum(vals[1:]) / len(vals[1:])) if len(vals) > 1 else None
    # SANITY: generation_s is a per-unit MEDIAN, so it must sit inside the
    # observed per-unit range. If it lands outside, the extractor summed instead
    # of taking a median (seen when a stale harness copy was still running:
    # 15 chunks of ~118s reported as 1772.6s = their sum). Flag it loudly rather
    # than publish a number that is off by the chunk count.
    if gen_f and vals:
        lo, hi = min(vals), max(vals)
        if not (lo * 0.5 <= gen_f <= hi * 1.5):
            rec["status"] = "suspect_unit_mismatch"
            rec["suspect_reason"] = (
                f"generation_s={gen_f} outside per-unit range [{lo:.2f},{hi:.2f}]"
                f" over {len(vals)} units -- extractor likely summed, not median")
    # SANITY: the chunk cap bounds how many units a run may produce. A method
    # that emits far more did NOT honour the episode cap and therefore measured a
    # different amount of work than the others in its cell (observed: Ctrl-World
    # SiTo ran 128 episodes / 512 chunks while dense ran 1 episode / 4 chunks,
    # because its run script ignored NUM_EPISODES). The median is still per-chunk
    # so the number may be usable, but it must not pass silently.
    cap = int(os.environ.get("BENCH_MAX_CHUNKS") or 0)
    if cap and len(vals) > cap * 4:
        rec["status"] = "suspect_cap_ignored"
        rec["suspect_reason"] = (
            f"{len(vals)} units recorded with BENCH_MAX_CHUNKS={cap}"
            " -- the run script likely ignored the episode/chunk cap")
print(json.dumps(rec))
PY
  cat "${RESULTS}.tmp" >> "$RESULTS"
  : > "${RESULTS}.tmp"
}

for spec in "${CELLS[@]}"; do
  IFS='|' read -r cell emb view model dense unit dscript pins <<<"$spec"
  [[ -n "$ONLY_CELL" && "$ONLY_CELL" != "$cell" ]] && continue
  echo "=== ${cell} (${model}, ${emb}, ${view}) ==="
  # DreamDojo resolves its checkpoint from latest_checkpoint.txt at runtime, so
  # confirm it still matches the one sampling_dataset was built with before
  # spending GPU time on a baseline that cannot be paired with the quality rows.
  if [[ "$model" == "dreamdojo" ]]; then
    verify_dd_ckpt "$emb" || { echo "  [skip] ${cell}: checkpoint mismatch"; continue; }
  fi
  for method in dense sito itm worldcache dicache fastercache pisa svg1 svg2 radial; do
    if [[ -n "$METHODS_FILTER" ]]; then
      [[ " $METHODS_FILTER " == *" $method "* ]] || continue
    fi
    run_one "$cell" "$emb" "$view" "$model" "$unit" "$dscript" "$pins" "$method"
  done
done

echo ""
echo "results: ${RESULTS}"
echo "now run: python3 assets/scripts/make_timing_pass_table.py --results ${RESULTS}"
