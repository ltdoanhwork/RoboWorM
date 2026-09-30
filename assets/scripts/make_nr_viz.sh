#!/usr/bin/env bash
# Build the full NR metric visualisation set: cache -> 12 figures -> 1 HTML dashboard.
#
# Everything downstream reads the cache, never the ~87k raw result JSONs, so a figure
# tweak is seconds instead of a 40-minute rescan. Re-run with REBUILD_CACHE=1 after a
# scoring wave adds clips.
#
#   bash assets/scripts/make_nr_viz.sh
#   RUN=metrics/WorldScore/results/dense_nr_full REBUILD_CACHE=1 bash assets/scripts/make_nr_viz.sh
#   ONLY=figures bash assets/scripts/make_nr_viz.sh      # skip cache + dashboard
set -euo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"

RUN="${RUN:-metrics/WorldScore/results/dense_nr_full}"
CACHE="${CACHE:-$RUN/figures/cache}"
OUT="${OUT:-$RUN/figures}"
ONLY="${ONLY:-all}"                  # all | cache | figures | dashboard
REBUILD_CACHE="${REBUILD_CACHE:-0}"
BOOT_B="${BOOT_B:-10000}"            # paper default; lower it for a quick look
WORKERS="${WORKERS:-16}"
FORMATS="${FORMATS:-pdf svg png}"   # space-separated (nargs)
SELFTEST="${SELFTEST:-1}"

# Pick an interpreter that has matplotlib. `.venv_edit` is the lightweight one used for
# the paper assets; the model venvs are the fallback because they definitely have it.
pick_python() {
  for cand in "$ROOT/.venv_edit/bin/python" \
              "$ROOT/models/dreamgen/.venv/bin/python" \
              "$ROOT/models/DreamDojo/.venv/bin/python"; do
    if [[ -x "$cand" ]] && "$cand" -c 'import matplotlib' 2>/dev/null; then
      echo "$cand"; return 0
    fi
  done
  echo "FATAL: no interpreter with matplotlib found (tried .venv_edit, dreamgen, DreamDojo)" >&2
  return 1
}
PY="${PY:-$(pick_python)}"
echo "python : $PY"
echo "run    : $RUN"
echo "cache  : $CACHE"
echo "out    : $OUT"

step() { printf '\n=== %s ===\n' "$1"; }

# ---------------------------------------------------------------- cache
if [[ "$ONLY" == all || "$ONLY" == cache ]]; then
  if [[ "$REBUILD_CACHE" == 1 || ! -f "$CACHE/agg.json" ]]; then
    step "cache (scan results/, gate, aggregate, B=$BOOT_B)"
    "$PY" assets/scripts/nr_viz_cache.py \
      --run "$RUN" --out "$CACHE" --workers "$WORKERS" --boot "$BOOT_B"
  else
    step "cache (reusing $CACHE -- set REBUILD_CACHE=1 to rescan)"
    "$PY" - "$CACHE" <<'PY'
import json, sys, pathlib
m = json.loads((pathlib.Path(sys.argv[1]) / 'meta.json').read_text())
print(f"  {m['scored_clips']:,} clips | protocol {m['protocol']} | built {m['generated_at']}")
if m.get('partial'):
    print(f"  PARTIAL: {m['scored_clips']:,}/{m['manifest_clips']:,} expected clips "
          f"({100 * m['coverage_frac']:.0f}%) -- every figure and the dashboard are "
          f"stamped PARTIAL so no number here is mistaken for the final wave")
PY
  fi
fi

# ---------------------------------------------------------------- figures
if [[ "$ONLY" == all || "$ONLY" == figures ]]; then
  step "figures (12 panels -> $FORMATS + audit CSVs)"
  "$PY" assets/scripts/make_nr_metric_figures.py \
    --run "$RUN" --cache "$CACHE" --out "$OUT" --formats $FORMATS ${FIGS:+--only $FIGS}
fi

# ---------------------------------------------------------------- dashboard
if [[ "$ONLY" == all || "$ONLY" == dashboard ]]; then
  step "dashboard (self-contained HTML)"
  "$PY" assets/scripts/make_nr_dashboard.py \
    --run "$RUN" --cache "$CACHE" --out "$OUT/nr_dashboard.html"

  # The dashboard re-implements median / bootstrap / Wilcoxon / pair-gating in JS, so it
  # can silently disagree with the paper tables. This check refuses that: it loads the
  # shipped HTML and compares its numbers against agg.json. Worth the 15 seconds.
  if [[ "$SELFTEST" == 1 ]] && command -v node >/dev/null 2>&1; then
    step "dashboard selftest (JS numbers vs Python tables)"
    if node --max-old-space-size=4096 assets/scripts/nr_dashboard/selftest.js \
         "$OUT/nr_dashboard.html" "$CACHE/agg.json"; then
      echo "  dashboard agrees with the Python aggregates"
    else
      echo "  SELFTEST FAILED -- the dashboard disagrees with agg.json; do not ship it" >&2
      exit 1
    fi
  elif [[ "$SELFTEST" == 1 ]]; then
    echo "  (node not found; skipping the JS-vs-Python agreement check)"
  fi
fi

step "done"
ls -1sh "$OUT"/f*.pdf "$OUT"/*.html 2>/dev/null | sed 's/^/  /' || true
echo
echo "  figures   : $OUT/f*.{pdf,svg,png}"
echo "  audit CSVs: $OUT/data/"
echo "  dashboard : $OUT/nr_dashboard.html  (open directly, no server needed)"
