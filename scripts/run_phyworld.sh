#!/usr/bin/env bash
# PhyWorld runner. Must start from repo root; we cd into metrics/WorldScore
# so vendored groundingdino / VFIMamba / SEA-RAFT relative paths resolve.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$REPO/models/dreamgen/.venv/bin/python}"
WS="$REPO/metrics/WorldScore"
TP="$WS/worldscore/benchmark/metrics/third_party"

cd "$WS"
export PYTHONPATH="$REPO:$REPO/scripts:$TP:$TP/SEA-RAFT/core:$TP/SEA-RAFT:$TP/VFIMamba:$TP/groundingdino:$TP/sam2:${PYTHONPATH:-}"

exec "$PY" "$REPO/scripts/compute_phyworld.py" "$@"
