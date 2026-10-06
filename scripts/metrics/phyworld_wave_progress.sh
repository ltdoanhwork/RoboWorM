#!/usr/bin/env bash
# Progress of a PhyWorld wave: per-episode sidecars scored vs available.
# Counts real files on disk, so it is accurate even after a container restart.
set -uo pipefail
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "$REPO"

EMBODIMENT="${EMBODIMENT:-single_arm}"
VIEW="${VIEW:-multiview}"
MODEL="${MODEL:-dreamgen}"

"${PY:-python3}" - "$EMBODIMENT" "$VIEW" "$MODEL" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from phyworld_index import walk

emb, view, model = sys.argv[1], sys.argv[2], sys.argv[3]
ds = Path("sampling_dataset")
methods = ["dense", "pisa", "svg1", "svg2", "radial", "sito", "itm",
           "worldcache", "dicache", "fastercache"]

print(f"=== PhyWorld progress: {emb} / {view} / {model} ===")
tot_done = tot_all = 0
for split in ("makovian", "non_makovian"):
    print(f"\n  [{split}]")
    for m in methods:
        rows = list(walk(ds, methods=[m], embodiment=emb, view=view,
                         model=model, split=split))
        if not rows:
            continue
        done = sum(1 for r in rows if Path(r["sidecar"]).exists())
        tot_done += done
        tot_all += len(rows)
        bar_n = int(24 * done / len(rows)) if rows else 0
        bar = "#" * bar_n + "." * (24 - bar_n)
        print(f"    {m:12s} [{bar}] {done:4d}/{len(rows):4d}")

pct = 100.0 * tot_done / tot_all if tot_all else 0.0
print(f"\n  TOTAL: {tot_done}/{tot_all}  ({pct:.1f}%)")
PY
