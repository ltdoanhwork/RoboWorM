# GPU agent: fill remaining `sampling_dataset` gaps

Repo root:

```
/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
```

Audited against sheet `distribution` of `[BENCHMARK] Accelerating Inference of Video World Models.xlsx` (in the repo root). **968 of 29343 planned episodes are missing, in exactly two cells.** Everything else is done.

| method | have | target | missing |
|---|---:|---:|---:|
| dicache | 3166 | 3166 | **0** |
| worldcache | 3166 | 3166 | **0** |
| dense | 3046 | 3166 | 120 |
| sito | 2290 | 2410 | 120 |
| itm | 2290 | 2410 | 120 |
| pisa | 3046 | 3166 | 120 |
| radial | 3046 | 3166 | 120 |
| svg1 | 3046 | 3166 | 120 |
| svg2 | 3046 | 3166 | 120 |
| fastercache | 2233 | 2361 | 128 |

All 120-episode gaps are the **same cell**: `single_arm / singleview / dreamgen / non_makovian`.

One GPU at a time. Scripts skip episodes whose output mp4 already exists (`FORCE_REGENERATE=0`). Do **not** write sweep output into `sampling_dataset/` — that tree is canonical.

```bash
cd /pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics
unset NUM_SHARDS SHARD_ID                      # leftover tmux env silently skips episodes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # box has one A100-80GB
bash scripts/jobs_fill_missing.sh              # dry-run catalog
WAVE=1 DRY_RUN=0 CUDA_VISIBLE_DEVICES=0 bash scripts/jobs_fill_missing.sh
```

---

## The canonical matrix

Two models per (embodiment, view). **The Ctrl-World variant depends on embodiment:**

| embodiment | view | model A | model B | makovian + non_makovian |
|---|---|---|---|---|
| bimanual | MV | dreamgen | `ctrlworld_4view_grid` (4-view) | 150 + 150 each |
| bimanual | SV | dreamgen | dreamdojo | 150 + 150 each |
| humanoid | MV | dreamgen | `ctrlworld_4view_grid` (4-view) | 98 + 150 each |
| humanoid | SV | dreamgen | dreamdojo | 130+130 / 118+98 |
| single_arm | MV | dreamgen | `ctrlworld` (**native 3-view**) | 128 + 129 each |
| single_arm | SV | dreamgen | dreamdojo | 120 + 120 each |

Per-family totals: DreamGen **1605**, DreamDojo **756**, Ctrl-World **805** → dense **3166**.

Consequences:

- Native `ctrlworld/` under **bimanual** and **humanoid** is *not* canonical. Output exists there (dense/dicache/worldcache, plus sito/itm on humanoid) — keep it as extra results, but do not extend or count it.
- `ctrlworld_4view_grid/` must not be created for **single_arm**.
- DreamDojo has no multiview; Ctrl-World has no singleview. Those are N/a, not gaps.

### Two deliberate exclusions encoded in the per-method targets

| method | target | decomposition | meaning |
|---|---:|---|---|
| sito, itm | 2410 | DreamGen 1605 + Ctrl-World 805 | **skip DreamDojo entirely** |
| fastercache | 2361 | DreamGen 1605 + DreamDojo 756 | **skip Ctrl-World entirely** |
| dense, dicache, worldcache, pisa, radial, svg1, svg2 | 3166 | full matrix | — |

These match the technical findings: SiTo/ITM have no DreamDojo integration (and the single_arm DreamDojo scripts point at the wrong checkpoint), and FasterCache is a known no-op on Ctrl-World's CFG-off SVD UNet. So roughly 2300 episodes that an earlier version of this doc listed as "waves 4–6" are **not work items**.

### Counts that look short but are correct

The sheet agrees with disk on all of these — do not try to fill them.

- **humanoid MV makovian = 98** — that split has 2 tasks, non_makovian has 4. Full dataset.
- **humanoid SV dreamdojo = 118 / 98** — GR1 is sampled at stride 2 and the loader needs `num_frames + 1 = 50` steps, i.e. `delta_indices` up to 98, so a clip must be **≥ 99 frames**. 12 makovian and 32 non_makovian clips are shorter (4–98 frames; ~10 are tasks literally labelled "invalid"). Every method matches dense. **Do not pad to 130.**
- **single_arm MV = 128 / 129** — episodes `015475`, `016705` (mak) and `015486` (non) are missing a camera in the source; dreamgen and ctrlworld skip the same three.
- **single_arm SV = 120** — `datasets/single_arm/singleview/{split}/meta/info.json` says 120. A spreadsheet value of 130 elsewhere would be wrong.

---

## WAVE 1 — single_arm SV DreamGen non_makovian (840 episodes)

The only cell where **dense itself is empty**, so nothing downstream can be compared. Needed for `dense sito itm pisa radial svg1 svg2` (120 each). DiCache / WorldCache / FasterCache already have their 120.

Cause: the scripts default `NUM_CHUNKS="2 3"`. Non-makovian clips are longer and live in the 4/6/7-chunk batches.

Checkpoint (same as the makovian dense run):

```
exp_logs/dreamgen_logs/posttraining/singleview/single_arm_static_singleview_16fps_v2/checkpoints_40_per/model/iter_000090000.pt
```

### Batch inputs live under a `global/` view subdir

```
sampling_dataset/dense/single_arm/input/singleview/dreamgen/global/non_makovian/
  batch_input_4chunks.json   60 ep
  batch_input_6chunks.json    6 ep
  batch_input_7chunks.json   54 ep     → 120 total
```

The `.../dreamgen/{makovian,non_makovian}/` dirs one level up are **empty leftovers** — they do not mean the inputs are missing. Output carries the same view level: `.../output/singleview/dreamgen/non_makovian/iter_000090000_static16fps/global/`.

`output_video` inside the dense JSONs currently points at the **`itm/`** tree (a previous ITM run shares this input dir and rewrote the paths). Each run script's prepare step rewrites it to its own method tree, so this self-corrects — **never pass `SKIP_PREPARE=1` here.**

Budget: makovian 2-chunk episodes take ~225 s, so non_makovian at 4/6/7 chunks is roughly **15–25 h for dense alone**, times seven methods.

```bash
WAVE=1 DRY_RUN=0 CUDA_VISIBLE_DEVICES=0 bash scripts/jobs_fill_missing.sh
```

One method at a time (resume-safe). Pin `ITER_NAME`; force dense to 1 GPU (its default is `NUM_GPUS=2`):

```bash
COMMON='SPLITS=non_makovian NUM_CHUNKS="4 6 7" ITER_NAME=iter_000090000 CUDA_VISIBLE_DEVICES=0'

NUM_GPUS=1 $COMMON bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_sito.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_itm.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_pisa.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_radial.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_svg1.sh
$COMMON bash scripts/run_single_arm_singleview_dreamgen_svg2.sh   # models/dreamgen/.venv_svg
```

Venv `models/dreamgen/.venv`, except SVG2 → `.venv_svg`.

Smoke-test one episode first (the full wave will then skip it):

```bash
MAX_EPISODES=1 SPLITS=non_makovian NUM_CHUNKS=4 ITER_NAME=iter_000090000 NUM_GPUS=1 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_arm_singleview_dreamgen_static16fps.sh
```

Done when each of these holds **120** mp4:

```
sampling_dataset/{dense,sito,itm,pisa,radial,svg1,svg2}/single_arm/output/singleview/dreamgen/non_makovian/iter_000090000_static16fps/global/
```

---

## WAVE 2 — FasterCache bimanual SV DreamGen non_makovian (128 episodes)

**22 / 150** done, resumable. Checkpoint `bimanual_singleview_16fps` / `iter_000110000`, output tag `iter_000110000_bimanual16fps`.

Pin `NUM_CHUNKS="3 5 6 18"`. These are the current groups produced by
`prepare_singleview_dreamgen_batch.py` from the 30 fps source. Do not use
`NUM_CHUNKS=auto`: the input dir also contains stale overlapping `9/11/33`
chunk JSONs, so auto discovers two chunk assignments for 132 episodes.

Audit caveat: 22 pre-existing FasterCache outputs were generated from the
stale 11-chunk group (1013 frames each); current 30 fps grouping assigns those
episodes 6 chunks (553 frames before any GT-length trim). This fill preserves
them because the sheet's `22/150` count treats them as done. Regenerate those
22 separately if strict length comparability is required.

```bash
WAVE=2 DRY_RUN=0 CUDA_VISIBLE_DEVICES=0 bash scripts/jobs_fill_missing.sh
```

Done at 150 mp4 under `sampling_dataset/fastercache/bimanual/output/singleview/dreamgen/non_makovian/iter_000110000_bimanual16fps/`.

---

## Already cleaned — do not regenerate

### Dense humanoid SV DreamGen non_makovian

**Keep only** `sampling_dataset/dense/humanoid/output/singleview/dreamgen/non_makovian/iter_000090000_static16fps` — 130 mp4 + 130 txt, frames 69 × 185 + 61 × 277 (short 2/3-chunk @ 16fps).

`scripts/cleanup_humanoid_sv_dreamgen_nm_iters.sh` moved the 130 shorts out of an untagged `iter_000090000` folder and deleted the long-rollout `iter_000070000` plus empty probe dirs (`base_2b`, `iter_0000{10,50,60,80,100,120}000_static16fps`). The batch JSONs already pointed at `_static16fps`. The script is idempotent and refuses to delete anything unless 130 mp4 are safely in place.

Checkpoint proof: `exp_logs/dreamgen_logs/posttraining/singleview/humanoid_singleview_16fps/checkpoints_40_per/model/iter_000090000.pt`. Only 90k and 120k exist on disk — **`iter_000050000.pt` does not**. `scripts/run_humanoid_singleview_dreamgen_static16fps.sh` now defaults `ITER_NAME=iter_000090000` (it previously defaulted to the non-existent 50k and would fail immediately).

---

## Leftover directories worth deleting (no GPU)

These inflate naive `find -name '*.mp4' | wc -l` counts. In every case the canonical sibling is complete, so nothing is lost.

| path | contents | note |
|---|---|---|
| `svg1/humanoid/.../dreamgen/{makovian,non_makovian}/iter_000090000` | 130 each | duplicate of `_static16fps` |
| `pisa/humanoid/.../dreamgen/makovian/iter_000090000` | 34 | partial |
| `radial/humanoid/.../dreamgen/makovian/iter_000090000` | 6 | partial |
| `svg2/humanoid/.../dreamgen/makovian/iter_000090000` | 0 | empty |
| `fastercache/single_arm/.../dreamdojo/{makovian,non_makovian}/iter_000100000` | 125 / 130 | **wrong ckpt** (2B_DROID 320×512 / 49 frames); correct `iter_000030000` already has 120 |
| `sito`, `itm` `/single_arm/.../dreamdojo/{split}/` | empty | DreamDojo is out of scope for these methods — safe to remove |

---

## Re-verify anytime (~2 min)

```bash
python3 - <<'PY'
import os
from pathlib import Path
ROOT=Path("sampling_dataset")
def units(d,depth=0):
    if depth>3 or not d.is_dir(): return 0
    try: e=list(os.scandir(d))
    except OSError: return 0
    if any(x.name=="full_pred.mp4" for x in e): return 1
    n=0
    for x in e:
        if x.is_file() and x.name.endswith(".mp4") and not any(
            s in x.name for s in ("full_gt","full_merged","gt_all","pred_all")
        ) and not x.name.startswith(("view_","seg_","gt_")):
            n+=1
        elif x.is_dir(): n+=units(Path(x.path),depth+1)
    return n
CELLS=[("bimanual","multiview","dreamgen","DG",150,150),
       ("bimanual","multiview","ctrlworld_4view_grid","CW",150,150),
       ("bimanual","singleview","dreamgen","DG",150,150),
       ("bimanual","singleview","dreamdojo","DD",150,150),
       ("humanoid","multiview","dreamgen","DG",98,150),
       ("humanoid","multiview","ctrlworld_4view_grid","CW",98,150),
       ("humanoid","singleview","dreamgen","DG",130,130),
       ("humanoid","singleview","dreamdojo","DD",118,98),
       ("single_arm","multiview","dreamgen","DG",128,129),
       ("single_arm","multiview","ctrlworld","CW",128,129),
       ("single_arm","singleview","dreamgen","DG",120,120),
       ("single_arm","singleview","dreamdojo","DD",120,120)]
SKIP={"sito":{"DD"},"itm":{"DD"},"fastercache":{"CW"}}
TARGET={"dense":3166,"dicache":3166,"worldcache":3166,"fastercache":2361,
        "sito":2410,"itm":2410,"pisa":3166,"radial":3166,"svg1":3166,"svg2":3166}
grand=0
for m,tgt in TARGET.items():
    have=0; gaps=[]
    for emb,view,model,fam,mak,non in CELLS:
        if fam in SKIP.get(m,set()): continue
        for split,plan in (("makovian",mak),("non_makovian",non)):
            got=min(units(ROOT/m/emb/"output"/view/model/split), plan)
            have+=got
            if got<plan: gaps.append(f"{emb}/{view}/{model}/{split} {got}/{plan}")
    grand+=tgt-have
    print(f"{m:13} {have:6}/{tgt:<6} missing {tgt-have:5}  " + ("; ".join(gaps) or "COMPLETE"))
print("TOTAL missing:", grand)
PY
```

`min(got, plan)` is deliberate — it stops the leftover iter dirs above from masking a real gap.

---

## Order

1. **WAVE 1 dense** — unblocks the only empty dense cell.
2. WAVE 1 accel: sito → itm → pisa → radial → svg1 → svg2.
3. WAVE 2 (short).
4. Delete the leftover dirs.

After that the matrix is 29343 / 29343.
