# Speedup coverage audit + prepared timing jobs

Status: **audit done; uniform pass RUN for 1 of 12 cells (`dreamdojo_humanoid_sv`,
8 methods incl. dense), remaining 11 cells NOT RUN** (GPUs must stay free so the
other in-flight measurement job is not skewed — Ctrl-World/DreamDojo timings
inflate ~15-20% under GPU contention, measured).

## DECISION (2026-09-23): report BOTH provenances, marked

The paper does **not** wait for the full uniform pass. Reason: the single A100 on
this host is at 100% / 79 GB from the in-flight job, and
`scripts/_bench_uniform_timing.sh` refuses to start on a contended card by design
(that refusal is what makes the measured ratios trustworthy). Blocking the paper
on ~8.3 GPU-hours that cannot be scheduled would buy nothing the marking does not
already give.

So every speedup carries its provenance, and the two are **never merged into one
undifferentiated number**:

| mark | provenance | what it means | count |
|---|---|---|---|
| `‡` | `timing_provenance=uniform_pass` | dense control re-run in the same batch, same episodes, same pinned config, idle GPU | 14 of 112 rows |
| `†` | `timing_provenance=tuning_notes` | per-cell tuning run: same script + config, own episode count, own dense reference | 86 of 112 rows |
| `--` | `timing_provenance=unavailable` | never instrumented; **never** interpolated from a neighbouring cell | 12 of 112 rows |

Where this is enforced:

- `scripts/make_speedup_summary.py` — tags each row (`timing_provenance` column in
  `metrics/reports/speedup_summary_full.csv`), emits `‡`/`†` in
  `metrics/reports/speedup_summary_appendix.tex`, and carries per-method counts
  (`n_cells_uniform_pass` / `n_cells_tuning_notes` / `n_cells_persisted`) into
  `metrics/reports/speedup_summary_by_method.csv` so a pooled mean cannot be read
  as one calibrated measurement.
- `assets/paper/appendix/3_implemetation_details.tex` — paragraph *A uniform
  re-measurement, specified and begun*, plus `appendix/tables/timing_pass` (now
  `\input`, since it holds real rows rather than a placeholder).
- `assets/paper/sections/4_experiments.tex` — the *Speedup measurement* paragraph
  states the coverage and the non-interpolation rule.

When GPU time frees up, extend the pass cell by cell; each extension only moves
rows from `†` to `‡` and requires no prose change beyond the count.

> **SCOPE (updated):** the Ctrl-World **3-view strip is no longer reported** for
> **bimanual** and **humanoid** — those two embodiments are reported on their
> 4-view 2x2 grid, which supersedes the strip on the same episodes. **single_arm
> keeps its 3-view strip** (it has no grid variant). The single source of truth is
> `assets/scripts/benchmark_cells.py`; every table/figure/CSV generator imports it.
> Reported grid = **12 cells / 105 method-cell configs**. Rows below for
> `bimanual MV (3-view)` and `humanoid MV (3-view)` are kept for history but are
> **out of scope — do not spend GPU time on them.**

Sources: the 14 `.cursor/rules/*-insights.mdc` tuning rules (ground truth for
every number below) + `metrics/reports/speedup_summary_full.csv` +
a filesystem audit of `sampling_dataset/`.

---

## 1. Speedup per cell (default / shipped config of each run script)

`—` = never measured. `n/a` = no run script for that cell (real hole in the grid).
Ctrl-World's 3-view strip and 4-view 2x2 grid are SEPARATE cells (different
checkpoint + latent canvas → very different cache behaviour; never mix them).

### Ctrl-World (SVD UNet, guidance 1.0, 50 steps)

| cell | DiCache | WorldCache | FasterCache | SiTo | ITM | PISA | SVG1 | SVG2 | Radial |
|---|---|---|---|---|---|---|---|---|---|
| single_arm MV (3-view strip) | **—** | **—** | **n/a\*** | ~1.00x | **—** | 0.95x | 0.88x | 0.65x | 0.93x |
| bimanual MV (3-view) | 1.23x | **1.27x** | 1.00x (no-op) | n/a | n/a | n/a | n/a | n/a | n/a |
| bimanual MV (4-view grid) | 1.72x | **1.81x** | 1.02x (no-op) | 1.08x | 1.04x | 0.98x | 0.88x | 0.74x | 0.94x |
| humanoid MV (3-view) | 1.11x | **1.16x** | n/a | 1.05x | 1.25x | **—** | **—** | **—** | **—** |
| humanoid MV (4-view grid) | 1.56x | **1.60x** | n/a | 1.11x | 1.11x | 0.96x | 0.85x | 0.66x | 0.92x |

\* script exists but `sampling_dataset/fastercache/single_arm/output/multiview/ctrlworld/`
does not exist → that cell has never produced output (the 4 stale episodes that
used to be counted in the CSV are gone).

Bigger lever than any backend on the 4-view grid: **`NUM_INFERENCE_STEPS`**.
25 steps = 1.74x free (== dense quality); 25 steps + DiCache 0.03 = **2.44x**;
15 steps + DiCache 0.03 = **2.95x**, still == dense visually.

### DreamGen (Cosmos-Predict2 DiT)

| cell | DiCache | WorldCache | FasterCache | SiTo | ITM | PISA | SVG1 | SVG2 | Radial |
|---|---|---|---|---|---|---|---|---|---|
| humanoid SV (2B) | **1.95x** | 1.89x | 1.15x | **—** | **—** | 1.45x | 1.28x | 1.15x | 1.36x |
| humanoid MV (2B grid) | **2.28x** | 2.08x | 1.16x | 1.20x | 0.96x | 1.44x | 1.29x | 1.19x | 1.36x |
| bimanual SV (2B) | **1.32x** | 1.31x | 0.76x | 1.29x | 0.95x | 1.14x | 0.77x | 0.84x | 1.10x |
| bimanual MV (2B grid) | **1.66x** | 1.56x | 0.94x | 1.19x | 0.95x | 1.46x | 1.45x | 1.40x | 1.41x |
| single_arm SV (2B) | **—** | **—** | **—** | 1.24x | 0.91x | 1.37x | **1.39x** | 1.26x | 1.34x |
| single_arm MV (14B) | **—** | **—** | **—** | **—** | **—** | **1.32x** | 1.22x | 1.02x | 1.29x |

### DreamDojo (action-conditioned 2B; self-attn only 15-18% of gen → Amdahl ~1.2x)

| cell | DiCache | WorldCache | FasterCache | SiTo | ITM | PISA | SVG1 | SVG2 | Radial |
|---|---|---|---|---|---|---|---|---|---|
| humanoid SV | **—** | **—** | **—** | **—**† | **—**† | 0.99x | 0.93x | 0.59x | **1.03x** |
| bimanual SV | **—** | **—** | **—** | n/a | n/a | **1.03x** | 0.97x | 0.69x | 1.02x |
| single_arm SV | **—** | **—** | **—** | **—**† | **—**† | **1.02x** | 0.91x | 0.69x | 1.01x |

† script exists but is wired to a DIFFERENT entrypoint
(`models.DreamDojo.examples.action_conditioned` with `--num-samples`, not the
canonical `infer_*_singleview_*.py`), and the canonical infer scripts do **not**
accept `--use-sito/--use-itm`. Its timing is therefore NOT comparable to
dense/cache/sparse on the same cell, and no output exists in the canonical tree.

---

## 2. Gap inventory

### (a) Timing missing (run scripts exist, rollouts mostly exist) — 23 in-scope cells

| cell | missing | prepared job |
|---|---|---|
| CW single_arm MV | DiCache, WorldCache, FasterCache, ITM | `scripts/_timing_ctrlworld_sa_mv.sh` |
| ~~CW humanoid MV 3-view~~ | ~~PISA, SVG1, SVG2, Radial~~ | **OUT OF SCOPE** (strip not reported; use the 4-view grid) |
| DG humanoid SV | SiTo, ITM | `_timing_dreamgen_gaps.sh hum_sv` |
| DG single_arm SV | DiCache, WorldCache, FasterCache | `_timing_dreamgen_gaps.sh sa_sv` |
| DG single_arm MV (14B) | DiCache, WorldCache, FasterCache, SiTo, ITM | `_timing_dreamgen_gaps.sh sa_mv` |
| DD humanoid SV | DiCache, WorldCache, FasterCache | `_timing_dreamdojo_sv_cache.sh humanoid` |
| DD bimanual SV | DiCache, WorldCache, FasterCache | `_timing_dreamdojo_sv_cache.sh bimanual` |
| DD single_arm SV | DiCache, WorldCache, FasterCache | `_timing_dreamdojo_sv_cache.sh single_arm` |

**Highest value = the 9 DreamDojo cache cells.** Sparse attention there is
kernel+Amdahl-capped (0.59-1.03x measured); whole-DiT step caching is the only
plausible real speedup and has *never* been timed or tuned on DreamDojo.

### (b) Quality metrics missing (rollouts exist, but video-only)

`sampling_dataset/{dense,dicache,worldcache,fastercache,pisa,svg1,svg2,radial}/`
`{humanoid,bimanual}/output/singleview/dreamdojo/**` contain the full rollouts
(216 humanoid / 300 bimanual mp4 each) but **no `metrics.json`** — those runs used
`SAVE_MODE=video_only`. So the appendix table has no PSNR/SSIM/LPIPS for 8 methods
x 2 cells. Fixing needs a scoring pass over existing mp4s (no re-generation).

### (c) Never run at all (script exists, zero output)

- FasterCache on Ctrl-World single_arm MV (directory absent).
- SiTo / ITM on DreamDojo humanoid SV and single_arm SV (0 outputs; see † above).

### (d) No script at all (holes in the 9-method grid)

- FasterCache on Ctrl-World humanoid MV — 3-view **and** 4-view grid
  (bimanual has one; humanoid does not). Expected result: no-op (~1.00x) like the
  other Ctrl-World cells (0 `attn_processors`, CFG off) — still worth shipping as
  a valid negative point per the acceleration-benchmark rule.
- SiTo / ITM on DreamDojo bimanual SV.
- Ctrl-World bimanual 3-view pruning/sparse (deliberate: superseded by the
  4-view-grid track).

---

## 3. Fixed: `scripts/make_speedup_summary.py` was mis-reporting

Two real bugs, both fixed (regenerated CSV/TeX committed):

1. **4-view grid rows inherited 3-view numbers.** `SPEEDUP_LOOKUP` was keyed on
   the collapsed display family (`ctrlworld`), so every `ctrlworld_4view_grid`
   row got the 3-view speedup. It now keys on the raw `model`:
   WorldCache humanoid grid **1.16x → 1.60x**, DiCache **1.11x → 1.56x**,
   SiTo **1.05x → 1.11x**, ITM **1.25x → 1.11x**.
2. **Most documented numbers were never loaded.** The lookup only covered 9 of
   the 14 insight rules — Ctrl-World bimanual (3-view + grid), Ctrl-World
   single_arm sparse, the grid sparse/pruning sets, DreamGen bimanual MV/SV
   caching + sparse + pruning, and DreamGen humanoid MV pruning were all absent,
   leaving their cells blank. All 14 rules are now transcribed (85 entries).

Result: rows with a speedup went **58/127 → 106/124**. The remaining 18 blanks are
exactly gap (a) above and are intentionally left blank (never estimated).

**After the 3-view exclusion** (bimanual/humanoid strip dropped from reporting),
the regenerated CSV is **112 rows / 56 method-cell pairs, 47 timed, 9 blank** —
the 9 blanks are CW single_arm MV (DiCache, WorldCache, ITM) and the 6 DreamDojo
SV cache cells. Mean speedups rose because the weaker strip cells left the pool
(DiCache 1.59x → 1.80x, WorldCache 1.58x → 1.76x); this is a change of scope,
not a re-measurement.

The claim in `docs/main_results_speedup_and_checkpoint.md` that "69 of 127 cells
have quality but no speedup" was therefore an artifact of the loader, not of the
benchmark — the honest number is **18 of 124**.

---

## 4. The job to run — `scripts/_bench_uniform_timing.sh`

A uniform timing harness already exists and is the right tool; this task
**extended it with the missing config pins** rather than adding a parallel one
(three ad-hoc helpers written during this task were deleted once the overlap was
found — one harness, one protocol).

```bash
bash scripts/_bench_uniform_timing.sh --estimate              # cost only, no GPU
bash scripts/_bench_uniform_timing.sh --all --gpu 0           # full pass
bash scripts/_bench_uniform_timing.sh --cell dreamdojo_humanoid_sv --gpu 0
EPS_PER_CATEGORY=1 bash scripts/_bench_uniform_timing.sh --all --gpu 0   # half the cost
python3 assets/scripts/make_timing_pass_table.py --results /tmp/uniform_timing/results.jsonl
```

### Sample protocol: 1 makovian + 1 non_makovian per (cell, method)

`EPS_PER_CATEGORY=1` (default) -> 2 episodes per run, the SAME episodes for dense
and for every method. Both categories are always covered because makovian and
non_makovian differ in length and motion and generation time scales with length,
so a makovian-only sample misstates any method whose cost is length-dependent
(caching skips more steps on slow-motion clips). Set `EPS_PER_CATEGORY=2` for the
fuller 2+2 protocol when the extra ~13 h of GPU is available.

Cost on one GPU (`--estimate`):

| protocol | eps/run | est. GPU time | Ctrl-World | DreamGen | DreamDojo |
|---|---|---|---|---|---|
| `EPS_PER_CATEGORY=1` (default) | 2 | **14.7 h** | 1.5 h | 10.0 h | 3.2 h |
| `EPS_PER_CATEGORY=2` | 4 | **27.4 h** | 2.6 h | 18.9 h | 5.8 h |

117 runs either way (12 cells x 10 methods, 3 pairs have no script). DreamGen
dominates because its per-episode cost is 226-492 s; Ctrl-World is cheap (59-75 s).
A practical split is `EPS_PER_CATEGORY=2` on Ctrl-World + DreamDojo (8.4 h) and
`EPS_PER_CATEGORY=1` on DreamGen (10.0 h).

The harness is resumable (already-measured `cell x method` pairs are kept),
refuses to start on a busy GPU, reads the family's authoritative log line instead
of wall clock, and reports the first segment/chunk separately so SVG/PISA warm-up
does not flatter dense.

To close only the never-timed cells first, run the DreamDojo cells (the
highest-value gap: sparse attention there is Amdahl-capped at ~1.2x, so caching
is the only plausible win and has never been timed):

```bash
for c in dreamdojo_humanoid_sv dreamdojo_single_arm_sv dreamdojo_bimanual_sv; do
  bash scripts/_bench_uniform_timing.sh --cell "$c" --gpu 0
done
```

### Three capping bugs found while wiring the 2+2 protocol (fixed)

Each one would have made a "4-episode" run silently measure something else:

1. **`SPLIT=makovian` is invalid for Ctrl-World bimanual/humanoid.** Their infer
   script's `--split` is `val|train|all` (a train/val selector), and the
   makovian/non_makovian category is parsed out of the *episode stem*
   (`split_category()`), not the split. The harness was passing
   `SPLIT=makovian`, which argparse rejects outright.
2. **`--num_episodes` cannot express "N per category".** It slices a stem-sorted
   list in which every `makovian_*` precedes every `non_makovian_*`, so
   `--num_episodes 4` returns 4 makovian episodes and 0 non_makovian. Added
   `--num_episodes_per_category` to `infer_bimanual_multiview_ctrlworld.py` (+ the
   `NUM_EPISODES_PER_CATEGORY` env knob on all 13 Ctrl-World bimanual/humanoid run
   scripts) which takes the first N of each category and logs the resulting mix.
3. **The 19 `_4view_grid` wrappers forward an explicit var whitelist**, so
   `NUM_EPISODES` / `NUM_INFERENCE_STEPS` / `GUIDANCE` were dropped for all but 3
   of them — a capped run would have quietly executed the entire split. Now
   forwarded everywhere.

While fixing (3) a latent shell hazard was also removed: a comment placed inside
a `VAR=v \`-continuation chain **silently discards every assignment before it**,
and `bash -n` still reports the file as valid. Verified:

```
FOO="a" \
# comment
BAR="b" \
  env    ->  BAR=b          (FOO is gone)
```

An initial version of the fix put the explanatory comment inside those chains,
which would have dropped `CONFIG`/`CKPT`/`DATASET_NAME` and run the 4-view grids
against the 3-view config. The comment was moved out of all 19 chains and a
repo-wide scan confirms no run script now has a comment following a
line-continuation inside an env-assignment chain.

### Why a re-run is unavoidable (audited, not assumed)

`mean_wall_time_s` is persisted in only **6 of 130** `all_summary.json` files and
in **ZERO dense ones**:

| method | all_summary | with `mean_wall_time_s` |
|---|---|---|
| dense | 16 | **0** |
| dicache / worldcache | 16 / 16 | 2 / 2 (Ctrl-World bimanual 3-view only) |
| pisa / radial / svg1 / svg2 | 12 each | 2 each (Ctrl-World single_arm MV only) |
| sito / itm | 10 each | 0 |
| fastercache | 4 | 0 |

Per-episode `metrics.json` never carries timing at all — its keys are
`episode_id, instruction, interact_num, psnr, ssim, lpips, per_view, mode,
num_frames_{gt,pred}`. With no dense wall time anywhere, **no cell's speedup can
be derived from disk**, which is exactly why the harness re-runs `dense` as a
control inside every cell.

### Config provenance — what actually generated `sampling_dataset`

Recovered from the artifacts, then encoded as the harness's per-cell `env_pins`:

| cell | evidence in `sampling_dataset` | pin |
|---|---|---|
| CW single_arm MV (3-view) | 128 makovian / 129 non_makovian `metrics.json`, all `mode: replay` | `NUM_INFERENCE_STEPS=50` |
| CW humanoid / bimanual 4-view grid | `output/multiview/ctrlworld_4view_grid/...` | none — the `_4view_grid` wrappers already pin ckpt-140000 / ckpt-110000 + `CONFIG=*_grid` |
| DG single_arm SV | dir tag `iter_000090000_static16fps` | `ITER_NAME=iter_000090000` |
| DG humanoid SV | dir tag `iter_000090000_static16fps` | `ITER_NAME=iter_000090000` |
| DG bimanual SV | dir tag `iter_000110000_bimanual16fps` | `ITER_NAME=iter_000110000` |
| DG bimanual / humanoid MV | dir tags `iter_000120000` / `iter_000100000` | none — the `_full.sh` scripts already default to these |
| DG single_arm MV | flat `episode_*.mp4`, no iter tag | none — 14B + `--gr00t_variant droid`, FPS 10 |
| DD single_arm / bimanual / humanoid SV | dir tags `iter_000030000` / `iter_000100000` / `iter_000050000` | `verify_dd_ckpt()` aborts the cell if `latest_checkpoint.txt` no longer matches |

Three traps these pins defuse — each would have produced a wrong number:

1. **DG humanoid SV dense would use the WRONG iter.** Its dense script defaults
   to `iter_000050000`, and a newer `iter_000120000.pt` now exists on disk, but
   `sampling_dataset` was generated with `iter_000090000`.
2. **DG bimanual SV dense would also use the WRONG iter** (`iter_000050000`
   default vs `iter_000110000` in the tree).
3. **CW `sito`/`itm` single_arm run at 25 steps while dense uses 50.** Timing
   them as-is reports a fake ~2x; the cell is pinned to 50 for all methods.
   Corollary: the SiTo/ITM *quality* already in the table for that cell was also
   produced at 25 steps, so it is not comparable to dense either — follow-up 5.

DreamDojo needs a guard rather than a pin because its infer scripts resolve
`latest_checkpoint.txt` at runtime; all three currently still match the tags in
`sampling_dataset` (verified), so the guard passes today and will fail loudly the
day a new checkpoint lands.

One more dirty-state find while verifying provenance:
`dense/single_arm/output/multiview/ctrlworld/makovian/all_summary.json` was
overwritten by a **10-episode `pi05` policy-predicted** run
(`"mode": "policy_predicted", "policy_type": "pi05"`), while the 128 per-episode
`metrics.json` beside it are all `"mode": "replay"`. The per-episode files are the
trustworthy ones; that summary should be regenerated from them (follow-up 6).

### Two hard blockers found by the first smoke run (both FIXED)

The DreamDojo cells could not run at all before this. Both failures were config
resolution, not GPU capacity, and both are now verified fixed:

1. **Gated HF repo, checkpoint.** `ActionConditionedSetupArguments` runs a
   pydantic `validate_model` hook that fills `checkpoint_path` from
   `MODEL_CHECKPOINTS[...]` when the caller omits it — and that default is a
   *property* that eagerly downloads from `nvidia/Cosmos-Predict2.5-2B`:

   ```
   GatedRepoError: 401 ... robot/action-cond/38c6c645-..._ema_bf16.pt
   ```

   It aborted before a single frame, even though `--checkpoints-dir` pointed at a
   local finetuned checkpoint that the script resolves *later* and hands to
   `Video2WorldInference` anyway. Fix: resolve the local checkpoint FIRST and pass
   `checkpoint_path=` into the constructor, which keeps the validator off the
   network. Applied to all three DreamDojo infer scripts
   (`infer_humanoid_singleview_full_episode.py`,
   `infer_single_arm_singleview_autoregressive.py`,
   `infer_bimanual_singleview_single_segment.py`). Proven GPU-free:
   with `checkpoint_path` -> no download; without -> still 401.

2. **Gated HF repo, tokenizer.** The next call wanted
   `Cosmos-Predict2.5-2B/tokenizer.pth` (the Wan2.1 VAE) from the same gated repo.
   `checkpoint_db._find_local_hf_repo_path()` already prefers a local mirror at
   `checkpoints/nvidia/<repo-name>/<filename>`, and the identical file is already
   on disk three times over — `Cosmos-Predict2-{2B,14B}-Video2World` and
   `...-GR00T-Dreams-GR1` all ship byte-identical copies (507,609,880 B, same
   head+tail hash). Fix: `checkpoints/nvidia/Cosmos-Predict2.5-2B/tokenizer.pth`
   as a symlink to the 2B copy. No download, no duplicated 0.5 GB.

This means the appendix claim that DreamDojo is excluded from the checkpoint
ablation because of "a pre-existing account-level access restriction" is only
half right: the 401 was real, but it was **avoidable** — the gated files were
either unnecessary (the DiT) or already on disk (the tokenizer).
`docs/main_results_speedup_and_checkpoint.md` should be revisited once the
DreamDojo cells have actually been measured.

After both fixes the run reached real generation (`[Timing] Generated 1 episodes
in 107.21s`) before being stopped for an unrelated reason — see below.

### Contention, not capacity, is the current blocker

The first real attempt died with `torch.OutOfMemoryError` while **16 foreign
processes** (PIDs 918xxx / 923xxx, 2.8-7.7 GB each) held the whole 80 GB card.
`require_idle_gpu()` exists precisely for this and must not be overridden with
`FORCE_BUSY_GPU=1` for numbers that will be reported: a speedup measured against
a contended dense baseline is a measurement of the co-tenant, not of the method.
Wait for an idle card.

### Reported scope

`assets/scripts/benchmark_cells.py` is the source of truth: the Ctrl-World 3-view
strip is **not reported** for bimanual/humanoid (superseded by the 4-view canvas
on the same episodes); single_arm keeps its strip because it has no grid variant.
The harness's `CELLS` list already matches that, so §1's humanoid/bimanual
3-view rows are historical context, not cells to re-measure.

### Script env knobs added so the harness can drive these cells

- `run_single_arm_multiview_ctrlworld{,_dicache,_worldcache,_fastercache,_itm}.sh`:
  `CKPT`, `NUM_EPISODES`, `NUM_INFERENCE_STEPS`, `OUTPUT_ROOT`, `INPUT_ROOT`,
  `SKIP_PREPARE` (mirrors the `_pisa.sh` pattern that already had them). Without
  `OUTPUT_ROOT`/`NUM_EPISODES` these five could not be capped to a scratch tree
  at all, so the cell was unmeasurable.
- `run_single_arm_singleview_dreamgen_{static16fps,dicache,worldcache,fastercache}.sh`:
  `MAX_EPISODES` (caps the prepared batch JSON, same mechanism as `_sito.sh`), and
  `INPUT_ROOT` is now overridable on the dense script so a scratch control run
  cannot overwrite the canonical dense batch JSONs.

---

## 5. Follow-ups that need code, not just GPU time

1. Add `--use-sito/--use-itm` to the DreamDojo canonical infer scripts
   (`infer_{humanoid_singleview_full_episode,single_arm_singleview_autoregressive,
   bimanual_singleview_single_segment}.py`) so DD pruning becomes comparable, then
   create the two missing `run_bimanual_singleview_dreamdojo_{sito,itm}.sh`.
2. Create `run_humanoid_multiview_ctrlworld{,_4view_grid}_fastercache.sh` (thin
   mirrors of the bimanual ones) to close the FasterCache column.
3. Score the DreamDojo humanoid/bimanual SV video-only rollouts to fill gap (b).
4. Run FasterCache on Ctrl-World single_arm MV (cell currently has no output).
5. **Re-run Ctrl-World single_arm SiTo/ITM quality at 50 steps.** Those two
   scripts default to `NUM_INFERENCE_STEPS=25` while dense uses 50, so the PSNR
   already in the table (SiTo 21.98/21.52, ITM 14.91/13.66) was produced at half
   the step budget — the *quality* comparison is as apples-to-oranges as the
   timing one. The timing job forces 50 for both, but the canonical rollouts on
   disk still hold the 25-step quality.
6. **Regenerate `dense/single_arm/output/multiview/ctrlworld/makovian/all_summary.json`**
   from its 128 per-episode `metrics.json` — it currently holds a 10-episode
   `pi05` policy-predicted run, not the 128-episode replay run it sits next to.
