# Paper asset generators

Every table, figure, and evidence file in `assets/paper/` is produced by a script in
this folder. Nothing in the paper is hand-typed: numbers come from `datasets/`,
`sampling_dataset/`, and `metrics/`, and a gate re-checks the prose against them.

Run everything from the **repository root** (each script resolves the repo root from
its own location, so the working directory only matters for relative CLI arguments):

```bash
python3 assets/scripts/<name>.py
```

## Dataset section (Method §Dataset + Appendix §Benchmark Dataset)

Run in this order — the later steps consume the JSON the earlier ones emit.

| # | Script | Writes |
|---|--------|--------|
| 1 | `make_benchmark_stats.py --json metrics/reports/benchmark_stats.json` | skill/scene/prompt statistics |
| 2 | `make_dataset_appendix_tables.py` | `appendix/tables/dataset_{cells,taxonomy}.tex`, `dataset_appendix_evidence.json` |
| 3 | `audit_sampling_distribution.py --json metrics/reports/sampling_audit.json` | audited rollout counts, verified against the design scope |
| 4 | `make_cell_matrix.py` | `metrics/reports/cell_matrix.csv`, `docs/datasets/cell_matrix.md` |
| 5 | `plot_benchmark_figures.py` | `paper/figures/fig_{coverage,composition,diversity}` |
| 6 | `make_hero_figure.py` | `paper/figures/fig_hero` — teaser: frame montage + 4 stat panels |
| 7 | `check_dataset_appendix_claims.py` | **gate** — fails if prose numbers or style drift |

Step 7 is the important one: it traces every multi-digit number in the dataset prose
back to measured evidence and rejects code paths / raw filenames / internal directory
names leaking into the paper. Run it after editing that prose.

`make_hero_figure.py` needs `ffmpeg`/`ffprobe` and samples one frame per episode
with an **equal quota per (embodiment, view) cell**, caching them in
`metrics/reports/_hero_frames/` (gitignored, regenerable). Sampling per *unit*
skews badly — bimanual is split into ~51 task directories while humanoid and
single-arm have one or two, which once produced a montage that was 56/60 bimanual.

Two background styles:

```bash
# default: sparse rotated-thumbnail cloud on white (BridgeData V2 teaser look)
python3 assets/scripts/make_hero_figure.py --scatter-count 300

# alternative: dense tiled mosaic behind translucent cards
python3 assets/scripts/make_hero_figure.py --bg-style grid --dim 1.12 --blur 1.0
```

In `scatter` mode the thumbnails are placed in an **annulus** (inner radius 0.62)
with size and opacity decaying outward, so the cloud thins toward the edges
instead of forming a hard ring, and the centre stays clear for the panels. Knobs:
`--scatter-count`, `--fade`, `--frames`, `--seed`. `grid` mode keeps `--dim`/`--blur`.

The run also exports the **action examples** — one ground-truth frame per
manipulation primitive — to `metrics/reports/_hero_frames/primitives/`, in two
versions per action: the plain crop (`1_pick_place.jpg`) and a **self-explanatory
copy with the action name and counts burnt in** (`labeled_1_pick_place.jpg`), plus
`captions.md` and `index.tsv` (episode, camera, legibility score, instruction).

Selection is a three-stage funnel, each stage fixing a failure the previous
version shipped:

1. **On-label (hard filter).** The instruction must contain the primitive's own
   verb. The classifier maps synonyms for *counting* (`spoon` → pour, `brush` →
   wipe / clean), which once illustrated "pour" with a gripper dropping a block
   into a bowl. This is a filter, not a score term — when it was only weighted,
   legibility outvoted it and that frame came back.
2. **Camera ranking.** Head / high / exterior / global first, **wrist last**.
   Plain alphabetical sorting picked `cam_front_rgb` on the humanoid rig — a
   body-mounted camera angled down at the robot's own grippers.
3. **Legibility score** over ~12 on-label candidates × 5 timestamps:
   `sharpness × motion-size-band × centrality × compactness`. Sharpness alone
   (variance of Laplacian) removes motion blur; the motion band (2–35% of frame)
   rejects both a speck-sized arm and a fully moving ego view; compactness
   rejects diffuse change, e.g. a swaying curtain that scored high with no robot
   visible.

Known limit: the score measures *where pixels change*, not *whether the gripper
is in frame*, so a frame where the arm has just left the view can still win.

**The counts burnt into the strip do not sum to the benchmark total, by design.**
Adding the nine "episodes" numbers gives less than the $1{,}605$ episode
instances. `captions.md` ends with the full reconciliation, regenerated on every
run: shown + the five primitives the strip has no room for
(handover, stack, hang/cover, draw/trace, sort) + the episodes with no usable
instruction (humanoid singleview episodes labelled by scene, e.g. `grid_Home`,
plus two `No action` placeholders). Likewise the instruction column sums to
fewer than the 507 distinct instructions. Each episode is attributed to one
primary primitive, so the groups partition rather than overlap — the larger
multi-label mention counts in the appendix are a different view of the same data.
Re-run `make_hero_figure.py` after any change to the episode set: the exact
residuals are recomputed there rather than restated here.

One near-miss total worth not confusing: **1,605** is *also* the number of dense
DreamGen rollouts (of 3,166 dense across three models). That coincidence is not
accidental — DreamGen is the only model covering all 12 cells, so it rolls out
every episode instance exactly once.

`make_stat_donuts.py` writes the three headline donuts as **separate files with a
transparent background** (no white box), for slides or a project page:

```bash
python3 assets/scripts/make_stat_donuts.py
# -> figures/stat_embodiment.{png,pdf}, stat_view.*, stat_regime.*
```

### Why step 3 is the only valid source of rollout counts

A plain directory walk over `sampling_dataset/` **overcounts**: it picks up stale
output trees (superseded iteration tags, `_fps_compare` debug dirs) and treats the
two Ctrl-World tree variants as two separate models. `audit_sampling_distribution.py`
applies those exclusions, merges the Ctrl-World variants into one model column, and
then checks the result against the intended design scope — it exits non-zero on any
mismatch. The figure script refuses to plot unless `scope_matches_design` is true, so
a figure can never show a number the audit rejects.

Current audited totals: **3,166** dense rollouts (DreamGen 1,605 / DreamDojo 756 /
Ctrl-World 805) and **29,343** rollouts including all nine acceleration methods.

## Results tables

| Script | Purpose |
|--------|---------|
| `table_dense_nr.py` | per-stratum medians with bootstrap CI + paired GT-delta |
| `table_paper_main.py` | main grid, both unpaired-absolute and paired-delta views |
| `table_main_results.py` | main-results matrix: (model, embodiment, view) x method |
| `table_main_options.py` | candidate normalisations for the main table slot |
| `make_accel_config_table.py` | appendix expansion: block x model x camera track x embodiment |

These four need **Python >= 3.12** (they use backslashes inside f-string
expressions, which is a syntax error on 3.10/3.11). The system `python3` here is
3.10, so run them with a 3.12 interpreter, e.g.:

```bash
models/dreamgen/.venv/bin/python assets/scripts/table_main_results.py --run <run-dir>
```

The dataset-section scripts above run fine on the system `python3`.

## Qualitative figures

| Script | Purpose |
|--------|---------|
| `make_fig_embodiments.py` | embodiment overview + teaser variant |
| `make_markov_figure.py` | Markovian vs non-Markovian task contrast |
| `make_skill_figure.py` | skill taxonomy illustrated with real frames |
| `make_speedup_star_chart.py` | speedup/quality star chart |
| `make_checkpoint_finetune_comparison.py` | checkpoint / finetune comparison |
| `make_teaser_tiles.py` | teaser tiles (also a helper module for the packs below) |
| `make_figure_slots.py` | per-camera-slot PNGs for the teaser diagram |
| `make_task_image_pack.py` | one folder per task, timeline frames as PNGs |
| `make_humanoid_short_prompt_pack.py` | shortest-instruction humanoid pack |
| `make_real_video_gallery.py` | ground-truth video gallery |
| `make_method_comparison_gallery.py` | one clip across every acceleration method |

## Appendix C: training & acceleration implementation

Generators for `assets/paper/appendix/3_implemetation_details.tex`. Run them in
this order; the gate must be last, because it re-derives every numeric claim in
the prose from the JSON the others emit and fails the build on any mismatch.

| # | Script | Emits |
|---|--------|-------|
| 1 | `make_finetune_config_table.py` | `metrics/reports/finetune_config.json`, `appendix/tables/{finetune_hyperparams,generation_config}.tex` |
| 2 | `make_training_data_table.py` | `metrics/reports/training_data_prep.json`, `appendix/tables/training_data_prep.tex` |
| 3 | `make_impl_config_tables.py` | `metrics/reports/impl_accel_configs.json`, `appendix/tables/accel_hyperparams_{token,cache,sparse}.tex` |
| 4 | `make_time_budget_figure.py` | `metrics/reports/time_budget.json`, `figures/fig_time_budget.pdf` |
| 5 | `make_knee_curves_figure.py` | `metrics/reports/knee_curves.json`, `figures/fig_knee_curves.pdf` |
| 6 | `make_finetune_qualitative_figure.py` | `metrics/reports/finetune_qualitative.json`, `figures/fig_finetune_qualitative.pdf` |
| 7 | `make_timing_pass_table.py` | `metrics/reports/timing_pass.{json,csv}`, `appendix/tables/timing_pass.tex` |
| 8 | `check_impl_appendix_claims.py` | nothing; exits non-zero if the prose and the data disagree |

Notes:

- `make_finetune_qualitative_figure.py` renders the base-vs-finetuned comparison
  behind the PSNR-gain claim. It only uses episodes where a base-checkpoint and a
  finetuned rollout exist at the *same* episode id, samples all three videos at
  the same fractions of the shortest of them (so panels are time-aligned rather
  than indexed by each clip's own length), and reads PSNR from each rollout's
  recorded metrics instead of recomputing it, so the figure and the tables cannot
  disagree. Frames are pulled to `/tmp/ft_qual_frames`.
- `make_training_data_table.py` measures the corpora on disk with `ffprobe`
  (frame counts come from container metadata, not `-count_frames`, which would
  decode every byte over the network filesystem). Measurements are cached in
  `metrics/reports/.training_data_cache.json`; pass `--remeasure` to refresh.
  It also reports discrepancies against the pipeline's own comments and any
  corpus whose on-disk frame rate disagrees with the rate the dataloader
  reports, both of which the appendix discloses rather than silently fixes.
- `make_timing_pass_table.py` consumes the JSONL written by
  `scripts/tools/tune/_bench_uniform_timing.sh`, which must run on an idle GPU; it refuses
  to start if another process is resident, since a contended measurement
  describes the contention.
- `fig_accel_taxonomy.pdf` is hand-drawn and is **not** generated. The `\input`
  target `appendix/figures/fig_accel_taxonomy.tex` holds a placeholder until the
  PDF is exported.

### Previewing the PDF

This box has no system TeX, so `build_paper_preview.sh` fetches `tectonic` (one
static binary, caches only the packages the document needs) on first use:

```bash
bash assets/scripts/build_paper_preview.sh appendixC --png   # one section
bash assets/scripts/build_paper_preview.sh full              # whole paper
```

Output lands in `assets/paper/_build/` (git-ignored). Section previews reuse the
real preamble from `iclr2027_conference.tex`, so spacing and fonts match the
paper rather than approximating it. The script runs two passes (so `\cref`
resolves) and then prints a layout report: undefined references, the worst
overfull boxes, and any float too large for its page — the last of which silently
*clips* a table in the PDF, so it should always be treated as an error rather
than a warning. `--png` also rasterises each page via `pymupdf`
(`pip3 install --target /tmp/pylibs pymupdf`).

The wrapper must load `\usepackage[OT1]{fontenc}`. `times.sty` is an OT1-era
package: without that encoding the Times fonts fail to load and `tectonic`
silently falls back to Latin Modern *and* drops the bold series, so every
`\textbf` renders at regular weight and the preview misrepresents the paper. A
quick check on any rebuild — bold Times reports as `NimbusRomNo9L-Medi`, not
`...-Bold`:

```bash
PYTHONPATH=/tmp/pylibs python3 -c "
import pymupdf, collections
d = pymupdf.open('assets/paper/_build/appendixC.pdf')
c = collections.Counter(s['font'] for p in d
    for b in p.get_text('dict')['blocks'] for l in b.get('lines', [])
    for s in l['spans'])
print(c.most_common(6))"
```

## Note on moving these files

`make_teaser_tiles`, `make_benchmark_stats`, `make_task_image_pack`, and
`make_real_video_gallery` are imported as modules by the packs/galleries, which add
their own directory to `sys.path`. Keep these scripts together in one folder, and if
the folder depth changes, update the repo-root expression at the top of each file.
