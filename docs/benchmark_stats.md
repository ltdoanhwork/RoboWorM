# Benchmark statistics — measured, not estimated

Every number here is produced by `assets/scripts/make_benchmark_stats.py` reading
`datasets/` directly (machine-readable copy: `assets/figures/benchmark_stats.json`).
Regenerate with:

```bash
python3 assets/scripts/make_benchmark_stats.py --json assets/figures/benchmark_stats.json
```

Counting rule: an episode is counted only if it **has a video on disk**. Each
LeRobot unit is a dir with `meta/episodes.jsonl`; `bimanual` and
`humanoid/multiview` are partitioned into per-task units that **reuse episode
indices**, so everything is keyed by `(unit, episode_index)`. Keying by index
alone silently collapses them (this is why a naive count gives 524 prompts
instead of 507).

## Headline numbers

| quantity | measured |
|---|---|
| source datasets | **5** |
| embodiments | **3** (single-arm, bimanual, humanoid) |
| view tracks | **2** (singleview, multiview) |
| cells (embodiment x track x split) | **12** |
| task units (LeRobot dirs) | **120** |
| **evaluation episodes** | **1,605** |
| **video files** (episodes x synced views) | **3,805** |
| **distinct language prompts** | **507** |
| **manipulation skills** (canonical primitives) | **14** |
| scenes (GR1 humanoid, explicit `grid_*` labels) | **38** |

## Per-cell breakdown

| embodiment | track | split | tasks | episodes | videos | views | prompts | skills |
|---|---|---|---|---|---|---|---|---|
| single_arm | singleview | makovian | 1 | 120 | 120 | 1 | 2 | 2 |
| single_arm | singleview | non_makovian | 1 | 120 | 120 | 1 | 2 | 2 |
| single_arm | multiview | makovian | 1 | 128 | 384 | **3** | 126 | 8 |
| single_arm | multiview | non_makovian | 1 | 129 | 387 | **3** | 127 | 12 |
| bimanual | singleview | makovian | 37 | 150 | 150 | 1 | 34 | 8 |
| bimanual | singleview | non_makovian | 17 | 150 | 150 | 1 | 17 | 4 |
| bimanual | multiview | makovian | 37 | 150 | 600 | **4** | 34 | 8 |
| bimanual | multiview | non_makovian | 17 | 150 | 600 | **4** | 17 | 4 |
| humanoid | singleview | makovian | 1 | 130 | 130 | 1 | 118 | 1 |
| humanoid | singleview | non_makovian | 1 | 130 | 130 | 1 | 73 | 3 |
| humanoid | multiview | makovian | 2 | 98 | 392 | **4** | 2 | 1 |
| humanoid | multiview | non_makovian | 4 | 150 | 642 | **4** | 6 | 1 |
| **total** | | | **120** | **1,605** | **3,805** | | **507** | **14** |

## Skill taxonomy (by #distinct prompts)

Derived by splitting every measured prompt on clause boundaries (`,` / `and` /
`then`) and mapping each clause's leading verb (+ particle, + `use X to VERB`
pattern) onto a canonical primitive. The lexicon lives in `SKILL` / `PHRASE` in
the script and is auditable: anything unmatched is reported as `unmapped`
rather than silently bucketed (currently **0 / 507**).

Two columns because 59 / 507 prompts (12%) ask for more than one primitive
("*remove the pen from the box **and** put it on the table*"): **primary** counts
each prompt once by its first verb (sums to 507), **mentions** counts every
primitive a prompt asks for (multi-label, sums to 570).

| skill | primary | mentions |
|---|---|---|
| pick & place | 304 | 339 |
| push / slide | 56 | 69 |
| remove / extract | 44 | 45 |
| insert / assemble | 23 | 23 |
| open / close | 21 | 22 |
| fold / unfold | 15 | 17 |
| press / actuate | 15 | 16 |
| wipe / clean | 12 | 16 |
| pour | 7 | 11 |
| stack | 4 | 5 |
| handover | 3 | 3 |
| sort | 1 | 2 |
| hang / cover | 1 | 1 |
| draw / trace | 1 | 1 |
| **distinct skills** | **14** | **14** |

Audit trail (`assets/scripts/make_benchmark_stats.py --json`): the JSON now ships
`prompt_skills`, the per-prompt label list, so every count above is traceable to
the exact strings that produced it. Three fixes came out of that audit:
`transfer` was mis-mapped to `handover` (it is "*transfer beans from the bowl to
the jar*" = pick/pour, so `handover` drops 8 → 3 and is now only the genuine
bimanual `handover *` prompts); `insert / assemble` and `fold / unfold` were
inflated by nouns matched outside verb position (44 → 23, 25 → 15); and the
`hang / cover` bucket existed only via a source typo (`Clover the corner of the
bed`), now normalised in `TYPO` instead of carried as a fake verb.

## Synced views per embodiment (from `meta/info.json` `features`)

Measured camera keys, not assumed:

| embodiment | views | camera keys |
|---|---|---|
| single-arm (DROID) | **3** | `exterior_1_left`, `exterior_2_left`, `wrist_left` |
| bimanual (AIST/ALOHA) | **4** | `cam_high`, `cam_low`, `cam_left_wrist`, `cam_right_wrist` |
| humanoid (RoboCOIN) layout A | **4** | `cam_front_rgb`, `cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb` |
| humanoid (RoboCOIN) layout B | **4** | `cam_high_rgb`, `cam_third_view`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb` |
| single-arm singleview (RoboChallenge) | 1 | `global` (static external) |
| humanoid singleview (GR00T GR1) | 1 | `ego_view_freq20` |

The singleview tracks are **not** "1 of the N multiview cameras" in general:

- `bimanual/singleview` **is** derived — it is `cam_high` of the same episodes
  (`scripts/tools/data/export_bimanual_singleview_cam_high.py` symlinks them), so the two
  bimanual tracks share episodes and prompts.
- `single_arm/singleview` and `humanoid/singleview` are **different source
  datasets** from their multiview counterparts (RoboChallenge vs DROID; GR00T
  GR1 vs RoboCOIN), so they contribute independent episodes.

## Source datasets (5)

| # | embodiment / track | source | id / URL |
|---|---|---|---|
| 1 | single_arm / multiview | DROID (Franka) | `cadene/droid_1.0.1` |
| 2 | single_arm / singleview | RoboChallenge | `Traly/RoboChallenge-lerobot` |
| 3 | bimanual / both | AIST Bimanual Manipulation (ALOHA) | https://aistairc.github.io/aist_bimanip_site/ |
| 4 | humanoid / singleview | NVIDIA GR00T Teleop GR1 (GR1-T2) | `nvidia/PhysicalAI-Robotics-GR00T-Teleop-GR1` |
| 5 | humanoid / multiview | RoboCOIN (AIRBOT MMK2), BAAI — arXiv 2511.17441 | `RoboCOIN/Airbot_MMK2_*`, `RoboCOIN/AIRBOT_MMK2_*` |

Humanoid multiview **layouts A and B are the same source dataset** (RoboCOIN),
differing only in which camera set that task recorded — do not count them as two.

## Figures

- `assets/figures/fig_skills.png|pdf` — 7 skills x 5 frames, verbatim
  instructions, spanning 4 source datasets (`assets/scripts/make_skill_figure.py`).
- `assets/figures/fig_markov_vs_nonmarkov.png|pdf` — Markovian vs non-Markovian
  contrast on one embodiment (`assets/scripts/make_markov_figure.py`).

## Deltas vs. the planning spreadsheet

The plan sheet says "100 videos x 12 cells = 1200"; the delivered benchmark is
**1,605 episodes / 3,805 videos**. Three specific corrections:

1. **single_arm singleview is RoboChallenge, not DROID.** The sheet lists
   `cadene/droid_1.0.1` and the note "take 1 view from 3 given views"; on disk
   `robot_type` is `RoboChallenge`, 480x640 `observation.images.global`, with
   `source_task` / `source_episode` provenance fields. DROID is only the
   multiview single-arm source.
2. **humanoid multiview is all AIRBOT MMK2, no G1.** The sheet lists
   `RoboCOIN/G1edu-u3_pullBowl_storage_bread_unordered_b` for the makovian row;
   the 6 task dirs on disk are all `Airbot_MMK2_*` / `AIRBOT_MMK2_*`.
   (`RoboCOIN/leju_robot_*` in `clone_dataset.md` is likewise cloned-but-unused.)
3. **bimanual = 4 views**, resolving the sheet's open question "Nên dùng 4 view
   hay 3 view?" — all 54 task units expose exactly `cam_high`, `cam_low`,
   `cam_left_wrist`, `cam_right_wrist`.

Also note the per-cell episode counts are not a uniform 100: they range from 98
(humanoid multiview makovian) to 150, and `bimanual` episode counts differ
between the split's *parquet* rows (150) and episodes *with video* (103 / 85).

## Suggested paper phrasing

> The benchmark comprises **1,605 evaluation episodes** (**3,805 videos**,
> counting synchronized views) drawn from **5 source robot datasets** across
> **3 embodiments** — single-arm, bimanual and humanoid — under both single-view
> and multi-view (3-4 synchronized cameras) conditions, and split into Markovian
> and non-Markovian regimes. Episodes carry **507 distinct language
> instructions** spanning **14 manipulation primitives** (pick & place,
> push/slide, remove/extract, insert/assemble, open/close, fold/unfold,
> press/actuate, wipe/clean, pour, stack, handover, ...), and **12% of the
> instructions are compositional**, chaining two or more primitives in a single
> command; the humanoid subset alone is recorded across **38 annotated scenes**
> (retail, kitchen, workshop, office, ...).

Do **not** write "6 source datasets" (RoboCOIN layout A/B are one dataset),
"1,200 videos" (that was the plan target, not the delivered count), or claim
`handover` is a large bucket (it is 3 prompts — `transfer` verbs belong to
pick/pour).
