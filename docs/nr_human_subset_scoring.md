# Scoring the human-rated subsets with the adapted NR metrics

Goal: run the adapted PhyWorld-NR metrics on the two sets that carry human labels, so the
metrics can be correlated against human judgement.

    geometry     300 clips   sampling_dataset/data_for_web/geometry_final_labels.json
    hallucinate  291 videos  sampling_dataset/data_for_web/hallucinate_final_labels.json

## Status

| set | rated | resolvable to a real mp4 | ready to score |
|---|---|---|---|
| geometry | 300 | **300 / 300** | **yes** |
| hallucinate | 291 | 291 named, but the mp4 is a RE-RENDER (see below) | **no — blocked** |

## geometry: ready

`clips[].video_path` is relative to `sampling_dataset/`, e.g.

    dense/bimanual/output/multiview/dreamgen/non_makovian/iter_000120000/
      brush_screws_into_dustpan_human_brush_left_hold/episode_000039/full_pred.mp4

All 300 exist on disk (verified). Each record also carries the human label and the machine
features already computed for the geometry check:

    human_score (1|2|3), human_rating (bad|medium|good), consensus, conflict,
    n_raters, rater_scores, checker_scores, arbiter_score,
    machine_score, combined_score,
    geometric_consistency, pose_stability, disjoint_frame_ratio

Embodiment split is exactly 100 bimanual / 100 humanoid / 100 single_arm, and the human
scale is well balanced:

    human_score   1 (bad) 93   |   2 (medium) 100   |   3 (good) 107
    n_raters      2 on every clip
    conflict      132 of 300 had the two raters disagree (resolved by a checker)
    view          multiview on all 300
    model         dreamgen 198, ctrlworld 102

Method trees present: dense 151, worldcache 86, tmp 39, augment 24 — i.e. the set
deliberately mixes dense, an acceleration method, and weak augmentations so the human scale
spans good-to-bad.

### Two things to handle before scoring

**1. 39 clips live under `tmp/`** (`sampling_dataset/tmp/lowquality_multiview/...`). Those
are scratch renders, not canonical benchmark outputs. They were rated by humans so they
belong in a correlation study, but they must not be folded into any per-method benchmark
table.

**2. 4 clips are the GT-stacked `full_pred.mp4`.** The DR wave established that
Ctrl-World's `full_pred.mp4` is the ground truth stacked ON TOP of the prediction (verified:
PSNR(top half, GT) = 38.4 dB vs PSNR(bottom half, GT) = 23.7 dB). Four rated clips are such
a file — identified as `full_pred.mp4` that has a `pred_all_views.mp4` sibling, all four
`model=ctrlworld`. Scoring them as-is would compute every NR metric on a frame that is half
ground truth, which inflates Visual Integrity and corrupts the motion metrics.

Fix: for those 4, score the `pred_all_views.mp4` sibling instead. The remaining 98
Ctrl-World clips in the set already point at `pred_all_views.mp4` directly, so this only
normalises an inconsistency in how the web set was assembled.

## hallucinate: BLOCKED, and why

`videos[].video_path` is a WEB path that only exists in the HuggingFace bucket:

    bimanual/markov/high/1.mp4
    -> https://huggingface.co/buckets/twanghcmut/hallucinate-bench/resolve/web/300vid/...

The local `hallucinate_bench/300vid/` directory holds only metadata (`key.json`,
`frames.json`, VLM results) — **no mp4s**. The mapping back to disk is in
`300vid/<robot>/key.json`, whose `videos` dict covers **291/291** of the rated paths and
gives `bundle` (method tree) + `video` (flattened episode id), e.g.

    bimanual/markov/high/1.mp4 -> bundle=itm  video=insert_usb_cable_fix_parts__episode_000029

Matching those tokens against the method trees resolved 286/291 paths — but **218 of them
were ambiguous** (more than one candidate file), so the mapping was verified rather than
trusted. Reading the real frame count of each candidate and comparing with the rated
`n_frames`:

    sample of 60:  20 match, 40 MISMATCH

**Two thirds of the mapping is wrong**, so it cannot be used: a wrong pick silently pairs a
human score with a different clip, and every correlation computed afterwards would be
meaningless while looking perfectly healthy.

The reason the frame counts cannot match is visible in the label file itself. `n_frames`
takes only a handful of distinct values shared by many videos:

    185: 87 videos,  277: 63,  1013: 55,  93: 27,  645: 26,  369: 16,  461: 11,  1657: 2

and it does not equal `n_segments x segment_len_frames` either (64 x 16 = 1024 vs
n_frames = 1013). The reference file explains it: *"The clip pauses at every segment
boundary; the rater scores that segment."* So the rated mp4 is a **re-render of the source
clip with pauses inserted** — a different video from anything in `sampling_dataset/`, with
a different length and a different frame sequence.

Consequence: scoring a source clip would not be scoring what the humans saw. Motion-based
metrics (Actor Activity, Active Motion Continuity) are the ones most affected — inserted
pauses change per-frame flow directly, which is exactly what they measure.

## To unblock hallucinate, one of these is needed

1. **The rated mp4s themselves.** Download the 291 files from the HF bucket
   (`hallucinate-bench`, branch `web`, prefix `300vid/`) and score those. This is the only
   option that scores exactly what the raters saw.
2. **The render recipe.** If the pause-insertion script exists, regenerate the 291 videos
   from their sources and score the regenerated files.
3. **An explicit path map** (web path -> source mp4) produced by whoever built the web set,
   plus a decision to accept that the source clip is a proxy for the paused render. Only
   defensible for the appearance metrics (Visual Integrity, Text Alignment); the motion
   metrics should be withheld under this option.

## Run command for geometry (ready now)

The NR runner takes an explicit id list, so the subset is scored without touching the full
manifest:

```bash
# build a manifest restricted to the 300 rated clips, then score it
models/dreamgen/.venv/bin/python scripts/nr_score_subset.py \
  --labels sampling_dataset/data_for_web/geometry_final_labels.json \
  --out metrics/WorldScore/results/nr_human_geometry
```

Metrics produced per clip: Visual Integrity, Text Alignment, Actor Activity, Active Motion
Continuity (the four reported columns; Active Rate is computed as the AMC gate only).

Correlation against `human_score` is then a straight Spearman per metric, with the caveat
that `human_score` is a 3-level ordinal (1/2/3) so ties dominate — report rho with its
bootstrap CI and the n per level (74 bad / 78 medium / 139 good for hallucinate; geometry's
distribution should be checked the same way before quoting anything).
