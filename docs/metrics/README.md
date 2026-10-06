# Metrics

The paper's metric set (Sec. 3, App. B) and where each one lives. The code-level
reference — registry, CLI flags, Python API, checkpoint overrides — is
[metrics/README.md](../../metrics/README.md); this page is the "how do I score the
benchmark" guide.

| paper metric | measures | direction | code | env |
|---|---|---|---|---|
| **KIVA** (Intra-View Kinematic Stability) | embodiment hallucination: rigidity + jerk violation rate of robot keypoints, as excess over the real-video anchor | ↓ | `metrics/kiva/` | `hdmi` conda env (`HAL_PYTHON`) |
| **RIGIS** (Robotic Inter-view Geometry & Intrinsic-Extrinsic Stability) | cross-view 3D consistency + temporal stability of camera extrinsics / intrinsics (VGGT-Omega) | ↑ | `metrics/rigis/` | `metrics/rigis/.venv` (uv) |
| **Visual Integrity** | CLIP-IQA+ (pyiqa `clipiqa+`), per-clip p10 over frames, ×100 | ↑ | `metrics/quality/quality_control_probe.py`, NR wave `scripts/metrics/run_dense_nr.py` | `models/dreamgen/.venv` |
| **Text Alignment** | CLIPScore (ViT-B/16), task clause only, [0,100] | ↑ | `metrics/clip_score.py` | `models/dreamgen/.venv` |
| **FVD** | I3D, non-overlapping 16-frame windows at 5 fps, shared per-camera real pool | ↓ | `metrics/fvd.py` | `models/dreamgen/.venv` |
| **PSNR / SSIM / LPIPS vs dense** | fidelity of an accelerated rollout to the dense rollout of the same episode | ↑ ↑ ↓ | `metrics/fidelity.py`, DR wave `scripts/metrics/compute_dr_metrics.py` | `models/dreamgen/.venv` |
| **Speedup** | dense / method internal per-segment timer, paired episodes | ↑ | see [../inference/README.md](../inference/README.md#measuring-speedup) | — |

Full write-ups:

- [hallucination_metric.md](hallucination_metric.md) — KIVA: readers, anchors, launcher, aggregation.
- [multiview_consistency_metric.md](multiview_consistency_metric.md) — RIGIS: frame extraction, scoring, aggregation.
- [phyworld_nr_adapted_metrics.md](phyworld_nr_adapted_metrics.md) — Visual Integrity / Text Alignment
  (and the adapted RBench / WorldScore dynamics probes used as RQ baselines).
- [nr_paper_output_schema.md](nr_paper_output_schema.md) — stored fields, validity gates, aggregation rules of the NR wave.
- [phyworld_metrics.md](phyworld_metrics.md) — inventory of every metric in the repo and its JSON key.

## One CLI for the registered metrics

```bash
PY=models/dreamgen/.venv/bin/python
export HF_HUB_OFFLINE=1
CELL=sampling_dataset/dense/single_arm/output/singleview/dreamgen/makovian

# PSNR / SSIM / LPIPS + CLIPScore over one cell
$PY -m metrics.evaluate --pred-root $CELL --metrics fidelity clip_score --out results/metrics/dense_sa_sv_mk

# FVD (I3D, 16-frame windows, target_fps=5 default)
$PY -m metrics.evaluate --pred-root $CELL --metrics fvd --out results/metrics/dense_sa_sv_mk_fvd

# KIVA, with the real-video anchor scored in the same call (launches the hdmi env)
$PY -m metrics.evaluate --pred-root $CELL --metrics kiva \
    --opt kiva.embodiment=single_arm --opt kiva.score_ref=true --out results/metrics/dense_sa_sv_mk_kiva

# RIGIS over extracted multiview episode dirs (<ep>/view1, <ep>/view2, ...)
$PY -m metrics.evaluate \
    --pred-root sampling_dataset/multiview_consistency/frames/dense/bimanual/ctrlworld/non_makovian/<task> \
    --pred-glob '*' --ref-template '' --text-template '' --metrics rigis --out results/metrics/dense_bi_rigis
```

Each `--out` dir gets `<metric>.json`, `per_sample.csv` and `summary.json`.

## Full-benchmark sweeps (what the paper tables read)

```bash
# KIVA: anchor first, then every method variant (writes results/hallucination/)
ROLE=gt VARIANTS="dense pisa radial svg1 svg2 sito itm worldcache dicache fastercache" bash metrics/kiva/run.sh --tmux
VARIANTS="dense pisa radial svg1 svg2 sito itm worldcache dicache fastercache" bash metrics/kiva/run.sh --tmux
python metrics/kiva/aggregate.py --help

# RIGIS: extract frames per variant, then score (writes results/multiview_consistency/)
python metrics/rigis/extract_frames.py \
    --videos-root sampling_dataset/pisa/bimanual/output/multiview/ctrlworld_4view_grid \
    --out-root    sampling_dataset/multiview_consistency/frames/pisa/bimanual/ctrlworld \
    --video-name pred_all_views.mp4 --layout 2x2 --fps 2
VARIANTS="dense pisa radial svg1 svg2" bash metrics/rigis/run.sh --tmux
python metrics/rigis/aggregate.py --help

# Visual Integrity + Text Alignment (no-reference wave over every rollout, GPU, shardable)
bash scripts/metrics/run_dense_nr.sh            # see header for MACHINES / WORKERS / --methods

# PSNR / SSIM / LPIPS / FVD vs the paired dense rollout (distortion-reference wave)
MACHINES=1 bash scripts/metrics/run_dr_metrics.sh 0
models/dreamgen/.venv/bin/python scripts/metrics/compute_dr_metrics.py --out <run> --fvd-only
```

Protocol rules that the tables depend on:

- Fidelity / PhyFid-DR reference = the **dense prediction** of the same episode (same
  resolution and fps), never the GT video; it is computed for accelerated methods only.
- FVD for multiview cells uses a **shared per-camera real pool** (`assets/scripts/fvd_camera_match.py`,
  `pool_perview_fvd.py`, `verify_shared_pool_fairness.py`); see `.cursor/rules/fvd-table-reporting.mdc`.
- NR scores (Visual Integrity, Text Alignment) are aggregated only within the same model,
  embodiment, camera mount and split ([nr_paper_output_schema.md](nr_paper_output_schema.md)).
- Human-alignment studies (App. D): human labels are in
  `sampling_dataset/data_for_web/{geometry,hallucinate}_final_labels.json`; agreement with
  KIVA / RIGIS / VLM judges is computed by the RQ3 producers listed in
  [../reproduce/README.md](../reproduce/README.md).
