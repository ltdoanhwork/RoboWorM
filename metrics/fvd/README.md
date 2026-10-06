# FVD — Fréchet Video Distance

**Source:** T. Unterthiner, S. van Steenkiste, K. Kurach, R. Marinier, M. Michalski,
S. Gelly. *Towards Accurate Generative Models of Video: A New Metric & Challenges.*
arXiv:1812.01717, 2018.

FVD fits a Gaussian to the I3D features of a set of real clips and another to a set of
generated clips, and reports the Fréchet distance between them. **Lower is better.**
It is a **set** statistic: one number per cell, never per rollout (paper App. B.3.3).

```
metrics/fvd/
├── __init__.py   re-exports everything below: `from metrics.fvd import FVD, frechet, ...`
├── core.py       I3D loading + window features, 5 fps decimation, mosaic split, Fréchet, FVD metric
├── pool.py       shared real-pool protocol: dedup, tile→camera map, budgeted n, split-half floor
└── README.md     this file
```

## 1. Setup

Interpreter: `models/dreamgen/.venv` (torch + scipy + cv2). No extra package.

I3D weights: the TorchScript export of the Kinetics-400 I3D used by StyleGAN-V / DIGAN
(400-d pre-logit features, called with `rescale=False, resize=False`).

| file | default path | override |
|---|---|---|
| `i3d_torchscript.pt` (49 MB, md5 `8d45542982dc92282a73cb852e3c53d4`) | `checkpoints/metrics/fvd/i3d_torchscript.pt` | `FVD_I3D_CKPT=/path/to.pt` |

To fetch it on a new machine (the URL the StyleGAN-V code downloads from):

```bash
mkdir -p checkpoints/metrics/fvd
wget -O checkpoints/metrics/fvd/i3d_torchscript.pt \
  'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
md5sum checkpoints/metrics/fvd/i3d_torchscript.pt   # compare with the table above
```

## 2. What one clip contributes

1. Decode with cv2 (the decoder every published number used).
2. **Decimate to 5 fps** by nearest-frame selection, never interpolation
   (`core.resample_fps`; 5 fps is the lowest ground-truth rate of any cell, so clips are only
   ever decimated). Interpolated frames would be synthetic motion, and motion is what I3D sees.
3. **Multi-view:** cut the mosaic into camera tiles (`core.split_mosaic`; 1×4 / 1×3 strips of
   320×192 for Ctrl-World, 2×2 grid of 384×216 for DreamGen).
4. Cut into **consecutive, non-overlapping 16-frame windows**; drop the trailing partial window.
5. Resize each window bilinearly to **224×224**, map to [−1, 1], run I3D → one 400-d vector.

## 3. The shared real pool (`pool.py`)

* **Key:** `(embodiment, camera, regime)`. Every world model and every acceleration method of
  that setting is scored against the **same** pool, so the column is comparable across them.
* **Dedup:** the same GT video is hard-linked into every method tree. Pooling by path inflated
  one pool ~10× (56,668 windows from ~150 real videos). `ref_identity()` strips the
  `sampling_dataset/<method>/` prefix so each real video enters once.
* **Camera identity** comes from data (`assets/scripts/fvd_camera_match.py`), not from stored
  view names — humanoid Ctrl-World metadata carries none. `TILE_CAMERA` is the result;
  `RESTRICT_CAMERAS` drops cameras whose own GT sits far above the pool's split-half floor
  (a framing mismatch between models, `assets/scripts/verify_shared_pool_fairness.py`).
* **Multi-view cell value** = mean over the cameras common to every configuration of the cell
  (`mean_over_common_cameras`).
* **Single-view** tracks keep a **per-configuration** reference (each model vs its own GT):
  `assets/scripts/pool_singleview_fvd.py` measured that only bimanual can share a pool
  (cross-model real-vs-real FVD 7.6 vs floor 86); humanoid / single-arm real sets differ in
  episodes and framing. Single-view FVD is therefore not comparable across models (Table 2b).
* **Noise floor:** FVD between two disjoint halves of the pool (`SharedRealPool.floor`). It is
  an episode-diversity upper bound on estimator noise, not a value no method can beat.

## 4. Sample-size caveat (read before comparing numbers)

FVD is biased by the number of windows `n`. Re-scoring the 24 dense settings against the same
reference while only varying `n` inflates FVD by a median **4.1× at n = 128**, and matching
`n = 512` changes the rank of **14 of 24** settings (paper Fig. B3,
`assets/scripts/fvd_sample_size_probe.py`). Hence:

* a side with fewer than **64 windows** gets `None` (`FVD_MIN_CLIPS`; the 400×400 covariance is
  too rank-deficient);
* compare FVD only between configurations that share the **same reference and sampling setup**,
  and report `n` (`budgeted()` draws a deterministic equal-`n` subset when needed);
* one value per cell means effective n = 1: no CI and no signed-rank test exist for FVD.

## 5. Running it

Small sets (one cell vs its own GT, or vs a saved pool):

```bash
PY=models/dreamgen/.venv/bin/python
$PY -m metrics.evaluate --pred-root <cell dir> --metrics fvd --out results/metrics/<run>
$PY -m metrics.evaluate --pred-root <cell dir> --ref-template '' --metrics fvd \
    --opt fvd.ref_pool=pool_feats.npz --out results/metrics/<run>   # npz with a `feats` array
```

`--opt fvd.target_fps=null` keeps native fps (the per-configuration DR protocol).

```python
from metrics.fvd import load_i3d, read_video, resample_fps, video_fps, i3d_features, SharedRealPool
i3d = load_i3d("cuda")
feats = i3d_features(resample_fps(read_video(p), video_fps(p)), i3d, "cuda")   # list of (400,)
```

Full benchmark (what the paper tables read; features are stored under
`metrics/WorldScore/results/`):

```bash
PY=models/dreamgen/.venv/bin/python
# per-configuration DR pass (PSNR/SSIM/LPIPS + mosaic I3D features vs dense and vs GT)
MACHINES=1 bash scripts/metrics/run_dr_metrics.sh 0
# multi-view shared pool: camera assignment -> per-camera 5 fps features -> pooled FVD
$PY assets/scripts/fvd_camera_match.py            # -> results/reports/fvd_camera_map.json
MACHINES=1 bash scripts/metrics/_perview_wave_fg.sh 0   # -> metrics/WorldScore/results/fvd_perview/
$PY assets/scripts/pool_perview_fvd.py            # -> results/reports/fvd_perview_pooled.json
$PY assets/scripts/verify_shared_pool_fairness.py # -> results/reports/shared_pool_fairness.json
# single-view: per-configuration reference
$PY assets/scripts/pool_singleview_fvd.py         # -> results/reports/fvd_singleview_pooled.json
# sample-size sensitivity (Fig. B3)
$PY assets/scripts/fvd_sample_size_probe.py       # -> results/reports/fvd_sample_size_probe.json
```
