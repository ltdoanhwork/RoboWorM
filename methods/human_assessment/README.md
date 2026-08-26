# Human-assessment visuals (from `thanhnv/human-assessment`)

Original branch: optical-flow magnitude + optional CoTracker3 keypoints, written
as a side-by-side `view1|view2` overlay to help raters see motion.

## What we use for labeling

| piece | used? | why |
|---|---|---|
| Farneback flow heatmap (`optical_flow.py`) | **yes** | no seeds needed; highlights where pixels move |
| CoTracker3 keypoints (`keypoints.py` + `point_picker.py`) | later | needs a human to click seeds per episode |
| PNG `view1/view2/frame_*.png` layout | original only | labeling pipeline decodes mp4 instead |

The labeling builder (`scripts/build_assessment_multiview.py`) crops the two
**static** cameras out of a 3-view strip / 4-view 2×2 grid, runs the flow
heatmap on each, then hstacks them into one clip.

## Original CLI (frame folders)

```
episode/
  view1/frame_0000.png ...
  view2/frame_0000.png ...

python point_picker.py <episode_dir>     # writes seed_points.json
python extract.py <episode_dir>          # writes features.npz (flow + tracks)
python overlay.py <episode_dir>          # writes overlay.mp4
```

Needs the DreamGen venv (working OpenCV): `models/dreamgen/.venv/bin/python`.
