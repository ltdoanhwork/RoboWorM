# Human-assessment visuals (from `thanhnv/human-assessment`)

Original branch: optical-flow magnitude + optional CoTracker3 keypoints, written
as a side-by-side `view1|view2` overlay to help raters see motion.

## What we use for labeling

| piece | used? | why |
|---|---|---|
| Farneback flow heatmap (`optical_flow.py`) | fallback | used when a curated seed entry has no points |
| CoTracker3 keypoints (`keypoints.py` + `point_picker.py`) | **yes** | initialized from `assets/seed_points.json` |
| PNG `view1/view2/frame_*.png` layout | original only | labeling pipeline decodes mp4 instead |

The labeling builder (`build_assessment_multiview.py`, not shipped with this release) reads the exact
300 paths from `assets/all_videos.md`, crops the two **static** cameras out of a
3/4-view strip or 2×2 grid, then writes a **raw** hstack and a **CoTracker**
hstack at original length. It also exports the 9 examples in
`assets/examples.md` as a public frontend manifest.

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
