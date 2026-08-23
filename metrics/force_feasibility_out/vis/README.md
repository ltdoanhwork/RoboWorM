# Force-feasibility visualisation

Rendered proxy scenes for the force-feasibility tier: the **same MuJoCo model `newton.py`
queries for contacts**, driven by the **same lifted trajectory** that produced the reported
numbers. Not a re-implementation for pictures — if the render looks wrong, the measurement is
wrong.

Regenerate any of these with:

```bash
MUJOCO_GL=egl python -m metrics.force_feasibility.render_sim \
    --tracks metrics/sim2real_out/dense/humanoid_singleview_dreamgen/makovian/episode_002346/tracks.npz \
    --out metrics/force_feasibility_out/vis/ep002346 --which both
```

The `*_lift.npz` files are cached lifts (mask -> MoGe -> metric 3D), so re-rendering needs **no
GPU and no MoGe pass** (~10 s instead of ~2 min). Delete them or pass `--force-lift` to redo the
vision stage.

## Files

| file | what |
|---|---|
| `ep*_compare.gif` | GT (top) vs pred (bottom), each `video \| fitted sim`. 10 fps, 620 px |
| `ep*_{gt,pred}.mp4` | full quality, 1120x368 @ 16 fps (**gitignored** by the repo-wide `*.mp4` rule) |
| `ep*_{gt,pred}_panel.png` | 3 stills (start / peak score / end) + per-frame score plot |
| `ep*_{gt,pred}*_lift.npz` | cached metric-3D lift, reused by re-renders |

`ep002346_pred_ref_lift.npz` (with `_ref`) is the prediction lifted under the **shared** object
scale; `ep002346_pred_lift.npz` without it is the older independently-scaled lift, kept only
because it is the artefact of the retracted result described below.

## Reading a panel

Orange/yellow box = the manipulated object. Two blue boxes = the gripper's contact bodies.
Checkered plane = the fitted support surface. Everything is inferred from masks + monocular
depth + a metric scale from GR1 forward kinematics — **no CAD assets, no 3D annotation, no
per-object setup**, which is what lets the tier run on every episode of `sampling_dataset`.

Check, in order of how badly each failure corrupts the measurement:

1. **object sits on the plane**, neither floating nor sunk — a wrong `support_z` turns every
   resting frame into a violation;
2. **fingers touch when the video shows contact** — the score is conditioned on contact, so
   frames where the blue boxes drift off the robot's hand are not trustworthy;
3. **object size is plausible in metres and matches between GT and pred** — it sets the moment
   arm and, through the metric scale, the magnitude of grip demand itself.

## Why these renders exist

Point 3 is not hypothetical. These panels are what caught a **2.4x object-size disagreement**
between GT and pred for the same mustard bottle, which invalidated the tier's first headline
result ("prediction needs 3.6x the grip force"). Grip demand is proportional to metric scale
(measured 2.52 / 3.80 / 6.35 / 11.44 at s = 0.5 / 1 / 2 / 4), each side had been scaled by its
own gripper path length, and the "signal" correlated with the scale ratio at **r = +0.943**.
After anchoring both sides to one object size the excess fell from +1.007 to **+0.179 ± 0.172**.

A residual table cannot show a units bug. A picture of the scene can. That is the whole
argument for keeping this in the repo.

## Known limitations, visible in these very GIFs

- **The object barely rotates in sim.** `aliased_frac = 0.994`: orientation from monocular PCA
  is almost never certifiable on this data, so the torque rows are mostly disabled and grip
  demand is driven by translation. Rotation you can see in the real video is *not* reflected.
- **This is a proxy scene, not a digital twin.** The object is a box matched in size and
  orientation; the gripper is two boxes. Detailed meshes would not change the force balance but
  would require per-object assets that `sampling_dataset` does not have, so they would not
  generalise.
- Two of the five scorable episodes sit inside noise. n = 5 is a direction, not a result.
