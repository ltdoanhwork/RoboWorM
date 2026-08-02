# real2sim — reconstructing the scene that is measured

Status: **blocked on segmentation quality, upstream of anything here.** What follows is what
was built, what was measured, and the diagnosis that stopped it. Nothing in this module should
be used for a reported number yet.

## Why this module exists

`force_feasibility` applies Newton to a *proxy* scene: a box for the object, two boxes for the
fingers. That needs no per-object asset, which is why it was chosen. The objection to it is
fair — real contact geometry and real inertia need the real shape — and single-image-to-3D
models now make the shape obtainable automatically, so the "we would need per-object assets"
argument no longer holds.

## What works: object mesh from one masked frame

`external/TripoSR` (vendored, two local patches) turns the object's own masked crop into a
watertight mesh. Measured on `humanoid_singleview/makovian/episode_002346` (mustard bottle):

| quantity | value |
|---|---|
| reconstruction time | **2.0 s** on one A100 |
| mesh | 48 128 verts / 96 244 faces, **watertight** |
| bbox (TripoSR units) | 0.470 x 0.504 x 0.978 (aspect 0.48 : 0.52 : 1.0) |
| frame chosen | 101, picked automatically as max area x (1 - gripper occlusion); occlusion **0.000** |

Watertight matters: it is the precondition for MuJoCo to use the mesh as a collision geom and
for `trimesh` to give a defensible inertia tensor, so this output is directly usable by the
force tier rather than being a picture.

Two patches were needed, both recorded in the vendored source:

- `tsr/models/isosurface.py` — `torchmcubes` is a CUDA extension needing a local build;
  `skimage.measure.marching_cubes` is already installed and is the same algorithm. The two
  differ in axis convention, so the `[2, 1, 0]` permute is applied only for the original
  backend. Getting that wrong silently mirrors every mesh.
- `tsr/utils.py` — `rembg` is imported at module level but only used by `remove_background`,
  which this pipeline never calls: the masks the metric already computes are a better matte
  here, because they are the same masks the physics is measured from.

## What does not work: metric scale, and now we know why

Reconstruction gives shape in a normalised unit cube. It cannot give **metres** — and metres
is what the force tier is most sensitive to, since grip demand is proportional to scale. Three
rulers were tried, each replacing the last after it was measured and found wanting:

| ruler | how it failed |
|---|---|
| gripper **path length** | biased by tracker jitter: +20 mm of noise inflates it **7.05x** (radius of gyration: 1.009x) |
| gripper **radius of gyration** | jitter-invariant, but matched the mask centroid to FK's *end effector*, while the mask is the whole arm — 2.65x inflation, mustard bottle came out 37.5 cm |
| **static arm extent** (`probe_static_scale.py`) | plausible sizes at last (2.2-13.7 cm, all graspable) but **not reproducible** — see below |

The static ruler measures the arm's 3D size per frame, so it needs no differencing (immune to
the jitter that killed path length) and no assumption about which material point the mask
centroid is. Its unknown is where the mask stops proximally, so all contiguous distal subsets
of the FK chain are computed and the answer is reported as their median with their spread as
the uncertainty. That part behaved: **candidate subsets agree to within 8-26%**, so the unknown
cut-off is not a large error term.

### The test it failed, and the diagnosis

Two episodes in this set manipulate **the same object** — `episode_000829` and
`episode_003151` are both "green bok choy". Same object, same robot, independent episodes, so
they must return the same size. They returned **9.1 cm and 2.2 cm: a 4.1x disagreement.**

Decomposing the extent into image-plane versus depth axis localises it:

| episode | arm extent XY | arm extent **Z** | Z/XY | obj/arm ratio (XY) |
|---|---|---|---|---|
| 000829 | 0.857 | 0.461 | 0.54 | 0.241 |
| 003151 | 1.757 | **2.051** | **1.17** | 0.056 |

The `obj/arm` ratio is **dimensionless** — scale cancels out of it — yet it differs by 4.3x.
So the error is not in the ruler's calibration; it is in the quantity being measured.

Rendering the masks shows what it is (`sim2real_out/r2s/mask_compare.png`):

- the "gripper" mask covers **both arms plus background**, not one gripper. Its extent is
  therefore partly the *distance between the two arms*, which is an episode-specific pose,
  not a fixed length. In `003151` it reaches 10.5% of the image and spans 2.05 units in depth
  — larger than its own width — because it unions two arms at two different depths.
- the **object mask is essentially empty in `003151`** (0.39% of image) and does not cover
  the bok choy at all.

## Conclusion

Every scale estimator tried so far computes a different statistic of the *same broken input*,
which is why each failed in a different way while none succeeded. No estimator can repair a
mask that unions both arms with the background, and no reconstruction can be done for an
episode whose object mask is empty. The mesh above is good because frame 101 of `002346`
happens to have a clean object mask (0.000 occlusion); that is not a property that holds.

The fix belongs upstream, in the segmentation tier:

1. separate left/right gripper and select the interacting one, instead of unioning them;
2. validate the object mask per episode and flag or drop the empty ones;
3. only then re-run the static ruler — at which point the two bok choy episodes become a real
   pass/fail test rather than a diagnostic.

## Files

- `probe_static_scale.py` — static arm-extent ruler, with the shift control that shows the
  time alignment is only ~1.6 sigma above chance (which is why the estimator uses a median
  ratio, needing no alignment, rather than a regression over time).

## Honest limitations

- The shift control is reported, not hidden: correlations of +0.7 against a null of
  +0.38 +- 0.19 do **not** establish frame alignment between video and parquet rows.
- The FK-row-to-video-frame map is proportional, an assumption, because the video is
  temporally resampled relative to the parquet.
- Prediction videos have no joint angles, so FK-anchored scale exists only for GT. Whatever
  scale is settled on must be transferred to the prediction through the shared object, as
  `force_feasibility.run_episodes` already does.
