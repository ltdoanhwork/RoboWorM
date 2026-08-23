"""Is the robot's configuration recoverable from the pixels at all?

The execution-grounded tier stands or falls on one question: given frames from a
generated video, can we recover the robot state/action that produced them? Paper
§4.4.2 had to answer it for their Franka IDM (71.4% replay success from real data
alone, 95.7% with sim pretraining) before any of their numbers meant anything. This
probe answers it for our embodiments *before* an IDM is trained, so a negative
result costs hours instead of GPU-days.

Method: freeze a CLIP vision encoder, extract features for a frame pair
``(t, t+k)``, and fit closed-form ridge regression to the target. Ridge on frozen
features is a **lower bound** on what a trained IDM achieves; if the lower bound is
already high the task is easy, and if the ridge R^2 for a target is ~0 while the
sanity targets are high, that target is close to unidentifiable from two frames.

Two targets, because measurement showed they are wildly different in difficulty:

``state_abs``   the absolute configuration at time t
``state_delta`` the change over the horizon, ``s[t+k] - s[t]``

Every action space in ``datasets/`` turned out to be an **absolute position target**
rather than a delta (``mean |a_t - s_t|`` is 0.0015 for single_arm singleview and
exactly 0.0 for humanoid multiview, where the action column is a verbatim copy of
the state). So "recover the action" really means "pose the robot from pixels", and
the per-step action is obtained by differencing the recovered state. That is the
easier and better-posed problem, which is why it is measured separately here.

Held-out split is **by episode**, never by frame: neighbouring frames of one episode
are near-duplicates, so a frame-level split leaks and reports a meaningless R^2.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CLIP_DIR = _REPO_ROOT / "checkpoints/openai/clip-vit-base-patch32"

# which camera to read per (embodiment, view_type); exterior/static views only —
# a wrist camera moves with the arm, so its pixels confound pose with viewpoint
DEFAULT_CAMERA = {
    ("single_arm", "multiview"): "observation.images.exterior_1_left",
    ("single_arm", "singleview"): "observation.images.exterior_1_left",
    ("bimanual", "singleview"): "observation.images.cam_high",
    ("bimanual", "multiview"): "observation.images.cam_high",
    ("humanoid", "multiview"): "observation.images.cam_head_rgb",
    ("humanoid", "singleview"): "observation.images.ego_view_freq20",
}

# Which slice of ``observation.state`` a camera can actually see.
#
# Read off each group's ``meta/modality.json``. This matters more than it looks:
# bimanual's state is 42-D = qpos(0:14) + qvel(14:28) + effort(28:42), and velocity
# and joint torque are not readable off two frames while dominating the variance, so
# scoring against the full vector measures mostly noise. Humanoid drops the legs and
# keeps arms plus hands, since the ego view frames a tabletop.
TARGET_DIMS = {
    ("single_arm", "multiview"): [(0, 8)],                      # cartesian 6 + gripper
    ("single_arm", "singleview"): [(0, 7)],
    ("bimanual", "singleview"): [(0, 14)],                      # qpos only
    ("bimanual", "multiview"): [(0, 14)],
    ("humanoid", "singleview"): [(0, 13), (22, 35)],            # l arm+hand, r arm+hand
    ("humanoid", "multiview"): None,                            # no modality.json; use all
}


def select_dims(state: np.ndarray, embodiment: str, view_type: str) -> np.ndarray:
    blocks = TARGET_DIMS.get((embodiment, view_type))
    if not blocks:
        return state
    keep = np.concatenate([np.arange(a, min(b, state.shape[1])) for a, b in blocks])
    return state[:, keep]


def dataset_dirs(embodiment: str, view_type: str, split: str) -> list[Path]:
    """Leaf LeRobot dirs (those with a ``data/``) for this group."""
    base = _REPO_ROOT / "datasets" / embodiment / view_type / split
    if not base.is_dir():
        return []
    if (base / "data").is_dir():
        return [base]
    return sorted(d for d in base.iterdir() if d.is_dir() and (d / "data").is_dir())


def find_video(ds_dir: Path, episode_stem: str, camera_key: str) -> Path | None:
    hits = sorted(ds_dir.glob(f"videos/chunk-*/{camera_key}/{episode_stem}.mp4"))
    return hits[0] if hits else None


def available_cameras(ds_dir: Path) -> list[str]:
    return sorted({p.name for p in ds_dir.glob("videos/chunk-*/*") if p.is_dir()})


class ClipFeatures:
    """Frozen CLIP ViT-B/32 pooled image features."""

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.proc = CLIPImageProcessor.from_pretrained(str(CLIP_DIR))
        self.model = CLIPVisionModel.from_pretrained(str(CLIP_DIR)).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 64) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,D) float32."""
        out = []
        for i in range(0, len(frames), batch):
            px = self.proc(images=list(frames[i : i + batch]), return_tensors="pt")
            px = px["pixel_values"].to(self.device)
            out.append(self.model(pixel_values=px).pooler_output.float().cpu().numpy())
        return np.concatenate(out, 0)


def load_episode(ds_dir: Path, pq: Path, camera_key: str, stride: int, horizon: int,
                 embodiment: str, view_type: str):
    """Return (frames_t, frames_tk, state_t, state_tk) or None if unusable."""
    import imageio.v3 as iio

    vid = find_video(ds_dir, pq.stem, camera_key)
    if vid is None:
        return None
    df = pd.read_parquet(pq)
    state = np.stack(df["observation.state"].values).astype(np.float32)
    state = select_dims(state, embodiment, view_type)

    frames = np.asarray(iio.imread(vid, index=None))
    n = min(len(frames), len(state))
    if n <= horizon + 1:
        return None
    frames, state = frames[:n], state[:n]

    idx = np.arange(0, n - horizon, stride)
    if len(idx) == 0:
        return None
    return frames[idx], frames[idx + horizon], state[idx], state[idx + horizon]


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with intercept. Returns W of shape (D+1, K).

    sklearn is deliberately not used: installing it pulls a numpy upgrade that the
    repo's dependency note says breaks cosmos-predict2/DreamGen.
    """
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    D = Xb.shape[1]
    A = Xb.T @ Xb
    reg = alpha * np.eye(D, dtype=Xb.dtype)
    reg[-1, -1] = 0.0  # do not penalise the intercept
    return np.linalg.solve(A + reg, Xb.T @ Y)


def ridge_predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([X, np.ones((len(X), 1), dtype=X.dtype)], 1)
    return Xb @ W


def r2(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    """R^2 against the *test* mean, per dim and aggregate over total variance."""
    resid = ((pred - true) ** 2).sum(0)
    var = ((true - true.mean(0)) ** 2).sum(0)
    per_dim = 1.0 - resid / np.maximum(var, 1e-12)
    agg = 1.0 - resid.sum() / max(var.sum(), 1e-12)
    return float(agg), per_dim


def run_group(embodiment, view_type, split, camera_key, n_episodes, stride, horizon,
              alpha, test_frac, device, verbose=True):
    dirs = dataset_dirs(embodiment, view_type, split)
    if not dirs:
        return {"error": "no dataset dirs"}

    if camera_key is None:
        camera_key = DEFAULT_CAMERA.get((embodiment, view_type))
        cams = available_cameras(dirs[0])
        if camera_key not in cams:
            if not cams:
                return {"error": "no video dirs"}
            camera_key = cams[0]

    episodes = []
    for d in dirs:
        for pq in sorted(glob.glob(str(d / "data/chunk-*/episode_*.parquet"))):
            episodes.append((d, Path(pq)))
    if not episodes:
        return {"error": "no parquet"}
    # even spread across task dirs rather than the first N of the first task
    step = max(1, len(episodes) // n_episodes)
    episodes = episodes[::step][:n_episodes]

    enc = ClipFeatures(device=device)
    per_ep = []
    for d, pq in episodes:
        got = load_episode(d, pq, camera_key, stride, horizon, embodiment, view_type)
        if got is None:
            continue
        f_t, f_tk, s_t, s_tk = got
        feat_t = enc(f_t)
        feat_tk = enc(f_tk)
        per_ep.append({"feat_t": feat_t, "feat_tk": feat_tk, "s_t": s_t, "s_tk": s_tk})
        if verbose:
            print(f"  {pq.parent.parent.parent.name}/{pq.stem}: {len(f_t)} pairs", flush=True)
    if len(per_ep) < 4:
        return {"error": f"only {len(per_ep)} usable episodes"}

    n_test = max(1, int(round(test_frac * len(per_ep))))
    train, test = per_ep[:-n_test], per_ep[-n_test:]

    def cat(eps, key):
        return np.concatenate([e[key] for e in eps], 0)

    # z-score features on train statistics so one ridge alpha suits every group
    Ftr_t, Ftr_tk = cat(train, "feat_t"), cat(train, "feat_tk")
    mu, sd = Ftr_t.mean(0), Ftr_t.std(0) + 1e-6

    def design(eps, two_frame: bool):
        a = (cat(eps, "feat_t") - mu) / sd
        if not two_frame:
            return a
        b = (cat(eps, "feat_tk") - mu) / sd
        return np.concatenate([a, b, b - a], 1)

    targets = {
        "state_abs": (cat(train, "s_t"), cat(test, "s_t")),
        "state_delta": (cat(train, "s_tk") - cat(train, "s_t"),
                        cat(test, "s_tk") - cat(test, "s_t")),
    }

    result = {
        "embodiment": embodiment, "view_type": view_type, "split": split,
        "camera_key": camera_key, "horizon": horizon, "stride": stride,
        "n_episodes_used": len(per_ep), "n_train_ep": len(train), "n_test_ep": len(test),
        "n_train_pairs": int(len(Ftr_t)), "n_test_pairs": int(len(cat(test, "feat_t"))),
        "feature_dim": int(Ftr_t.shape[1]), "alpha": alpha,
        "state_dim_used": int(per_ep[0]["s_t"].shape[1]),
        "targets": {},
    }

    for tname, (Ytr, Yte) in targets.items():
        entry = {"dim": int(Ytr.shape[1]),
                 "target_std_train": float(Ytr.std()),
                 "target_std_test": float(Yte.std())}
        for feats, label in ((False, "one_frame"), (True, "two_frame")):
            Xtr, Xte = design(train, feats), design(test, feats)
            W = ridge_fit(Xtr, Ytr, alpha)
            agg_tr, _ = r2(ridge_predict(Xtr, W), Ytr)
            agg_te, per_dim = r2(ridge_predict(Xte, W), Yte)
            entry[label] = {
                "r2_train": agg_tr, "r2_test": agg_te,
                "r2_test_per_dim_median": float(np.median(per_dim)),
                "r2_test_per_dim_min": float(per_dim.min()),
                "r2_test_per_dim_max": float(per_dim.max()),
            }
        result["targets"][tname] = entry

    result["within_episode"] = _within_episode(per_ep, alpha)
    return result


def _within_episode(per_ep: list[dict], alpha: float) -> dict:
    """Fit and test inside each episode, on disjoint interleaved frame subsets.

    This is the regime the metric actually runs in. Every generated video resolves to
    its own GT episode (``episode_index.py`` resolves 99.9% of the tree), so the
    decoder never has to transfer across episodes: it can be conditioned on the GT
    episode it is being compared against. Across-episode transfer is the harder
    problem and, for DROID, an ill-posed one — ``camera_extrinsics.exterior_1_left``
    is fixed within an episode (std ~1e-6) but is re-placed between episodes (yaw std
    0.82 rad, position range 1.44 m), so pixels do not determine base-frame pose
    without knowing the camera.

    Alternate frames go to train/test rather than a prefix/suffix cut, so the test
    poses are interpolations rather than an unseen phase of the task; this measures
    "are the pixels informative about pose", not "can it extrapolate in time".
    """
    scores = {"state_abs": [], "state_delta": []}
    for e in per_ep:
        n = len(e["feat_t"])
        if n < 8:
            continue
        tr, te = np.arange(0, n, 2), np.arange(1, n, 2)
        mu, sd = e["feat_t"][tr].mean(0), e["feat_t"][tr].std(0) + 1e-6
        A = (e["feat_t"] - mu) / sd
        B = (e["feat_tk"] - mu) / sd
        X = np.concatenate([A, B, B - A], 1)
        for tname, Y in (("state_abs", e["s_t"]),
                         ("state_delta", e["s_tk"] - e["s_t"])):
            W = ridge_fit(X[tr], Y[tr], alpha)
            agg, _ = r2(ridge_predict(X[te], W), Y[te])
            scores[tname].append(agg)
    return {
        t: {"n_episodes": len(v),
            "r2_mean": float(np.mean(v)) if v else None,
            "r2_median": float(np.median(v)) if v else None,
            "r2_min": float(np.min(v)) if v else None,
            "r2_max": float(np.max(v)) if v else None}
        for t, v in scores.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--groups", nargs="*", default=[
        "single_arm/multiview", "single_arm/singleview",
        "bimanual/singleview", "humanoid/multiview", "humanoid/singleview",
    ], help="embodiment/view_type pairs")
    ap.add_argument("--split", default="makovian")
    ap.add_argument("--camera-key", default=None)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--stride", type=int, default=8, help="frame subsample within an episode")
    ap.add_argument("--horizon", type=int, default=5, help="k in (t, t+k)")
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_res = {}
    for g in args.groups:
        emb, vt = g.split("/")
        print(f"\n=== {g} (split={args.split}) ===", flush=True)
        res = run_group(emb, vt, args.split, args.camera_key, args.episodes,
                        args.stride, args.horizon, args.alpha, args.test_frac, args.device)
        all_res[g] = res
        if "error" in res:
            print("  ERROR:", res["error"])
            continue
        print(f"  camera={res['camera_key']}  eps={res['n_episodes_used']}"
              f" (train {res['n_train_ep']} / test {res['n_test_ep']})"
              f"  pairs={res['n_train_pairs']}/{res['n_test_pairs']}"
              f"  state_dim={res['state_dim_used']}")
        for tname, e in res["targets"].items():
            print(f"  {tname:<12} dim={e['dim']:<3} std={e['target_std_test']:.4f}")
            for label in ("one_frame", "two_frame"):
                d = e[label]
                print(f"     {label:<10} R2 train={d['r2_train']:+.3f}"
                      f"  test={d['r2_test']:+.3f}"
                      f"  per-dim med={d['r2_test_per_dim_median']:+.3f}")
        we = res.get("within_episode", {})
        for tname, d in we.items():
            if d["r2_mean"] is not None:
                print(f"  within-ep {tname:<12} R2 mean={d['r2_mean']:+.3f}"
                      f" median={d['r2_median']:+.3f}"
                      f" [{d['r2_min']:+.3f},{d['r2_max']:+.3f}] over {d['n_episodes']} eps")

    print("\n============== SUMMARY: across-episode vs within-episode (test R2) ==============")
    print(f"{'group':<24}{'abs across':>13}{'abs within':>13}{'delta across':>14}{'delta within':>14}")
    for g, res in all_res.items():
        if "error" in res:
            print(f"{g:<24}{res['error']:>54}")
            continue
        we = res.get("within_episode", {})
        row = f"{g:<24}"
        for t in ("state_abs", "state_delta"):
            across = res["targets"][t]["two_frame"]["r2_test"]
            within = we.get(t, {}).get("r2_mean")
            row += f"{across:>+13.3f}" if t == "state_abs" else ""
            if t == "state_abs":
                row += f"{within:>+13.3f}" if within is not None else f"{'-':>13}"
        for t in ("state_delta",):
            across = res["targets"][t]["two_frame"]["r2_test"]
            within = we.get(t, {}).get("r2_mean")
            row += f"{across:>+14.3f}"
            row += f"{within:>+14.3f}" if within is not None else f"{'-':>14}"
        print(row)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_res, f, indent=2)
        print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
