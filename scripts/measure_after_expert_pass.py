#!/usr/bin/env python3
"""Re-measure every predictor after the expert pass moved the human labels.

The expert (doanh_check) re-rates segments while seeing our metric's band and the
current human label. Their score overrides raters and checkers, so the reference
every predictor is scored against has moved. This script re-measures ours AND the
VLMs on the new reference and reports the shift.

It reports each predictor on four slices:
  full         all resolved segments (the headline number)
  expert       only segments the expert has re-rated so far
  changed      only segments whose final score actually moved
  untouched    segments the expert has not reached (a control: nothing should move)

and inside `expert` it splits the two strata the queue was built from:
  disagree     our band differed from the human label before the pass
  control      our band already matched

Why the split matters. The expert sees our band before scoring, so on a segment
where we disagreed with the humans, agreeing with us is one click away and looks
identical to a genuine correction. The VLMs were never shown to the expert. So a
rise in OUR overlap on the `disagree` slice is the one number here that cannot be
read as evidence about our metric's quality — it is the mechanical consequence of
the display. The same rise for a VLM, or a rise on `control`, is informative,
because nothing in the UI pushed those.

Usage:
    python3 scripts/measure_after_expert_pass.py [--before /tmp/hallu_before.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
VLM_DIR = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300vid/vlm/results"
EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
CATALOG = REPO / "tmp/20261005_refactor/web/.main-edit/webpage/labeling/catalog.segment.json"
STRATA = REPO / "sampling_dataset/data_for_web/expert_review_strata.json"
OUT = REPO / "results/hallucinate_final"

EMBODIMENTS = ["bimanual", "humanoid", "single_arm"]
LEVELS = ["low", "medium", "high"]
NUM = {"low": 0, "medium": 1, "high": 2}
H2L = {3: "low", 2: "medium", 1: "high"}
# pipeline cuts on label_ratio (real-data p95 multiples)
CUTS = (1.0, 1.5)


def band(ratio: float, lo: float = CUTS[0], hi: float = CUTS[1]) -> str | None:
    if ratio is None or not np.isfinite(ratio):
        return None
    return "low" if ratio < lo else ("medium" if ratio < hi else "high")


def segment_key_map() -> dict[str, str]:
    """item_id (hv000_s000) -> bucket segment_id (video_path#index).

    The human export is keyed by catalog item id; the VLM CSVs are keyed by the
    source video path plus segment index. The catalog is the only place that
    holds both, so the join has to go through it.
    """
    cat = json.loads(CATALOG.read_text())
    out = {}
    for it in cat["items"]:
        for s in it["segments"]:
            # the catalog stores only the index; the item id is composed the same
            # way the labeling UI composes it (VGP.segmentId)
            out[f"{it['id']}_s{s['index']:03d}"] = f"{it['video_path']}#{s['index']}"
    return out


def load_humans(path: Path, keymap: dict[str, str]) -> pd.DataFrame:
    exp = json.loads(path.read_text())
    rows = []
    for v in exp["videos"]:
        for s in v["segments"]:
            sid = keymap.get(s["item_id"])
            if sid is None:
                continue
            rows.append({
                "segment_id": sid,
                "item_id": s["item_id"],
                "video_id": v["id"],
                "embodiment": v["embodiment"],
                "score": s["human_score"],
                "consensus": s["consensus"],
            })
    df = pd.DataFrame(rows)
    df["human"] = df["score"].map(H2L)
    return df


def load_preds() -> pd.DataFrame:
    """segment_id x model -> predicted level, plus ours from label_ratio."""
    frames = []
    for model_dir in sorted(p for p in VLM_DIR.iterdir() if p.is_dir()):
        for emb in EMBODIMENTS:
            f = model_dir / f"{emb}.csv"
            if not f.exists():
                continue
            raw = pd.read_csv(f)
            raw = raw[raw["score"].astype(str) != "score"] if "score" in raw else raw
            runs = [c for c in ("run1", "run2", "run3") if c in raw.columns]
            if "final" not in raw.columns:
                raw["final"] = raw[runs[0]]
            sub = raw[["segment_id", "label", "label_ratio", "final"]].copy()
            sub["model"] = model_dir.name
            frames.append(sub)
    long = pd.concat(frames, ignore_index=True)

    # ours: one row per segment, recomputed from label_ratio through the cuts
    ours = long.drop_duplicates("segment_id")[["segment_id", "label_ratio"]].copy()
    ours["label_ratio"] = pd.to_numeric(ours["label_ratio"], errors="coerce")
    ours["ours"] = ours["label_ratio"].map(lambda r: band(r))

    wide = long.pivot_table(
        index="segment_id", columns="model", values="final", aggfunc="first"
    ).reset_index()
    wide.columns.name = None
    out = ours.merge(wide, on="segment_id", how="outer")
    for c in wide.columns:
        if c != "segment_id":
            out[c] = out[c].astype(str).str.strip().str.lower().where(
                out[c].astype(str).str.strip().str.lower().isin(LEVELS)
            )
    return out


def score(pred: pd.Series, human: pd.Series) -> dict:
    m = pred.notna() & human.notna()
    p, h = pred[m].astype(str), human[m].astype(str)
    if len(p) == 0:
        return {"N": 0}
    pn, hn = p.map(NUM).astype(float), h.map(NUM).astype(float)
    rho = float(stats.spearmanr(pn, hn).statistic) if pn.nunique() > 1 and hn.nunique() > 1 else np.nan
    return {
        "N": int(len(p)),
        "overlap": float((p == h).mean()),
        "kappa": float(cohen_kappa_score(h, p, labels=LEVELS)),
        "balanced_acc": float(balanced_accuracy_score(h, p)),
        "macro_f1": float(f1_score(h, p, labels=LEVELS, average="macro", zero_division=0)),
        "spearman": rho,
    }


def table(df: pd.DataFrame, models: list[str], human_col: str) -> pd.DataFrame:
    rows = []
    for m in models:
        r = score(df[m], df[human_col])
        r["predictor"] = m
        rows.append(r)
    return pd.DataFrame(rows).set_index("predictor")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="/tmp/hallu_before.json",
                    help="pre-expert export, for the delta")
    args = ap.parse_args()

    keymap = segment_key_map()
    now = load_humans(EXPORT, keymap)
    preds = load_preds()
    df = now.merge(preds, on="segment_id", how="inner")

    before_path = Path(args.before)
    if before_path.exists():
        old = load_humans(before_path, keymap)[["segment_id", "human"]].rename(
            columns={"human": "human_before"})
        df = df.merge(old, on="segment_id", how="left")
    else:
        df["human_before"] = df["human"]

    models = ["ours"] + sorted(c for c in preds.columns
                               if c not in ("segment_id", "label_ratio", "ours"))

    df["is_expert"] = df["consensus"] == "expert"
    df["changed"] = df["human"] != df["human_before"]

    # strata, from the offline key (never shipped to the UI)
    strata = {}
    if STRATA.exists():
        sj = json.loads(STRATA.read_text())
        raw = sj.get("segments", sj)
        if isinstance(raw, dict):
            strata = {k: (v.get("stratum") if isinstance(v, dict) else v)
                      for k, v in raw.items()}
    df["stratum"] = df["item_id"].map(strata)
    # fall back to reconstructing it: the queue's disagree stratum is exactly
    # where our band differed from the pre-pass human label
    miss = df["stratum"].isna() & df["is_expert"]
    df.loc[miss, "stratum"] = np.where(
        df.loc[miss, "ours"] != df.loc[miss, "human_before"], "disagree", "control")

    slices = {
        "full": df,
        "expert": df[df["is_expert"]],
        "changed": df[df["changed"]],
        "untouched": df[~df["is_expert"]],
        "expert/disagree": df[df["is_expert"] & (df["stratum"] == "disagree")],
        "expert/control": df[df["is_expert"] & (df["stratum"] == "control")],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    bundle = {"n_segments": int(len(df)),
              "n_expert": int(df["is_expert"].sum()),
              "n_changed": int(df["changed"].sum()),
              "cuts": {"low_medium": CUTS[0], "medium_high": CUTS[1]},
              "slices": {}}

    print(f"segments matched      : {len(df)}")
    print(f"expert-rated so far   : {df['is_expert'].sum()}"
          f"  ({df.loc[df['is_expert'],'video_id'].nunique()} videos)")
    print(f"final score changed   : {df['changed'].sum()}")

    for name, sub in slices.items():
        if not len(sub):
            continue
        after = table(sub, models, "human")
        before = table(sub, models, "human_before")
        cmp = after[["N", "overlap", "kappa", "balanced_acc", "macro_f1", "spearman"]].copy()
        cmp["overlap_before"] = before["overlap"]
        cmp["d_overlap"] = cmp["overlap"] - cmp["overlap_before"]
        cmp = cmp.sort_values("overlap", ascending=False)
        bundle["slices"][name] = {
            "n": int(len(sub)),
            "rows": json.loads(cmp.reset_index().to_json(orient="records")),
        }
        print(f"\n=== {name}  (n={len(sub)}) ===")
        show = cmp[["N", "overlap_before", "overlap", "d_overlap",
                    "kappa", "balanced_acc", "macro_f1", "spearman"]]
        print(show.to_string(float_format=lambda x: f"{x:+.4f}" if abs(x) < 1 and x < 0 else f"{x:.4f}"))
        rank = list(cmp.index).index("ours") + 1
        print(f"  ours rank by overlap: #{rank} of {len(cmp)}")

    (OUT / "after_expert_pass.json").write_text(json.dumps(bundle, indent=2) + "\n")
    df.to_csv(OUT / "after_expert_pass_segments.csv", index=False)
    print(f"\nwrote {(OUT / 'after_expert_pass.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
