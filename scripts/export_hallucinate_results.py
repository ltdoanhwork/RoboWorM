#!/usr/bin/env python3
"""Export the final hallucinate results in one consumable bundle.

Reads the resolved human labels from
`sampling_dataset/data_for_web/hallucinate_final_labels.json` (produced by
tmp/20261005_refactor/web/scripts/export_hallucinate_final_labels.js, which calls the deployed
VGP.resolveSegments) and the VLM/kinescore predictions from the bucket, then
writes every number in one place so figures/tables can be built without
recomputing anything.

Cut points are imported from compare_ours_vs_vlm_segment_overlap.calibrate_cuts
rather than reimplemented, so the two scripts cannot disagree.

Outputs (results/hallucinate_final/):
  results.json               everything, machine-readable
  metrics_segment.csv        per predictor: overlap/kappa/spearman/balanced/macroF1
  metrics_video.csv          video-level overlap (291 videos)
  metrics_by_embodiment.csv  overlap split by embodiment
  metrics_by_subset.csv      overlap under each label subset
  cut_points.csv             what each pair of kinescore cuts yields
  RESULTS.md                 the same, readable

Usage:
    python3 scripts/export_hallucinate_results.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score

REPO = Path(__file__).resolve().parents[1]
VLM_DIR = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300vid/vlm/results"
CATALOG = REPO / "tmp/20261005_refactor/web/.main-edit/webpage/labeling/catalog.segment.json"
EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
OUT = REPO / "results/hallucinate_final"

EMBODIMENTS = ["bimanual", "humanoid", "single_arm"]
LEVELS = ["low", "medium", "high"]
NUM = {"low": 0, "medium": 1, "high": 2}
# humans rate quality (3=Good); the bucket rates amount of hallucination
H2L = {3: "low", 2: "medium", 1: "high"}
FLAG = {"low": 1.0, "medium": 0.5, "high": 0.0}
# raters whose calibration is an outlier on HUMAN-only evidence: a third human
# overruled them in 72-73% of the conflicts they were part of, vs ~50% for
# everyone else, and minh_loi's mean score is a full point below the group.
LOW_REL = {"minh_loi", "doanh_1", "doanh_2"}


def _calibrate():
    spec = importlib.util.spec_from_file_location(
        "cmp", REPO / "scripts/compare_ours_vs_vlm_segment_overlap.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.calibrate_cuts


def load() -> tuple[pd.DataFrame, list[str], dict]:
    cat = json.loads(CATALOG.read_text())
    smap = {}
    for it in cat["items"]:
        for s in it["segments"]:
            smap[f"{it['id']}_s{s['index']:03d}"] = (
                f"{it['video_path']}#{s['index']}", it["embodiment"], it["split"]
            )

    exp = json.loads(EXPORT.read_text())
    rows = []
    for v in exp["videos"]:
        for s in v["segments"]:
            if s["human_score"] is None:
                continue
            sid, emb, split = smap[s["item_id"]]
            names = {k for k, x in s["rater_scores"].items() if x is not None}
            rows.append({
                "segment_id": sid, "item_id": s["item_id"], "video_id": v["id"],
                "embodiment": emb, "split": split,
                "human": H2L[s["human_score"]], "human_score": s["human_score"],
                "consensus": s["consensus"],
                "low_rel": bool(names & LOW_REL),
            })
    base = pd.DataFrame(rows)

    frames = []
    for md in sorted(p for p in VLM_DIR.iterdir() if p.is_dir()):
        for emb in EMBODIMENTS:
            f = md / f"{emb}.csv"
            if not f.exists():
                continue
            raw = pd.read_csv(f)
            raw = raw[raw["score"].astype(str) != "score"] if "score" in raw else raw
            runs = [c for c in ("run1", "run2", "run3") if c in raw.columns]
            if "final" not in raw.columns:
                raw["final"] = raw[runs[0]]
            frames.append(raw[["segment_id", "label", "label_ratio", "final"]]
                          .assign(model=md.name))
    vlm = pd.concat(frames, ignore_index=True)
    models = sorted(vlm["model"].unique())

    ours = vlm.drop_duplicates("segment_id")[["segment_id", "label", "label_ratio"]]
    wide = base.merge(ours.rename(columns={"label": "ours_pipeline"}),
                      on="segment_id", how="inner")
    for m in models:
        wide = wide.merge(
            vlm[vlm["model"] == m][["segment_id", "final"]].rename(columns={"final": m}),
            on="segment_id", how="left")
    return wide, models, exp["counts"]


def metrics(pred: pd.Series, human: pd.Series) -> dict:
    m = pred.notna() & human.notna()
    p, h = pred[m].astype(str), human[m]
    if not len(p):
        return {}
    return {
        "N": int(len(p)),
        "overlap": round(float((p == h).mean()), 4),
        "kappa": round(float(cohen_kappa_score(h, p, labels=LEVELS)), 4),
        "spearman": round(float(stats.spearmanr(p.map(NUM), h.map(NUM)).statistic), 4),
        "balanced_acc": round(float(balanced_accuracy_score(h, p)), 4),
        "macro_f1": round(float(f1_score(h, p, labels=LEVELS, average="macro",
                                         zero_division=0)), 4),
        "pred_low_pct": round(100 * float((p == "low").mean()), 1),
        "pred_medium_pct": round(100 * float((p == "medium").mean()), 1),
        "pred_high_pct": round(100 * float((p == "high").mean()), 1),
    }


def video_level(wide: pd.DataFrame, col: str) -> dict:
    """Video verdict = mean of segment flags, rebanded — the rule humans use."""
    def band(x):
        return "low" if x >= 0.75 else ("medium" if x >= 0.4 else "high")
    hs, ps = [], []
    for _, g in wide.groupby("video_id"):
        p = g[col].dropna()
        if not len(p):
            continue
        hs.append(band(g["human"].map(FLAG).mean()))
        ps.append(band(p.map(FLAG).mean()))
    if not hs:
        return {}
    hs, ps = pd.Series(hs), pd.Series(ps)
    return {
        "N_videos": len(hs),
        "overlap": round(float((hs == ps).mean()), 4),
        "kappa": round(float(cohen_kappa_score(hs, ps, labels=LEVELS)), 4),
        "spearman": round(float(stats.spearmanr(ps.map(NUM), hs.map(NUM)).statistic), 4),
    }


def main() -> None:
    wide, models, counts = load()
    calibrate_cuts = _calibrate()

    # the two kinescore cut points, refit against humans (out-of-fold by video)
    wide["ours_accuracy_fit"], cuts_acc = calibrate_cuts(
        wide.assign(human_level=wide["human"]), mode="accuracy")
    wide["ours_marginal_match"], cuts_marg = calibrate_cuts(
        wide.assign(human_level=wide["human"]), mode="marginal")

    ours_cols = ["ours_pipeline", "ours_marginal_match", "ours_accuracy_fit"]
    preds = ours_cols + models
    nice = {
        "ours_pipeline": "ours — pipeline cuts 1.00/1.50 (never fitted to humans)",
        "ours_marginal_match": f"ours — cuts matched to human class shares {cuts_marg[0]:.2f}/{cuts_marg[1]:.2f} (out-of-fold)",
        "ours_accuracy_fit": f"ours — cuts fitted for accuracy {cuts_acc[0]:.2f}/{cuts_acc[1]:.2f} (out-of-fold)",
    }

    N = len(wide)
    majority = float(wide["human"].value_counts(normalize=True).max())
    human_mix = {k: round(100 * v, 1) for k, v in
                 wide["human"].value_counts(normalize=True).items()}

    seg = pd.DataFrame([{"predictor": p, "label": nice.get(p, p), **metrics(wide[p], wide["human"])}
                        for p in preds]).sort_values("overlap", ascending=False)
    vid = pd.DataFrame([{"predictor": p, "label": nice.get(p, p), **video_level(wide, p)}
                        for p in preds]).sort_values("overlap", ascending=False)

    emb_rows = []
    for emb, g in wide.groupby("embodiment"):
        r = {"embodiment": emb, "N": len(g),
             "majority_baseline": round(float(g["human"].value_counts(normalize=True).max()), 4)}
        for p in preds:
            r[p] = metrics(g[p], g["human"]).get("overlap")
        emb_rows.append(r)
    emb = pd.DataFrame(emb_rows)

    subsets = {
        "all segments": wide,
        "drop unreviewed segments of outlier raters": wide[~(wide["consensus"].eq("raters") & wide["low_rel"])],
        "drop every segment touched by an outlier rater": wide[~wide["low_rel"]],
        "only checker-adjudicated segments": wide[wide["consensus"].eq("checker")],
        "only clean rater agreement": wide[wide["consensus"].eq("raters") & ~wide["low_rel"]],
        "binary — human medium dropped": wide[wide["human"].ne("medium")],
    }
    sub_rows = []
    for name, g in subsets.items():
        r = {"subset": name, "N": len(g)}
        for p in preds:
            r[p] = metrics(g[p], g["human"]).get("overlap")
        sub_rows.append(r)
    sub = pd.DataFrame(sub_rows)

    cut_rows = []
    for t1, t2, name in [(1.00, 1.50, "pipeline (real-segment p95)"),
                         (cuts_marg[0], cuts_marg[1], "matched to human class shares"),
                         (cuts_acc[0], cuts_acc[1], "fitted for accuracy")]:
        p = pd.Series(np.where(wide["label_ratio"] < t1, "low",
                      np.where(wide["label_ratio"] < t2, "medium", "high")), index=wide.index)
        cut_rows.append({"cuts": f"{t1:.2f}/{t2:.2f}", "how": name,
                         "note": "fit on all data, optimistic" if name != "pipeline (real-segment p95)" else "",
                         **metrics(p, wide["human"])})
    cuts = pd.DataFrame(cut_rows)

    hi = wide[wide["human"] == "high"]
    diag = {
        "label_ratio_median": round(float(wide["label_ratio"].median()), 3),
        "label_ratio_min": round(float(wide["label_ratio"].min()), 3),
        "label_ratio_max": round(float(wide["label_ratio"].max()), 3),
        "human_high_below_1.50_pct": round(100 * float((hi["label_ratio"] < 1.5).mean()), 1),
        "human_high_below_1.50_n": int((hi["label_ratio"] < 1.5).sum()),
        "human_high_n": int(len(hi)),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    seg.to_csv(OUT / "metrics_segment.csv", index=False)
    vid.to_csv(OUT / "metrics_video.csv", index=False)
    emb.to_csv(OUT / "metrics_by_embodiment.csv", index=False)
    sub.to_csv(OUT / "metrics_by_subset.csv", index=False)
    cuts.to_csv(OUT / "cut_points.csv", index=False)

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "track": "hallucinate",
        "human_labels": {
            "source": str(EXPORT.relative_to(REPO)),
            "n_videos": counts["videos"], "n_segments": counts["segments"],
            "resolved_segments": counts["segments_resolved"],
            "segment_consensus": counts["segment_consensus"],
            "class_mix_pct": human_mix,
            "majority_baseline": round(majority, 4),
        },
        "note": (
            "Our method emits a CONTINUOUS label_ratio; low/medium/high needs two cut "
            "points, so the same method yields different overlap under different cuts. "
            "The pipeline cuts (1.00/1.50) come from the real-segment p95 and were never "
            "fitted to human labels. Fitted cuts are out-of-fold (GroupKFold over videos). "
            "The accuracy-fitted variant reaches its number by shrinking `medium` to 3%, "
            "the same minority-class refusal the VLMs use, so the marginal-matched variant "
            "is the more defensible one to cite."
        ),
        "cut_points": {"pipeline": [1.0, 1.5],
                       "marginal_match": list(cuts_marg),
                       "accuracy_fit": list(cuts_acc)},
        "diagnostics": diag,
        "segment_level": seg.to_dict("records"),
        "video_level": vid.to_dict("records"),
        "by_embodiment": emb.to_dict("records"),
        "by_subset": sub.to_dict("records"),
        "cut_point_sweep": cuts.to_dict("records"),
    }
    (OUT / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    def rank(df, pred="ours_pipeline", metric="overlap"):
        o = list(df.sort_values(metric, ascending=False)["predictor"])
        return f"{o.index(pred) + 1}/{len(o)}" if pred in o else "n/a"

    md = [
        "# Hallucinate — final results",
        "",
        f"Human labels: **{counts['videos']} videos / {counts['segments']} segments**, "
        f"all resolved ({counts['segment_consensus']['raters']} by rater agreement, "
        f"{counts['segment_consensus']['checker']} adjudicated by a checker).",
        f"Human class mix: {human_mix}. Majority-class baseline **{majority:.4f}**.",
        "",
        "Our method emits a continuous `label_ratio`; turning it into low/medium/high",
        "needs two cut points, so the same method gives different overlap under",
        "different cuts. The pipeline cuts 1.00/1.50 come from the real-segment p95 and",
        "were never fitted to human labels.",
        "",
        "## Segment level (N=%d)" % N, "",
        seg.drop(columns=["label"]).to_markdown(index=False), "",
        f"Ours (pipeline cuts) ranks **{rank(seg)}** on overlap, "
        f"**{rank(seg, metric='macro_f1')}** on macro-F1, "
        f"**{rank(seg, metric='balanced_acc')}** on balanced accuracy.", "",
        "## Video level (mean of segment flags, rebanded)", "",
        vid.drop(columns=["label"]).to_markdown(index=False), "",
        "## By embodiment (overlap)", "",
        emb.to_markdown(index=False), "",
        "## By label subset (overlap)", "",
        sub.to_markdown(index=False), "",
        "## What each pair of cut points yields", "",
        cuts.to_markdown(index=False), "",
        f"`medium` is {human_mix.get('medium')}% of human labels. The accuracy-fitted cuts",
        f"reach their overlap by emitting only {cuts.iloc[2]['pred_medium_pct']}% medium — the same",
        "minority-class refusal the VLMs use — so the marginal-matched cuts are the more",
        "defensible variant to cite.", "",
        f"Diagnostic: {diag['human_high_below_1.50_pct']}% of human-`high` segments "
        f"({diag['human_high_below_1.50_n']}/{diag['human_high_n']}) have `label_ratio` below the",
        "1.50 cut, so the pipeline calls them medium/low.", "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(md))

    print(f"human labels: {counts['videos']} videos / {counts['segments']} segments\n")
    print(seg.drop(columns=["label", "pred_low_pct", "pred_high_pct"]).to_string(index=False))
    print(f"\nours(pipeline) rank: overlap {rank(seg)}, macro-F1 {rank(seg, metric='macro_f1')}, "
          f"balanced {rank(seg, metric='balanced_acc')}")
    print(f"\nwrote:\n  " + "\n  ".join(str(p.relative_to(REPO)) for p in sorted(OUT.iterdir())))


if __name__ == "__main__":
    main()
