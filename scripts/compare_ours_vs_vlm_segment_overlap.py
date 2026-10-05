#!/usr/bin/env python3
"""Ours (kinescore band) vs VLM judges — overlap with HUMAN segment labels.

The bucket's own `overlap/summary.csv` measures each VLM against the
*kinescore band*, i.e. against OUR method — so by construction ours would
score 100% there and the table says nothing about who is closer to humans.
This script instead uses the live human segment labels from the labeling
site as the reference and reports, on the SAME segments:

    overlap %  = exact 3-way agreement with the human level
    kappa      = chance-corrected agreement (exact-match is inflated by the
                 majority class; kappa is not)
    Spearman   = ordinal correlation (low < medium < high)

Scale mapping: humans score 1 Bad / 2 Medium / 3 Good on *quality*, the
bucket levels are *hallucination amount*, so Good <-> low and Bad <-> high.

Data:
  ours + VLM : sampling_dataset/data_for_web/hallucinate_bench/300vid/vlm/
               results/<model>/<embodiment>.csv  (label = kinescore band,
               run1..run3 + final = median of runs)
  human      : Supabase `labels`, item_id = <videoId>_s<index>
  mapping    : tmp/20261005_refactor/web/.main-edit/webpage/labeling/catalog.segment.json (the DEPLOYED
               291-video catalog; the copy in this branch's working tree is an
               older 300-prune pool with 8370 segments and will not join)

Usage:
    python3 scripts/compare_ours_vs_vlm_segment_overlap.py
    python3 scripts/compare_ours_vs_vlm_segment_overlap.py --offline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix

REPO = Path(__file__).resolve().parents[1]
VLM_DIR = REPO / "sampling_dataset/data_for_web/hallucinate_bench/300vid/vlm/results"
CATALOG = REPO / "tmp/20261005_refactor/web/.main-edit/webpage/labeling/catalog.segment.json"
HALLU_EXPORT = REPO / "sampling_dataset/data_for_web/hallucinate_final_labels.json"
OUT_DIR = REPO / "results/ours_vs_vlm_segment"

REVIEWERS = {"quang_check", "thanh_check"}
EMBODIMENTS = ["bimanual", "humanoid", "single_arm"]
LEVELS = ["low", "medium", "high"]
LEVEL_NUM = {"low": 0, "medium": 1, "high": 2}
# humans rate quality (3=Good); the bucket rates hallucination amount
HUMAN_TO_LEVEL = {3: "low", 2: "medium", 1: "high"}


def load_labels(offline: bool) -> list[dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "geo_corr", REPO / "scripts/analyze_geometry_label_correlation.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    return geo.load_labels(offline)


def segment_map() -> pd.DataFrame:
    """catalog segment -> bucket segment_id."""
    cat = json.loads(CATALOG.read_text())
    recs = []
    for item in cat["items"]:
        for seg in item["segments"]:
            if not seg.get("rateable", True):
                continue
            recs.append(
                {
                    "item_id": f"{item['id']}_s{seg['index']:03d}",
                    "segment_id": f"{item['video_path']}#{seg['index']}",
                    "embodiment": item["embodiment"],
                    "split": item["split"],
                    "video_id": item["id"],
                }
            )
    return pd.DataFrame(recs)


def resolved_human_segments(rows: list[dict]) -> pd.DataFrame:
    """Final per-segment human labels, read from the canonical export.

    This used to re-derive resolution from raw label rows, which drifted from
    the site's own rules (it accepted a checker score wherever one existed, so a
    checker could override a segment both raters agreed on). The export produced
    by scripts/export_hallucinate_final_labels.js calls the deployed resolver
    (VGP.resolveSegments), so reading it keeps exactly one source of truth.

    `rows` is unused now; kept so the caller/CLI stays unchanged.
    """
    if not HALLU_EXPORT.exists():
        raise SystemExit(
            f"missing {HALLU_EXPORT}\n"
            "run: node scripts/export_hallucinate_final_labels.js"
        )
    payload = json.loads(HALLU_EXPORT.read_text())
    recs = []
    for video in payload["videos"]:
        for seg in video["segments"]:
            if seg["human_score"] is None:
                continue
            recs.append(
                {
                    "item_id": seg["item_id"],
                    "human": int(seg["human_score"]),
                    # "checker"/"checkers" both mean adjudicated
                    "how": "raters" if seg["consensus"] == "raters" else "checker",
                    "n_raters": len([v for v in seg["rater_scores"].values() if v is not None]),
                }
            )
    return pd.DataFrame(recs)


def load_vlm_segments() -> pd.DataFrame:
    """Long: segment_id, model, ours_label, run1..3, final."""
    frames = []
    for model_dir in sorted(p for p in VLM_DIR.iterdir() if p.is_dir()):
        for emb in EMBODIMENTS:
            f = model_dir / f"{emb}.csv"
            if not f.exists():
                continue
            raw = pd.read_csv(f)
            runs = [c for c in ("run1", "run2", "run3") if c in raw.columns]
            # qwen3.6-plus-half has a single run and therefore no `final`
            if "final" not in raw.columns:
                raw["final"] = raw[runs[0]]
            keep = ["segment_id", "label", "label_ratio", "final"] + runs
            sub = raw[keep].copy()
            sub["model"] = model_dir.name
            sub["embodiment"] = emb
            frames.append(sub)
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={"label": "ours_label"})


def agreement(pred: pd.Series, human: pd.Series) -> dict:
    m = pred.notna() & human.notna()
    p, h = pred[m].astype(str), human[m].astype(str)
    if len(p) == 0:
        return {}
    overlap = float((p == h).mean())
    kappa = float(cohen_kappa_score(h, p, labels=LEVELS))
    pn = p.map(LEVEL_NUM).astype(float)
    hn = h.map(LEVEL_NUM).astype(float)
    rho = float(stats.spearmanr(pn, hn).statistic) if pn.nunique() > 1 else float("nan")
    return {
        "N": int(len(p)),
        "overlap": round(overlap, 4),
        "kappa": round(kappa, 4),
        "spearman": round(rho, 4) if rho == rho else float("nan"),
    }


def bootstrap_delta(df: pd.DataFrame, col_a: str, col_b: str,
                    n_boot: int, seed: int) -> dict:
    """Paired bootstrap over videos (segments inside a video are correlated)."""
    sub = df[[col_a, col_b, "human_level", "video_id"]].dropna()
    vids = sub["video_id"].unique()
    rng = np.random.default_rng(seed)
    by_vid = {v: g for v, g in sub.groupby("video_id")}
    d_ov, d_ka = [], []
    for _ in range(n_boot):
        pick = rng.choice(vids, size=len(vids), replace=True)
        s = pd.concat([by_vid[v] for v in pick], ignore_index=True)
        a = agreement(s[col_a], s["human_level"])
        b = agreement(s[col_b], s["human_level"])
        if not a or not b:
            continue
        d_ov.append(a["overlap"] - b["overlap"])
        d_ka.append(a["kappa"] - b["kappa"])
    d_ov, d_ka = np.array(d_ov), np.array(d_ka)
    return {
        "d_overlap": round(float(d_ov.mean()), 4),
        "d_overlap_lo": round(float(np.percentile(d_ov, 2.5)), 4),
        "d_overlap_hi": round(float(np.percentile(d_ov, 97.5)), 4),
        "d_kappa": round(float(d_ka.mean()), 4),
        "d_kappa_lo": round(float(np.percentile(d_ka, 2.5)), 4),
        "d_kappa_hi": round(float(np.percentile(d_ka, 97.5)), 4),
        "n_boot": int(len(d_ov)),
    }


def plot_bars(tbl: pd.DataFrame, out_png: Path, majority: float, n: int) -> None:
    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, metric, ref, reflabel in (
        (axes[0], "overlap", majority, f"majority-class baseline ({majority:.3f})"),
        (axes[1], "kappa", 0.0, "0 = chance"),
    ):
        d = tbl.sort_values(metric, ascending=False)
        colors = ["#2d6cdf" if p.startswith("ours") else "#b0b6c0" for p in d["predictor"]]
        ax.barh(d["label"], d[metric], color=colors)
        ax.axvline(ref, color="#d9534f" if metric == "overlap" else "#888", lw=1.2, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric}   ({reflabel})")
        for y, v in enumerate(d[metric]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
    fig.suptitle(
        f"Segment-level agreement with HUMAN labels — ours (blue) vs VLM judges (N={n} segments)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_confusions(df: pd.DataFrame, preds: list[tuple[str, str]], out_png: Path) -> None:
    sns.set_theme(style="white", font_scale=0.85)
    fig, axes = plt.subplots(1, len(preds), figsize=(3.1 * len(preds), 3.4))
    for ax, (label, col) in zip(axes, preds):
        m = df[col].notna() & df["human_level"].notna()
        cm = confusion_matrix(df.loc[m, "human_level"], df.loc[m, col], labels=LEVELS)
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", vmin=0, vmax=1, cbar=False,
                    xticklabels=LEVELS, yticklabels=LEVELS, ax=ax)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("predicted")
        ax.set_ylabel("human" if ax is axes[0] else "")
    fig.suptitle("Row-normalised confusion vs human level (counts annotated)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def calibrate_cuts(df: pd.DataFrame, n_splits: int = 5, seed: int = 0,
                   mode: str = "accuracy") -> tuple[pd.Series, tuple]:
    """Re-fit the two kinescore cut points against human labels.

    The cuts used in the pipeline (1.0 / 1.5) come from the real-segment p95
    definition, not from human agreement, and they make `medium` too wide: we
    emit 28.1% medium where humans use 18.6%, and 56.7% of human-`high`
    segments (1372/2418) sit below the 1.5 cut and so get called medium/low.
    Cuts are fitted with GroupKFold over VIDEOS and every segment is scored by
    a fold that never saw its video, so the reported number is out-of-fold and
    not the optimum-on-itself (fitting on everything reads ~0.01 higher).

    mode="accuracy" maximises exact agreement (1.20/1.25 -> 0.599 out-of-fold).
    It wins by shrinking `medium` to 3.0%, i.e. the same minority-class refusal
    we criticise the VLMs for, so it is the weaker number to stand behind.
    mode="marginal" only matches the human class proportions (quantile mapping,
    1.02/1.32 -> 0.579, medium 18.9% vs human 18.6%) and never looks at
    per-segment correctness -- the stricter, harder-to-criticise variant.

    Numbers above are on the complete 8093-segment label set.
    """
    from sklearn.model_selection import GroupKFold

    r = df["label_ratio"].to_numpy()
    y = df["human_level"].to_numpy()
    grp = df["video_id"].to_numpy()

    def cut(ratio, t1, t2):
        return np.where(ratio < t1, "low", np.where(ratio < t2, "medium", "high"))

    def fit(idx):
        if mode == "marginal":
            q_low = (y[idx] == "low").mean()
            q_med = q_low + (y[idx] == "medium").mean()
            return tuple(np.quantile(r[idx], [q_low, q_med]))
        grid = [
            (a, b)
            for a in np.arange(0.4, 1.45, 0.05)
            for b in np.arange(a + 0.05, 2.6, 0.05)
        ]
        return max(grid, key=lambda t: (cut(r[idx], *t) == y[idx]).mean())

    oof = np.empty(len(df), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(r, y, grp):
        oof[te] = cut(r[te], *fit(tr))
    full_best = fit(np.arange(len(df)))
    return pd.Series(oof, index=df.index), tuple(round(float(x), 2) for x in full_best)


def plot_spearman(tbl: pd.DataFrame, out_png: Path, n: int) -> None:
    """The threshold-free result: ordinal correlation with humans.

    Unlike overlap/kappa this needs no cut points at all — it reads only the
    ORDER of our continuous kinescore, so it is identical under the pipeline and
    the human-fitted cuts and involves no fitting against human labels. That makes it
    the one number directly comparable to a zero-shot VLM judge.
    """
    d = tbl[~tbl["predictor"].isin(["ours_cal", "ours_marg"])].sort_values("spearman", ascending=False)
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    colors = ["#2d6cdf" if p.startswith("ours") else "#b0b6c0" for p in d["predictor"]]
    bars = ax.barh(
        [l.replace(" (median of runs)", "").replace(" (kinescore band, pipeline cuts)", "")
         for l in d["label"]],
        d["spearman"], color=colors,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.62, float(d["spearman"].max()) * 1.18))
    ax.set_xlabel("Spearman correlation with human segment label  (threshold-free)")
    for bar, v in zip(bars, d["spearman"]):
        ax.text(v + 0.008, bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=10,
                fontweight="bold" if bar.get_facecolor()[2] > 0.7 else "normal")
    # Rank is read off the data, never asserted: on the complete label set the
    # top slot is not necessarily ours.
    rank = list(d["predictor"]).index("ours") + 1 if "ours" in set(d["predictor"]) else None
    ax.set_title(
        f"Ordinal agreement with human hallucination labels — ours ranks "
        f"#{rank} of {len(d)}\nkinescore vs VLM judges, N={n} human-labeled segments",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def plot_headline(tbl: pd.DataFrame, out_png: Path, n: int, majority: float) -> None:
    """The two-panel headline: overlap % and Spearman."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.0))

    def clean(lbl: str) -> str:
        return lbl.replace(" (median of runs)", "")

    panels = [
        (axes[0], "overlap", majority,
         f"dashed = majority-class baseline ({majority:.3f})",
         "Overlap % with human label"),
        (axes[1], "spearman", None, "threshold-free",
         "Spearman correlation with human label"),
    ]
    for ax, metric, ref, note, title in panels:
        d = tbl.copy()
        if metric == "spearman":
            # threshold choice cannot change a rank correlation, so the three
            # "ours" rows are one and the same number — show it once
            d = d[~d["predictor"].isin(["ours_cal", "ours_marg"])]
        d = d.sort_values(metric, ascending=False)
        colors = ["#2d6cdf" if p.startswith("ours") else "#b0b6c0" for p in d["predictor"]]
        bars = ax.barh([clean(l) for l in d["label"]], d[metric], color=colors)
        if ref is not None:
            ax.axvline(ref, color="#d9534f", ls="--", lw=1.3)
        ax.invert_yaxis()
        ax.set_xlim(0, float(d[metric].max()) * 1.2)
        ax.set_xlabel(f"{title}\n({note})")
        for bar, v in zip(bars, d[metric]):
            ax.text(v + float(d[metric].max()) * 0.012,
                    bar.get_y() + bar.get_height() / 2, f"{v:.3f}",
                    va="center", fontsize=9.5,
                    fontweight="bold" if bar.get_facecolor()[2] > 0.7 else "normal")
    def rank_of(metric: str, pred: str = "ours") -> str:
        d = tbl if metric != "spearman" else tbl[~tbl["predictor"].isin(["ours_cal", "ours_marg"])]
        order = list(d.sort_values(metric, ascending=False)["predictor"])
        return f"#{order.index(pred) + 1}/{len(order)}" if pred in order else "n/a"

    fig.suptitle(
        f"Segment-level agreement with human hallucination labels — ours (blue) is "
        f"{rank_of('overlap')} on overlap, {rank_of('spearman')} on Spearman "
        f"(N={n} human-labeled segments)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    smap = segment_map()
    human = resolved_human_segments([])
    vlm = load_vlm_segments()

    print(f"[data] catalog rateable segments: {len(smap)}")
    print(f"[data] resolved human segments  : {len(human)} "
          f"{human['how'].value_counts().to_dict()}")
    print(f"[data] VLM models found         : {sorted(vlm['model'].unique())}")

    # wide: one row per segment, columns = ours + each model's final/run1
    base = smap.merge(human, on="item_id", how="inner")
    ours = vlm.drop_duplicates("segment_id")[["segment_id", "ours_label", "label_ratio"]]
    wide = base.merge(ours, on="segment_id", how="inner")
    wide["human_level"] = wide["human"].map(HUMAN_TO_LEVEL)

    models = sorted(vlm["model"].unique())
    for m in models:
        sub = vlm[vlm["model"] == m][["segment_id", "final", "run1"]]
        sub = sub.rename(columns={"final": f"vlm_{m}", "run1": f"vlm_{m}_r1"})
        wide = wide.merge(sub, on="segment_id", how="left")

    n = len(wide)
    if n == 0:
        raise SystemExit("no segments joined — check catalog/bucket path mapping")
    print(f"[data] joined segments (human ∩ bucket): {n}")
    print(f"[data] human level mix: {wide['human_level'].value_counts().to_dict()}\n")

    majority = float(wide["human_level"].value_counts(normalize=True).max())

    wide["ours_cal"], full_cuts = calibrate_cuts(wide, seed=args.seed, mode="accuracy")
    wide["ours_marg"], marg_cuts = calibrate_cuts(wide, seed=args.seed, mode="marginal")
    print(f"[calib] kinescore cuts refit vs human (out-of-fold below):")
    print(f"        pipeline cuts  1.00 / 1.50  (real-segment p95; never fitted to humans)")
    print(f"        accuracy-fit   {full_cuts[0]:.2f} / {full_cuts[1]:.2f}")
    print(f"        marginal-match {marg_cuts[0]:.2f} / {marg_cuts[1]:.2f}  "
          f"(matches human class shares only)\n")

    rows = [{"predictor": "ours", "label": "ours (pipeline cuts 1.00/1.50, not fitted to humans)",
             **agreement(wide["ours_label"], wide["human_level"])},
            {"predictor": "ours_marg", "label": "ours (marginal-matched cuts, out-of-fold)",
             **agreement(wide["ours_marg"], wide["human_level"])},
            {"predictor": "ours_cal", "label": "ours (accuracy-fitted cuts, out-of-fold)",
             **agreement(wide["ours_cal"], wide["human_level"])}]
    for m in models:
        rows.append({"predictor": m, "label": f"{m} (median of runs)",
                     **agreement(wide[f"vlm_{m}"], wide["human_level"])})
        rows.append({"predictor": f"{m}_r1", "label": f"{m} (run1)",
                     **agreement(wide[f"vlm_{m}_r1"], wide["human_level"])})
    tbl = pd.DataFrame([r for r in rows if "overlap" in r])

    print("=" * 92)
    print(f"AGREEMENT WITH HUMAN SEGMENT LABELS (N={n}); "
          f"majority-class baseline = {majority:.4f}")
    print("=" * 92)
    print(tbl.sort_values("overlap", ascending=False).to_string(index=False))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT_DIR / "overlap_vs_human.csv", index=False)

    # per embodiment
    print("\n" + "=" * 92)
    print("PER EMBODIMENT (overlap % vs human)")
    print("=" * 92)
    per = []
    for emb, g in wide.groupby("embodiment"):
        r = {"embodiment": emb, "N": len(g)}
        a = agreement(g["ours_label"], g["human_level"])
        r["ours"] = a.get("overlap")
        for m in models:
            r[m] = agreement(g[f"vlm_{m}"], g["human_level"]).get("overlap")
        per.append(r)
    per_df = pd.DataFrame(per)
    print(per_df.to_string(index=False))
    per_df.to_csv(OUT_DIR / "overlap_by_embodiment.csv", index=False)

    main_tbl = tbl[~tbl["predictor"].str.endswith("_r1")]
    plot_bars(main_tbl, OUT_DIR / "overlap_ours_vs_vlm.png", majority, n)
    plot_spearman(main_tbl, OUT_DIR / "spearman_vs_human.png", n)
    plot_headline(main_tbl, OUT_DIR / "overlap_and_spearman_vs_human.png", n, majority)
    preds = [("ours (pipeline cuts)", "ours_label"),
             ("ours (marginal-matched)", "ours_marg")] + [(m, f"vlm_{m}") for m in models]
    plot_confusions(wide, preds, OUT_DIR / "confusion_vs_human.png")

    print("\n" + "=" * 92)
    print("HEAD-TO-HEAD — paired bootstrap over VIDEOS (ours CALIBRATED minus VLM)")
    print("=" * 92)
    deltas = []
    for m in models:
        d = bootstrap_delta(wide, "ours_cal", f"vlm_{m}", args.n_boot, args.seed)
        verdict = ("OURS higher" if d["d_overlap_lo"] > 0 else
                   "VLM higher" if d["d_overlap_hi"] < 0 else "tie (CI spans 0)")
        deltas.append({"vlm": m, **d, "verdict_overlap": verdict})
    dd = pd.DataFrame(deltas)
    print(dd[["vlm", "d_overlap", "d_overlap_lo", "d_overlap_hi", "verdict_overlap",
              "d_kappa", "d_kappa_lo", "d_kappa_hi"]].to_string(index=False))
    dd.to_csv(OUT_DIR / "head_to_head.csv", index=False)

    print("\n" + "=" * 92)
    print("BINARY VIEW — drop the medium class (it is only 12% of human labels, "
          "and the VLMs mostly refuse to emit it)")
    print("=" * 92)
    bin_rows = []
    keep = wide["human_level"] != "medium"
    for label, col in [("ours (pipeline cuts)", "ours_label"),
                       ("ours (marginal-matched)", "ours_marg"),
                       ("ours (accuracy-fitted)", "ours_cal")] + \
                      [(m, f"vlm_{m}") for m in models]:
        pred = wide.loc[keep, col]
        truth = wide.loc[keep, "human_level"]
        ok = pred.notna() & (pred != "medium")
        bin_rows.append({"predictor": label,
                         "N_scored": int(ok.sum()),
                         "N_total": int(keep.sum()),
                         "abstained_medium": int((pred == "medium").sum()),
                         "overlap": round(float((pred[ok] == truth[ok]).mean()), 4)})
    bin_df = pd.DataFrame(bin_rows).sort_values("overlap", ascending=False)
    print(bin_df.to_string(index=False))
    bin_df.to_csv(OUT_DIR / "overlap_binary.csv", index=False)

    o = tbl[tbl["predictor"] == "ours"].iloc[0]
    best = (
        main_tbl[~main_tbl["predictor"].str.startswith("ours")]
        .sort_values("overlap", ascending=False)
        .iloc[0]
    )
    oc = tbl[tbl["predictor"] == "ours_cal"].iloc[0]
    print(f"\n[verdict] ours pipeline  : overlap={o['overlap']:.4f} kappa={o['kappa']:+.4f} "
          f"spearman={o['spearman']:+.4f}")
    print(f"[verdict] ours calibrated: overlap={oc['overlap']:.4f} kappa={oc['kappa']:+.4f} "
          f"spearman={oc['spearman']:+.4f}")
    print(f"[verdict] best VLM : {best['label']}: overlap={best['overlap']:.4f} kappa={best['kappa']:+.4f}")
    print(f"[verdict] ours(pipeline)   - bestVLM: dOverlap={o['overlap']-best['overlap']:+.4f} "
          f"dKappa={o['kappa']-best['kappa']:+.4f}")
    print(f"[verdict] ours(calibrated) - bestVLM: dOverlap={oc['overlap']-best['overlap']:+.4f} "
          f"dKappa={oc['kappa']-best['kappa']:+.4f}")
    print(f"\n[note] the bucket's own overlap/summary.csv scores VLMs against the "
          f"kinescore band = OUR method, so ours is 100% there by construction; "
          f"this table uses HUMAN labels as the reference instead.")
    print(f"[out] {OUT_DIR}")


if __name__ == "__main__":
    main()
