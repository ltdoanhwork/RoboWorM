#!/usr/bin/env python3
"""Geometry-consistency human labels vs. machine combined_score.

Pulls the LIVE human labels from the Supabase `labels` table used by the
labeling site (tmp/20261005_refactor/web/webpage/labeling), keeps only the geometry-consistency
(multiview, numeric item_id) track, joins them to `combined_score` from
`sampling_dataset/data_for_web/data_for_web.csv` via the clip metadata
(`assessment/multiview_static2/clips/<id>/meta.json` -> `source_path`), then

  1. prints Spearman / Kendall / Pearson per robot type, and
  2. renders the per-robot box plot (combined_score vs human score 1/2/3).

Usage:
    python3 scripts/tables/analyze_geometry_label_correlation.py            # fetch live
    python3 scripts/tables/analyze_geometry_label_correlation.py --offline  # cached json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "sampling_dataset/data_for_web/data_for_web.csv"
CLIPS_DIR = REPO / "sampling_dataset/data_for_web/assessment/multiview_static2/clips"
CONFIG_JS = REPO / "tmp/20261005_refactor/web/webpage/labeling/config.js"
OUT_DIR = REPO / "results/geometry_correlation"
LABEL_CACHE = OUT_DIR / "labels_server.json"

SCORE_NAMES = {1: "Bad", 2: "Medium", 3: "Good"}
PALETTE = {1: "#e06666", 2: "#e8b44a", 3: "#5cb87f"}
ROBOT_ORDER = ["bimanual", "humanoid", "single_arm"]


# --------------------------------------------------------------------------- #
# server


def read_supabase_config() -> tuple[str, str]:
    text = CONFIG_JS.read_text()
    url = re.search(r'SUPABASE_URL:\s*"([^"]+)"', text)
    key = re.search(r'SUPABASE_PUBLISHABLE_KEY:\s*"([^"]+)"', text)
    if not url or not key:
        raise SystemExit(f"could not parse Supabase config from {CONFIG_JS}")
    return url.group(1), key.group(1)


def fetch_labels(page: int = 1000) -> list[dict]:
    base, key = read_supabase_config()
    url = (
        f"{base}/rest/v1/labels?select=rater_name,item_id,score,updated_at"
        "&order=item_id.asc,rater_name.asc"
    )
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows: list[dict] = []
    start = 0
    while True:
        req = urllib.request.Request(
            url, headers={**headers, "Range": f"{start}-{start + page - 1}"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            chunk = json.load(resp)
        rows.extend(chunk)
        if len(chunk) < page:
            break
        start += page
        if start > 200_000:
            break
    return rows


def load_labels(offline: bool) -> list[dict]:
    if offline:
        if not LABEL_CACHE.exists():
            raise SystemExit(f"no cached labels at {LABEL_CACHE}; run without --offline")
        return json.loads(LABEL_CACHE.read_text())
    rows = fetch_labels()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_CACHE.write_text(json.dumps(rows))
    return rows


# --------------------------------------------------------------------------- #
# join


def extract_robot_type(path_str: str) -> str:
    parts = Path(path_str).parts
    for r_type in ("bimanual", "humanoid", "single_arm"):
        if r_type in parts:
            return r_type
    return "unknown"


def clip_index() -> pd.DataFrame:
    recs = []
    for meta_path in sorted(CLIPS_DIR.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        recs.append(
            {
                "item_id": m["id"],
                "video_path": m.get("source_path"),
                "task": m.get("task"),
                "model": m.get("model"),
                "split": m.get("split"),
                "pool": m.get("pool"),
                "episode": m.get("episode"),
            }
        )
    return pd.DataFrame(recs)


def build_frames(label_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame(label_rows)
    # geometry-consistency track = whole-video multiview items ("00042");
    # hallucination track uses segment ids ("hv014_s011").
    labels = labels[labels["item_id"].str.fullmatch(r"\d+")].copy()

    metrics = pd.read_csv(CSV_PATH)
    # the csv already carries robot_type; path parsing only as a fallback
    # (39 single_arm rows live under tmp/lowquality_multiview/ and have no
    # embodiment component in their path).
    from_path = metrics["video_path"].map(extract_robot_type)
    metrics["robot_type"] = metrics["robot_type"].fillna("unknown").where(
        metrics["robot_type"].notna(), from_path
    )
    clips = clip_index()

    metrics = metrics.rename(columns={"score": "csv_score"})
    per_label = labels.merge(clips, on="item_id", how="inner").merge(
        metrics, on="video_path", how="inner"
    )

    # one row per video = mean human vote (only items where raters agree keep an
    # integer bin; disagreement -> .5 which we drop from the binned plot).
    agg = (
        per_label.groupby(["item_id", "video_path", "robot_type", "combined_score"])
        .agg(score_mean=("score", "mean"), n_raters=("score", "size"))
        .reset_index()
    )
    agg["score"] = agg["score_mean"]
    consensus = agg[agg["score_mean"].mod(1) == 0].copy()
    consensus["score"] = consensus["score_mean"].astype(int)
    return per_label, consensus


# --------------------------------------------------------------------------- #
# stats


def evaluate_correlation_by_robot(data_df: pd.DataFrame, metric_col: str = "combined_score"):
    robots = sorted(data_df["robot_type"].unique())
    summary_rows = []
    for robot in robots:
        sub_df = data_df[data_df["robot_type"] == robot]
        n_samples = len(sub_df)
        spearman_corr, spearman_p = stats.spearmanr(sub_df[metric_col], sub_df["score"])
        kendall_corr, kendall_p = stats.kendalltau(sub_df[metric_col], sub_df["score"])
        pearson_corr, pearson_p = stats.pearsonr(sub_df[metric_col], sub_df["score"])

        if spearman_p > 0.05:
            status = "Không có ý nghĩa thống kê (p > 0.05)"
        elif abs(spearman_corr) >= 0.7:
            status = "Tương đồng RẤT CAO"
        elif abs(spearman_corr) >= 0.4:
            status = "Tương đồng TRUNG BÌNH"
        else:
            status = "Tương đồng YẾU"

        print(f"🤖 Robot Type: [{robot.upper()}] (N = {n_samples})")
        print(f"   • Spearman: {spearman_corr:.4f} (p-val: {spearman_p:.4e})")
        print(f"   • Kendall : {kendall_corr:.4f} (p-val: {kendall_p:.4e})")
        print(f"   • Pearson : {pearson_corr:.4f} (p-val: {pearson_p:.4e})")
        print(f"  => Final by Spearman {status}\n")

        summary_rows.append(
            {
                "Robot": robot,
                "N": n_samples,
                "Spearman": round(spearman_corr, 4),
                "Spearman p-val": round(spearman_p, 4),
                "Kendall": round(kendall_corr, 4),
                "Pearson": round(pearson_corr, 4),
                "Assessment": status,
            }
        )
    return pd.DataFrame(summary_rows)


# --------------------------------------------------------------------------- #
# plot


def boxplot_by_robot(
    data_df: pd.DataFrame,
    out_png: Path,
    metric_col: str = "combined_score",
    title: str | None = None,
):
    robots = [r for r in ROBOT_ORDER if r in set(data_df["robot_type"])]
    robots += [r for r in sorted(set(data_df["robot_type"])) if r not in robots]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, len(robots), figsize=(4.2 * len(robots), 4.0), sharey=True)
    if len(robots) == 1:
        axes = [axes]

    order = [1, 2, 3]
    for ax, robot in zip(axes, robots):
        sub = data_df[data_df["robot_type"] == robot]
        sns.boxplot(
            data=sub,
            x="score",
            y=metric_col,
            order=order,
            hue="score",
            hue_order=order,
            palette=PALETTE,
            legend=False,
            width=0.6,
            fliersize=3,
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="score",
            y=metric_col,
            order=order,
            hue="score",
            hue_order=order,
            palette=PALETTE,
            legend=False,
            size=3.5,
            alpha=0.75,
            jitter=0.22,
            linewidth=0,
            dodge=False,
            ax=ax,
        )
        counts = sub["score"].value_counts()
        ax.set_title(robot)
        ax.set_xlabel("Human Score")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            [f"{s}\n{SCORE_NAMES[s]} (n={int(counts.get(s, 0))})" for s in order]
        )
        ax.set_ylabel(metric_col if ax is axes[0] else "")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use cached labels json")
    ap.add_argument(
        "--level",
        choices=["label", "consensus", "both"],
        default="both",
        help="label = one row per (rater, video); consensus = one row per video",
    )
    args = ap.parse_args()

    rows = load_labels(args.offline)
    per_label, consensus = build_frames(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_label.to_csv(OUT_DIR / "geometry_labels_joined.csv", index=False)
    consensus.to_csv(OUT_DIR / "geometry_consensus_joined.csv", index=False)

    n_raters = per_label["rater_name"].nunique()
    print(
        f"[data] {len(rows)} label rows on server | geometry: {len(per_label)} votes "
        f"on {per_label['item_id'].nunique()}/300 videos by {n_raters} raters"
    )
    print(f"[data] consensus rows (raters agree / single rater): {len(consensus)}")
    agree = per_label.groupby("item_id")["score"].nunique()
    print(
        f"[data] videos with >=2 raters: {int((per_label.groupby('item_id').size() >= 2).sum())}, "
        f"of which disagreeing: {int((agree > 1).sum())}\n"
    )

    if args.level in ("label", "both"):
        print("=" * 78)
        print("LABEL LEVEL — one row per (rater, video)")
        print("=" * 78)
        summary = evaluate_correlation_by_robot(per_label)
        print(summary.to_string(index=False))
        summary.to_csv(OUT_DIR / "correlation_summary_label_level.csv", index=False)
        boxplot_by_robot(
            per_label,
            OUT_DIR / "geometry_boxplot_label_level.png",
            title="combined_score vs human geometry score (one row per rater vote)",
        )
        print()

    if args.level in ("consensus", "both"):
        print("=" * 78)
        print("VIDEO LEVEL — one row per video (rater consensus)")
        print("=" * 78)
        summary_c = evaluate_correlation_by_robot(consensus)
        print(summary_c.to_string(index=False))
        summary_c.to_csv(OUT_DIR / "correlation_summary_consensus.csv", index=False)
        boxplot_by_robot(
            consensus,
            OUT_DIR / "geometry_boxplot_consensus.png",
            title="combined_score vs human geometry score (video-level consensus)",
        )


if __name__ == "__main__":
    main()
