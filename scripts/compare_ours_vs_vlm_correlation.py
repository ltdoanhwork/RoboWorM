#!/usr/bin/env python3
"""Ours (combined_score) vs VLM judges — which agrees more with humans?

Human reference = the resolved geometry label used by the labeling site:
two agreeing raters, else the geometry arbiter (thanh_check's full 300-clip
pass, i.e. the `score` column of data_for_web.csv). See
docs/labeling-site-architecture.md section 6.

VLM predictions live in sampling_dataset/data_for_web/relative_path/
{baseline,cot}/turn{1,2,3}/{gemini,gemma,qwen_open,qwen_close}.csv, keyed by
`video_path` on the same 1/2/3 scale as humans.

Fairness note: our metric is continuous while a single VLM turn emits only
3 levels, and heavy ties deflate Spearman. So we report
  * Spearman / Kendall tau-b  (standard rank correlation), and
  * pairwise ranking accuracy (AUC-style, ties = 0.5, over human-discordant
    pairs) which is tie-fair and comparable across continuous/discrete, and
  * the 3-turn mean per VLM, which gives each judge the benefit of
    self-consistency ensembling before we compare.

Usage:
    python3 scripts/compare_ours_vs_vlm_correlation.py            # fetch live labels
    python3 scripts/compare_ours_vs_vlm_correlation.py --offline  # cached
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CSV_PATH = REPO / "sampling_dataset/data_for_web/data_for_web.csv"
CLIPS_DIR = REPO / "sampling_dataset/data_for_web/assessment/multiview_static2/clips"
VLM_ROOT = REPO / "sampling_dataset/data_for_web/relative_path"
OUT_DIR = REPO / "results/ours_vs_vlm"

REVIEWERS = {"quang_check", "thanh_check"}
GEO_ARBITER = "thanh_check"

OURS = "combined_score"
VLM_MODELS = ["gemini", "gemma", "qwen_open", "qwen_close"]
STRATEGIES = ["baseline", "cot"]
TURNS = [1, 2, 3]

SCORE_NAMES = {1: "Bad", 2: "Medium", 3: "Good"}
PALETTE = {1: "#e06666", 2: "#e8b44a", 3: "#5cb87f"}
ROBOT_ORDER = ["bimanual", "humanoid", "single_arm"]


# --------------------------------------------------------------------------- #
# human reference


def load_labels(offline: bool) -> list[dict]:
    """Reuse the fetch/cache from the sibling geometry script."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "geo_corr", REPO / "scripts/analyze_geometry_label_correlation.py"
    )
    geo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geo)
    return geo.load_labels(offline)


def resolved_human(label_rows: list[dict], arbiter_by_item: dict[str, int]) -> pd.DataFrame:
    """One row per geometry item: the site's resolved human score."""
    labels = pd.DataFrame(label_rows)
    labels = labels[labels["item_id"].str.fullmatch(r"\d+")].copy()

    recs = []
    for item_id, grp in labels.groupby("item_id"):
        regular = grp[~grp["rater_name"].isin(REVIEWERS)]
        scores = regular["score"].tolist()
        uniq = set(scores)
        arb = arbiter_by_item.get(item_id)

        if len(scores) >= 2 and len(uniq) == 1:
            human, consensus = scores[0], "raters"
        elif len(scores) >= 2 and arb is not None:
            human, consensus = arb, "arbiter"
        elif len(scores) >= 2:
            human, consensus = None, "unresolved"
        else:
            human, consensus = None, "insufficient"

        recs.append(
            {
                "item_id": item_id,
                "human": human,
                "consensus": consensus,
                "n_raters": len(scores),
                "rater_scores": ",".join(str(s) for s in sorted(scores)),
            }
        )
    return pd.DataFrame(recs)


def clip_index() -> pd.DataFrame:
    recs = []
    for meta_path in sorted(CLIPS_DIR.glob("*/meta.json")):
        m = json.loads(meta_path.read_text())
        recs.append({"item_id": m["id"], "video_path": m.get("source_path")})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# VLM predictions


def load_vlm() -> pd.DataFrame:
    """Long frame: video_path, strategy, turn, model, vlm_score."""
    frames = []
    for strat in STRATEGIES:
        for turn in TURNS:
            for model in VLM_MODELS:
                path = VLM_ROOT / strat / f"turn{turn}" / f"{model}.csv"
                if not path.exists():
                    raise SystemExit(f"missing VLM file: {path}")
                raw = pd.read_csv(path)
                # some runs were resumed and re-emitted the header mid-file
                raw = raw[raw["score"].astype(str) != "score"].copy()
                raw["score"] = pd.to_numeric(raw["score"], errors="coerce")
                bad = int(raw["score"].isna().sum())
                raw = raw.dropna(subset=["score"])
                raw = raw.drop_duplicates(subset=["video_path"], keep="last")
                frames.append(
                    pd.DataFrame(
                        {
                            "video_path": raw["video_path"],
                            "strategy": strat,
                            "turn": turn,
                            "model": model,
                            "vlm_score": raw["score"].astype(int),
                        }
                    )
                )
                if bad:
                    print(f"[warn] {strat}/turn{turn}/{model}: dropped {bad} unparsable score rows")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# tie-fair agreement


def ranking_accuracy(pred: np.ndarray, human: np.ndarray) -> tuple[float, int]:
    """P(pred orders a pair correctly | humans disagree); ties count 0.5.

    Equivalent to a multi-class AUC / rescaled Somers' D. Unlike Spearman it
    does not punish a predictor for having only 3 distinct levels, so a
    continuous metric and a 3-level VLM are compared on equal footing.
    """
    pred = np.asarray(pred, dtype=float)
    human = np.asarray(human, dtype=float)
    conc = 0.0
    total = 0
    for i, j in itertools.combinations(range(len(human)), 2):
        if human[i] == human[j]:
            continue
        total += 1
        hi_is_i = human[i] > human[j]
        if pred[i] == pred[j]:
            conc += 0.5
        elif (pred[i] > pred[j]) == hi_is_i:
            conc += 1.0
    return (conc / total if total else float("nan")), total


def score_predictor(df: pd.DataFrame, col: str) -> dict:
    sub = df[[col, "human"]].dropna()
    if len(sub) < 5 or sub[col].nunique() < 2:
        return {}
    sp, sp_p = stats.spearmanr(sub[col], sub["human"])
    kt, kt_p = stats.kendalltau(sub[col], sub["human"], variant="b")
    pe, pe_p = stats.pearsonr(sub[col], sub["human"])
    acc, npairs = ranking_accuracy(sub[col].to_numpy(), sub["human"].to_numpy())
    return {
        "N": len(sub),
        "Spearman": round(float(sp), 4),
        "Spearman_p": float(f"{sp_p:.3g}"),
        "Kendall_b": round(float(kt), 4),
        "Kendall_p": float(f"{kt_p:.3g}"),
        "Pearson": round(float(pe), 4),
        "RankAcc": round(float(acc), 4),
        "n_pairs": npairs,
        "n_levels": int(sub[col].nunique()),
    }


def bootstrap_delta(df: pd.DataFrame, col_a: str, col_b: str, n_boot: int, seed: int) -> dict:
    """Paired bootstrap on the SAME clips: Spearman/RankAcc advantage of a over b."""
    sub = df[[col_a, col_b, "human"]].dropna()
    rng = np.random.default_rng(seed)
    idx = np.arange(len(sub))
    a, b, h = (sub[col_a].to_numpy(), sub[col_b].to_numpy(), sub["human"].to_numpy())
    d_sp, d_acc = [], []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        hs = h[pick]
        if len(np.unique(hs)) < 2:
            continue
        sa = stats.spearmanr(a[pick], hs).statistic
        sb = stats.spearmanr(b[pick], hs).statistic
        if np.isnan(sa) or np.isnan(sb):
            continue
        d_sp.append(sa - sb)
        d_acc.append(
            ranking_accuracy(a[pick], hs)[0] - ranking_accuracy(b[pick], hs)[0]
        )
    d_sp, d_acc = np.array(d_sp), np.array(d_acc)
    return {
        "d_spearman": round(float(d_sp.mean()), 4),
        "d_spearman_lo": round(float(np.percentile(d_sp, 2.5)), 4),
        "d_spearman_hi": round(float(np.percentile(d_sp, 97.5)), 4),
        "p_ours_worse_sp": round(float((d_sp <= 0).mean()), 4),
        "d_rankacc": round(float(d_acc.mean()), 4),
        "d_rankacc_lo": round(float(np.percentile(d_acc, 2.5)), 4),
        "d_rankacc_hi": round(float(np.percentile(d_acc, 97.5)), 4),
        "p_ours_worse_acc": round(float((d_acc <= 0).mean()), 4),
        "n_boot": int(len(d_sp)),
    }


# --------------------------------------------------------------------------- #
# plots


def _panel(ax, sub, ycol, title, ylabel, discrete):
    order = [1, 2, 3]
    sub = sub.copy()
    # human arrives as float (NaN-carrying column); order=[1,2,3] would then
    # match nothing and seaborn silently draws no boxes.
    sub["human"] = sub["human"].astype(int)
    sns.boxplot(
        data=sub, x="human", y=ycol, order=order, hue="human", hue_order=order,
        palette=PALETTE, legend=False, width=0.6, fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=sub, x="human", y=ycol, order=order, hue="human", hue_order=order,
        palette=PALETTE, legend=False, size=3.2, alpha=0.6,
        jitter=0.3 if discrete else 0.22, linewidth=0, dodge=False, ax=ax,
    )
    counts = sub["human"].value_counts()
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Human score (resolved)")
    ax.set_xticks(range(len(order)))
    # short labels: the full "Medium (n=68)" form collides between panels
    ax.set_xticklabels([f"{SCORE_NAMES[s]}\nn={int(counts.get(s,0))}" for s in order])
    ax.set_ylabel(ylabel)


def boxplot_ours_vs_vlm(df: pd.DataFrame, stats_tbl: pd.DataFrame, out_png: Path,
                        strategy: str) -> None:
    """Row 1: ours. Row 2: each VLM (3-turn mean). x = human score."""
    cols = [("ours", OURS, "combined_score (ours)")]
    cols += [(m, f"vlm_{m}", f"{m} (3-turn mean)") for m in VLM_MODELS]

    sns.set_theme(style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, len(cols), figsize=(3.5 * len(cols), 4.2))
    for ax, (key, col, label) in zip(axes, cols):
        sub = df[[col, "human"]].dropna().rename(columns={col: "y"})
        row = stats_tbl[stats_tbl["predictor"] == key]
        sp = row["Spearman"].iloc[0] if len(row) else float("nan")
        acc = row["RankAcc"].iloc[0] if len(row) else float("nan")
        _panel(ax, sub, "y", f"{label}\nρ={sp:+.3f}  rankAcc={acc:.3f}",
               "score" if ax is axes[0] else "", key != "ours")
    fig.suptitle(
        f"Agreement with human geometry labels — ours vs VLM judges ({strategy}, N={int(df['human'].notna().sum())})",
        fontsize=12,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def boxplot_by_robot(df: pd.DataFrame, out_png: Path, strategy: str) -> None:
    """Ours vs best VLM, split per robot type."""
    robots = [r for r in ROBOT_ORDER if r in set(df["robot_type"])]
    cols = [("ours", OURS, "ours")] + [(m, f"vlm_{m}", m) for m in VLM_MODELS]

    sns.set_theme(style="whitegrid", font_scale=0.85)
    fig, axes = plt.subplots(len(cols), len(robots),
                             figsize=(3.4 * len(robots), 2.9 * len(cols)))
    for r_i, (key, col, label) in enumerate(cols):
        for c_i, robot in enumerate(robots):
            ax = axes[r_i][c_i]
            sub = df[(df["robot_type"] == robot)][[col, "human"]].dropna()
            sub = sub.rename(columns={col: "y"})
            if len(sub) >= 5 and sub["y"].nunique() >= 2 and sub["human"].nunique() >= 2:
                sp = stats.spearmanr(sub["y"], sub["human"]).statistic
                acc = ranking_accuracy(sub["y"].to_numpy(), sub["human"].to_numpy())[0]
                ttl = f"{label} · {robot}\nρ={sp:+.3f} acc={acc:.3f}"
            else:
                ttl = f"{label} · {robot}\n(n<5)"
            _panel(ax, sub, "y", ttl, label if c_i == 0 else "", key != "ours")
            if r_i != len(cols) - 1:
                ax.set_xlabel("")
    fig.suptitle(f"Per-robot agreement with human labels ({strategy})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


def barplot_summary(all_stats: pd.DataFrame, out_png: Path) -> None:
    """Ours vs every VLM/strategy/turn on both metrics."""
    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ref in ((axes[0], "Spearman", 0.0), (axes[1], "RankAcc", 0.5)):
        d = all_stats.sort_values(metric, ascending=False)
        colors = ["#2d6cdf" if p == "ours" else "#b0b6c0" for p in d["label"]]
        ax.barh(d["label"], d[metric], color=colors)
        ax.axvline(ref, color="#888", lw=1, ls="--")
        ax.invert_yaxis()
        ax.set_xlabel(metric + ("  (0 = no correlation)" if metric == "Spearman"
                                else "  (0.5 = chance)"))
        for y, v in enumerate(d[metric]):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
    fig.suptitle("Ours (blue) vs VLM judges — agreement with resolved human labels", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out_png}")


# --------------------------------------------------------------------------- #


def build(offline: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(CSV_PATH).rename(columns={"score": "arbiter_score"})
    clips = clip_index()
    arb = metrics.merge(clips, on="video_path", how="inner")
    arbiter_by_item = dict(zip(arb["item_id"], arb["arbiter_score"]))

    human = resolved_human(load_labels(offline), arbiter_by_item)
    base = clips.merge(metrics, on="video_path", how="inner").merge(
        human, on="item_id", how="left"
    )

    vlm = load_vlm()
    # per-turn wide + 3-turn mean per model, per strategy
    wide = {}
    for strat in STRATEGIES:
        w = base.copy()
        for model in VLM_MODELS:
            sub = vlm[(vlm["strategy"] == strat) & (vlm["model"] == model)]
            piv = sub.pivot_table(index="video_path", columns="turn", values="vlm_score")
            piv.columns = [f"vlm_{model}_t{t}" for t in piv.columns]
            piv[f"vlm_{model}"] = piv.mean(axis=1)
            w = w.merge(piv.reset_index(), on="video_path", how="left")
        wide[strat] = w
    return wide, vlm


def stats_table(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    rows = []
    ours = score_predictor(df, OURS)
    if ours:
        rows.append({"predictor": "ours", "label": "ours", "strategy": "-", "turn": "-", **ours})
    for model in VLM_MODELS:
        m = score_predictor(df, f"vlm_{model}")
        if m:
            rows.append({"predictor": model, "label": f"{model} ({strategy}, 3-turn mean)",
                         "strategy": strategy, "turn": "mean", **m})
        for t in TURNS:
            col = f"vlm_{model}_t{t}"
            if col not in df.columns:
                continue
            s = score_predictor(df, col)
            if s:
                rows.append({"predictor": f"{model}_t{t}",
                             "label": f"{model} ({strategy}, turn{t})",
                             "strategy": strategy, "turn": t, **s})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--human",
        choices=["resolved", "raters", "arbiter"],
        default="resolved",
        help="resolved = site rule (2 agreeing raters, else arbiter); "
        "raters = only clips where 2 raters agreed, so the label is fully "
        "independent of the CSV that also holds combined_score (leakage check); "
        "arbiter = thanh_check's full 300-clip pass",
    )
    args = ap.parse_args()

    wide, vlm_long = build(args.offline)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.human == "raters":
        wide = {k: v[v["consensus"] == "raters"].copy() for k, v in wide.items()}
    elif args.human == "arbiter":
        for k, v in wide.items():
            v = v.copy()
            v["human"] = v["arbiter_score"]
            wide[k] = v

    suffix = "" if args.human == "resolved" else f"_{args.human}"

    ref = wide["baseline"]
    n_res = int(ref["human"].notna().sum())
    print(f"[data] 300 geometry clips | human reference = {args.human} | N={n_res}")
    print("[data] consensus mix:", ref["consensus"].value_counts().to_dict())
    print(f"[data] VLM rows: {len(vlm_long)} "
          f"({vlm_long['model'].nunique()} models x {len(STRATEGIES)} strategies x {len(TURNS)} turns)\n")

    all_stats = []
    for strat in STRATEGIES:
        df = wide[strat]
        tbl = stats_table(df, strat)
        print("=" * 96)
        print(f"{strat.upper()} — agreement with resolved human label (N={n_res})")
        print("=" * 96)
        cols = ["label", "N", "n_levels", "Spearman", "Spearman_p", "Kendall_b", "RankAcc"]
        print(tbl.sort_values("Spearman", ascending=False)[cols].to_string(index=False))
        print()
        tbl.to_csv(OUT_DIR / f"stats_{strat}{suffix}.csv", index=False)
        boxplot_ours_vs_vlm(df, tbl, OUT_DIR / f"boxplot_ours_vs_vlm_{strat}{suffix}.png", strat)
        boxplot_by_robot(df, OUT_DIR / f"boxplot_by_robot_{strat}{suffix}.png", strat)
        keep = tbl[tbl["turn"].isin(["mean", "-"])].copy()
        keep["strategy"] = strat
        all_stats.append(keep)

    combined = pd.concat(all_stats, ignore_index=True)
    combined = combined.drop_duplicates(subset=["label"])
    combined.to_csv(OUT_DIR / f"stats_all{suffix}.csv", index=False)
    barplot_summary(combined, OUT_DIR / f"summary_ours_vs_vlm{suffix}.png")

    print("=" * 96)
    print("HEAD-TO-HEAD — paired bootstrap on the same clips (ours minus VLM)")
    print("=" * 96)
    deltas = []
    for strat in STRATEGIES:
        df = wide[strat]
        for model in VLM_MODELS:
            d = bootstrap_delta(df, OURS, f"vlm_{model}", args.n_boot, args.seed)
            verdict = ("OURS higher" if d["d_spearman_lo"] > 0 else
                       "VLM higher" if d["d_spearman_hi"] < 0 else "tie (CI spans 0)")
            deltas.append({"strategy": strat, "vlm": model, **d, "verdict_spearman": verdict})
    dd = pd.DataFrame(deltas)
    show = ["strategy", "vlm", "d_spearman", "d_spearman_lo", "d_spearman_hi",
            "verdict_spearman", "d_rankacc", "d_rankacc_lo", "d_rankacc_hi"]
    print(dd[show].to_string(index=False))
    dd.to_csv(OUT_DIR / f"head_to_head{suffix}.csv", index=False)

    best = combined[combined["predictor"] != "ours"].sort_values("Spearman", ascending=False)
    ours_row = combined[combined["predictor"] == "ours"].iloc[0]
    print(f"\n[verdict] ours Spearman={ours_row['Spearman']:+.4f} rankAcc={ours_row['RankAcc']:.4f}")
    if len(best):
        b = best.iloc[0]
        print(f"[verdict] best VLM = {b['label']}: Spearman={b['Spearman']:+.4f} rankAcc={b['RankAcc']:.4f}")
        print(f"[verdict] ours - bestVLM: dSpearman={ours_row['Spearman']-b['Spearman']:+.4f} "
              f"dRankAcc={ours_row['RankAcc']-b['RankAcc']:+.4f}")
    print(f"\n[out] {OUT_DIR}")


if __name__ == "__main__":
    main()
