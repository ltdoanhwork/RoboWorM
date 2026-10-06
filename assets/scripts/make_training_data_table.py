#!/usr/bin/env python3
"""Emit the appendix training-data-preparation table.

The appendix promises "training data preparation" alongside the hyperparameter
matrices. Unlike the hyperparameters, the interesting facts here are not in a
config file: the corpora are built by a multi-stage ffmpeg/T5 pipeline whose
effect is only visible in the clip counts it produces. This script therefore
does two things:

  1. Reads the *recipe* from code (never restated by hand):
       * DreamGen  -- the ``L(Dataset)`` registrations in
                      ``models/dreamgen/.../configs/base/experiment/singleview.py``
                      give each released checkpoint's corpus directory,
                      ``num_frames``, ``video_size`` and ``override_fps``; the
                      grid / segment / re-encode stage parameters come from
                      ``finetuning/dreamgen/scripts/*.py`` argparse defaults.
       * Ctrl-World -- the latent-precomputation fields of
                      ``finetuning/ctrlworld/config.py``.

  2. *Measures* the corpora on disk with ffprobe, because the script comments
     are stale. The humanoid comment claims segmentation "recovers ~204/248
     clips"; the measured figure is 132/248 usable before and 373/373 after.
     We report the measurement and record the discrepancy.

The frame filter is the load-bearing quantity: the dataloader can only sample a
clip that has at least ``num_frames`` frames, so a corpus's *usable* size is not
its file count. Measuring it is what turns the pipeline from an undocumented
preprocessing detail into a justified design decision.

ffprobe over ~4k clips takes a few minutes, so measurements are cached in the
output JSON and reused unless --remeasure is passed.

    python3 assets/scripts/make_training_data_table.py [--remeasure]

Writes:
    results/reports/training_data_prep.json
    assets/paper/appendix/tables/training_data_prep.tex
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DG_EXP = os.path.join(
    REPO, "models/dreamgen/cosmos_predict2/configs/base/experiment/singleview.py")
DG_SCRIPTS = os.path.join(REPO, "finetuning/dreamgen/scripts")
CW_CONFIG = os.path.join(REPO, "finetuning/ctrlworld/config.py")
FT_JSON = os.path.join(REPO, "results/reports/finetune_config.json")

OUT_JSON = os.path.join(REPO, "results/reports/training_data_prep.json")
OUT_TEX = os.path.join(REPO, "assets/paper/appendix/tables/training_data_prep.tex")

# Released DreamGen checkpoints -> the corpus stage chain that produced them.
# The chain is expressed as dataset *directory* names so every entry is checked
# against the registrations parsed out of the experiment file.
DG_CHAINS = {
    "humanoid_multiview_seg16": [
        ("$2{\\times}2$ grid stitch", "humanoid_multiview_grid_dreamgen"),
        ("segment", "humanoid_multiview_grid_seg_dreamgen"),
        ("re-encode 16\\,fps", "humanoid_multiview_grid_seg16_dreamgen"),
    ],
    "bimanual_multiview_seg16": [
        ("$2{\\times}2$ grid stitch", "bimanual_multiview_grid_dreamgen"),
        ("segment", "bimanual_multiview_grid_seg_dreamgen"),
        ("re-encode 16\\,fps", "bimanual_multiview_grid_seg16_dreamgen"),
    ],
    "humanoid_singleview_16fps": [
        ("transcode 16\\,fps", "humanoid_dreamgen_16fps"),
    ],
    "bimanual_singleview_16fps": [
        ("transcode 16\\,fps", "bimanual_dreamgen_16fps"),
    ],
    "single_arm_static_singleview_16fps": [
        ("transcode 16\\,fps + segment", "single_arm_static_dreamgen_16fps"),
    ],
}


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def parse_dg_datasets():
    """dataset_dir / num_frames / video_size / override_fps per registration."""
    src = open(DG_EXP).read()
    out = {}
    for m in re.finditer(r"_dataset_(\w+) = L\(Dataset\)\((.*?)\n\)", src, re.S):
        body = m.group(2)
        d = re.search(r'"([\w]+)"\)', body)
        nf = re.search(r"num_frames=(\d+)", body)
        vs = re.search(r"video_size=\((\d+), (\d+)\)", body)
        ofps = re.search(r"override_fps=(\d+)", body)
        out[m.group(1)] = {
            "dir": d.group(1) if d else None,
            "num_frames": int(nf.group(1)) if nf else None,
            "video_size": "%sx%s" % (vs.group(1), vs.group(2)) if vs else None,
            "override_fps": int(ofps.group(1)) if ofps else None,
        }
    return out


def parse_stage_params():
    """argparse defaults of the prep stages (window, stride, grid fps)."""
    def defaults(fname, keys):
        path = os.path.join(DG_SCRIPTS, fname)
        if not os.path.exists(path):
            return {}
        src = open(path).read()
        got = {}
        for k in keys:
            m = re.search(r'"--%s",\s*type=\w+,\s*default=([0-9.]+)' % k, src)
            if not m:
                m = re.search(r'"--%s", type=\w+, default=([0-9.]+)' % k, src)
            if m:
                got[k] = float(m.group(1))
        return got

    seg_h = defaults("segment_humanoid_multiview_grid_dreamgen.py",
                     ["win", "stride", "min-keep-whole"])
    seg_b = defaults("segment_bimanual_multiview_grid_dreamgen.py",
                     ["win", "stride", "min-keep-whole"])
    grid_src = open(os.path.join(
        DG_SCRIPTS, "prepare_humanoid_multiview_grid_dreamgen.py")).read()
    grid_fps = re.search(r'"--fps", type=int,\s*default=(\d+)', grid_src)
    layouts = re.findall(r'"(TL|TR|BL|BR)": \("observation\.images\.(\w+)"', grid_src)
    return {
        "segment_humanoid": seg_h,
        "segment_bimanual": seg_b,
        "grid_fps": int(grid_fps.group(1)) if grid_fps else None,
        "grid_slots": layouts,
    }


def parse_ctrlworld():
    tree = ast.parse(open(CW_CONFIG).read())
    fields = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FinetuneConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                    try:
                        fields[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, SyntaxError):
                        pass
    return fields


def stream_probe(path):
    """(frame_count, fps) from container metadata."""
    raw = sh("ffprobe -v error -select_streams v:0 "
             "-show_entries stream=nb_frames,avg_frame_rate,duration "
             "-of default=nw=1 '%s'" % path)
    f = {}
    for line in raw.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            f[k.strip()] = v.strip()
    fps = None
    try:
        num, den = (f["avg_frame_rate"].split("/") + ["1"])[:2]
        if float(den):
            fps = float(num) / float(den)
    except (KeyError, ValueError):
        pass
    return frame_count_from(f, path), fps


def frame_count_from(f, path):
    nb = f.get("nb_frames", "")
    if nb.isdigit() and int(nb) > 0:
        return int(nb)
    try:
        num, den = (f["avg_frame_rate"].split("/") + ["1"])[:2]
        return int(round(float(f["duration"]) * float(num) / float(den)))
    except (KeyError, ValueError, ZeroDivisionError):
        pass
    slow = sh("ffprobe -v error -select_streams v:0 -count_frames "
              "-show_entries stream=nb_read_frames -of csv=p=0 '%s'" % path).strip()
    try:
        return int(slow)
    except ValueError:
        return -1


def frame_count(path):
    """Frame count from container metadata.

    ``-count_frames`` decodes every frame, which on a network filesystem means
    reading every corpus byte and takes hours over ~6.7k clips. The mp4s here
    are written by ffmpeg and carry an exact ``nb_frames`` in the stream header,
    which agrees with ``nb_read_frames`` and costs one header read. We fall back
    to duration x frame rate, and only then to a full decode, so a container
    without the tag still yields a correct number rather than a silent zero.

    Fields are parsed by *name*: ffprobe emits them in stream order, not in the
    order requested, so positional parsing silently confuses nb_frames with
    duration.
    """
    return stream_probe(path)[0]


def measure(dirname, min_frames):
    """(n_clips, n_usable, has_t5) for a corpus directory."""
    root = os.path.join(REPO, "datasets", dirname)
    if not os.path.isdir(root):
        return None
    clips = []
    t5 = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.endswith(".mp4"):
                clips.append(os.path.join(dp, fn))
            elif fn.endswith((".pickle", ".pkl")):
                t5 += 1
    if not clips:
        return None
    with ThreadPoolExecutor(max_workers=32) as ex:
        probes = list(ex.map(stream_probe, clips))
    counts = [p[0] for p in probes]
    rates = [p[1] for p in probes if p[1]]
    # The stitch fps differs per corpus (humanoid was downsampled to 10 fps,
    # bimanual kept its native 30 fps) and that asymmetry is the reason only one
    # of them needed the length fix, so it is measured rather than assumed.
    fps_vals = sorted({round(r, 3) for r in rates})
    return {
        "clips": len(clips),
        "usable": sum(1 for c in counts if c >= min_frames),
        "t5_embeddings": t5,
        "min_frames": min_frames,
        "median_frames": sorted(counts)[len(counts) // 2],
        "fps": fps_vals[0] if len(fps_vals) == 1 else fps_vals,
    }


CACHE_PATH = os.path.join(REPO, "results/reports/.training_data_cache.json")


def _save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)


def build(remeasure):
    ds = parse_dg_datasets()
    stages = parse_stage_params()
    cw = parse_ctrlworld()
    by_dir = {v["dir"]: v for v in ds.values() if v["dir"]}

    cache = {}
    if not remeasure:
        for src in (CACHE_PATH, OUT_JSON):
            if not os.path.exists(src):
                continue
            try:
                blob = json.load(open(src))
                cache = blob.get("measurements", blob) or {}
                if cache:
                    break
            except (ValueError, OSError):
                cache = {}

    chains = {}
    for exp, chain in DG_CHAINS.items():
        final_dir = chain[-1][1]
        nf = (by_dir.get(final_dir) or {}).get("num_frames") or 93
        rows = []
        for stage, dirname in chain:
            if dirname in cache:
                m = cache[dirname]
            else:
                print("  measuring %s ..." % dirname, flush=True)
                m = measure(dirname, nf)
                if m:
                    cache[dirname] = m
                    # Persist after every corpus: the measurement is the
                    # expensive part and must survive an interrupt.
                    _save_cache(cache)
            rows.append({"stage": stage, "dir": dirname, **(m or {})})
        chains[exp] = {
            "num_frames": nf,
            "video_size": (by_dir.get(final_dir) or {}).get("video_size"),
            "override_fps": (by_dir.get(final_dir) or {}).get("override_fps"),
            "stages": rows,
        }
    return ds, stages, cw, chains, cache


CELL_LABEL = {
    "humanoid_multiview_seg16": ("humanoid", "MV"),
    "bimanual_multiview_seg16": ("bimanual", "MV"),
    "humanoid_singleview_16fps": ("humanoid", "SV"),
    "bimanual_singleview_16fps": ("bimanual", "SV"),
    "single_arm_static_singleview_16fps": ("single-arm", "SV"),
}


def emit_table(chains, stages, path):
    seg = stages["segment_humanoid"]
    win = seg.get("win", 10.0)
    stride = seg.get("stride", 5.0)
    hum = [s for s in chains["humanoid_multiview_seg16"]["stages"] if "clips" in s]
    bim = [s for s in chains["bimanual_multiview_seg16"]["stages"] if "clips" in s]
    hum_gain = hum[-1]["usable"] / float(hum[0]["usable"])
    bim_gain = bim[-1]["usable"] / float(bim[0]["usable"])
    out = [
        "% Generated by assets/scripts/make_training_data_table.py -- do not edit by hand.",
        "% Recipe: models/dreamgen/.../experiment/singleview.py + finetuning/dreamgen/scripts/*.py",
        "% Counts: measured with ffprobe over the corpora on disk (not copied from comments).",
        r"\begin{table}[t]\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{\textbf{Training corpus construction for the finetuned DreamGen"
        r" checkpoints.} Each block traces one released checkpoint's corpus through"
        r" the preprocessing stages that built it. \emph{Clips} counts the video"
        r" files a stage emits; \emph{usable} counts those the dataloader can"
        r" actually sample, i.e.\ with at least the $%d$ frames a training example"
        r" requires. The two quantities come apart whenever a corpus is stitched"
        r" below its source frame rate: the humanoid grid was written at"
        r" $10$\,fps, leaving only $%d$ of $%d$ clips long enough to sample, while"
        r" the bimanual grid kept its native $30$\,fps and lost none. Segmenting"
        r" into overlapping $%g$\,s windows at a $%g$\,s stride and re-encoding to"
        r" a true $16$\,fps therefore serves two different purposes: it is a"
        r" $%.1f\times$ repair of usable data on the humanoid corpus, and a"
        r" $%.1f\times$ augmentation on the bimanual one. In both cases it also"
        r" aligns the frame rate on disk with the $16$\,fps base checkpoint, which"
        r" removes a train/inference mismatch that no clip count would reveal."
        r" Counts and frame rates are measured with \texttt{ffprobe} over the"
        r" corpora on disk rather than copied from the pipeline's logs.}"
        % (hum[0]["min_frames"], hum[0]["usable"], hum[0]["clips"],
           win, stride, hum_gain, bim_gain),
        r"\label{tab:supp_training_data}",
        r"\begin{tabular}{@{}llrrrrr@{}}\toprule",
        r"Corpus & Stage & fps & Clips & Usable & \%\ usable & Median fr. \\",
        r"\midrule",
    ]
    first = True
    for exp, info in chains.items():
        if not first:
            out.append(r"\addlinespace[2pt]")
        first = False
        emb, track = CELL_LABEL.get(exp, (exp, ""))
        label = "%s %s" % (emb, track)
        for i, st in enumerate(info["stages"]):
            if "clips" not in st:
                continue
            pct = 100.0 * st["usable"] / st["clips"]
            fps = st.get("fps")
            fps_s = ("$%g$" % fps) if isinstance(fps, (int, float)) else (
                "$%s$" % "/".join("%g" % v for v in (fps or [])))
            out.append("%s & %s & %s & $%d$ & $%d$ & $%.0f$ & $%d$ \\\\" % (
                label if i == 0 else "", st["stage"], fps_s, st["clips"],
                st["usable"], pct, st["median_frames"]))
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", action="store_true",
                    help="re-run ffprobe instead of reusing cached counts")
    args = ap.parse_args()

    ds, stages, cw, chains, cache = build(args.remeasure)
    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    emit_table(chains, stages, OUT_TEX)

    # The stale comment is itself a finding worth recording: the appendix must
    # cite the measurement, not the comment.
    h = chains["humanoid_multiview_seg16"]["stages"]
    b = chains["bimanual_multiview_seg16"]["stages"]
    discrepancies = [{
        "location": "finetuning/dreamgen/train_humanoid_multiview_grid_16fps.sh "
                    "and configs/base/experiment/singleview.py",
        "claim": "segmenting + re-encoding recovers ~204/248 clips",
        "measured": "%d/%d usable before, %d/%d after" % (
            h[0]["usable"], h[0]["clips"], h[-1]["usable"], h[-1]["clips"]),
        "resolution": "the comment is stale; the appendix reports the measured "
                      "counts and the ratio derived from them",
    }]
    findings = [{
        "finding": "the stitch frame rate differs per embodiment and explains why "
                   "only the humanoid corpus needed the length fix",
        "humanoid_grid_fps": h[0].get("fps"),
        "bimanual_grid_fps": b[0].get("fps"),
        "humanoid_usable_before": "%d/%d" % (h[0]["usable"], h[0]["clips"]),
        "bimanual_usable_before": "%d/%d" % (b[0]["usable"], b[0]["clips"]),
        "effect": "on humanoid the pipeline is a %.1fx repair of usable data; on "
                  "bimanual, where nothing was filtered, it is a %.1fx "
                  "augmentation" % (h[-1]["usable"] / float(h[0]["usable"]),
                                    b[-1]["usable"] / float(b[0]["usable"])),
    }]

    # A corpus whose on-disk frame rate differs from the rate the dataloader
    # reports to the model is a train/inference mismatch that no clip count
    # reveals, so it is derived here rather than left to inspection.
    fps_mismatch = []
    for exp, info in chains.items():
        st = [s for s in info["stages"] if "clips" in s]
        if not st:
            continue
        disk = st[-1].get("fps")
        reported = info.get("override_fps") or disk
        if isinstance(disk, (int, float)) and isinstance(reported, (int, float)) \
                and abs(disk - reported) > 1e-6:
            fps_mismatch.append({
                "experiment": exp,
                "corpus": st[-1]["dir"],
                "fps_on_disk": disk,
                "fps_reported_to_model": reported,
                "relative_error": round(abs(disk - reported) / disk, 4),
                "note": "override_fps relabels the clip rather than resampling it, "
                        "so the model is conditioned on a frame rate the content "
                        "does not have",
            })

    cw_prep = {
        "source": cw.get("dataset_root"),
        "processed_root": cw.get("processed_root"),
        "retained_on_disk": os.path.isdir(
            os.path.join(REPO, cw.get("processed_root", "///"))),
        "source_fps": cw.get("source_fps"),
        "target_fps": cw.get("fps"),
        "downsample_stride": cw.get("downsample_stride"),
        "resolution": "%dx%d" % (cw.get("image_width", 0), cw.get("image_height", 0)),
        "n_views": cw.get("n_views"),
        "num_history": cw.get("num_history"),
        "num_frames": cw.get("num_frames"),
        "val_ratio": cw.get("val_ratio"),
        "action_key": cw.get("action_key"),
        "action_dim": cw.get("action_dim"),
        "precomputed": "VAE latents per view (.pt) + annotation JSON + dataset "
                       "statistics for action normalisation",
    }

    payload = {
        "generated_by": "assets/scripts/make_training_data_table.py",
        "purpose": "Training corpus construction: preprocessing recipe read from "
                   "code, clip counts measured on disk.",
        "dreamgen": {
            "stage_params": stages,
            "chains": chains,
            "pipeline": ["stitch 4 camera streams into a 2x2 grid",
                         "segment long clips into overlapping windows",
                         "precompute T5 text embeddings",
                         "re-encode to a true 16 fps"],
        },
        "ctrlworld": cw_prep,
        "dreamdojo": {
            "note": "DreamDojo consumes the LeRobot datasets directly through its "
                    "MultiVideoActionDataset; actions are encoded by a frozen "
                    "latent action model. The per-experiment dataloader wiring "
                    "resolves from the vendored Cosmos package at run time and is "
                    "not present in this repository.",
            "lam_ckpt": "LAM_400k.ckpt",
        },
        "discrepancies": discrepancies,
        "findings": findings,
        "fps_mismatch": fps_mismatch,
        "measurements": cache,
        "gaps": [{
            "quantity": "Ctrl-World precomputed latent corpus",
            "note": "the intermediate latent corpus (%s) was not retained after "
                    "training, so its clip counts cannot be measured; the "
                    "construction recipe is reported from config instead."
                    % cw.get("processed_root"),
        }],
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    print("wrote", os.path.relpath(OUT_TEX, REPO))
    print("wrote", os.path.relpath(OUT_JSON, REPO))
    for exp, info in chains.items():
        st = [s for s in info["stages"] if "clips" in s]
        if st:
            print("  %-34s %s -> %s usable" % (
                exp, "%d/%d" % (st[0]["usable"], st[0]["clips"]),
                "%d/%d" % (st[-1]["usable"], st[-1]["clips"])))
    print("\ndiscrepancies found vs script comments:")
    for d in discrepancies:
        print("  - claim: %s\n    measured: %s" % (d["claim"], d["measured"]))
    if fps_mismatch:
        print("\nframe-rate mismatches (disclose, do not silently fix):")
        for f in fps_mismatch:
            print("  - %s: %g fps on disk, reported as %g (%.0f%% off)" % (
                f["corpus"], f["fps_on_disk"], f["fps_reported_to_model"],
                100 * f["relative_error"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
