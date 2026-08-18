"""Build the augment batch_input JSONs for the DreamGen branches.

DreamGen inference is driven by a `batch_input*.json` list of
`{input_video (a PNG), prompt, output_video}`. This script produces the augment
version of those lists for one config / category / branch:

  input_video   edit_only, combo -> the recolored PNG from
                edit_dreamgen_first_frames.py; prompt_only, task_swap -> the
                DENSE frame, reused as-is (a copy would be dead weight and
                could drift).
  prompt        augment_prompt.compose() — edit_only reuses the dense caption
                verbatim, prompt_only/combo/task_swap inject the perturbation
                into the config's caption template.
  output_video  augment_paths.output_path() -> .../{dense stem}__{branch}.mp4

The number of autoregressive chunks per episode is REUSED from the dense batch
jsons (the value is a property of the episode's GT length, so recomputing it from
ffprobe would risk disagreeing with dense and generating a different-length video).

`dreamgen_bimanual_multiview` is intentionally unsupported: that config runs
through infer_bimanual_multiview_rollout_augment.py, which reads the manifest
itself (segment rollout, not a batch json).

Usage:
  python scripts/augment/prepare_dreamgen_augment_batch.py \
      --config dreamgen_single_arm_singleview --split makovian --branch edit_only \
      --iter iter_000090000_static16fps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment_paths as AP  # noqa: E402
import augment_prompt as PR  # noqa: E402
from edit_dreamgen_first_frames import _dense_frame, edited_dir  # noqa: E402

# Configs that consume a batch json (i.e. everything except the rollout config).
BATCH_CONFIGS = [k for k in AP.CONFIGS
                 if k.startswith("dreamgen_") and k != "dreamgen_bimanual_multiview"]

# Multiview single_arm runs single-shot (one batch_input.json, no chunking);
# the singleview configs group episodes by required chunk count.
CHUNKED = {"dreamgen_single_arm_singleview", "dreamgen_bimanual_singleview"}


def dense_chunk_counts(config: str, split: str) -> dict[str, int]:
    """{dense episode stem -> num_chunks} read from the dense batch jsons.

    `*_17s` / `*_23s` files are the legacy capped variants of the same episodes,
    so they are skipped to avoid a stem mapping to two different chunk counts.
    """
    d = AP.dreamgen_input_dir(config, split, root=AP.DENSE_ROOT)
    out: dict[str, int] = {}
    for f in sorted(d.glob("batch_input_*chunks.json")):
        m = re.fullmatch(r"batch_input_(\d+)chunks\.json", f.name)
        if not m:
            continue
        for item in json.loads(f.read_text()):
            out[Path(item["input_video"]).stem] = int(m.group(1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, choices=BATCH_CONFIGS)
    ap.add_argument("--split", required=True, choices=["makovian", "non_makovian"])
    ap.add_argument("--branch", required=True, choices=list(AP.DREAMGEN_BRANCHES))
    ap.add_argument("--manifest", default=str(AP.REPO_ROOT / "augment_manifest.json"))
    ap.add_argument("--iter", dest="iter_name", default="",
                    help="Output iter dir name (required for configs that have one).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only take the first N entries (mirrors the editor's --limit "
                         "so a quick verification run does not report the un-edited rest "
                         "as missing).")
    ap.add_argument("--finalize", action="store_true",
                    help="Do not prepare anything; just drop the provenance sidecar next "
                         "to every video that actually got generated. Run AFTER inference "
                         "so the output dir never shows a sidecar without its mp4.")
    args = ap.parse_args()

    cfg = AP.get(args.config)
    if cfg.has_iter and not args.iter_name:
        raise SystemExit(f"{args.config} needs --iter (e.g. iter_000090000_static16fps)")

    manifest = json.loads(Path(args.manifest).read_text())
    entries = manifest.get(args.config, {}).get(args.split, {}).get(args.branch, [])
    if not entries:
        print(f"[prep] no entries for {args.config}/{args.split}/{args.branch}")
        return
    if args.limit:
        entries = entries[:args.limit]

    chunk_of = dense_chunk_counts(args.config, args.split) if args.config in CHUNKED else {}
    out_kw = {"iter_name": args.iter_name} if cfg.has_iter else {}
    if cfg.has_view_name:
        out_kw["view_name"] = "global"

    groups: dict[int, list[dict]] = defaultdict(list)
    input_dir = AP.dreamgen_input_dir(args.config, args.split)
    input_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    finalized = 0
    for entry in entries:
        stem = AP.episode_stem(args.config, entry)
        leaf = AP.leaf_name(args.config, entry, args.branch)

        if args.branch in ("prompt_only", "task_swap"):
            frame = _dense_frame(args.config, args.split, entry)
        else:
            frame = edited_dir(args.config, args.split, entry) / f"{leaf}.png"
        if not frame.exists() and not args.finalize:
            print(f"  [SKIP] {leaf}: frame missing ({frame})")
            skipped += 1
            continue

        out_mp4 = AP.output_path(args.config, args.split, entry, args.branch, **out_kw)
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        prompt = PR.compose(args.config, entry, args.branch)
        # 1 = "not chunked"; the single-shot configs ignore num_chunks entirely.
        nc = chunk_of.get(stem, 2) if args.config in CHUNKED else 1

        sidecar = {
            "config": args.config,
            "split": args.split,
            "branch": args.branch,
            "dense_stem": stem,
            "leaf": leaf,
            "object": entry.get("object"),
            "edit_instruction": entry.get("edit_instruction"),
            "source_instruction": entry.get("source_instruction"),
            "swap_task": entry.get("swap_task"),
            "swap_instruction": entry.get("swap_instruction"),
            "prompt": prompt,
            "dense_prompt": PR.dense_prompt(args.config, entry),
            "input_frame": str(frame),
            "num_chunks": nc,
        }
        sidecar_path = out_mp4.with_suffix(".augment.json")

        # --finalize only attaches provenance to videos that exist, so the output
        # dir never advertises an episode that was never generated.
        if args.finalize:
            if out_mp4.is_file():
                sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))
                finalized += 1
            elif sidecar_path.is_file():
                sidecar_path.unlink()   # left over from an aborted run
            continue

        groups[nc].append({
            "input_video": str(frame),
            "prompt": prompt,
            "output_video": str(out_mp4),
        })
        if out_mp4.is_file():
            # Already generated (resumed run) -> safe to write provenance now.
            sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False))

    if args.finalize:
        print(f"[prep --finalize] {args.config}/{args.split}/{args.branch}: "
              f"{finalized} sidecar(s) written")
        return

    total = 0
    for nc, items in sorted(groups.items()):
        name = (f"batch_input_{args.branch}_{nc}chunks.json" if args.config in CHUNKED
                else f"batch_input_{args.branch}.json")
        (input_dir / name).write_text(json.dumps(items, indent=2, ensure_ascii=False))
        total += len(items)
        print(f"  {name}: {len(items)} episodes")
    print(f"[prep] {args.config}/{args.split}/{args.branch}: {total} episodes "
          f"({skipped} skipped) -> {input_dir}")


if __name__ == "__main__":
    main()
