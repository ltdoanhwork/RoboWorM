"""
Split humanoid singleview data into makovian / non_makovian folders.

Sources (in priority order):
  1. DreamDojo-HV_Eval  (classified via gr1_task_labels_cache.jsonl)
  2. GR1_robot          (classified via gr1robot_task_labels.csv)

Target structure (mirrors source LeRobot layout):
  datasets/humanoid/singleview/makovian/
    data/chunk-XXX/episode_XXXXXX.parquet   → symlink or copy
    videos/chunk-XXX/observation.images.ego_view_freq20/episode_XXXXXX.mp4 → symlink or copy
    meta/  → copied from DreamDojo-HV_Eval/meta, then `repair_humanoid_singleview_split_meta.py`
             rewrites episodes.jsonl / info.json to match on-disk episodes (mixed GR1 + DreamDojo).

Usage:
    python scripts/tools/data/split_humanoid_singleview.py
    python scripts/tools/data/split_humanoid_singleview.py --n-per-label 130 --seed 42
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


CAMERA = "observation.images.ego_view_freq20"


def clean_prompt(text: str) -> str:
    return text.replace("locked waist: ", "").replace("grid_", "").replace("_", " ").strip()


def load_dreamdojo_labels(cache_jsonl: Path) -> dict[str, str]:
    """episode_name -> label, only DreamDojo source."""
    labels = {}
    if not cache_jsonl.exists():
        return labels
    with open(cache_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source") == "DreamDojo-HV_Eval":
                ep = obj.get("episode", "")
                lbl = obj.get("label", "")
                if ep and lbl in ("markovian", "non_markovian"):
                    labels[ep] = lbl
    return labels


def load_gr1robot_task_labels(task_csv: Path) -> dict[str, str]:
    """task_text -> label from gr1robot_task_labels.csv."""
    import csv
    labels = {}
    if not task_csv.exists():
        return labels
    with open(task_csv) as f:
        for row in csv.DictReader(f):
            lbl = row["label"].strip()
            if lbl in ("markovian", "non_markovian"):
                labels[row["task"].strip()] = lbl
    return labels


def link_episode(ep_idx: int, src_root: Path, dst_root: Path) -> bool:
    """Create symlinks for parquet + video. Returns True if parquet existed."""
    chunk_id = ep_idx // 1000

    # parquet
    parquet_src = src_root / f"data/chunk-{chunk_id:03d}/episode_{ep_idx:06d}.parquet"
    if not parquet_src.exists():
        return False
    parquet_dst = dst_root / f"data/chunk-{chunk_id:03d}/episode_{ep_idx:06d}.parquet"
    parquet_dst.parent.mkdir(parents=True, exist_ok=True)
    if not parquet_dst.exists():
        parquet_dst.symlink_to(parquet_src.resolve())

    # video
    vid_src = src_root / f"videos/chunk-{chunk_id:03d}/{CAMERA}/episode_{ep_idx:06d}.mp4"
    if vid_src.exists():
        vid_dst = dst_root / f"videos/chunk-{chunk_id:03d}/{CAMERA}/episode_{ep_idx:06d}.mp4"
        vid_dst.parent.mkdir(parents=True, exist_ok=True)
        if not vid_dst.exists():
            vid_dst.symlink_to(vid_src.resolve())

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-label", type=int, default=130)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/VLA/doanh/video_world/video_gen_physics"),
    )
    args = parser.parse_args()

    ws = args.workspace
    GR1_BASE    = ws / "datasets/humanoid/singleview/PhysicalAI-Robotics-GR00T-Teleop-GR1"
    DST_BASE    = ws / "datasets/humanoid/singleview"
    MKV_DST     = DST_BASE / "makovian"
    NMKV_DST    = DST_BASE / "non_makovian"

    random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. DreamDojo episodes (have parquet+video)
    # -----------------------------------------------------------------------
    dd_labels = load_dreamdojo_labels(ws / "tmp/20261005_refactor/misc/data_labels/gr1_task_labels_cache.jsonl")
    print(f"[dreamdojo] Classified episodes: {len(dd_labels)}")

    # map episode_name -> episode_index via meta
    dd_ep_idx_map: dict[str, int] = {}
    dd_meta = GR1_BASE / "DreamDojo-HV_Eval/meta/episodes.jsonl"
    with open(dd_meta) as f:
        for line in f:
            obj = json.loads(line)
            ep_name = f"episode_{obj['episode_index']:06d}"
            dd_ep_idx_map[ep_name] = obj["episode_index"]

    dd_pool: dict[str, list[int]] = defaultdict(list)  # label -> [ep_idx]
    for ep_name, lbl in dd_labels.items():
        ep_idx = dd_ep_idx_map.get(ep_name)
        if ep_idx is not None:
            dd_pool[lbl].append(ep_idx)

    print(f"[dreamdojo] markovian={len(dd_pool['markovian'])}, non_markovian={len(dd_pool['non_markovian'])}")

    # -----------------------------------------------------------------------
    # 2. GR1_robot episodes (classified by task text)
    # -----------------------------------------------------------------------
    gr1_task_labels = load_gr1robot_task_labels(ws / "tmp/20261005_refactor/misc/data_labels/gr1robot_task_labels.csv")
    print(f"[gr1_robot] Classified tasks: {len(gr1_task_labels)}")

    gr1_pool: dict[str, list[int]] = defaultdict(list)
    gr1_meta = GR1_BASE / "GR1_robot/meta/episodes.jsonl"
    with open(gr1_meta) as f:
        for line in f:
            obj = json.loads(line)
            ep_idx = obj["episode_index"]
            tasks = [clean_prompt(t.strip()) for t in obj.get("tasks", []) if t.strip()]
            if not tasks:
                continue
            lbl = gr1_task_labels.get(tasks[0])
            if lbl in ("markovian", "non_markovian"):
                gr1_pool[lbl].append(ep_idx)

    print(f"[gr1_robot] markovian={len(gr1_pool['markovian'])}, non_markovian={len(gr1_pool['non_markovian'])}")

    # -----------------------------------------------------------------------
    # 3. Sample n_per_label from combined pool
    # -----------------------------------------------------------------------
    N = args.n_per_label
    selected: dict[str, list[tuple[str, int]]] = defaultdict(list)  # label -> [(source, ep_idx)]

    for lbl in ("markovian", "non_markovian"):
        dd_eps = dd_pool[lbl]
        gr1_eps = gr1_pool[lbl]
        combined = [("DreamDojo", ep) for ep in dd_eps] + [("GR1_robot", ep) for ep in gr1_eps]
        if len(combined) < N:
            print(f"[warn] Only {len(combined)} episodes for {lbl}, need {N}")
            selected[lbl] = combined
        else:
            selected[lbl] = random.sample(combined, N)
        print(f"[sample] {lbl}: {len(selected[lbl])} episodes")

    # -----------------------------------------------------------------------
    # 4. Rebuild destination directories
    # -----------------------------------------------------------------------
    for dst in [MKV_DST, NMKV_DST]:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)

    src_map = {
        "DreamDojo": GR1_BASE / "DreamDojo-HV_Eval",
        "GR1_robot": GR1_BASE / "GR1_robot",
    }

    stats = defaultdict(lambda: defaultdict(int))  # label -> {parquet, video}
    for lbl, episodes in selected.items():
        dst = MKV_DST if lbl == "markovian" else NMKV_DST
        for source, ep_idx in episodes:
            src_root = src_map[source]
            ok = link_episode(ep_idx, src_root, dst)
            if ok:
                stats[lbl]["parquet"] += 1
                vid = src_root / f"videos/chunk-{ep_idx//1000:03d}/{CAMERA}/episode_{ep_idx:06d}.mp4"
                if vid.exists():
                    stats[lbl]["video"] += 1

        # Copy meta from DreamDojo as a template; episodes.jsonl/info.json are fixed by repair script.
        meta_dst = dst / "meta"
        if meta_dst.exists():
            shutil.rmtree(meta_dst)
        shutil.copytree(GR1_BASE / "DreamDojo-HV_Eval" / "meta", meta_dst)

    repair = ws / "scripts" / "prepare" / "repair_humanoid_singleview_split_meta.py"
    subprocess.run([sys.executable, str(repair), "--workspace", str(ws)], check=True)

    # -----------------------------------------------------------------------
    # 5. Report
    # -----------------------------------------------------------------------
    print("\n=== Final structure ===")
    for lbl in ("markovian", "non_markovian"):
        folder = "makovian" if lbl == "markovian" else "non_makovian"
        dst = MKV_DST if lbl == "markovian" else NMKV_DST
        n_p = sum(1 for _ in (dst / "data").rglob("*.parquet")) if (dst / "data").exists() else 0
        n_v = sum(1 for _ in (dst / "videos").rglob("*.mp4")) if (dst / "videos").exists() else 0
        print(f"  {folder}/  {n_p} parquets, {n_v} videos")


if __name__ == "__main__":
    main()
