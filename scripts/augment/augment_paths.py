"""Single source of truth for `sampling_dataset/augment/` paths.

The augment tree mirrors the dense tree level-for-level (see
`.cursor/rules/sampling-dataset-layout.mdc` and
`sampling_dataset/augment/README.md`); only two things differ:

  1. the top-level method folder is `augment` instead of `dense`, and
  2. the episode leaf carries a `__{tag}` suffix identifying the augmentation
     (DreamGen branch `edit_only|prompt_only|combo`, Ctrl-World recolor variant
     `v00_red`, ...).

Drop the suffix from an augment path and you have the dense path of the same
episode, which is what makes augment-vs-dense comparison trivial. Every run
script / metric script must build paths through here instead of hardcoding, so
the layout can never drift.

CLI (for bash run scripts):
  python scripts/augment/augment_paths.py --config ctrlworld_bimanual_multiview \
      --io output --split makovian
  python scripts/augment/augment_paths.py --config dreamgen_single_arm_singleview \
      --io output --split makovian --iter iter_000090000_static16fps --view-name global
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AUGMENT_ROOT = REPO_ROOT / "sampling_dataset" / "augment"
DENSE_ROOT = REPO_ROOT / "sampling_dataset" / "dense"

# Branch tags used by the DreamGen configs (5 videos each -> 15 per config, 20
# for dreamgen_bimanual_singleview which additionally has task_swap, plus a
# variable-count hand-curated object_relocation branch on that same config).
DREAMGEN_BRANCHES = ("edit_only", "prompt_only", "combo", "task_swap", "object_relocation")


@dataclass(frozen=True)
class AugConfig:
    """Describes one benchmark config's augment sub-layout."""

    embodiment: str          # single_arm | bimanual | humanoid
    view: str                # singleview | multiview
    model: str               # dreamgen | ctrlworld | dreamdojo
    leaf_kind: str           # "flat_mp4" (episode_x.mp4 + .txt) | "episode_dir"
    has_iter: bool           # is there an iter_* level under {split}?
    has_view_name: bool      # is there a {view_name} level (e.g. global)?
    has_task: bool           # is there a {task} level (bimanual multiview)?
    task_in_stem: bool       # is the task part of a flat file stem instead?
    tag_kind: str            # "branch" (DreamGen) | "variant" (Ctrl-World)

    def rel_root(self, io: str) -> Path:
        """`{embodiment}/{io}/{view}/{model}` — the part shared with dense."""
        assert io in ("input", "output"), io
        return Path(self.embodiment) / io / self.view / self.model


# Keys match the section names in augment_manifest.json.
CONFIGS: dict[str, AugConfig] = {
    # --- Ctrl-World: action-conditioned, per-episode dirs, recolor variants ---
    "ctrlworld_single_arm_multiview": AugConfig(
        embodiment="single_arm", view="multiview", model="ctrlworld",
        leaf_kind="episode_dir", has_iter=False, has_view_name=False,
        has_task=False, task_in_stem=False, tag_kind="variant",
    ),
    "ctrlworld_bimanual_multiview": AugConfig(
        embodiment="bimanual", view="multiview", model="ctrlworld",
        leaf_kind="episode_dir", has_iter=False, has_view_name=False,
        # bimanual/humanoid Ctrl-World annotation ids already embed the task
        # (`makovian_close_cardboard_box__000026`), so the dense leaf is a flat
        # `episode_{task}__{id}` dir — no separate {task} directory level.
        has_task=False, task_in_stem=True, tag_kind="variant",
    ),
    "ctrlworld_humanoid_multiview": AugConfig(
        embodiment="humanoid", view="multiview", model="ctrlworld",
        leaf_kind="episode_dir", has_iter=False, has_view_name=False,
        has_task=False, task_in_stem=True, tag_kind="variant",
    ),
    # --- Ctrl-World 4-VIEW GRID: same recolor pipeline, different finetuned
    # checkpoint/config (num_views=4, 2x2 latent canvas). Dense keeps this as a
    # SEPARATE method dir (`ctrlworld_4view_grid`) alongside the 3-view
    # `ctrlworld`, not a replacement — same here, so augment mirrors dense 1:1.
    # single_arm has no 4-view grid finetune (it stays on the pretrained 3-view
    # DROID checkpoint), so there is no single_arm entry here.
    "ctrlworld_bimanual_multiview_4view_grid": AugConfig(
        embodiment="bimanual", view="multiview", model="ctrlworld_4view_grid",
        leaf_kind="episode_dir", has_iter=False, has_view_name=False,
        has_task=False, task_in_stem=True, tag_kind="variant",
    ),
    "ctrlworld_humanoid_multiview_4view_grid": AugConfig(
        embodiment="humanoid", view="multiview", model="ctrlworld_4view_grid",
        leaf_kind="episode_dir", has_iter=False, has_view_name=False,
        has_task=False, task_in_stem=True, tag_kind="variant",
    ),
    # --- DreamGen: text-conditioned, 3 branches ---
    "dreamgen_single_arm_singleview": AugConfig(
        embodiment="single_arm", view="singleview", model="dreamgen",
        leaf_kind="flat_mp4", has_iter=True, has_view_name=True,
        has_task=False, task_in_stem=False, tag_kind="branch",
    ),
    "dreamgen_single_arm_multiview": AugConfig(
        embodiment="single_arm", view="multiview", model="dreamgen",
        leaf_kind="flat_mp4", has_iter=False, has_view_name=False,
        has_task=False, task_in_stem=False, tag_kind="branch",
    ),
    "dreamgen_bimanual_singleview": AugConfig(
        embodiment="bimanual", view="singleview", model="dreamgen",
        leaf_kind="flat_mp4", has_iter=True, has_view_name=False,
        has_task=False, task_in_stem=True, tag_kind="branch",
    ),
    "dreamgen_bimanual_multiview": AugConfig(
        embodiment="bimanual", view="multiview", model="dreamgen",
        leaf_kind="episode_dir", has_iter=True, has_view_name=False,
        has_task=True, task_in_stem=False, tag_kind="branch",
    ),
}


def get(config: str) -> AugConfig:
    if config not in CONFIGS:
        raise KeyError(f"unknown augment config '{config}'. Known: {sorted(CONFIGS)}")
    return CONFIGS[config]


def dreamgen_input_dir(config: str, split: str, task: str | None = None,
                       *, root: Path | None = None) -> Path:
    """Dir holding a DreamGen config's prepared inputs (frames + batch jsons).

    This mirrors how the DENSE *input* tree is actually laid out, which differs
    from the output tree in two ways:
      - there is no `iter_*` level (a first frame does not depend on a checkpoint),
      - single_arm singleview nests `{view_name}/{split}`, not `{split}/{view_name}`.
    """
    cfg = get(config)
    if cfg.model != "dreamgen":
        raise ValueError(f"{config} is not a DreamGen config")
    base = (root or AUGMENT_ROOT) / cfg.rel_root("input")
    if config == "dreamgen_single_arm_singleview":
        return base / "global" / split
    base = base / split
    if cfg.has_task:
        if not task:
            raise ValueError(f"{config} needs task")
        base = base / task
    return base


def base_dir(config: str, io: str = "output", *, root: Path | None = None) -> Path:
    """`{root}/{embodiment}/{io}/{view}/{model}` — the dir ABOVE `{split}`.

    The bimanual/humanoid Ctrl-World inference scripts derive `{category}` from
    the annotation stem themselves, so they take this split-less base as
    `--save_dir` instead of a per-split dir.
    """
    return (root or AUGMENT_ROOT) / get(config).rel_root(io)


def _split_dir(
    config: str,
    io: str,
    split: str,
    *,
    iter_name: str | None = None,
    view_name: str | None = None,
    task: str | None = None,
    root: Path | None = None,
) -> Path:
    cfg = get(config)
    base = (root or AUGMENT_ROOT) / cfg.rel_root(io) / split
    if cfg.has_iter:
        if not iter_name:
            raise ValueError(f"{config} needs iter_name (e.g. iter_000090000_static16fps)")
        base = base / iter_name
    if cfg.has_view_name:
        base = base / (view_name or "global")
    if cfg.has_task:
        if not task:
            raise ValueError(f"{config} needs task (bimanual multiview task dir)")
        base = base / task
    return base


def output_dir(config: str, split: str, **kw) -> Path:
    """Directory that directly contains the episode leaf (file or dir)."""
    return _split_dir(config, "output", split, **kw)


def input_dir(config: str, split: str, **kw) -> Path:
    """Mirror of `output_dir` under `input/` — holds first frames / batch jsons."""
    return _split_dir(config, "input", split, **kw)


def dense_output_dir(config: str, split: str, **kw) -> Path:
    """Same sub-path under `sampling_dataset/dense/` (for side-by-side compare)."""
    return _split_dir(config, "output", split, root=DENSE_ROOT, **kw)


def episode_stem(config: str, entry: dict) -> str:
    """Dense-style episode stem WITHOUT the augmentation tag.

    Ctrl-World bimanual/humanoid ids already contain split+task
    (`makovian_close_cardboard_box__000026`); the dense leaf strips the leading
    split so it reads `episode_close_cardboard_box__000026`.
    """
    cfg = get(config)
    ep = str(entry["episode"])
    if cfg.model.startswith("ctrlworld"):
        if cfg.task_in_stem:
            for prefix in ("non_makovian_", "makovian_"):
                if ep.startswith(prefix):
                    ep = ep[len(prefix):]
                    break
        return f"episode_{ep}"
    # DreamGen: numeric episode ids, zero-padded to 6.
    ep_num = f"episode_{int(ep):06d}"
    if cfg.task_in_stem:
        task = entry.get("task")
        if not task:
            raise ValueError(f"{config} entry needs 'task': {entry}")
        return f"{task}__{ep_num}"
    return ep_num


def tag(config: str, entry: dict, branch: str | None = None) -> str:
    """The `__{tag}` suffix value: DreamGen branch name or Ctrl-World variant."""
    cfg = get(config)
    if cfg.tag_kind == "variant":
        variant = entry.get("variant")
        if not variant:
            raise ValueError(f"{config} entry needs 'variant': {entry}")
        return variant
    if branch not in DREAMGEN_BRANCHES:
        raise ValueError(f"{config} needs branch in {DREAMGEN_BRANCHES}, got {branch!r}")
    return branch


def leaf_name(config: str, entry: dict, branch: str | None = None) -> str:
    """Episode leaf name (file stem for flat_mp4, dir name for episode_dir)."""
    return f"{episode_stem(config, entry)}__{tag(config, entry, branch)}"


def output_path(config: str, split: str, entry: dict, branch: str | None = None, **kw):
    """Full path of the primary generated video for one manifest entry."""
    cfg = get(config)
    if cfg.has_task and "task" not in kw:
        kw["task"] = entry.get("task")
    d = output_dir(config, split, **kw)
    leaf = leaf_name(config, entry, branch)
    return d / f"{leaf}.mp4" if cfg.leaf_kind == "flat_mp4" else d / leaf / "full_pred.mp4"


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--io", default="output", choices=["input", "output"])
    ap.add_argument("--split", default=None,
                    help="Category (makovian|non_makovian). Omit with --base.")
    ap.add_argument("--iter", dest="iter_name", default=None)
    ap.add_argument("--view-name", default=None)
    ap.add_argument("--task", default=None)
    ap.add_argument("--base", action="store_true",
                    help="Print the split-less base dir ({emb}/{io}/{view}/{model}).")
    ap.add_argument("--dense", action="store_true", help="Print the dense counterpart.")
    a = ap.parse_args()
    root = DENSE_ROOT if a.dense else AUGMENT_ROOT
    if a.base:
        print(base_dir(a.config, a.io, root=root))
        return
    if not a.split:
        ap.error("--split is required unless --base is given")
    print(_split_dir(a.config, a.io, a.split, iter_name=a.iter_name,
                     view_name=a.view_name, task=a.task, root=root))


if __name__ == "__main__":
    _cli()
