"""Prompt composition for the DreamGen augment branches.

Each DreamGen config was trained on a very specific caption template (single
sentence for singleview, a 2x2-grid description for multiview). The augment
branches must respect that:

  edit_only    the prompt is the DENSE prompt, byte-identical — only the image
               changed, so any text difference would confound the comparison.
  prompt_only  the recorded actions stay, the prompt describes something else.
  combo        the recolored object is named in the prompt.
  task_swap    (dreamgen_bimanual_singleview ONLY) the frame stays the dense
               frame, but the prompt is a DIFFERENT real task's caption
               (borrowed from elsewhere in the dataset) — tests whether the
               model follows the text or just continues what it saw in frame 0.
  object_relocation  (dreamgen_bimanual_singleview ONLY, hand-curated, few
               entries) frame stays the dense frame, prompt names a REAL loose
               object AND a REAL container both visible in that same frame —
               tests actionable "pick X, place in Y" following on a scene that
               plausibly supports it.

For the multiview configs the caption also tells the model that the frame is a
2x2 grid of specific camera views. Dropping that would change the OUTPUT FORMAT
(the model may stop producing a grid), which is not the perturbation we want to
measure — so a perturbed prompt is injected INTO the template's `{task}` slot
rather than replacing the whole caption.

The manifest stores perturbed prompts as full sentences ("A robot arm lifts the
cup ..."), which is exactly right for the singleview templates but needs the
leading agent phrase stripped before it can go into a `{task}` slot ("... shows
that a robot lifts the cup ...") — that is what `task_phrase()` does.
"""

from __future__ import annotations

import re

# Copied verbatim from the dense prepare scripts. Keep in sync:
#   prepare_single_arm_multiview_dreamgen_batch.py::PROMPT_TEMPLATE
#   prepare_bimanual_multiview_dreamgen_batch.py::PROMPT_4VIEW
#   prepare_singleview_dreamgen_batch.py::bimanual_task_to_caption
#   run_single_arm_singleview_dreamgen_static16fps.sh::build_caption
SINGLE_ARM_MULTIVIEW_TEMPLATE = (
    "A multi-view video shows that a robot {task}. "
    "The video is split into four views: "
    "The top-left view shows the robotic arm from the left side, "
    "the top-right view shows it from another left exterior viewing angle, "
    "the bottom-left view shows a first-person perspective from the "
    "robot's end-effector (gripper), "
    "and the bottom-right view is a black screen (inactive view). "
    "The robot {task}."
)

BIMANUAL_MULTIVIEW_TEMPLATE = (
    "A multi-view video of a bimanual ALOHA robot performing: {task}. "
    "The video shows four synchronized camera views arranged in a 2x2 grid: "
    "top-left shows the high overhead view, top-right shows the low front view, "
    "bottom-left shows the left wrist view, and bottom-right shows the right wrist view."
)

BIMANUAL_SINGLEVIEW_TEMPLATE = "A bimanual robot performing the task: {task}."

# single_arm singleview rotates a robot prefix by episode id and rephrases a few
# known tasks into 3rd person (build_caption in the dense run script).
SA_SV_PREFIXES = [
    "A robot arm",
    "A single robotic arm",
    "A tabletop robot arm",
    "A robot manipulator",
]
SA_SV_REPHRASINGS = {
    "place the cup on the coaster": "places the cup on the coaster",
    "turn on the light switch": "turns on the light switch",
    "press the pink, blue, and green buttons in sequence":
        "presses the pink, blue, and green buttons in sequence",
    "pull out a tissue, wipe the stains on the table clean, "
    "and then discard the tissue into the trash bin":
        "pulls out a tissue, wipes the stains on the table clean, "
        "and then discards the tissue into the trash bin",
}
SA_SV_GENERIC = "A robot arm performing a manipulation task."

# Agent phrases the manifest's perturbed prompts start with.
_AGENT_PREFIXES = ("a robot arm", "the robot arm", "the bimanual robot", "a bimanual robot")


def task_phrase(sentence: str) -> str:
    """Turn a full sentence into a `{task}` slot value.

    "A robot arm lifts the cup high above the table." ->
    "lifts the cup high above the table"
    """
    s = (sentence or "").strip()
    low = s.lower()
    for p in _AGENT_PREFIXES:
        if low.startswith(p):
            s = s[len(p):].lstrip()
            break
    return s.rstrip(" .;,")


def sa_singleview_caption(raw_task: str, ep_id: int) -> str:
    """Reproduce the dense single_arm singleview caption (build_caption)."""
    if not raw_task or raw_task == SA_SV_GENERIC:
        return SA_SV_GENERIC
    prefix = SA_SV_PREFIXES[ep_id % len(SA_SV_PREFIXES)]
    phrase = SA_SV_REPHRASINGS.get(raw_task, raw_task)
    return f"{prefix} {phrase}."


def bimanual_task_caption(task_name: str) -> str:
    """Reproduce the dense bimanual singleview caption."""
    name = task_name.replace("_", " ")
    name = name[0].upper() + name[1:] if name else name
    return BIMANUAL_SINGLEVIEW_TEMPLATE.format(task=name)


def dense_prompt(config: str, entry: dict) -> str:
    """The prompt the DENSE run used for this episode (edit_only must reuse it)."""
    instr = (entry.get("source_instruction") or "").strip().rstrip(".")
    if config == "dreamgen_single_arm_singleview":
        return sa_singleview_caption(instr, int(entry["episode"]))
    if config == "dreamgen_single_arm_multiview":
        return SINGLE_ARM_MULTIVIEW_TEMPLATE.format(task=instr)
    if config == "dreamgen_bimanual_singleview":
        return bimanual_task_caption(entry["task"])
    if config == "dreamgen_bimanual_multiview":
        return BIMANUAL_MULTIVIEW_TEMPLATE.format(task=instr)
    raise KeyError(f"no dense prompt template for {config}")


def perturbed_prompt(config: str, entry: dict) -> str:
    """The prompt for prompt_only / combo / task_swap.

    Singleview: the manifest sentence verbatim (already in template style —
    for task_swap this is the donor task's own dense-style caption).
    Multiview: injected into the grid template so the 2x2 layout description —
    which the model needs to keep emitting a grid — survives the perturbation.
    """
    new = (entry.get("prompt") or "").strip()
    if not new:
        raise ValueError(f"manifest entry has no 'prompt': {entry}")
    if config == "dreamgen_single_arm_multiview":
        return SINGLE_ARM_MULTIVIEW_TEMPLATE.format(task=task_phrase(new))
    if config == "dreamgen_bimanual_multiview":
        return BIMANUAL_MULTIVIEW_TEMPLATE.format(task=task_phrase(new))
    return new


def compose(config: str, entry: dict, branch: str) -> str:
    """Prompt for one manifest entry on one branch."""
    if branch == "edit_only":
        return dense_prompt(config, entry)
    if branch in ("prompt_only", "combo", "task_swap", "object_relocation"):
        return perturbed_prompt(config, entry)
    raise ValueError(f"unknown branch {branch!r}")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
