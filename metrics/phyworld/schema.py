"""Canonical PhyWorld names, paper columns, and embodiment lexicons.

No model imports. JSON keys on disk match the constants below.
"""

from __future__ import annotations

# --- tracks ---
TRACK_QUAL = "qual"  # PhyQual-NR
TRACK_DYN = "dyn"    # PhyDyn-NR
TRACK_FID = "fid"    # PhyFid-DR

# --- JSON keys ---
SCENE_INTEGRITY = "scene_integrity"
TASK_LOCK = "task_lock"
ENTITY_PRESENCE = "entity_presence"
LOOK_SCORE = "look_score"
PHY_QUAL = "phy_qual"

KINETIC_BUDGET = "kinetic_budget"
ACTOR_DYNAMICS = "actor_dynamics"
ACTOR_MOTION = "actor_motion"  # subject-only kinetic budget (RBench MAS, adapted)
ACTOR_MOTION_ACTIVE_RATE = "actor_motion_active_rate"
MOTION_CONTINUITY = "motion_continuity"
PHY_DYN = "phy_dyn"

MOTION_FIDELITY = "motion_fidelity"
KINETIC_DRIFT = "kinetic_drift"
ACTOR_MOTION_DRIFT = "actor_motion_drift"
CONTINUITY_GAP = "continuity_gap"
LOOK_DRIFT = "look_drift"
PHY_FID = "phy_fid"

# WorldScore lineage (comments / debug only — never paper columns)
WORLDSCORE_LINEAGE = {
    SCENE_INTEGRITY: "clip_iqa+",
    LOOK_SCORE: "clip_aesthetic",
    TASK_LOCK: "clip_score",
    ENTITY_PRESENCE: "object_detection",
    KINETIC_BUDGET: "motion_magnitude",
    ACTOR_DYNAMICS: "motion_accuracy",
    ACTOR_MOTION: "RBench Motion-Aware Score (MAS), adapted — flow-on-mask not CoTracker",
    ACTOR_MOTION_ACTIVE_RATE: "PhyWorld no-ref actor-motion temporal coverage",
    MOTION_CONTINUITY: "motion_smoothness",
    MOTION_FIDELITY: "flow_epe vs dense (ours)",
}

# Paper tables. ActorMotion sits beside KineticBudget (raw, reported not averaged) —
# Continuity rewards stillness, ActorMotion penalizes it; folding both into PhyDyn's
# mean would cancel the signal. Same reasoning keeps ActorMotionDrift out of PhyFid's mean.
PAPER_QUAL = [SCENE_INTEGRITY, TASK_LOCK, ENTITY_PRESENCE, PHY_QUAL]
PAPER_DYN = [KINETIC_BUDGET, ACTOR_MOTION, ACTOR_MOTION_ACTIVE_RATE, ACTOR_DYNAMICS, MOTION_CONTINUITY, PHY_DYN]
PAPER_FID = [MOTION_FIDELITY, KINETIC_DRIFT, ACTOR_MOTION_DRIFT, CONTINUITY_GAP, LOOK_DRIFT, PHY_FID]

# GroundingDINO phrases for EntityPresence / ActorDynamics
EMBODIMENT_PHRASES = {
    "single_arm": ["robot arm", "gripper"],
    "bimanual": ["robot arm", "gripper"],
    "humanoid": ["robot", "hand", "arm"],
}

METHODS = (
    "dense", "worldcache", "dicache", "fastercache",
    "sito", "itm", "pisa", "radial", "svg1", "svg2",
)
EMBODIMENTS = ("single_arm", "bimanual", "humanoid")
VIEWS = ("singleview", "multiview")
MODELS = ("ctrlworld", "dreamgen", "dreamdojo")
SPLITS = ("makovian", "non_makovian")
