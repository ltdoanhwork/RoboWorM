"""Sim2real object-interaction metrics: physics predicates without a simulator.

See metrics/sim2real/README.md. Public entry points:

    from metrics.sim2real import episode_metrics, view_metrics   # scoring
    from metrics.sim2real import track_episode, save_tracks, load_tracks  # tracking
    from metrics.sim2real import episode_panel, violation_clip   # visual evidence
"""

from metrics.sim2real.interaction_metrics import (
    DEFAULTS, SCORE_TERMS, SCORE_GROUPS_BY_CONDITIONING,
    CONTACT_YES, CONTACT_NO, CONTACT_UNKNOWN,
    boundary_gap, gripper_tip, track_series,
    contact_from_gap, contact_bool, and_gate,
    levitation, grasp_follow, object_integrity, contact_chatter, penetration,
    contact_agreement, trajectory_agreement, gap_agreement,
    derive_nodes_from_gt, check_nodes,
    interaction_score, view_metrics, episode_metrics,
)
from metrics.sim2real.interaction_viz import (
    overlay_masks, episode_panel, violation_clip,
)
from metrics.sim2real.lift_backend import (
    build_lift, mask_centroid, mask_scale, mask_extent,
)

__all__ = [
    "DEFAULTS", "SCORE_TERMS", "SCORE_GROUPS_BY_CONDITIONING",
    "CONTACT_YES", "CONTACT_NO", "CONTACT_UNKNOWN",
    "boundary_gap", "gripper_tip", "track_series",
    "contact_from_gap", "contact_bool", "and_gate",
    "levitation", "grasp_follow", "object_integrity", "contact_chatter",
    "penetration", "contact_agreement", "trajectory_agreement", "gap_agreement",
    "derive_nodes_from_gt", "check_nodes",
    "interaction_score", "view_metrics", "episode_metrics",
    "overlay_masks", "episode_panel", "violation_clip",
    "build_lift", "mask_centroid", "mask_scale", "mask_extent",
]
