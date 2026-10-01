"""GR-1 keypoint sets: 22 (arms and fingertips) and the 12-keypoint arms-only set.

Requires ``pytorch_kinematics`` + a resolvable GR-1 URDF; skipped entirely
when unavailable.
"""
import pytest
import torch

pytest.importorskip("pytorch_kinematics")


def _gr1(**kwargs):
    from kinescore.robots.gr1.spec import GR1Spec
    try:
        return GR1Spec(**kwargs)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"GR-1 URDF / KINESCORE_ASSETS unavailable: {exc}")


def test_arms_only_is_the_arm_prefix_of_each_side():
    full, arms = _gr1(), _gr1(n_keypoints=12)
    assert len(full.keypoint_links) == 22 and len(arms.keypoint_links) == 12
    assert arms.keypoint_links == full.keypoint_links[:6] + full.keypoint_links[11:17]

    q = torch.zeros(2, 3, full.n_joints)
    p_full, p_arms = full.forward_kinematics(q), arms.forward_kinematics(q)
    assert p_arms.shape == (2, 3, 12, 3)
    assert torch.equal(p_arms, torch.cat([p_full[:, :, :6], p_full[:, :, 11:17]], dim=2))


def test_arms_only_bones_and_end_effectors_index_the_twelve():
    arms = _gr1(n_keypoints=12)
    assert int(arms.rigid_bone_pairs.max()) < 12
    assert arms.ee_sites() == (5, 11)


def test_other_keypoint_counts_are_rejected():
    from kinescore.robots.gr1.spec import GR1Spec
    with pytest.raises(ValueError, match="12 or 22"):
        GR1Spec(n_keypoints=18)
