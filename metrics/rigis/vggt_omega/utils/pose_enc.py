# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .rotation import quat_to_mat

def encoding_to_camera(pose_encoding, image_size_hw, build_intrinsics=True):
    """Decode VGGT-Omega pose encoding into extrinsics and intrinsics."""
    T = pose_encoding[..., :3]
    quat = pose_encoding[..., 3:7]
    fov_h = pose_encoding[..., 7]
    fov_w = pose_encoding[..., 8]

    R = quat_to_mat(quat)
    extrinsics = torch.cat([R, T[..., None]], dim=-1)

    intrinsics = None
    if build_intrinsics:
        H, W = image_size_hw
        fy = (H / 2.0) / torch.tan(fov_h / 2.0)
        fx = (W / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = W / 2
        intrinsics[..., 1, 2] = H / 2
        intrinsics[..., 2, 2] = 1.0

    return extrinsics, intrinsics
