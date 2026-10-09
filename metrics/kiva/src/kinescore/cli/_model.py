"""Load a reader checkpoint on its own: the robot and the view come from the file."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any

#: The detectors ``calibrate`` and ``score`` run.
DETECTOR_NAMES: tuple[str, ...] = ("rigidity", "jerk")


@dataclass
class Model:
    path: str
    robot: Any
    view: Any
    reader: Any
    device: str

    def read(self, path: str, frame_chunk: int = 32, max_frames: int = 0, seed: int = 0):
        """Keypoints for one clip, as the detectors take them.

        The diffusion head starts from Gaussian noise, so the RNG is reset per
        clip: the same clip gives the same keypoints in any run and any order.
        """
        import torch

        from kinescore.core.context import ClipContext
        from kinescore.video.probe import resolve_timebase
        from kinescore.video.reader import load_rgb

        clip = resolve_timebase(path, view_layout=self.view.layout())
        self.view.check_frame_size(clip.width, clip.height)
        frames = load_rgb(clip, max_frames=max_frames)
        torch.manual_seed(seed)
        with torch.no_grad():
            out = self.reader.read(frames.to(self.device), frame_chunk=frame_chunk)
        return ClipContext(dt=clip.dt, P=out.P.float().cpu(), robot=self.robot,
                           flags={"path": path})


def load_model(path: str, device: str) -> Model:
    """Build the reader in ``path``, with the robot and view its ``cfg`` names."""
    from kinescore.readers.checkpoint import load_reader, read_cfg
    from kinescore.registry.views import load_views
    from kinescore.robots import get_robot

    if not os.path.isfile(path):
        raise SystemExit(f"no model checkpoint at {path}")
    cfg = read_cfg(path)
    robot_name, view_id = cfg.get("robot"), cfg.get("view_id")
    if not robot_name or not view_id:
        raise SystemExit(f"{path}: checkpoint does not name its robot and view_id")
    views = load_views()
    if view_id not in views:
        raise SystemExit(f"{path}: view {view_id!r} is not in views.yaml "
                         f"(have: {', '.join(sorted(views))})")
    view = views[view_id]
    robot = get_robot(robot_name)
    reader = load_reader(path, robot=robot, view_layout=view.layout(), device=device,
                         reader_id=os.path.splitext(os.path.basename(path))[0])
    if reader.n_keypoints != len(robot.keypoint_links):
        # an older reader of this robot that predicts fewer keypoints
        robot = get_robot(robot_name, n_keypoints=reader.n_keypoints)
    return Model(path=path, robot=robot, view=view, reader=reader, device=device)


def build_scorer():
    """A scorer running only :data:`DETECTOR_NAMES`."""
    from kinescore.violations import JerkDetector, RigidityDetector, ViolationScorer

    return ViolationScorer([RigidityDetector(), JerkDetector()])


def find_clips(root: str, pattern: str = "*.mp4", limit: int = 0) -> list[str]:
    """Every clip under ``root`` whose basename matches ``pattern``, sorted."""
    if not os.path.isdir(root):
        raise SystemExit(f"no directory {root}")
    found = sorted(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return found[:limit] if limit else found


def add_read_arguments(parser) -> None:
    """Flags every command that runs the reader takes."""
    parser.add_argument("--model", required=True, help="reader checkpoint (.pt)")
    parser.add_argument("--pattern", default="*.mp4",
                        help="basename glob a clip must match (default: *.mp4)")
    parser.add_argument("--limit", type=int, default=0, help="cap clips read (0 = all)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="cap frames per clip (0 = all)")
    parser.add_argument("--frame-chunk", type=int, default=32,
                        help="frames encoded per backbone call (0 = whole clip at once)")
    parser.add_argument("--seed", type=int, default=0,
                        help="noise seed of the diffusion reader, reset per clip")
    parser.add_argument("--device", default="cuda")
