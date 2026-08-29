"""Tool chọn keypoints trên nhiều episode: chuột + mũi tên, Next/Prev folder."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from seed_points import JSON_PATH, DEFAULT_DATA_ROOT as DEFAULT_ROOT

__all__ = [
    "intersect_frames",
    "list_frame_paths",
    "load_frame_bgr",
    "frame_path",
    "EpisodeReader",
    "list_episode_dirs",
]


def list_frame_paths(view_dir: str | Path) -> list[Path]:
    p = Path(view_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Missing view folder: {view_dir}")
    return sorted(p.glob("frame_*.png"))


def frame_index(path: Path) -> int:
    m = re.search(r"frame_(\d+)\.png$", path.name)
    if not m:
        raise ValueError(f"Unexpected frame filename: {path.name}")
    return int(m.group(1))


def intersect_frames(views: list[str], episode_dir: str | Path) -> list[int]:
    per_view = []
    for v in views:
        frames = list_frame_paths(Path(episode_dir) / v)
        per_view.append({frame_index(f) for f in frames})
    common = set.intersection(*per_view) if per_view else set()
    return sorted(common)


def frame_path(episode_dir: Path | str, view: str, frame_id: int) -> Path:
    return Path(episode_dir) / view / f"frame_{frame_id:04d}.png"


def load_frame_bgr(path: Path | str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def episode_json_key(episode_dir: Path, root: Path | None = None) -> str:
    """Relative path from root (POSIX), stable across symlinked absolute paths."""
    episode_dir = Path(episode_dir).resolve()
    if root is not None:
        return episode_dir.relative_to(Path(root).resolve()).as_posix()
    return str(episode_dir)


def legacy_abs_to_rel_key(old_key: str, root: Path | None = None) -> str | None:
    """Map old absolute episode paths to new relative layout."""
    marker = "data_for_web_init_frame"
    key = old_key.replace("\\", "/")
    if marker in key:
        rel = key.split(marker, 1)[1].lstrip("/")
    elif root is not None:
        try:
            return Path(old_key).resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            return None
    else:
        return None

    parts = Path(rel).parts
    if len(parts) < 7 or parts[3] != "multiview":
        return None

    video_type, robot, _out, _mv, model, task_type, *rest = parts
    prefix = Path(video_type, robot, "output", "multiview", model, task_type)

    if model == "ctrlworld" and robot in ("bimanual", "humanoid"):
        if len(rest) != 2:
            return None
        task_name, ep = rest
        if ep.startswith("episode_"):
            return str(prefix / ep).replace("\\", "/")
        return str(prefix / f"episode_{task_name}__{ep}").replace("\\", "/")

    if model == "ctrlworld" and robot == "single_arm":
        if len(rest) != 1:
            return None
        ep = rest[0]
        if ep.startswith("episode_"):
            return str(prefix / ep).replace("\\", "/")
        return str(prefix / f"episode_{ep}").replace("\\", "/")

    if model == "dreamgen" and robot == "single_arm":
        if len(rest) != 1:
            return None
        ep = rest[0]
        if ep.startswith("episode_"):
            return str(prefix / ep).replace("\\", "/")
        return str(prefix / f"episode_{ep}").replace("\\", "/")

    if model == "dreamgen" and robot == "bimanual":
        if len(rest) == 4 and rest[0].startswith("iter_"):
            return str(Path(*parts)).replace("\\", "/")
        if len(rest) != 2:
            return None
        task_name, ep = rest
        ep_name = ep if ep.startswith("episode_") else f"episode_{ep}"
        return str(prefix / "iter_000120000" / task_name / ep_name).replace("\\", "/")

    if model == "dreamgen" and robot == "humanoid":
        if len(rest) == 4 and rest[0].startswith("iter_"):
            return str(Path(*parts)).replace("\\", "/")
        if len(rest) != 2:
            return None
        task_name, ep = rest
        ep_name = ep if ep.startswith("episode_") else f"episode_{ep}"
        return str(prefix / "iter_000100000" / task_name / ep_name).replace("\\", "/")

    if parts[0] == "tmp" and parts[1] == "lowquality_multiview":
        ep = Path(rel).stem
        ep_name = ep if ep.startswith("episode_") else f"episode_{ep}"
        return f"tmp/lowquality_multiview/{ep_name}"

    return None


def migrate_seed_json(root: Path | None = None) -> dict:
    """Rewrite seed_points.json keys to relative paths under root."""
    root = Path(root or DEFAULT_ROOT).resolve()
    db = load_seed_json()
    migrated: dict = {}

    for old_key, entry in db.items():
        new_key = legacy_abs_to_rel_key(old_key, root)
        if new_key is None:
            candidate = Path(old_key)
            if not candidate.is_absolute():
                new_key = old_key.replace("\\", "/")
            else:
                try:
                    new_key = candidate.resolve().relative_to(root).as_posix()
                except ValueError:
                    print(f"⚠️ Bỏ qua key không migrate được: {old_key}")
                    continue

        payload = {
            "episode_dir": new_key,
            "name": Path(new_key).name,
            "view1": entry.get("view1", []),
            "view2": entry.get("view2", []),
        }
        if new_key in migrated:
            existing = migrated[new_key]
            if not existing.get("view1") and not existing.get("view2"):
                migrated[new_key] = payload
        else:
            migrated[new_key] = payload

    save_seed_json(migrated)
    on_disk = sum(
        1 for k in migrated if (root / k).is_dir() and (root / k / "view1").is_dir()
    )
    print(f"Migrated {len(db)} -> {len(migrated)} entries ({on_disk} có folder trên disk)")
    return migrated


def get_seed_entry(db: dict, key: str) -> dict:
    return db.get(key, {})


def is_episode_done(episode_dir: Path, db: dict, root: Path) -> bool:
    return episode_json_key(episode_dir, root) in db


def advance_to_next_pending(
    episodes: list[Path], idx: int, db: dict, root: Path, *, skip_done: bool
) -> int:
    if not skip_done:
        return idx
    while idx < len(episodes) and is_episode_done(episodes[idx], db, root):
        idx += 1
    return idx


def list_episode_dirs(root: str | Path) -> list[Path]:
    """Mọi folder có view1/ + view2/ (leaf episode), sort theo path."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing root: {root}")
    episodes: list[Path] = []
    for view1 in sorted(root.rglob("view1")):
        if not view1.is_dir():
            continue
        ep = view1.parent
        if (ep / "view2").is_dir():
            episodes.append(ep.resolve())
    return episodes


class EpisodeReader:
    """Sequential access to synchronized frames across views."""

    def __init__(
        self,
        episode_dir: Path | str,
        views: tuple[str, str] = ("view1", "view2"),
        *,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        self.episode_dir = Path(episode_dir)
        self.views = views
        self.stride = max(1, int(stride))
        frame_ids = intersect_frames(list(views), self.episode_dir)
        if max_frames is not None:
            frame_ids = frame_ids[: int(max_frames)]
        self.frame_ids = frame_ids[:: self.stride]
        if not self.frame_ids:
            raise ValueError(f"No common frames under {self.episode_dir}")

    def __len__(self) -> int:
        return len(self.frame_ids)

    def bgr(self, index: int, view: str) -> np.ndarray:
        fid = self.frame_ids[index]
        return load_frame_bgr(frame_path(self.episode_dir, view, fid))

    def gray(self, index: int, view: str) -> np.ndarray:
        return cv2.cvtColor(self.bgr(index, view), cv2.COLOR_BGR2GRAY)

    def frame_id(self, index: int) -> int:
        return self.frame_ids[index]


def load_seed_json() -> dict:
    if JSON_PATH.exists():
        with open(JSON_PATH, "r") as f:
            return json.load(f)
    return {}


def save_seed_json(data: dict):
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Đã lưu seed points vào {JSON_PATH}")


class KeypointAnnotator:
    def __init__(
        self,
        img_v1: np.ndarray,
        img_v2: np.ndarray,
        *,
        pts_v1: list[list[float]] | None = None,
        pts_v2: list[list[float]] | None = None,
        banner: str = "",
    ):
        self.img1 = img_v1.copy()
        self.img2 = img_v2.copy()
        self.h, self.w, _ = img_v1.shape
        self.banner = banner

        self.pts_v1: list[list[float]] = [list(map(float, p)) for p in (pts_v1 or [])]
        self.pts_v2: list[list[float]] = [list(map(float, p)) for p in (pts_v2 or [])]

        self.active_view = 1
        self.selected_idx = -1

        self.colors = [
            (0, 255, 255),
            (255, 0, 255),
            (0, 255, 0),
            (0, 165, 255),
            (255, 255, 0),
            (255, 192, 203),
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < self.w:
                curr_x, curr_y = x, y
                pts = self.pts_v1
                self.active_view = 1
            else:
                curr_x, curr_y = x - self.w, y
                pts = self.pts_v2
                self.active_view = 2

            clicked_near_idx = -1
            for i, p in enumerate(pts):
                if np.hypot(p[0] - curr_x, p[1] - curr_y) < 10:
                    clicked_near_idx = i
                    break

            if clicked_near_idx != -1:
                self.selected_idx = clicked_near_idx
            else:
                pts.append([float(curr_x), float(curr_y)])
                self.selected_idx = len(pts) - 1

    def _draw_view_points(
        self, img: np.ndarray, pts: list[list[float]], is_active_view: bool
    ):
        for i, (x, y) in enumerate(pts):
            c = self.colors[i % len(self.colors)]
            is_selected = is_active_view and (i == self.selected_idx)
            pt = (int(round(x)), int(round(y)))

            if is_selected:
                cv2.drawMarker(
                    img,
                    pt,
                    c,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=16,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
            else:
                cv2.circle(img, pt, 4, c, -1)

            cv2.putText(
                img,
                f"p{i+1}:({int(x)},{int(y)})",
                (pt[0] + 6, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                c,
                1,
                cv2.LINE_AA,
            )

    def draw(self) -> np.ndarray:
        v1 = self.img1.copy()
        v2 = self.img2.copy()

        self._draw_view_points(
            v1, self.pts_v1, is_active_view=(self.active_view == 1)
        )
        self._draw_view_points(
            v2, self.pts_v2, is_active_view=(self.active_view == 2)
        )

        canvas = np.hstack([v1, v2])

        n1, n2 = len(self.pts_v1), len(self.pts_v2)
        sel_str = f"p{self.selected_idx+1}" if self.selected_idx != -1 else "None"
        line1 = self.banner
        line2 = (
            f"Points: V1={n1}, V2={n2} | Selected: View{self.active_view} ({sel_str}) | "
            "ARROWS: Fine-tune | S: Save+Next | X: No points | N: Skip | P: Prev | R: Reset | Q: Quit"
        )

        for i, text in enumerate((line1, line2)):
            if not text:
                continue
            y = 20 + i * 22
            cv2.putText(
                canvas,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas

    def run(self, win_name: str) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        while True:
            canvas = self.draw()
            cv2.imshow(win_name, canvas)
            key = cv2.waitKeyEx(30)

            if key in (27, ord("q")):
                return ("quit", None, None)

            if key == ord("n"):
                return ("next", None, None)

            if key == ord("p"):
                return ("prev", None, None)

            if key in (ord("x"), ord("X")):
                return (
                    "save",
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 2), dtype=np.float32),
                )

            if key == ord("s"):
                n1, n2 = len(self.pts_v1), len(self.pts_v2)
                if n1 == 0 and n2 == 0:
                    return (
                        "save",
                        np.zeros((0, 2), dtype=np.float32),
                        np.zeros((0, 2), dtype=np.float32),
                    )
                if n1 == 0 or n2 == 0:
                    print(
                        "⚠️ Chọn đủ cặp ở CẢ HAI view, hoặc nhấn X để lưu episode không có điểm!"
                    )
                    continue
                if n1 != n2:
                    print(
                        f"⚠️ Số điểm View 1 ({n1}) và View 2 ({n2}) không bằng nhau!"
                    )
                    continue
                return (
                    "save",
                    np.array(self.pts_v1, dtype=np.float32),
                    np.array(self.pts_v2, dtype=np.float32),
                )

            if key == ord("r"):
                self.pts_v1.clear()
                self.pts_v2.clear()
                self.selected_idx = -1
                continue

            if self.selected_idx == -1:
                continue

            pts = self.pts_v1 if self.active_view == 1 else self.pts_v2
            step = 1.0

            if key in (65361, 2424832, 81):
                pts[self.selected_idx][0] -= step
            elif key in (65363, 2555904, 83):
                pts[self.selected_idx][0] += step
            elif key in (65362, 2490368, 82):
                pts[self.selected_idx][1] -= step
            elif key in (65364, 2621440, 84):
                pts[self.selected_idx][1] += step
            elif key == 9 and len(pts) > 0:
                self.selected_idx = (self.selected_idx + 1) % len(pts)


def save_episode_points(
    episode_dir: Path, pts_v1: np.ndarray, pts_v2: np.ndarray, root: Path
) -> None:
    key = episode_json_key(episode_dir, root)
    db = load_seed_json()
    db[key] = {
        "episode_dir": key,
        "name": episode_dir.name,
        "view1": pts_v1.tolist(),
        "view2": pts_v2.tolist(),
    }
    save_seed_json(db)
    n = len(pts_v1)
    if n == 0:
        print(f"   (không chọn điểm) key: {key}")
    else:
        print(f"   ({n} cặp) key: {key}")


def annotate_episodes(
    root: str | Path = DEFAULT_ROOT,
    *,
    skip_done: bool = False,
    start: int = 0,
) -> None:
    root = Path(root).resolve()
    episodes = list_episode_dirs(root)
    if not episodes:
        raise FileNotFoundError(f"Không tìm thấy episode (view1/view2) dưới {root}")

    db = load_seed_json()
    idx = max(0, min(int(start), len(episodes) - 1))
    idx = advance_to_next_pending(episodes, idx, db, root, skip_done=skip_done)
    if idx >= len(episodes):
        print("Tất cả episode đã có seed points.")
        return

    win_name = "Keypoint Annotator (multi-episode)"
    print(f"Tìm thấy {len(episodes)} episode dưới {root}")
    print(f"JSON: {JSON_PATH}")
    if skip_done:
        done_count = sum(1 for ep in episodes if episode_json_key(ep, root) in db)
        print(f"Đã annotate: {done_count}/{len(episodes)}")

    while 0 <= idx < len(episodes):
        episode_dir = episodes[idx]
        rel = episode_dir.relative_to(root)
        key = episode_json_key(episode_dir, root)
        existing = get_seed_entry(db, key)
        if key in db and not existing.get("view1") and not existing.get("view2"):
            done = " [done, no points]"
        elif key in db:
            done = " [done]"
        else:
            done = ""
        banner = f"[{idx + 1}/{len(episodes)}] {rel}{done}"
        print(banner)

        try:
            reader = EpisodeReader(episode_dir, max_frames=1)
            img_v1 = reader.bgr(0, "view1")
            img_v2 = reader.bgr(0, "view2")
        except (FileNotFoundError, ValueError) as e:
            print(f"⚠️ Bỏ qua {episode_dir}: {e}")
            idx += 1
            continue

        annotator = KeypointAnnotator(
            img_v1,
            img_v2,
            pts_v1=existing.get("view1"),
            pts_v2=existing.get("view2"),
            banner=banner,
        )
        action, pts_v1, pts_v2 = annotator.run(win_name)

        if action == "quit":
            break
        if action == "save":
            save_episode_points(episode_dir, pts_v1, pts_v2, root)
            db = load_seed_json()
            idx += 1
            idx = advance_to_next_pending(episodes, idx, db, root, skip_done=skip_done)
            continue
        if action == "next":
            idx += 1
            idx = advance_to_next_pending(episodes, idx, db, root, skip_done=skip_done)
            continue
        if action == "prev":
            idx = max(0, idx - 1)
            continue

    cv2.destroyAllWindows()
    if idx >= len(episodes):
        print("Hết danh sách episode.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Chọn keypoints lần lượt trên các episode folder."
    )
    p.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Thư mục gốc chứa nhiều episode (mặc định: data_for_web_init_frame).",
    )
    p.add_argument(
        "--skip-done",
        action="store_true",
        help="Bỏ qua episode đã có trong JSON, bắt đầu từ episode chưa annotate.",
    )
    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Chỉ số episode bắt đầu (0-based).",
    )
    p.add_argument(
        "--migrate-json",
        action="store_true",
        help="Migrate seed_points.json sang relative path (chạy một lần).",
    )
    args = p.parse_args()
    if args.migrate_json:
        migrate_seed_json(args.root)
    else:
        annotate_episodes(args.root, skip_done=args.skip_done, start=args.start)
