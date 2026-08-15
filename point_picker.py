"""Tool chọn và tinh chỉnh keypoints bằng chuột + phím mũi tên."""

from __future__ import annotations

import json
import re
from pathlib import Path
import cv2
import numpy as np

JSON_PATH = Path("seed_points.json")



__all__ = [
    "intersect_frames",
    "list_frame_paths",
    "load_frame_bgr",
    "frame_path",
    "EpisodeReader",
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
    print(f"✅ Đã lưu seed points vào {JSON_PATH.resolve()}")


class KeypointAnnotator:

    def __init__(self, img_v1: np.ndarray, img_v2: np.ndarray):
        self.img1 = img_v1.copy()
        self.img2 = img_v2.copy()
        self.h, self.w, _ = img_v1.shape

        self.pts_v1: list[list[float]] = []
        self.pts_v2: list[list[float]] = []

        # State management
        self.active_view = 1  # 1: view1, 2: view2
        self.selected_idx = -1  # Chỉ số point đang được chọn để tinh chỉnh

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
            if x < self.w:  # Click View 1
                curr_x, curr_y = x, y
                pts = self.pts_v1
                self.active_view = 1
            else:  # Click View 2
                curr_x, curr_y = x - self.w, y
                pts = self.pts_v2
                self.active_view = 2

            # Kiểm tra xem có đang click gần một điểm cũ để "Chọn" không
            clicked_near_idx = -1
            for i, p in enumerate(pts):
                if np.hypot(p[0] - curr_x, p[1] - curr_y) < 10:  # R = 10px
                    clicked_near_idx = i
                    break

            if clicked_near_idx != -1:
                # Click gần điểm cũ -> Chọn điểm đó để tinh chỉnh
                self.selected_idx = clicked_near_idx
            else:
                # Click chỗ trống -> Thêm điểm mới (không giới hạn n_points)
                pts.append([float(curr_x), float(curr_y)])
                self.selected_idx = len(pts) - 1

    def _draw_view_points(
        self, img: np.ndarray, pts: list[list[float]], is_active_view: bool
    ):
        """Hàm hỗ trợ vẽ các điểm lên từng View."""
        for i, (x, y) in enumerate(pts):
            c = self.colors[i % len(self.colors)]
            is_selected = is_active_view and (i == self.selected_idx)
            pt = (int(round(x)), int(round(y)))

            if is_selected:
                # Vẽ dấu + (Cross) rực rỡ tại chính xác tâm điểm đang chọn
                cv2.drawMarker(
                    img,
                    pt,
                    c,
                    markerType=cv2.MARKER_CROSS,  # Đổi sang MARKER_TILTED_CROSS nếu muốn dấu 'X'
                    markerSize=16,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )
            else:
                # Điểm bình thường vẽ chấm tròn nhỏ
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

        # Thanh Hướng dẫn & Trạng thái
        n1, n2 = len(self.pts_v1), len(self.pts_v2)
        sel_str = f"p{self.selected_idx+1}" if self.selected_idx != -1 else "None"
        info_str = f"Points: V1={n1}, V2={n2} | Selected: View{self.active_view} ({sel_str}) | "
        info_str += (
            "ARROWS: Fine-tune | 'S': Save | 'R': Reset | 'Q': Cancel"
        )

        cv2.putText(
            canvas,
            info_str,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def run(self) -> tuple[np.ndarray, np.ndarray] | None:
        win_name = "Keypoint Annotator (+ Marker for Selected Point)"
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, self.mouse_callback)

        while True:
            canvas = self.draw()
            cv2.imshow(win_name, canvas)
            key = cv2.waitKeyEx(30)

            if key in (27, ord("q")):  # ESC hoặc Q -> Thoát không lưu
                cv2.destroyAllWindows()
                return None

            elif key == ord("s"):  # S -> Lưu kết quả
                n1, n2 = len(self.pts_v1), len(self.pts_v2)

                # Kiểm tra số lượng điểm từng cặp ở 2 view
                if n1 == 0 or n2 == 0:
                    print(
                        "⚠️ Vui lòng chọn ít nhất 1 điểm ở CẢ HAI view trước khi lưu!"
                    )
                    continue
                if n1 != n2:
                    print(
                        f"⚠️ Số lượng điểm ở View 1 ({n1}) và View 2 ({n2}) không bằng nhau!"
                    )
                    continue

                cv2.destroyAllWindows()
                return np.array(self.pts_v1, dtype=np.float32), np.array(
                    self.pts_v2, dtype=np.float32
                )

            elif key == ord("r"):  # R -> Reset chọn lại từ đầu
                self.pts_v1.clear()
                self.pts_v2.clear()
                self.selected_idx = -1

            # Tinh chỉnh từng Pixel bằng các phím mũi tên
            elif self.selected_idx != -1:
                pts = self.pts_v1 if self.active_view == 1 else self.pts_v2
                step = 1.0  # Tốc độ nhích pixel

                # Phím mũi tên (Mã Keycode trong cv2.waitKeyEx)
                if key in (65361, 2424832, 81):  # Left
                    pts[self.selected_idx][0] -= step
                elif key in (65363, 2555904, 83):  # Right
                    pts[self.selected_idx][0] += step
                elif key in (65362, 2490368, 82):  # Up
                    pts[self.selected_idx][1] -= step
                elif key in (65364, 2621440, 84):  # Down
                    pts[self.selected_idx][1] += step

                # TAB để chuyển đổi giữa các point
                elif key == 9 and len(pts) > 0:
                    self.selected_idx = (self.selected_idx + 1) % len(pts)


def annotate_episode(episode_dir: str | Path) -> dict[str, np.ndarray] | None:
    episode_dir = Path(episode_dir)
    reader = EpisodeReader(episode_dir, max_frames=1)

    img_v1 = reader.bgr(0, "view1")
    img_v2 = reader.bgr(0, "view2")

    annotator = KeypointAnnotator(img_v1, img_v2)
    res = annotator.run()

    if res is not None:
        pts_v1, pts_v2 = res
        db = load_seed_json()
        db[episode_dir.name] = {
            "view1": pts_v1.tolist(),
            "view2": pts_v2.tolist(),
        }
        save_seed_json(db)
        return {"view1": pts_v1, "view2": pts_v2}
    return None


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("episode_dir", type=str)
    args = p.parse_args()

    annotate_episode(args.episode_dir)