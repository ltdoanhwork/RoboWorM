import cv2
import os
from glob import glob

def parse_video_path_real(video_path):
    """
    Parse:
      bimanual: {ROOT_BASE}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-XXX/{view_name}/episode_{num}.mp4
      single_arm: {ROOT_BASE}/{video_type}/{robot_type}/multiview/{task_type}/videos/chunk-XXX/{view_name}/episode_{num}.mp4
      humanoid: {root_base}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-*/{view_name}/episode_{episode_num}.mp4
                {root_base}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-000/{view_name}/episode_000000.mp4
    """
    parts = os.path.normpath(video_path).split(os.sep)
    try:
        mv_idx = parts.index("multiview")
        videos_idx = parts.index("videos", mv_idx)
    except ValueError:
        raise ValueError(f"Unexpected video path layout: {video_path}")

    parsed_robot = parts[mv_idx - 1]
    parsed_task_type = parts[mv_idx + 1]
    if parsed_robot == "bimanual":
        task_name = parts[mv_idx + 2]
    elif parsed_robot == "single_arm":
        task_name = None
    elif parsed_robot == "humanoid":
        task_name = parts[mv_idx + 2]

    view_name = parts[videos_idx + 2]  # chunk-XXX / view_name / episode_*.mp4
    episode_name = os.path.splitext(parts[-1])[0]  # episode_000026
    return parsed_robot, parsed_task_type, task_name, view_name, episode_name

import os

def parse_video_path_ctrlworld(video_path: str):
    """
    Parse the new desired path structure:
      {ROOT_BASE}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{task_name}__{episode_num}/pred_all_views.mp4
      {ROOT_BASE}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{episode_num}/pred_all_views.mp4
    
    Returns:
        tuple: (robot_type, model_type, task_type, task_name, episode_num, view_name)
    """
    parts = os.path.normpath(video_path).split(os.sep)
    
    try:
        mv_idx = parts.index("multiview")
    except ValueError:
        raise ValueError(f"Unexpected video path layout (missing 'multiview'): {video_path}")

    robot_type = parts[mv_idx - 2]
    
    model_type = parts[mv_idx + 1]
    
    task_type = parts[mv_idx + 2]
    
    episode_dir = parts[mv_idx + 3]
    if robot_type == "bimanual":
        if episode_dir.startswith("episode_") and "__" in episode_dir:
            core_info = episode_dir[len("episode_"):] 
            task_name, episode_num = core_info.split("__", 1)
        else:
            task_name = episode_dir
            episode_num = "unknown"
    elif robot_type == "humanoid":
        if episode_dir.startswith("episode_") and "__" in episode_dir:
            core_info = episode_dir[len("episode_"):] 
            task_name, episode_num = core_info.split("__", 1)
        else:
            task_name = episode_dir
            episode_num = "unknown"
    elif robot_type == "single_arm":
        if episode_dir.startswith("episode_"):
            episode_num = episode_dir[len("episode_"):]
            task_name=None
        else:
            episode_num = episode_dir
            task_name=None

    view_name = os.path.splitext(parts[-1])[0]
    
    return robot_type, model_type, task_type, task_name, episode_num, view_name

# def parse_video_path_dreamgen(video_path: str):
#     """
#     Parse the new desired path structure:
#       dreamgen: 
#         bimanual:
#             markovian: {ROOT_BASE}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/iter_000120000/{task_name}/episode_{episode_num}/full_pred.mp4
#             non_markovian: {ROOT_BASE}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/iter_000113000/{task_name}/episode_{episode_num}.mp4
#         single_arm:
#             markovian: {ROOT_BASE}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{episode_num}.mp4
#     Returns:
#         tuple: (robot_type, model_type, task_type, task_name, episode_num, view_name)
#     """
#     parts = os.path.normpath(video_path).split(os.sep)
    
#     try:
#         mv_idx = parts.index("multiview")
#     except ValueError:
#         raise ValueError(f"Unexpected video path layout (missing 'multiview'): {video_path}")

#     robot_type = parts[mv_idx - 2]
#     model_type = parts[mv_idx + 1]
#     task_type = parts[mv_idx + 2]
    
#     task_name = parts[mv_idx + 4]
#     print(task_type)

#     # if task_type == "makovian":
#     #     episode_dir = parts[mv_idx + 5]
#     #     if episode_dir.startswith("episode_"):
#     #         episode_num = episode_dir[len("episode_"):] 
#     #     else:
#     #         episode_num = episode_dir

#     #     view_name = os.path.splitext(parts[-1])[0]
#     # elif task_type == "non_makovian":
#     #     episode_num = os.path.splitext(parts[-1])[0]
#     #     view_name = None
#     episode_num = os.path.splitext(parts[-1])[0]
#     view_name = None
    
#     return robot_type, model_type, task_type, task_name, episode_num, view_name

def parse_video_path_dreamgen(video_path: str):
    """
    Parse the dreamgen path structures based on structural depth:
      1. bimanual + markovian:
         .../multiview/{model_type}/{task_type}/iter_xxxx/{task_name}/episode_{episode_num}/full_pred.mp4
      2. bimanual + non_markovian:
         .../multiview/{model_type}/{task_type}/iter_xxxx/{task_name}/episode_{episode_num}.mp4
      3. single_arm + markovian:
         .../multiview/{model_type}/{task_type}/episode_{episode_num}.mp4

    Returns:
        tuple: (robot_type, model_type, task_type, task_name, episode_num, view_name)
    """
    parts = os.path.normpath(video_path).split(os.sep)
    
    try:
        mv_idx = parts.index("multiview")
    except ValueError:
        raise ValueError(f"Unexpected video path layout (missing 'multiview'): {video_path}")

    # 1. Lấy thông tin cố định dựa trên vị trí của 'multiview'
    robot_type = parts[mv_idx - 2]
    model_type = parts[mv_idx + 1]
    task_type = parts[mv_idx + 2]

    # Khởi tạo các giá trị mặc định phòng trường hợp cấu trúc tối giản
    task_name = None
    episode_num = None
    view_name = None

    file_name_ext = parts[-1]
    file_name = os.path.splitext(file_name_ext)[0]

    # 2. Phân tách dựa trên độ sâu tính từ đuôi video ngược lên
    # Tính số lượng thư mục con đứng sau 'task_type'
    remaining_depth = len(parts) - 1 - (mv_idx + 2)

    if remaining_depth == 1:
        # Case 3: single_arm (markovian) -> .../{task_type}/episode_{episode_num}.mp4
        # Không có task_name, không có view_name riêng biệt
        if file_name.startswith("episode_"):
            episode_num = file_name[len("episode_"):]
        else:
            episode_num = file_name

    elif remaining_depth == 3:
        # Case 2: bimanual (non_markovian) -> .../iter_xxxx/{task_name}/episode_{episode_num}.mp4
        task_name = parts[-2]
        if file_name.startswith("episode_"):
            episode_num = file_name[len("episode_"):]
        else:
            episode_num = file_name

    elif remaining_depth == 4:
        # Case 1: bimanual (markovian) -> .../iter_xxxx/{task_name}/episode_{episode_num}/full_pred.mp4
        view_name = file_name
        
        episode_dir = parts[-2]
        if episode_dir.startswith("episode_"):
            episode_num = episode_dir[len("episode_"):]
        else:
            episode_num = episode_dir
            
        task_name = parts[-3]
        
    else:
        raise ValueError(f"Unknown dreamgen path depth structure for layout: {video_path}")

    return robot_type, model_type, task_type, task_name, episode_num, view_name

def get_frame_indices(total_frames, video_fps, extract_fps):
    """Sample frames at extract_fps, always including the last frame."""
    if total_frames <= 0:
        return []

    step = video_fps / extract_fps if extract_fps > 0 else 1.0
    frame_indices = []
    frame_to_seek_float = 0.0

    while True:
        idx = int(round(frame_to_seek_float))
        if idx >= total_frames:
            break
        if not frame_indices or frame_indices[-1] != idx:
            frame_indices.append(idx)
        frame_to_seek_float += step

    last_frame_idx = total_frames - 1
    if not frame_indices or frame_indices[-1] != last_frame_idx:
        frame_indices.append(last_frame_idx)

    return frame_indices

# def extract_views_from_video_multiview(video_path, output_root, extract_fps=1, split_layout="1x3", model_type="ctrlworld"): # split_layout: 2x2 / 1x3
#     video_name = os.path.splitext(os.path.basename(video_path))[0]

#     if model_type == "ctrlworld":
#         robot, model_type, ttype, task_name, episode_name,_ = parse_video_path_ctrlworld(video_path)
#     elif model_type == "dreamgen":
#         robot, model_type, ttype, task_name, episode_name,_ = parse_video_path_dreamgen(video_path)
#     else:
#         print(f"Unknown model_type: {model_type}")
#         return

#     if task_name is None:
#         video_output_dir = os.path.join(output_root, robot, model_type, ttype, episode_name)
#     else:   
#         video_output_dir = os.path.join(output_root, robot, model_type, ttype, task_name, episode_name)
#     os.makedirs(video_output_dir, exist_ok=True)

#     view_names = ["view1", "view2", "view3", "view4"] if split_layout == "2x2" else ["view1", "view2", "view3"]
#     view_dirs = [os.path.join(video_output_dir, vn) for vn in view_names]

#     for d in view_dirs:
#         os.makedirs(d, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"❌ Error: Không thể mở video {video_path}")
#         return

#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames <= 0:
#         print(f"⚠️ Warning: Video {video_name} không có frame nào.")
#         cap.release()
#         return

#     step = video_fps / extract_fps

#     frame_indices = []
#     frame_to_seek_float = 0.0

#     while True:
#         idx = int(round(frame_to_seek_float))
#         if idx >= total_frames:
#             break
#         if not frame_indices or frame_indices[-1] != idx:
#             frame_indices.append(idx)
#         frame_to_seek_float += step

#     last_frame_idx = total_frames - 1
#     if not frame_indices or frame_indices[-1] != last_frame_idx:
#         frame_indices.append(last_frame_idx)

#     saved_count = 0
#     for frame_to_seek in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek)
#         ret, frame = cap.read()
        
#         if not ret:
#             continue

#         h, w, _ = frame.shape
        
#         if split_layout == "2x2":
#             mid_h, mid_w = h // 2, w // 2
#             views = [
#                 frame[0:mid_h, 0:mid_w],      
#                 frame[0:mid_h, mid_w:w],      
#                 frame[mid_h:h, 0:mid_w],      
#                 frame[mid_h:h, mid_w:w]      
#             ]
#         elif split_layout == "1x3":
#             split_w = w // 3
#             views = [
#                 frame[0:h, 0:split_w],
#                 frame[0:h, split_w:2*split_w],
#                 frame[0:h, 2*split_w:w]
#             ]

#         # Lưu từng view
#         for i, view_img in enumerate(views):
#             save_path = os.path.join(view_dirs[i], f"frame_{saved_count:04d}.png")
#             cv2.imwrite(save_path, view_img)

#         saved_count += 1

#     cap.release()
#     print(f"{saved_count} frames ({extract_fps} fps + Last Frame) of {video_name}")

def parse_video_path_tmp(video_path: str):
    """
    Parse tmp low-quality multiview videos:
      tmp/lowquality_multiview/episode_{episode_num}.mp4
    """
    parts = os.path.normpath(video_path).split(os.sep)
    episode_name = os.path.splitext(parts[-1])[0]
    return "single_arm", "tmp", "lowquality", None, episode_name, None


def get_preserved_output_dir(video_path, video_root, output_root):
    """
    Mirror source video relative path under output_root.
    e.g. dense/bimanual/.../episode_000039/full_pred.mp4
      -> {output_root}/dense/bimanual/.../episode_000039
    """
    rel = os.path.relpath(os.path.normpath(video_path), os.path.normpath(video_root))
    basename = os.path.basename(rel)
    if basename in ("full_pred.mp4", "pred_all_views.mp4"):
        rel_dir = os.path.dirname(rel)
    else:
        rel_dir = os.path.splitext(rel)[0]
    return os.path.join(output_root, rel_dir)


def extract_views_from_video_multiview(
    video_path,
    output_root,
    extract_fps: float | None = None,
    split_layout="1x3",
    model_type="ctrlworld",
    first_frame_only=False,
    video_output_dir=None,
):
    """Extract mosaic video into per-view frames via ffmpeg (AV1-safe)."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if video_output_dir is None:
        if model_type == "ctrlworld":
            robot, model_type, ttype, task_name, episode_name, _ = parse_video_path_ctrlworld(video_path)
        elif model_type == "dreamgen":
            robot, model_type, ttype, task_name, episode_name, _ = parse_video_path_dreamgen(video_path)
        elif model_type == "tmp":
            robot, model_type, ttype, task_name, episode_name, _ = parse_video_path_tmp(video_path)
        else:
            print(f"Unknown model_type: {model_type}")
            return

        if task_name is None:
            video_output_dir = os.path.join(output_root, robot, model_type, ttype, episode_name)
        else:
            video_output_dir = os.path.join(output_root, robot, model_type, ttype, task_name, episode_name)
    else:
        video_output_dir = os.path.normpath(video_output_dir)

    os.makedirs(video_output_dir, exist_ok=True)

    view_names = ["view1", "view2", "view3", "view4"] if split_layout == "2x2" or split_layout == "1x4" else ["view1", "view2", "view3"]
    view_dirs = [os.path.join(video_output_dir, vn) for vn in view_names]
    for d in view_dirs:
        os.makedirs(d, exist_ok=True)
        for old in glob(os.path.join(d, "frame_*.png")):
            os.remove(old)

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH")
        return

    # Match previous cv2 crop logic:
    # 1x3: widths iw//3, iw//3, iw-2*(iw//3)
    # 2x2: quadrants iw//2 x ih//2
    frame_select = build_multiview_frame_prefix(extract_fps, first_frame_only)
    if split_layout == "2x2":
        filter_complex = (
            f"[0:v]{frame_select}split=4[a][b][c][d];"
            "[a]crop=iw/2:ih/2:0:0[v1];"
            "[b]crop=iw/2:ih/2:iw/2:0[v2];"
            "[c]crop=iw/2:ih/2:0:ih/2[v3];"
            "[d]crop=iw/2:ih/2:iw/2:ih/2[v4]"
        )
        maps = ["v1", "v2", "v3", "v4"]
    elif split_layout == "1x3":
        filter_complex = (
            f"[0:v]{frame_select}split=3[a][b][c];"
            "[a]crop=iw/3:ih:0:0[v1];"
            "[b]crop=iw/3:ih:iw/3:0[v2];"
            "[c]crop=iw-2*iw/3:ih:2*iw/3:0[v3]"
        )
        maps = ["v1", "v2", "v3"]
    elif split_layout == "1x4":
        filter_complex = (
            f"[0:v]{frame_select}split=4[a][b][c][d];"
            "[a]crop=iw/4:ih:0:0[v1];"
            "[b]crop=iw/4:ih:iw/4:0[v2];"
            "[c]crop=iw/4:ih:iw/2:0[v3];"
            "[d]crop=iw/4:ih:3*iw/4:0[v4]"
        )
        maps = ["v1", "v2", "v3", "v4"]
    else:
        print(f"Unknown split_layout: {split_layout}")
        return

    def _build_cmd(decoder=None):
        cmd = ["ffmpeg", "-y", "-hwaccel", "none"]
        if decoder:
            cmd += ["-c:v", decoder]
        cmd += ["-i", video_path, "-filter_complex", filter_complex]
        for label, view_dir in zip(maps, view_dirs):
            cmd += [
                "-map", f"[{label}]",
                "-start_number", "0",
            ]
            if first_frame_only:
                cmd += ["-frames:v", "1"]
            cmd += [os.path.join(view_dir, "frame_%04d.png")]
        return cmd

    r = subprocess.run(_build_cmd("libdav1d"), capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(_build_cmd(None), capture_output=True, text=True)

    if r.returncode != 0:
        print(f"Error: ffmpeg failed for {video_path}\n{r.stderr[-800:]}")
        return

    saved_count = len(glob(os.path.join(view_dirs[0], "frame_*.png")))
    fps_label = format_extract_fps_label(extract_fps, first_frame_only)
    if first_frame_only:
        print(f"frame_0000 x {len(view_dirs)} views of {video_name}")
    else:
        if extract_fps is None:
            native = get_video_fps(video_path)
            if native is not None:
                fps_label = f"native fps ({native:.3f})"
        print(f"{saved_count} frames ({fps_label}) of {video_name}")


# def extract_views_from_video_oneview(video_path, output_root, extract_fps=1, robot_type=None, task_type=None):
#     """
#     Extract frames from a single-view video (one camera per file).

#     video_path: {ROOT_BASE}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-000/{view_name}/episode_{num}.mp4
#     frame folder: {robot_type}/{task_type}/{task_name}/episode_{num}/{view_name}/frame_{frame_num}.png
#     """
#     try:
#         parsed_robot, parsed_task_type, task_name, view_name, episode_name = parse_video_path_real(video_path)
#     except ValueError as e:
#         print(f"Error: {e}")
#         return

#     robot = robot_type or parsed_robot
#     ttype = task_type or parsed_task_type

#     if task_name is None:
#         out_dir = os.path.join(output_root, robot, ttype, episode_name, view_name)
#     else:   
#         out_dir = os.path.join(output_root, robot, ttype, task_name, episode_name, view_name)
#     os.makedirs(out_dir, exist_ok=True)

#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: cannot open video {video_path}")
#         return

#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     if total_frames <= 0 or video_fps <= 0:
#         print(f"Warning: invalid video metadata for {video_path}")
#         cap.release()
#         return
#     print(video_fps)
#     frame_indices = get_frame_indices(total_frames, video_fps, extract_fps)

#     saved_count = 0
#     for frame_to_seek in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_seek)
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         save_path = os.path.join(out_dir, f"frame_{saved_count:04d}.png")
#         cv2.imwrite(save_path, frame)
#         saved_count += 1

#     cap.release()
#     print(f"{saved_count} frames ({extract_fps} fps) | {robot}/{ttype}/{task_name}/{episode_name}/{view_name}")

import subprocess
import shutil
import argparse


def get_video_fps(video_path: str) -> float | None:
    """Return native video FPS via ffprobe, or None if unavailable."""
    if not shutil.which("ffprobe"):
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    rate = r.stdout.strip()
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den)
        if den_f == 0:
            return None
        return float(num) / den_f
    try:
        return float(rate)
    except ValueError:
        return None


def build_multiview_frame_prefix(extract_fps: float | None, first_frame_only: bool) -> str:
    if first_frame_only:
        return "select=eq(n\\,0),"
    if extract_fps is None:
        return ""
    return f"fps={extract_fps},"


def build_oneview_vf(extract_fps: float | None, first_frame_only: bool) -> str | None:
    if first_frame_only:
        return "select=eq(n\\,0)"
    if extract_fps is None:
        return None
    return f"fps={extract_fps}"


def format_extract_fps_label(extract_fps: float | None, first_frame_only: bool) -> str:
    if first_frame_only:
        return "frame_0000"
    if extract_fps is None:
        return "native fps"
    return f"{extract_fps} fps"

def extract_views_from_video_oneview(
    video_path, output_root, extract_fps: float | None = None, robot_type=None, task_type=None, first_frame_only=False
):
    try:
        parsed_robot, parsed_task_type, task_name, view_name, episode_name = parse_video_path_real(video_path)
    except ValueError as e:
        print(f"Error: {e}")
        return

    robot = robot_type or parsed_robot
    ttype = task_type or parsed_task_type

    if task_name is None:
        out_dir = os.path.join(output_root, robot, "real", ttype, episode_name, view_name)
    else:
        out_dir = os.path.join(output_root, robot, "real", ttype, task_name, episode_name, view_name)
    os.makedirs(out_dir, exist_ok=True)

    # Clear old frames so numbering stays clean
    for old in glob(os.path.join(out_dir, "frame_*.png")):
        os.remove(old)

    pattern = os.path.join(out_dir, "frame_%04d.png")
    vf = build_oneview_vf(extract_fps, first_frame_only)
    frame_args = ["-frames:v", "1"] if first_frame_only else []

    def _build_cmd(decoder=None):
        cmd = ["ffmpeg", "-y", "-hwaccel", "none"]
        if decoder:
            cmd += ["-c:v", decoder]
        cmd += ["-i", video_path]
        if vf is not None:
            cmd += ["-vf", vf]
        cmd += ["-start_number", "0", *frame_args, pattern]
        return cmd

    # Prefer software AV1 decoder; fall back if unavailable
    cmd = _build_cmd(None)

    exe = shutil.which("ffmpeg")
    if not exe:
        print("Error: ffmpeg not found in PATH")
        return

    r = subprocess.run(_build_cmd("libdav1d"), capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run(cmd, capture_output=True, text=True)

    if r.returncode != 0:
        print(f"Error: ffmpeg failed for {video_path}\n{r.stderr[-500:]}")
        return

    saved_count = len(glob(os.path.join(out_dir, "frame_*.png")))
    fps_label = format_extract_fps_label(extract_fps, first_frame_only)
    if not first_frame_only and extract_fps is None:
        native = get_video_fps(video_path)
        if native is not None:
            fps_label = f"native fps ({native:.3f})"
    print(f"{saved_count} frames ({fps_label}) | {robot}/{ttype}/{task_name}/{episode_name}/{view_name}")

# def collect_video_files(root_base, video_type, robot_type, task_type, view_names):
#     """
#     Collect per-view episode videos under:
#     real:
#       {root_base}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-*/{view_name}/*.mp4
    
#     ctrlworld:
#         {root_base}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{task_name}__{episode_num}/pred_all_views.mp4
    
#     """
#     if video_type == "real":
#         base = os.path.join(root_base, video_type, robot_type, "multiview", task_type)
#     elif video_type == "ctrlworld":
#         base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
#     video_files = []
#     for view_name in view_names:
#         if video_type == "real":
#             pattern = os.path.join(base, "*", "videos", "chunk-*", view_name, "*.mp4")
#         elif video_type == "ctrlworld":
#             pattern = os.path.join(base, "episode_{task_name}__{episode_num}", "pred_all_views.mp4")
#         video_files.extend(glob(pattern))
#     return sorted(video_files)

import os
from glob import glob

def collect_video_files(root_base, video_type, robot_type, task_type, view_names=None, model_type="*"):
    """
    Collect per-view episode videos under:
    real:
      bimanual: {root_base}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-*/{view_name}/*.mp4
      single_arm: {root_base}/{video_type}/{robot_type}/multiview/{task_type}/videos/chunk-*/{view_name}/*.mp4
      humanoid: {root_base}/{video_type}/{robot_type}/multiview/{task_type}/{task_name}/videos/chunk-*/{view_name}/episode_{episode_num}.mp4

    dreamgen:
        single_arm:
            makovian: {root_base}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{episode_num}.mp4
        huamnoid:
            {root_base}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/iter_000100000/{task_name}/episode_{episode_num}/full_pred.mp4
        
    ctrlworld:
      {root_base}/{video_type}/{robot_type}/output/multiview/{model_type}/{task_type}/episode_{task_name}__{episode_num}/pred_all_views.mp4
      {root_base}/dense/bimanual/output/multiview/ctrlworld_4view_grid/makovian/episode_close_cardboard_box__000026/pred_all_views.mp4
    """
    video_files = []

    if video_type == "real":
        if not view_names:
            raise ValueError("view_names list must be provided for 'real' video_type.")
            
        base = os.path.join(root_base, video_type, robot_type, "multiview", task_type)
        
        for view_name in view_names:
            if robot_type == "bimanual":
                pattern = os.path.join(base, "*", "videos", "chunk-*", view_name, "*.mp4")
            elif robot_type == "single_arm":
                pattern = os.path.join(base, "videos", "chunk-*", view_name, "*.mp4")
            elif robot_type == "humanoid":
                pattern = os.path.join(base, "*", "videos", "chunk-*", view_name, "*.mp4")
            video_files.extend(glob(pattern))
            
    elif video_type == "dense":
        if model_type == "ctrlworld":
            base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
            
            pattern = os.path.join(base, "episode_*", "pred_all_views.mp4")
            video_files.extend(glob(pattern))

        elif model_type == "dreamgen":
            if robot_type == "bimanual":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                if task_type == "makovian":
                    pattern = os.path.join(base, "iter_000120000", "*", "episode_*", "full_pred.mp4")
                elif task_type == "non_makovian":
                    pattern = os.path.join(base, "iter_000120000", "*", "episode_*", "full_pred.mp4")
                    # pattern = os.path.join(base, "iter_000113000", "*", "episode_*.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "single_arm":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "episode_*.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "humanoid":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "iter_000100000", "*", "episode_*", "full_pred.mp4")
                video_files.extend(glob(pattern))
    elif video_type == "worldcache":
        if model_type == "ctrlworld":
            base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
            
            pattern = os.path.join(base, "episode_*", "pred_all_views.mp4")
            video_files.extend(glob(pattern))

        elif model_type == "dreamgen":
            if robot_type == "bimanual":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                if task_type == "makovian":
                    pattern = os.path.join(base, "iter_000120000", "*", "episode_*", "full_pred.mp4")
                elif task_type == "non_makovian":
                    pattern = os.path.join(base, "iter_000120000", "*",  "episode_*", "full_pred.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "single_arm":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "episode_*.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "humanoid":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "iter_000100000", "*", "episode_*", "full_pred.mp4")
                video_files.extend(glob(pattern))
    elif video_type == "augment":
        if model_type == "ctrlworld":
            base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
            pattern = os.path.join(base, "episode_*", "pred_all_views.mp4")

            video_files.extend(glob(pattern))

        elif model_type == "dreamgen":
            if robot_type == "bimanual":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                if task_type == "makovian":
                    pattern = os.path.join(base, "iter_000120000", "*", "episode_*", "full_pred.mp4")
                elif task_type == "non_makovian":
                    pattern = os.path.join(base, "iter_000120000", "*",  "episode_*", "full_pred.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "single_arm":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "episode_*.mp4")
                video_files.extend(glob(pattern))
            elif robot_type == "humanoid":
                base = os.path.join(root_base, video_type, robot_type, "output", "multiview", model_type, task_type)
                pattern = os.path.join(base, "iter_000100000", "*", "episode_*", "full_pred.mp4")
                video_files.extend(glob(pattern))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")   
    return sorted(list(set(video_files))) # set() để loại bỏ trùng lặp nếu có


def load_videos_from_md(md_path, video_root):
    """Load video paths listed in all_videos.md (one path per line)."""
    video_root = os.path.normpath(video_root)
    paths = []
    with open(md_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if os.path.isabs(line):
                paths.append(os.path.normpath(line))
            else:
                paths.append(os.path.normpath(os.path.join(video_root, line)))
    return paths


def infer_video_config(video_path, video_root=None):
    """Infer model_type and split_layout from a video path."""
    if video_root:
        rel = os.path.relpath(os.path.normpath(video_path), os.path.normpath(video_root))
    else:
        rel = os.path.normpath(video_path)
    parts = rel.split(os.sep)

    if parts[0] == "tmp" and "lowquality_multiview" in parts:
        return "tmp", "2x2"

    if "multiview" not in parts:
        return None, None

    mv_idx = parts.index("multiview")
    robot_type = parts[mv_idx - 2] if mv_idx >= 2 else None
    model_type = parts[mv_idx + 1] if mv_idx + 1 < len(parts) else None

    if model_type == "dreamgen":
        split_layout = "2x2"
    elif model_type == "ctrlworld":
        split_layout = "1x4" if robot_type in ("bimanual", "humanoid") else "1x3"
    else:
        return None, None

    return model_type, split_layout


def main():
    p = argparse.ArgumentParser(description="Extract multiview frames from videos listed in all_videos.md")
    p.add_argument(
        "--root",
        default="/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/3.Code/"
        "VideoWordModelBenchmark/thanhnv_multiview/data/video",
        help="Video root directory",
    )
    p.add_argument(
        "--videos-md",
        default="/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/3.Code/"
        "VideoWordModelBenchmark/human_assessment_feature/assets/all_videos.md",
        help="Markdown file listing relative video paths (one per line)",
    )
    p.add_argument(
        "--output",
        default="/home/thanhnv/data2_disk/VT/1.VinRobotics/1.Projects/3.Code/"
        "VideoWordModelBenchmark/thanhnv_multiview/data/data_for_web_init_frame",
        help="Output root (mirrors relative video path)",
    )
    p.add_argument(
        "--all-frames",
        action="store_true",
        help="Extract frames for the entire video (default: only frame_0000)",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Target extract FPS with --all-frames. Omit for native video FPS.",
    )
    args = p.parse_args()

    first_frame_only = not args.all_frames

    ROOT_BASE = args.root
    ALL_VIDEOS_MD = args.videos_md
    output_base = args.output
    extract_fps = args.fps

    mode = format_extract_fps_label(extract_fps, first_frame_only)
    print(f"Mode: {mode}")

    video_files = load_videos_from_md(ALL_VIDEOS_MD, ROOT_BASE)
    if not video_files:
        print(f"No videos found in {ALL_VIDEOS_MD}")
        return

    print(f"Found {len(video_files)} videos in {ALL_VIDEOS_MD}")
    skipped = 0
    for video_f in video_files:
        if not os.path.isfile(video_f):
            print(f"Skip (not found): {video_f}")
            skipped += 1
            continue

        model_type, split_layout = infer_video_config(video_f, ROOT_BASE)
        if model_type is None:
            print(f"Skip (unknown layout): {video_f}")
            skipped += 1
            continue

        extract_views_from_video_multiview(
            video_f,
            output_base,
            extract_fps=extract_fps,
            split_layout=split_layout,
            model_type=model_type,
            first_frame_only=first_frame_only,
            video_output_dir=get_preserved_output_dir(video_f, ROOT_BASE, output_base),
        )

    print(f"\nDone. processed={len(video_files) - skipped}, skipped={skipped}")


if __name__ == "__main__":
    main()
