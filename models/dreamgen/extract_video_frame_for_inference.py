import cv2
import os
from pathlib import Path

class VideoScreenshotExtractor:
    def __init__(self, video_folder_path):
        """
        Initialize the video screenshot extractor

        Args:
            video_folder_path (str): Path to the folder containing videos
        """
        self.video_folder_path = Path(video_folder_path)
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def single_frame_screenshot(self, video_path, frame_number=0, output_folder=None):
        """
        Extract a single frame from video

        Args:
            video_path (str): Path to the video file
            frame_number (int): Frame number to extract (default: 0 for first frame)
            output_folder (str): Custom output folder (optional)

        Returns:
            str: Path to saved screenshot or None if failed
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return None

        # Open video with additional codec options to handle AV1
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None

        # Get total frame count for validation
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number >= total_frames:
            print(f"Frame number {frame_number} exceeds total frames {total_frames}")
            frame_number = total_frames - 1 if total_frames > 0 else 0

        # Try different approaches to read the frame
        success = False
        frame = None

        # Method 1: Try to set frame position and read
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True

        # Method 2: If method 1 fails or frame_number is 0, try reading from start
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for i in range(frame_number + 1):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            if ret and frame is not None:
                success = True

        if not success:
            print(f"Could not read frame {frame_number} from {video_path}")
            cap.release()
            return None

        # Create output folder
        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = video_path.parent / f"{video_path.stem}_screenshots"

        output_dir.mkdir(exist_ok=True)

        # Save screenshot
        screenshot_name = f"{video_path.stem}_frame_{frame_number}.jpg"
        screenshot_path = output_dir / screenshot_name

        success = cv2.imwrite(str(screenshot_path), frame)
        cap.release()

        if success:
            print(f"Screenshot saved: {screenshot_path}")
            return str(screenshot_path)
        else:
            print(f"Failed to save screenshot: {screenshot_path}")
            return None

    def interval_screenshots(self, video_path, num_frames=10, output_folder=None):
        """
        Extract multiple frames at regular intervals from video

        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to extract
            output_folder (str): Custom output folder (optional)

        Returns:
            list: List of paths to saved screenshots
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            return []

        # Open video with FFMPEG backend for better codec support
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video info - Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")

        if total_frames == 0:
            print("Warning: Could not determine total frames, attempting sequential read")
            # Fallback method - read frames sequentially
            return self._sequential_frame_extraction(cap, video_path, num_frames, output_folder)

        # Calculate frame intervals
        if num_frames >= total_frames:
            frame_intervals = list(range(total_frames))
        else:
            frame_intervals = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]

        # Create output folder
        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = video_path.parent / f"{video_path.stem}_screenshots"

        output_dir.mkdir(exist_ok=True)

        saved_screenshots = []

        # Extract frames
        for i, frame_num in enumerate(frame_intervals):
            # Try different methods to get the frame
            success = False
            frame = None

            # Method 1: Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True

            # Method 2: If method 1 fails, try reading sequentially
            if not success:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for j in range(frame_num + 1):
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                if ret and frame is not None:
                    success = True

            if success:
                # Calculate timestamp
                timestamp = frame_num / fps if fps > 0 else i

                # Save screenshot
                screenshot_name = f"{video_path.stem}_frame_{i+1:03d}_t{timestamp:.1f}s.jpg"
                screenshot_path = output_dir / screenshot_name

                success_save = cv2.imwrite(str(screenshot_path), frame)
                if success_save:
                    saved_screenshots.append(str(screenshot_path))
                    print(f"Screenshot {i+1}/{len(frame_intervals)} saved: {screenshot_name}")
                else:
                    print(f"Failed to save screenshot: {screenshot_name}")
            else:
                print(f"Could not read frame {frame_num}")

        cap.release()
        print(f"Total screenshots saved: {len(saved_screenshots)}")
        return saved_screenshots

    def _sequential_frame_extraction(self, cap, video_path, num_frames, output_folder):
        """
        Fallback method to extract frames sequentially when frame positioning fails
        """
        # Create output folder
        if output_folder:
            output_dir = Path(output_folder)
        else:
            output_dir = video_path.parent / f"{video_path.stem}_screenshots"

        output_dir.mkdir(exist_ok=True)

        saved_screenshots = []
        frame_count = 0
        saved_count = 0

        # Read all frames and save every nth frame
        while saved_count < num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Save frame at regular intervals
            if frame_count % max(1, (frame_count // num_frames + 1)) == 0:
                screenshot_name = f"{video_path.stem}_frame_{saved_count+1:03d}.jpg"
                screenshot_path = output_dir / screenshot_name

                success = cv2.imwrite(str(screenshot_path), frame)
                if success:
                    saved_screenshots.append(str(screenshot_path))
                    print(f"Screenshot {saved_count+1}/{num_frames} saved: {screenshot_name}")
                    saved_count += 1

            frame_count += 1

        return saved_screenshots

    def process_all_videos(self, num_frames=10, single_frame_only=False):
        """
        Process all videos in the folder

        Args:
            num_frames (int): Number of frames to extract per video
            single_frame_only (bool): If True, extract only first frame from each video

        Returns:
            dict: Dictionary with video names as keys and screenshot paths as values
        """
        if not self.video_folder_path.exists():
            print(f"Folder not found: {self.video_folder_path}")
            return {}

        # Find all video files
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(self.video_folder_path.glob(f"*{ext}"))
            video_files.extend(self.video_folder_path.glob(f"*{ext.upper()}"))

        if not video_files:
            print(f"No video files found in {self.video_folder_path}")
            return {}

        print(f"Found {len(video_files)} video files")

        results = {}
        output_folder_root = f"{self.video_folder_path}/../Screenshots"
        for video_file in video_files:
            print(f"\nProcessing: {video_file.name}")
            # episode_001547-task_pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_screenshots
            output_folder = f"{output_folder_root}/{str(video_file.name).replace('.mp4', '_screenshots')}/"
            if single_frame_only:
                screenshot = self.single_frame_screenshot(video_file, 0, output_folder)
                results[video_file.name] = [screenshot] if screenshot else []
            else:
                screenshots = self.interval_screenshots(video_file, num_frames, output_folder)
                results[video_file.name] = screenshots

        return results

# Example usage
if __name__ == "__main__":
    # CORRECTED USAGE EXAMPLES

    # Example 1: Extract single frame from one video
    video_folder = "/projects/extern/kisski/kisski-spath/dir.project/VLA_3D/binh/dreamgen/dreamgen/robocasa_100demos_5tasks_data_infer_dg/right_view/original_videos"
    extractor = VideoScreenshotExtractor(video_folder)

    #single_video = "/work/HHRI-AI/POC/public/pretraining_data/Robotic-Data/Nvidia-Cosmos-Reasoning-SFT/Cosmos-Reason1-SFT-Dataset/test-sampling-data/115709854601196208.mp4"
    #result = extractor.single_frame_screenshot(single_video, frame_number=0)

    # Example 2: Extract multiple frames from one video
    #results = extractor.interval_screenshots(single_video, num_frames=5)

    #Example 3: Process all videos in folder (extract multiple frames)
    results = extractor.process_all_videos(num_frames=5)

    # Example 4: Process all videos in folder (single frame only)
    results = extractor.process_all_videos(single_frame_only=True)

    print("Video screenshot extractor completed!")
 