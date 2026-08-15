1. Run extract.py with a folder containing video frames (extract with exact original FPS of the video) --> save data into a npz file
    Data folder structure:
        episode/
            view1/frame_0000.png ...
            view2/frame_0000.png ...

2. overlay.py to write a mp4 video with overlay from 1

- use point_picker.py to pick the initial keypoint to track (should run on a local machine - monitor available) -> it will write to the seed_points.json