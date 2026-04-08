import os
from pathlib import Path
import subprocess

root=Path(__file__).parent.parent
base_dir=root/"data"/"injury_01"

for angle in os.listdir(base_dir):
    angle_path = os.path.join(base_dir, angle)

    if not os.path.isdir(angle_path):
        continue

    print(angle_path)

    video_file = None
    for file in os.listdir(angle_path):
        if file.endswith(".mp4"):
            video_file = file
            break

    if video_file is None:
        print(f"no clip found in {angle}")
        continue

    clip_path = os.path.join(angle_path, video_file)

    frames_dir = os.path.join(angle_path, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
        print(f"frames already exist for {angle}, skipping.")
        continue

    output_path=os.path.join(frames_dir, "frame_%03d.jpg")

    command = [
        "ffmpeg",
        "-i", clip_path,
        "-vf", "fps=5",
        output_path
    ]

    subprocess.run(command)

print("done")
