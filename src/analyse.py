import cv2
import mediapipe as mp
import os
from pathlib import Path

# --- Setup ---
mp_pose = mp.tasks.vision
mp_base = mp.tasks

root=Path(__file__).parent.parent
model_path = "pose_landmarker_lite.task"

base_options = mp_base.BaseOptions(
    model_asset_path=model_path
)

options = mp_pose.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_pose.RunningMode.IMAGE
)

pose = mp_pose.PoseLandmarker.create_from_options(options)

frames_dir = root/"data"/"injury_01"/"angle_03"/"frames"
output_dir = root/"data"/"injury_01"/"angle_03"/"output"
os.makedirs(output_dir, exist_ok=True)

for i in range(14, 28):  # your ±5 window
    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.jpg")

    image = cv2.imread(frame_path)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = pose.detect(mp_image)

    print(f"Frame {i}: {len(result.pose_landmarks)} poses detected")

    # --- Draw pose ---
    annotated = image.copy()

    if result.pose_landmarks:
        for landmark_list in result.pose_landmarks:
            for lm in landmark_list:
                h, w, _ = image.shape
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)

    cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.jpg"), annotated)

print("Done.")