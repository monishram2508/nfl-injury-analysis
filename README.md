# NFL Injury Analysis  
### DSAC Lab Project — End-to-End Computer Vision Pipeline for Injury Motion Tracking

## Overview
This project is an academic research and engineering effort developed in the **DSAC Lab** to study player movement dynamics around injury events in NFL footage.  

The core objective is to convert raw match clips into structured motion signals that can be analyzed quantitatively. To do that, the pipeline combines:
- frame extraction from game video,
- player detection + pose estimation,
- injured-player tracking across frames,
- optical-flow based motion metrics,
- and final analytical visualization of velocity/acceleration trends.

In simple terms: this repository turns unstructured video into injury-centric biomechanical evidence.

---

## Why this project matters
Injury analysis often relies heavily on manual review. This project demonstrates how computer vision can make that process:
- **more objective** (numerical measurements),
- **more repeatable** (consistent pipeline),
- **and more scalable** (automated per-frame processing).

This is especially useful for exploratory sports analytics, injury-event forensics, and future ML-based risk modeling.

---

## Key features
- **Automated frame extraction** from injury clips using FFmpeg.
- **Pose-aware person detection** with YOLOv8 pose model.
- **Interactive injured-player lock-on** (click-to-select target player in first frame).
- **Robust tracking strategy** using appearance similarity + IoU + spatial continuity.
- **Manual re-lock support** (`R` key) if identity drift occurs.
- **Optical flow velocity and acceleration extraction** in target ROI.
- **Annotated frame export** with bounding boxes and skeleton overlays.
- **Structured CSV output** for downstream data science workflows.
- **Visualization scripts** for velocity and acceleration profiles by injury and camera angle.

---

## Repository structure
```text
nfl-injury-analysis/
├── README.md
├── requirements.txt
├── pose_landmarker_lite.task
├── motion_data.csv
├── motion_data1.csv
├── report.pages
└── src/
    ├── retrieval.py      # Extract frames from video clips (FFmpeg)
    ├── track_pos.py      # Main detection + tracking + feature extraction pipeline
    ├── analyse.py        # MediaPipe pose experiment script
    ├── plot.py           # Velocity/acceleration visualization
    └── test_modules.py   # Quick dependency sanity check script
```

> Note: The runtime pipeline expects a dataset directory in the format `data/<injury_id>/<angle_id>/...`.

---

## Pipeline architecture
### 1) Frame Retrieval (`src/retrieval.py`)
- Scans injury-angle folders under `data/injury_01/`.
- Finds `.mp4` clips and extracts frames at **5 FPS**.
- Writes extracted images to `frames/` inside each angle directory.

### 2) Tracking + Motion Extraction (`src/track_pos.py`)
This is the core of the project:
1. Detect persons + 17 COCO keypoints per frame (YOLOv8 pose).
2. Let user click the injured player in frame 0.
3. Track that player frame-to-frame via weighted scoring:
   - appearance correlation (HSV histogram),
   - IoU continuity,
   - spatial consistency.
4. Compute ROI optical flow to derive velocity and acceleration.
5. Save annotated frames and write all metrics to `motion_data1.csv`.

### 3) Analytics and Visualization (`src/plot.py`)
- Loads `motion_data.csv`.
- Groups by `(injury_id, angle_id)`.
- Plots frame-wise velocity and acceleration trends.

### 4) Experimental Pose Module (`src/analyse.py`)
- Contains MediaPipe pose testing logic for selected frame windows.
- Useful as an exploratory module for alternate pose-estimation approaches.

---

## Data output schema
The generated CSV includes:
- identifiers: `injury_id`, `angle_id`, `frame_index`
- tracking box: `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`, `track_conf`
- pose keypoints: `kp_0_x ... kp_16_x`, `kp_0_y ... kp_16_y`, `kp_0_conf ... kp_16_conf`
- motion metrics: `velocity`, `acceleration`

This schema is analysis-ready for time-series modeling, feature engineering, and event comparison studies.

---

## Installation
### Prerequisites
- Python 3.10+
- FFmpeg installed and available in PATH
- GPU is recommended (for faster pose inference), but CPU is also possible

### Setup
```bash
cd nfl-injury-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to run
### Step 1 — Extract frames
```bash
python src/retrieval.py
```

### Step 2 — Run tracking + motion extraction
```bash
python src/track_pos.py
```
- Click the injured player in the selection window on first frame.
- Press `R` anytime to re-lock target.
- Press `Q` to skip clip selection when needed.

### Step 3 — Plot results
```bash
python src/plot.py
```

---

## Research and engineering highlights
- Blends **human-in-the-loop** interaction with automated tracking for practical reliability.
- Uses **adaptive appearance memory (EMA)** to reduce drift under changing lighting.
- Employs **optical flow in tracked ROI** for interpretable movement intensity metrics.
- Produces a reproducible bridge between vision outputs and data science analysis.

---

## Current limitations
- Initial target selection is manual.
- Pipeline sensitivity depends on clip quality, camera movement, and occlusions.
- Current scripts are optimized for a specific folder convention.
- No full automated test suite included yet.

---

## Future improvements
- Add multi-object tracking fallback (DeepSORT/ByteTrack style integration).
- Add automatic injured-player candidate ranking.
- Add temporal smoothing for keypoints and derived kinematics.
- Add model evaluation metrics across labeled benchmark clips.
- Package pipeline into a CLI + notebook workflow for easier reproducibility.

---

## DSAC Lab context
This project reflects the kind of practical, research-driven engineering done in a college data science/AI lab: identifying a real-world problem, building an end-to-end technical pipeline, and producing interpretable outputs that can support deeper analytics and future publication-grade work.

---

## Credits
Developed as part of a DSAC Lab initiative on sports injury analytics using computer vision, motion analysis, and applied machine learning workflows.
