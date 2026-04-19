"""
src/track.py
------------
Per-frame bounding box + pose skeleton for the one injured player per clip.

Pipeline:
  1. YOLOv8-pose detects all persons + 17 COCO keypoints in every frame
  2. A window opens on frame 0 — you CLICK the injured player to lock target
  3. Every subsequent frame: re-detect, pick best match using a blended score:
       a) Appearance similarity  (HSV histogram, primary)
       b) IoU with previous bbox (tiebreaker)
       c) Centroid distance      (spatial fallback)
  4. ref_appearance updates via EMA so gradual lighting changes don't cause drift
  5. Press R at any point to manually re-lock onto a new player
  6. Optical flow (Farneback) computed on the locked ROI
  7. Annotated frames saved to data/<injury>/<angle>/frames_annotated/
  8. All data written to motion_data1.csv in project root

Output CSV columns:
  injury_id, angle_id, frame_index,
  bbox_x1, bbox_y1, bbox_x2, bbox_y2, track_conf,
  kp_0_x ... kp_16_x, kp_0_y ... kp_16_y, kp_0_conf ... kp_16_conf,
  velocity, acceleration
"""

from pathlib import Path
import csv
import os
import cv2
import numpy as np

# -- paths ---------------------------------------------------------------------
ROOT       = Path(__file__).parent.parent
DATA       = ROOT / "data"
OUTPUT_CSV = ROOT / "motion_data1.csv"

# -- tuning constants ----------------------------------------------------------
YOLO_CONF   = 0.35   # minimum detection confidence
MAX_DRIFT   = 300    # max pixel distance before target is considered lost

# tracker blend weights (must sum to 1.0)
W_APPEARANCE = 0.55
W_IOU        = 0.25
W_SPATIAL    = 0.20

# EMA smoothing for appearance model
# 0.85 = keep 85% of stored appearance, absorb 15% of new each frame
EMA_ALPHA = 0.85

# -- keypoint names (COCO 17) --------------------------------------------------
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
N_KP = len(KP_NAMES)

EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# -- model ---------------------------------------------------------------------
_yolo = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8m-pose.pt")
        print("[INFO] YOLOv8-pose loaded.")
    return _yolo


# ==============================================================================
# DETECTION  — returns keypoints directly from the pose model
# ==============================================================================

def detect_persons(frame_bgr):
    """
    Returns list of dicts, one per detected person:
      {
        'bbox'     : [x1, y1, x2, y2],
        'conf'     : float,
        'kps'      : np.ndarray shape (17,2) or None,
        'kp_scores': np.ndarray shape (17,)  or None,
      }
    """
    yolo    = get_yolo()
    results = yolo(frame_bgr, classes=[0], verbose=False, conf=YOLO_CONF)[0]
    persons = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        kps, kp_scores = None, None
        if results.keypoints is not None and i < len(results.keypoints.data):
            kp_data   = results.keypoints.data[i].cpu().numpy()  # (17, 3)
            kps       = kp_data[:, :2]   # (17, 2)
            kp_scores = kp_data[:, 2]    # (17,)
        persons.append({
            'bbox'     : [x1, y1, x2, y2],
            'conf'     : conf,
            'kps'      : kps,
            'kp_scores': kp_scores,
        })
    return persons


# ==============================================================================
# APPEARANCE HELPERS
# ==============================================================================

def get_appearance(frame_bgr, bbox):
    """16x16 bin HSV histogram over the player crop. Returns flat float32 (256,)."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame_bgr[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)

def appearance_similarity(h1, h2):
    """Histogram correlation in [-1,1]. 1.0 = identical."""
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(
        h1.reshape(16, 16), h2.reshape(16, 16), cv2.HISTCMP_CORREL
    ))

def update_appearance_ema(stored, new, alpha=EMA_ALPHA):
    if new is None:
        return stored
    if stored is None:
        return new
    return alpha * stored + (1.0 - alpha) * new


# ==============================================================================
# IoU HELPER
# ==============================================================================

def box_iou(bbox_a, bbox_b):
    ax1,ay1,ax2,ay2 = bbox_a
    bx1,by1,bx2,by2 = bbox_b
    ix1 = max(ax1,bx1); iy1 = max(ay1,by1)
    ix2 = min(ax2,bx2); iy2 = min(ay2,by2)
    inter = max(0.0, ix2-ix1) * max(0.0, iy2-iy1)
    if inter == 0.0:
        return 0.0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter)


# ==============================================================================
# GEOMETRY HELPER
# ==============================================================================

def box_centre(bbox):
    x1,y1,x2,y2 = bbox
    return np.array([(x1+x2)/2, (y1+y2)/2])


# ==============================================================================
# TRACKER
# ==============================================================================

def pick_best_match(persons, prev_bbox, prev_centre, ref_appearance, frame_bgr):
    """
    Scores every detection and returns the best match + its crop histogram.
    Returns (person_dict, crop_hist) or (None, None) if nothing passes the drift gate.
    """
    best_score = -np.inf
    best_person = None
    best_crop   = None

    for p in persons:
        centre = box_centre(p['bbox'])
        dist   = np.linalg.norm(centre - prev_centre)
        if dist > MAX_DRIFT:
            continue

        crop_hist        = get_appearance(frame_bgr, p['bbox'])
        raw_sim          = appearance_similarity(ref_appearance, crop_hist)
        appearance_score = (raw_sim + 1.0) / 2.0       # [-1,1] -> [0,1]
        iou_score        = box_iou(prev_bbox, p['bbox'])
        spatial_score    = 1.0 - (dist / MAX_DRIFT)

        score = (W_APPEARANCE * appearance_score
               + W_IOU        * iou_score
               + W_SPATIAL    * spatial_score)

        if score > best_score:
            best_score  = score
            best_person = p
            best_crop   = crop_hist

    return best_person, best_crop


# ==============================================================================
# CLICK-TO-SELECT
# ==============================================================================

def select_player_on_frame(frame, persons, clip_label, reason="CLICK injured player"):
    """
    Draws all detections, waits for a mouse click, returns chosen person dict.
    Press Q to skip.
    """
    if not persons:
        print(f"  [WARN] No persons detected in frame for {clip_label}.")
        return None

    display = frame.copy()
    for i, p in enumerate(persons):
        x1,y1,x2,y2 = map(int, p['bbox'])
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.putText(display, f"#{i} {p['conf']:.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)

        # draw skeleton so you can identify the right player easily
        if p['kps'] is not None:
            for a, b in EDGES:
                if p['kp_scores'][a] > 0.3 and p['kp_scores'][b] > 0.3:
                    pa = tuple(map(int, p['kps'][a]))
                    pb = tuple(map(int, p['kps'][b]))
                    cv2.line(display, pa, pb, (255,180,0), 2)
            for j,(px,py) in enumerate(p['kps']):
                if p['kp_scores'][j] > 0.3:
                    cv2.circle(display, (int(px),int(py)), 4, (0,200,255), -1)

    h,w   = display.shape[:2]
    scale = min(1.0, 1200/w, 800/h)
    disp  = cv2.resize(display, (int(w*scale), int(h*scale)))

    chosen = {"person": None}

    def on_click(event, cx, cy, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy   = cx/scale, cy/scale
        click_pt = np.array([ox, oy])
        dists    = [np.linalg.norm(box_centre(p['bbox']) - click_pt) for p in persons]
        chosen["person"] = persons[int(np.argmin(dists))]
        cv2.destroyAllWindows()

    win = f"{reason}  [{clip_label}]  |  Q to skip"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    print(f"\n  >>> {reason} ({len(persons)} detected)  |  Q = skip clip")

    while True:
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or chosen["person"] is not None:
            break

    cv2.destroyAllWindows()

    if chosen["person"] is None:
        print("  [SKIP] No player selected.")
    else:
        b = chosen["person"]['bbox']
        print(f"  Locked: bbox=({int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])})  "
              f"conf={chosen['person']['conf']:.2f}")
    return chosen["person"]


# ==============================================================================
# OPTICAL FLOW
# ==============================================================================

def optical_flow_on_roi(prev_gray, curr_gray, bbox):
    x1,y1,x2,y2 = map(int, bbox)
    H,W = curr_gray.shape
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W,x2), min(H,y2)
    rp = prev_gray[y1:y2, x1:x2]
    rc = curr_gray[y1:y2, x1:x2]
    if rp.size == 0 or rc.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(rp, rc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))


# ==============================================================================
# DRAW
# ==============================================================================

def draw_annotations(frame, bbox, kps, kp_scores, velocity, frame_idx):
    out = frame.copy()
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(out, f"F{frame_idx}  v={velocity:.2f}",
                (x1, max(y1-8,12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 1)
    if kps is not None:
        CONF_T = 0.3
        for i,(px,py) in enumerate(kps):
            if kp_scores[i] >= CONF_T:
                cv2.circle(out, (int(px),int(py)), 4, (0,200,255), -1)
        for a,b in EDGES:
            if kp_scores[a] >= CONF_T and kp_scores[b] >= CONF_T:
                cv2.line(out,
                         tuple(map(int,kps[a])),
                         tuple(map(int,kps[b])),
                         (255,180,0), 2)
    return out


# ==============================================================================
# CSV SCHEMA
# ==============================================================================

def build_fieldnames():
    base    = ["injury_id","angle_id","frame_index",
               "bbox_x1","bbox_y1","bbox_x2","bbox_y2","track_conf"]
    kp_cols = ([f"kp_{i}_x"    for i in range(N_KP)] +
               [f"kp_{i}_y"    for i in range(N_KP)] +
               [f"kp_{i}_conf" for i in range(N_KP)])
    return base + kp_cols + ["velocity","acceleration"]

def kp_to_dict(kps, kp_scores):
    d = {}
    for i in range(N_KP):
        if kps is not None and kp_scores is not None and kp_scores[i] > 0:
            d[f"kp_{i}_x"]    = round(float(kps[i][0]),    2)
            d[f"kp_{i}_y"]    = round(float(kps[i][1]),    2)
            d[f"kp_{i}_conf"] = round(float(kp_scores[i]), 3)
        else:
            d[f"kp_{i}_x"] = d[f"kp_{i}_y"] = d[f"kp_{i}_conf"] = -1
    return d


# ==============================================================================
# PER-CLIP PROCESSING
# ==============================================================================

def process_clip(injury_id, angle_id, frame_paths):
    annotated_dir = DATA / injury_id / angle_id / "frames_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    clip_label = f"{injury_id}/{angle_id}"

    # ---- frame 0: initial lock -----------------------------------------------
    frame0 = cv2.imread(str(frame_paths[0]))
    if frame0 is None:
        print(f"  [ERROR] Cannot read frame 0 for {clip_label}")
        return []

    persons0 = detect_persons(frame0)
    locked   = select_player_on_frame(frame0, persons0, clip_label)
    if locked is None:
        return []

    # seed tracking state
    prev_bbox      = locked['bbox']
    prev_centre    = box_centre(locked['bbox'])
    ref_appearance = get_appearance(frame0, locked['bbox'])
    init_w = locked['bbox'][2] - locked['bbox'][0]
    init_h = locked['bbox'][3] - locked['bbox'][1]

    prev_gray = None
    prev_vel  = None
    rows      = []

    # ---- frame loop ----------------------------------------------------------
    for frame_idx, fp in enumerate(frame_paths):
        frame = cv2.imread(str(fp))
        if frame is None:
            print(f"  [SKIP] cannot read {fp}")
            continue

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons   = detect_persons(frame)

        # -- R-key: manual re-lock ---------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print(f"\n  [R] Manual re-lock at frame {frame_idx}")
            relock = select_player_on_frame(
                frame, persons, clip_label,
                reason=f"RE-LOCK at frame {frame_idx}"
            )
            if relock is not None:
                locked         = relock
                prev_bbox      = relock['bbox']
                prev_centre    = box_centre(relock['bbox'])
                ref_appearance = get_appearance(frame, relock['bbox'])

        # -- frame 0: already locked -------------------------------------------
        if frame_idx == 0:
            target      = locked
            target_crop = ref_appearance
        else:
            target, target_crop = pick_best_match(
                persons, prev_bbox, prev_centre, ref_appearance, frame
            )
            if target is None:
                # interpolate: hold bbox at last known centre
                print(f"  [WARN] lost target at frame {frame_idx}, interpolating")
                cx,cy = prev_centre
                target = {
                    'bbox'     : [cx-init_w/2, cy-init_h/2,
                                  cx+init_w/2, cy+init_h/2],
                    'conf'     : 0.0,
                    'kps'      : None,
                    'kp_scores': None,
                }
                target_crop = None   # don't update EMA on interpolated frame

        bbox = target['bbox']
        prev_bbox      = bbox
        prev_centre    = box_centre(bbox)
        ref_appearance = update_appearance_ema(ref_appearance, target_crop)

        kps       = target['kps']
        kp_scores = target['kp_scores']

        # -- optical flow ------------------------------------------------------
        if prev_gray is not None:
            velocity     = optical_flow_on_roi(prev_gray, curr_gray, bbox)
            acceleration = (velocity - prev_vel) if prev_vel is not None else None
        else:
            velocity, acceleration = 0.0, None

        prev_vel  = velocity
        prev_gray = curr_gray

        # -- annotate & save ---------------------------------------------------
        ann = draw_annotations(frame, bbox, kps, kp_scores, velocity, frame_idx)
        cv2.imwrite(str(annotated_dir / Path(fp).name), ann)

        # -- CSV row -----------------------------------------------------------
        x1,y1,x2,y2 = bbox
        row = {
            "injury_id":    injury_id,
            "angle_id":     angle_id,
            "frame_index":  frame_idx,
            "bbox_x1":      round(x1, 1),
            "bbox_y1":      round(y1, 1),
            "bbox_x2":      round(x2, 1),
            "bbox_y2":      round(y2, 1),
            "track_conf":   round(target['conf'], 3),
            "velocity":     round(velocity, 4),
            "acceleration": round(acceleration, 4) if acceleration is not None else "",
        }
        row.update(kp_to_dict(kps, kp_scores))
        rows.append(row)

        if frame_idx % 20 == 0:
            print(f"    frame {frame_idx:04d} | bbox=({int(x1)},{int(y1)}) "
                  f"conf={target['conf']:.2f}  vel={velocity:.3f}")

    cv2.destroyAllWindows()
    return rows


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    all_rows   = []
    fieldnames = build_fieldnames()

    for injury in sorted(os.listdir(DATA)):
        injury_path = DATA / injury
        if not injury_path.is_dir():
            continue

        for angle in sorted(os.listdir(injury_path)):
            frames_dir = injury_path / angle / "frames"
            if not frames_dir.exists():
                continue

            frame_paths = sorted(frames_dir.glob("*.jpg"))
            if not frame_paths:
                continue

            print(f"\nProcessing  {injury} / {angle}  ({len(frame_paths)} frames)")
            rows = process_clip(injury, angle, frame_paths)
            all_rows.extend(rows)
            print(f"  -> {len(rows)} rows written")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. {len(all_rows)} total rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()