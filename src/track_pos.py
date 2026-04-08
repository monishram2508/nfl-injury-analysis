"""
src/track.py
------------
Per-frame bounding box + pose skeleton for the one injured player per clip.

Pipeline:
  1. YOLOv8 detects all persons in frame 0
  2. A window opens showing frame 0 — you CLICK the injured player
  3. The detection closest to your click is locked as the target
  4. Every subsequent frame: YOLO re-detects, then we pick the best match using
     a blended score of:
       a) Appearance similarity  (HSV color histogram, primary)
       b) IoU with previous bbox (tiebreaker when appearances are close)
       c) Centroid distance      (spatial fallback)
  5. ref_appearance is updated each frame using an exponential moving average
     so gradual lighting changes don't cause drift
  6. Press R at any point to manually re-lock onto a new player (escape hatch)
  7. ViTPose estimates the 17-keypoint COCO skeleton inside the locked box
  8. Optical flow (Farneback) computed on the locked ROI
  9. Annotated frames saved to data/<injury>/<angle>/frames_annotated/
  10. All data written to motion_data1.csv in project root

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
YOLO_CONF   = 0.35   # minimum YOLO detection confidence to consider
MAX_DRIFT   = 300    # max pixels a centroid can move between frames before
                     # we consider the target lost entirely

# tracker blend weights (must sum to 1.0)
W_APPEARANCE = 0.55  # HSV histogram correlation  — most reliable across frames
W_IOU        = 0.25  # bbox overlap with prev frame — strong tiebreaker
W_SPATIAL    = 0.20  # centroid proximity          — fallback

# how strongly to trust the stored appearance vs this frame's crop
# 0.85 means "keep 85% of old appearance, absorb 15% of new" each frame
# raise toward 1.0 if lighting is stable; lower toward 0.6 if it changes fast
EMA_ALPHA = 0.85

# IoU threshold below which we ignore a detection as a spatial fallback only
IOU_CONFIRM_THRESH = 0.30

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

# -- lazy model loading --------------------------------------------------------
_yolo    = None
_vitpose = None

def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO("yolov8m.pt")
    return _yolo

def get_vitpose():
    global _vitpose
    if _vitpose is not None:
        return _vitpose
    try:
        from mmpose.apis import init_model, inference_topdown
        from mmpose.utils import register_all_modules
        register_all_modules()

        cfg  = "td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
        ckpt = (
            "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/"
            "topdown_heatmap/coco/"
            "td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
        )
        model = init_model(cfg, ckpt, device="cpu")

        def _infer(frame_bgr, bbox_xyxy):
            bboxes  = np.array([[*bbox_xyxy, 1.0]])
            results = inference_topdown(model, frame_bgr, bboxes)
            if results:
                kps    = results[0].pred_instances.keypoints[0]
                scores = results[0].pred_instances.keypoint_scores[0]
                return kps, scores
            return None, None

        _vitpose = _infer
        print("[INFO] ViTPose loaded successfully.")

    except Exception as e:
        print(f"[WARN] ViTPose not available ({e}). Keypoints will be -1.")
        _vitpose = lambda frame, bbox: (None, None)

    return _vitpose


# ==============================================================================
# APPEARANCE HELPERS
# ==============================================================================

def get_appearance(frame_bgr, bbox_xyxy):
    """
    Crop the player from the frame and compute a 16x16 bin HSV histogram.

    Why HSV and not BGR?
      Hue encodes the jersey colour independently of brightness — so the same
      jersey under different lighting conditions (shadow, stadium lights, sun)
      gives a similar histogram. Saturation catches white/grey/black jerseys
      that have low hue signal. We ignore Value (brightness) entirely by only
      using channels 0 and 1.

    Why 16x16 bins?
      256 values → 16 bins per channel. Fine enough to distinguish team colours
      (red vs blue vs green), coarse enough to be robust to small colour shifts.
      The resulting 256-element vector is cheap to compare.

    Returns a flat float32 array of length 256, or None if the crop is empty.
    """
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def appearance_similarity(h1, h2):
    """
    Correlation between two histograms.  Returns a value in [-1, 1].
      1.0  = identical distributions  (same player)
      0.0  = no correlation
     -1.0  = opposite distributions   (very different players)

    We use HISTCMP_CORREL because it's symmetric and scale-invariant —
    a brighter crop of the same jersey gives the same result as a darker one.
    """
    if h1 is None or h2 is None:
        return 0.0
    return float(cv2.compareHist(
        h1.reshape(16, 16),
        h2.reshape(16, 16),
        cv2.HISTCMP_CORREL,
    ))


def update_appearance_ema(stored, new, alpha=EMA_ALPHA):
    """
    Exponential moving average of the stored histogram.

    Each frame we blend:
        stored = alpha * stored + (1 - alpha) * new

    With alpha=0.85 this means the stored appearance changes slowly —
    15% of each new observation bleeds in per frame.  This handles:
      - Gradual lighting changes (stadium lights vs daylight)
      - Slight colour shifts as the camera angle changes
      - Partial occlusion (new crop may be partially blocked but still useful)

    We do NOT update if new is None (crop was empty / out of frame).
    """
    if new is None:
        return stored
    if stored is None:
        return new
    return alpha * stored + (1.0 - alpha) * new


# ==============================================================================
# IoU HELPER
# ==============================================================================

def box_iou(bbox_a, bbox_b):
    """
    Intersection-over-Union between two [x1,y1,x2,y2] boxes.

    Why use this as a tiebreaker?
      Between consecutive frames (~33ms apart at 30fps) the same player's bbox
      will almost always overlap heavily with the previous frame's bbox, because
      a player can't teleport.  Two different players at similar distances to
      the centroid will have low IoU with the stored box.  So when appearance
      similarity scores are close (e.g. both players wear similar-coloured kits),
      IoU reliably breaks the tie.
    """
    ax1, ay1, ax2, ay2 = bbox_a[:4]
    bx1, by1, bx2, by2 = bbox_b[:4]

    ix1   = max(ax1, bx1);  iy1 = max(ay1, by1)
    ix2   = min(ax2, bx2);  iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    if inter == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ==============================================================================
# CORE TRACKER
# ==============================================================================

def pick_best_match(detections, prev_bbox, prev_centre, ref_appearance,
                    frame_bgr, max_drift=MAX_DRIFT):
    """
    Given a list of YOLO detections in the current frame, find the one that
    most likely corresponds to the player we locked in the previous frame.

    Scoring (weighted blend — see constants at top of file):

      1. Appearance score  [0..1]
         HSV histogram correlation mapped from [-1,1] → [0,1].
         This is the primary signal.  Distinct jersey colours score near 1.0
         for the correct player and near 0.0–0.3 for others.

      2. IoU score  [0..1]
         Raw IoU between the candidate bbox and the previous bbox.
         Strong when the player hasn't moved much (standing, slow jog).
         Critical tiebreaker when two players wear similar colours and are
         next to each other.

      3. Spatial score  [0..1]
         1 - (distance / max_drift).  Closest centroid scores 1.0.
         This is a soft prior — nearby detections are preferred but not
         exclusively chosen.

    A detection is only considered if its centroid is within max_drift pixels
    of the previous centre.  Anything beyond that is almost certainly a
    different player.

    Returns:
        best_det  — the winning detection [x1,y1,x2,y2,conf], or None if no
                    candidate survived the drift gate
        crop_hist — the HSV histogram of the winning crop (for EMA update),
                    or None if best_det is None
    """
    if not detections:
        return None, None

    best_score    = -np.inf
    best_det      = None
    best_crop     = None

    for det in detections:
        centre = box_centre(*det[:4])
        dist   = np.linalg.norm(centre - prev_centre)

        # gate: ignore detections that are too far away
        if dist > max_drift:
            continue

        # --- appearance ---
        crop_hist        = get_appearance(frame_bgr, det[:4])
        raw_sim          = appearance_similarity(ref_appearance, crop_hist)
        appearance_score = (raw_sim + 1.0) / 2.0          # [-1,1] → [0,1]

        # --- IoU ---
        iou_score = box_iou(prev_bbox, det[:4])            # already in [0,1]

        # --- spatial ---
        spatial_score = 1.0 - (dist / max_drift)           # [0,1]

        # --- blend ---
        score = (W_APPEARANCE * appearance_score
               + W_IOU        * iou_score
               + W_SPATIAL    * spatial_score)

        if score > best_score:
            best_score = score
            best_det   = det
            best_crop  = crop_hist

    return best_det, best_crop


# ==============================================================================
# UTILITY HELPERS  (unchanged from original)
# ==============================================================================

def box_centre(x1, y1, x2, y2):
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def detect_persons(frame_bgr):
    yolo    = get_yolo()
    results = yolo(frame_bgr, classes=[0], verbose=False, conf=YOLO_CONF)[0]
    boxes   = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        conf = float(box.conf[0])
        boxes.append([x1, y1, x2, y2, conf])
    return boxes


def optical_flow_on_roi(prev_gray, curr_gray, bbox_xyxy):
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    H, W = curr_gray.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    rp = prev_gray[y1:y2, x1:x2]
    rc = curr_gray[y1:y2, x1:x2]
    if rp.size == 0 or rc.size == 0:
        return 0.0
    flow = cv2.calcOpticalFlowFarneback(rp, rc, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)))


def draw_annotations(frame, bbox_xyxy, kps, kp_scores, velocity, frame_idx):
    out = frame.copy()
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(out, f"F{frame_idx}  v={velocity:.2f}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    if kps is not None:
        CONF_T = 0.3
        for i, (px, py) in enumerate(kps):
            if kp_scores[i] >= CONF_T:
                cv2.circle(out, (int(px), int(py)), 4, (0, 200, 255), -1)
        for a, b in EDGES:
            if kp_scores[a] >= CONF_T and kp_scores[b] >= CONF_T:
                cv2.line(out,
                         tuple(map(int, kps[a])),
                         tuple(map(int, kps[b])),
                         (255, 180, 0), 2)
    return out


# ==============================================================================
# CLICK-TO-SELECT  (used on frame 0 AND on R-key re-lock)
# ==============================================================================

def select_player_on_frame(frame, detections, clip_label, reason="CLICK injured player"):
    """
    Opens a window, draws all detections, waits for a mouse click.
    Returns the detection nearest to the click, or None if Q is pressed.

    This function is reused for:
      - Initial lock on frame 0
      - Manual re-lock triggered by pressing R mid-clip
    """
    if not detections:
        print(f"  [WARN] No persons detected in frame for {clip_label}. Cannot select.")
        return None

    display = frame.copy()
    for i, (x1, y1, x2, y2, conf) in enumerate(detections):
        cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
        cv2.putText(display, f"#{i} {conf:.2f}",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    h, w  = display.shape[:2]
    scale = min(1.0, 1200 / w, 800 / h)
    disp  = cv2.resize(display, (int(w * scale), int(h * scale)))

    chosen = {"idx": None}

    def on_click(event, cx, cy, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        ox, oy   = cx / scale, cy / scale
        click_pt = np.array([ox, oy])
        dists    = [np.linalg.norm(box_centre(*d[:4]) - click_pt) for d in detections]
        chosen["idx"] = int(np.argmin(dists))
        cv2.destroyAllWindows()

    win = f"{reason}  [{clip_label}]  |  Q to skip"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)

    print(f"\n  >>> {reason} ({len(detections)} detected)  |  Q = skip")

    while True:
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or chosen["idx"] is not None:
            break

    cv2.destroyAllWindows()

    if chosen["idx"] is None:
        print("  [SKIP] No player selected.")
        return None

    sel = detections[chosen["idx"]]
    x1, y1, x2, y2, conf = sel
    print(f"  Locked: #{chosen['idx']}  bbox=({int(x1)},{int(y1)},{int(x2)},{int(y2)})  conf={conf:.2f}")
    return sel


# ==============================================================================
# CSV SCHEMA
# ==============================================================================

def build_fieldnames():
    base    = ["injury_id", "angle_id", "frame_index",
               "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "track_conf"]
    kp_cols = ([f"kp_{i}_x"    for i in range(N_KP)] +
               [f"kp_{i}_y"    for i in range(N_KP)] +
               [f"kp_{i}_conf" for i in range(N_KP)])
    return base + kp_cols + ["velocity", "acceleration"]


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
    """
    Main loop for one clip.

    Tracking state maintained across frames:
      prev_bbox       — [x1,y1,x2,y2,conf] from the last confirmed frame
      prev_centre     — centroid of prev_bbox
      ref_appearance  — EMA-smoothed HSV histogram of the target player
      prev_gray       — grayscale of previous frame (for optical flow)
      prev_vel        — velocity scalar from previous frame (for acceleration)

    Per-frame logic:
      1. Detect all persons with YOLO
      2. Check if user pressed R → if so, open click-to-select and re-lock
      3. Run pick_best_match to find the best candidate
      4. If no candidate found, interpolate by holding the previous bbox
      5. Update ref_appearance via EMA
      6. Run ViTPose on the locked bbox
      7. Compute optical flow velocity and acceleration
      8. Save annotated frame and append CSV row
    """
    annotated_dir = DATA / injury_id / angle_id / "frames_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    clip_label = f"{injury_id}/{angle_id}"

    # ---- frame 0: initial lock -----------------------------------------------
    frame0 = cv2.imread(str(frame_paths[0]))
    if frame0 is None:
        print(f"  [ERROR] Cannot read frame 0 for {clip_label}")
        return []

    detections0 = detect_persons(frame0)
    locked      = select_player_on_frame(frame0, detections0, clip_label)
    if locked is None:
        return []

    # seed tracking state from the locked detection
    prev_bbox       = locked[:4]
    prev_centre     = box_centre(*locked[:4])
    ref_appearance  = get_appearance(frame0, locked[:4])

    # remember initial box size for interpolation fallback
    init_w = locked[2] - locked[0]
    init_h = locked[3] - locked[1]

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

        # -- detect all persons in this frame ----------------------------------
        dets = detect_persons(frame)

        # -- R-key check (show preview briefly to catch keypress) --------------
        #
        # We briefly show the PREVIOUS annotated frame (if it exists) in a
        # small preview window.  waitKey(1) is non-blocking so it doesn't
        # slow the loop.  If the user presses R:
        #   - We immediately open the click-to-select window on THIS frame
        #   - The selected detection becomes the new lock
        #   - ref_appearance is reset to this frame's crop (hard reset, not EMA)
        #     because a manual re-lock means the user is telling us the
        #     appearance model has drifted — trust the new crop fully.
        #
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print(f"\n  [R] Manual re-lock requested at frame {frame_idx}")
            relock = select_player_on_frame(
                frame, dets, clip_label,
                reason=f"RE-LOCK at frame {frame_idx}"
            )
            if relock is not None:
                locked         = relock
                prev_bbox      = relock[:4]
                prev_centre    = box_centre(*relock[:4])
                ref_appearance = get_appearance(frame, relock[:4])  # hard reset
                print(f"  [R] Re-locked successfully.")

        # -- frame 0: already locked above, just use it ------------------------
        if frame_idx == 0:
            target      = locked
            target_crop = ref_appearance
        else:
            # ---- primary tracker: appearance + IoU + spatial -----------------
            target, target_crop = pick_best_match(
                dets,
                prev_bbox,
                prev_centre,
                ref_appearance,
                frame,
            )

            if target is None:
                # No candidate survived the drift gate — interpolate
                # Hold the bbox at the last known centre, keep previous conf=0
                print(f"  [WARN] lost target at frame {frame_idx}, interpolating")
                cx, cy = prev_centre
                target = [cx - init_w / 2, cy - init_h / 2,
                          cx + init_w / 2, cy + init_h / 2, 0.0]
                target_crop = None   # don't update EMA on an interpolated frame

        x1, y1, x2, y2, conf = target
        bbox_xyxy   = [x1, y1, x2, y2]
        prev_bbox   = bbox_xyxy
        prev_centre = box_centre(x1, y1, x2, y2)

        # -- EMA update of appearance ------------------------------------------
        # Only update on real detections, not interpolated frames.
        # This prevents the appearance model from slowly drifting toward
        # whatever background/player the interpolated crop catches.
        ref_appearance = update_appearance_ema(ref_appearance, target_crop)

        # -- pose --------------------------------------------------------------
        kps, kp_scores = get_vitpose()(frame, bbox_xyxy)

        # -- optical flow ------------------------------------------------------
        if prev_gray is not None:
            velocity     = optical_flow_on_roi(prev_gray, curr_gray, bbox_xyxy)
            acceleration = (velocity - prev_vel) if prev_vel is not None else None
        else:
            velocity, acceleration = 0.0, None

        prev_vel  = velocity
        prev_gray = curr_gray

        # -- annotate & save ---------------------------------------------------
        ann = draw_annotations(frame, bbox_xyxy, kps, kp_scores, velocity, frame_idx)
        cv2.imwrite(str(annotated_dir / Path(fp).name), ann)

        # -- CSV row -----------------------------------------------------------
        row = {
            "injury_id":    injury_id,
            "angle_id":     angle_id,
            "frame_index":  frame_idx,
            "bbox_x1":      round(x1,   1),
            "bbox_y1":      round(y1,   1),
            "bbox_x2":      round(x2,   1),
            "bbox_y2":      round(y2,   1),
            "track_conf":   round(conf, 3),
            "velocity":     round(velocity, 4),
            "acceleration": round(acceleration, 4) if acceleration is not None else "",
        }
        row.update(kp_to_dict(kps, kp_scores))
        rows.append(row)

        if frame_idx % 20 == 0:
            print(f"    frame {frame_idx:04d} | bbox=({int(x1)},{int(y1)}) "
                  f"conf={conf:.2f}  vel={velocity:.3f}")

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