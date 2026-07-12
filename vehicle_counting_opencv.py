"""
Vehicle Counting per Lane using OpenCV (NO YOLO / NO deep learning)
--------------------------------------------------------------------
Method: Background Subtraction + Contour Detection

  1. Learn a background model of the empty road (MOG2).
  2. For every new frame, subtract the background -> moving/parked
     vehicles show up as white "blobs" (foreground mask).
  3. Clean the mask with morphological operations (remove noise/shadows).
  4. Find contours; each contour big enough to be a vehicle is counted.
  5. Count separately inside a Region Of Interest (ROI) box drawn over
     EACH lane, so a single overhead ESP32-CAM watching the whole
     intersection can report per-lane counts (you only need ONE camera
     to start; add more later if you want per-lane cameras instead).

This is deliberately simple (CPU-only, no GPU, no model download) so it
runs fine on a laptop or a Raspberry Pi, and is easy to explain in a
viva/presentation. Accuracy is lower than YOLO, but is sufficient for
"how many vehicles roughly in each lane" -- which is all the scheduler
in traffic_light_controller.py needs.

Usage:
    python vehicle_counting_opencv.py --source 0                     # laptop webcam
    python vehicle_counting_opencv.py --source video.mp4              # recorded video
    python vehicle_counting_opencv.py --source http://<esp32-ip>:81/stream   # ESP32-CAM
"""

import argparse
import time
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency fallback
    cv2 = None
    np = None
    _cv2_IMPORT_ERROR = exc

# ---- Tunable parameters -------------------------------------------------
MIN_CONTOUR_AREA = 900      # pixels; ignore blobs smaller than this (noise)
LEARNING_RATE = -1          # -1 = let MOG2 pick automatically
HISTORY = 300               # frames used to build the background model
# No artificial cap: allow counts to reflect reality
MAX_LANE_COUNT = None

# Example lane ROIs (x1, y1, x2, y2) as FRACTIONS of frame width/height,
# so it still works if you change camera resolution.
# ADJUST these boxes to match where each lane appears in YOUR camera view
# (draw them by eye once, looking at a saved snapshot from your ESP32-CAM).
LANE_ROIS = {
    "North": (0.00, 0.00, 0.50, 0.50),
    "South": (0.50, 0.00, 1.00, 0.50),
    "East":  (0.00, 0.50, 0.50, 1.00),
    "West":  (0.50, 0.50, 1.00, 1.00),
}
CANONICAL_LANES = ["North", "South", "East", "West"]


def normalize_lane_counts(counts, lane_names=None):
    """Convert lane-count input into a canonical {North, South, East, West} dict."""
    lane_names = lane_names or CANONICAL_LANES
    normalized = {lane: 0 for lane in lane_names}
    if not isinstance(counts, dict):
        return normalized

    for lane in lane_names:
        if lane in counts:
            val = int(counts[lane])
            normalized[lane] = max(val, 0)
        else:
            for alt_key, alt_value in counts.items():
                if isinstance(alt_key, str) and alt_key.lower() == lane.lower():
                    val = int(alt_value)
                    normalized[lane] = max(val, 0)
                    break
    return normalized


def count_vehicles_from_simulation(vehicle_state):
    """Count vehicles from the simulation state and return canonical lane counts."""
    if hasattr(vehicle_state, "get_vehicle_counts"):
        vehicle_state = vehicle_state.get_vehicle_counts()

    if isinstance(vehicle_state, dict):
        return normalize_lane_counts(vehicle_state)

    return {lane: 0 for lane in CANONICAL_LANES}


def make_background_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=HISTORY, varThreshold=40, detectShadows=True
    )


def count_vehicles_in_mask(mask, min_area=MIN_CONTOUR_AREA):
    # remove shadow pixels (value 127 in MOG2 output) and clean noise
    _, clean = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8), iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_DILATE,
                              np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    count = sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    return count, contours


def roi_pixels(frame_shape, roi_fraction):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi_fraction
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def build_source_candidates(source):
    if isinstance(source, (int, float)):
        return [source]

    if not isinstance(source, str):
        return [source]

    raw_source = source.strip()
    if not raw_source:
        return [source]

    candidates = [raw_source]

    parsed = urlparse(raw_source)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or ""
        path = parsed.path or "/"
        query = f"?{parsed.query}" if parsed.query else ""
        fragment = f"#{parsed.fragment}" if parsed.fragment else ""

        if host:
            if parsed.port is None:
                candidates.append(f"{parsed.scheme}://{host}:81{path}{query}{fragment}")
            if path in {"", "/"}:
                candidates.append(f"{parsed.scheme}://{host}/stream{query}{fragment}")
                if parsed.port is None:
                    candidates.append(f"{parsed.scheme}://{host}:81/stream{query}{fragment}")
            elif path != "/stream":
                candidates.append(f"{parsed.scheme}://{host}{path}/stream{query}{fragment}")
                if parsed.port is None:
                    candidates.append(f"{parsed.scheme}://{host}:81{path}/stream{query}{fragment}")

    return list(dict.fromkeys(candidates))


def open_video_source(source):
    candidates = build_source_candidates(source)
    last_error = None

    for candidate in candidates:
        print(f"Trying video source: {candidate}")
        cap = cv2.VideoCapture(candidate)
        if cap.isOpened():
            print(f"Opened video source successfully: {candidate}")
            return cap

        cap.release()
        last_error = candidate

    raise RuntimeError(
        f"Could not open video source: {source}. Tried: {', '.join(str(c) for c in candidates)}"
    )


def main(source, show_window=True, post_url=None):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for webcam/video counting") from _cv2_IMPORT_ERROR

    cap = open_video_source(source)

    bg_subtractor = make_background_subtractor()

    print("Warming up background model on empty/near-empty road...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        bg_subtractor.apply(frame, learningRate=LEARNING_RATE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended / camera disconnected.")
            break

        fg_mask = bg_subtractor.apply(frame, learningRate=LEARNING_RATE)

        lane_counts = {}
        for lane_name, roi in LANE_ROIS.items():
            x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
            lane_mask = fg_mask[y1:y2, x1:x2]
            count, _ = count_vehicles_in_mask(lane_mask)
            lane_counts[lane_name] = int(max(count, 0))

            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"{lane_name}: {lane_counts[lane_name]}", (x1 + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ---- This dict is exactly what the greedy scheduler needs ----
        print(f"[COUNT] Vehicle counts detected: {lane_counts}")
        if post_url:
            try:
                import requests
                requests.post(post_url, json=lane_counts, timeout=1)
            except Exception as e:
                print("Could not send counts to controller:", e)

        if show_window:
            cv2.imshow("Lane vehicle counting (OpenCV, no YOLO)", frame)
            cv2.imshow("Foreground mask", fg_mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="0 for webcam, path to video file, or ESP32-CAM stream URL")
    parser.add_argument("--no-window", action="store_true",
                         help="run headless (e.g. on a Raspberry Pi without a monitor)")
    parser.add_argument("--post-url", default=None,
                         help="optional: URL of traffic_light_controller.py Flask server "
                              "e.g. http://127.0.0.1:5000/update_counts")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    main(src, show_window=not args.no_window, post_url=args.post_url)
