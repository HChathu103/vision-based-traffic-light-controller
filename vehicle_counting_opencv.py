"""
Vehicle Counting per Lane using OpenCV (NO YOLO / NO deep learning)
--------------------------------------------------------------------
Method: STATIC reference-frame differencing + contour detection
(this replaces MOG2 adaptive background subtraction - see note below)

  1. Capture a reference frame of the EMPTY road once at startup
     (median of the first ~30 frames, to reduce noise).
  2. For every new frame, compute absolute difference against that
     FIXED reference -> anything different from the truly-empty road
     (moving OR stationary vehicles) shows up as a white "blob".
  3. Clean the mask with morphological operations (remove noise/shadows).
  4. Find contours inside each lane's ROI (loaded from lane_rois.json,
     produced by calibrate_rois.py) and count contours big enough to be
     a vehicle.
  5. Counting happens per lane, all in the same pass over the frame, so
     all lanes are counted "at once" every frame - and a live bar chart
     (matplotlib) is updated in the SAME loop, in real time, right
     alongside the counting.

WHY NOT MOG2 (adaptive background subtraction)?
-------------------------------------------------
The original approach used cv2.createBackgroundSubtractorMOG2(), which
continuously updates ("learns") its background model from every new
frame. That's great for spotting motion, but it is the WRONG tool here:
a vehicle that stops and waits in a queue will, within a few seconds,
get absorbed into MOG2's background model and STOP being reported as
foreground - i.e. a stopped, waiting vehicle silently disappears from
the count. Since the whole point of this script is counting vehicles
that are STOPPED and WAITING at a lane, that's a direct conflict.

Using one FIXED reference frame of the empty road avoids this: a
stopped vehicle keeps differing from the empty-road reference for as
long as it sits there, so it keeps being counted. The trade-off is that
a fixed reference is more sensitive to lighting changes over time
(clouds passing, sunset, etc.) - press 'r' while a lane is confirmed
empty (e.g. right after its signal turns green and clears) to refresh
just that lane's slice of the reference frame.

Usage:
    python calibrate_rois.py --source Traffic.mp4      # do this FIRST, once

    # Works with a recorded video file, a webcam, OR a live network
    # camera (e.g. ESP32-CAM) - same script, just change --source:
    python vehicle_counting_opencv.py --source Traffic.mp4
    python vehicle_counting_opencv.py --source 0                     # webcam
    python vehicle_counting_opencv.py --source http://<esp32-ip>:81/stream

    # Send counts to the scheduler (traffic_light_controller.py) so it can
    # decide green-light durations per lane:
    python vehicle_counting_opencv.py --source Traffic.mp4 \\
        --post-url http://127.0.0.1:5000/update_counts

Controls while running:
    q  quit
    r  refresh the reference frame (do this when you can see the road
       is genuinely empty - fixes gradual lighting drift)

Live chart:
    A single matplotlib window with TWO panels updates in real time,
    in the same loop as the counting (nothing is queued/delayed):
      - LEFT:  bar chart of the current waiting count per lane
      - RIGHT: line chart of each lane's count over the last N seconds,
               so you can see queues building up / clearing over time
"""

import argparse
import json
import os
import time
from collections import deque
from urllib.parse import urlparse

try:
    import cv2
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency fallback
    cv2 = None
    np = None
    _cv2_IMPORT_ERROR = exc

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- Tunable parameters -------------------------------------------------
MIN_CONTOUR_AREA = 250        # pixels; ignore blobs smaller than this (noise)
DIFF_THRESHOLD = 35           # 0-255; how different from reference counts as "foreground"
REFERENCE_WARMUP_FRAMES = 30  # frames used to build the initial static reference
ROI_CONFIG_FILE = "lane_rois.json"

# Fallback ROIs (fractions of frame width/height) ONLY used if
# lane_rois.json doesn't exist yet. Strongly recommended: run
# calibrate_rois.py first and let it write the real file instead of
# relying on these guesses.
DEFAULT_LANE_ROIS = {
    "West":  (0.00, 0.40, 0.40, 0.55),   # queue on the west incoming road
    "South": (0.42, 0.55, 0.58, 1.00),   # queue on the south incoming road
}


def load_lane_rois(path=ROI_CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"Loaded lane ROIs from {path}: {list(data.keys())}")
        return data
    print(f"No {path} found - using rough DEFAULT_LANE_ROIS. "
          f"Run calibrate_rois.py first for accurate boxes.")
    return DEFAULT_LANE_ROIS


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
    for candidate in candidates:
        print(f"Trying video source: {candidate}")
        cap = cv2.VideoCapture(candidate)
        if cap.isOpened():
            print(f"Opened video source successfully: {candidate}")
            return cap
        cap.release()
    raise RuntimeError(
        f"Could not open video source: {source}. Tried: {', '.join(str(c) for c in candidates)}"
    )


def build_static_reference(cap, warmup_frames=REFERENCE_WARMUP_FRAMES):
    """Capture several frames of the (ideally empty) road and take the
    per-pixel median, which is far less noisy than trusting a single frame."""
    frames = []
    print("Building static background reference "
          f"(capturing {warmup_frames} frames - try to start when the road is clear)...")
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise RuntimeError("Could not read any frames to build a reference background.")
    stack = np.stack(frames, axis=0)
    reference = np.median(stack, axis=0).astype(np.uint8)
    return reference


def count_contours_in_mask(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    return count, contours


class LiveChart:
    """Live-updating chart with two panels, drawn in the same loop as the
    counting so both update every frame in real time, alongside counting:
      - a bar chart of the CURRENT count per lane
      - a line chart of each lane's count over the last `history_seconds`
        seconds, so queue build-up/clearing over time is visible too
    """

    def __init__(self, lane_names, history_seconds=60):
        self.enabled = plt is not None
        if not self.enabled:
            print("matplotlib not available - skipping live chart "
                  "(counts will still print to the console).")
            return

        self.lane_names = lane_names
        self.history_seconds = history_seconds
        self.start_time = time.time()
        self.times = deque()
        self.history = {name: deque() for name in lane_names}

        plt.ion()
        self.fig, (self.ax_bar, self.ax_line) = plt.subplots(1, 2, figsize=(10, 4))

        # Left panel: current counts, one bar per lane
        self.bars = self.ax_bar.bar(lane_names, [0] * len(lane_names), color="tab:orange")
        self.ax_bar.set_ylim(0, 10)
        self.ax_bar.set_ylabel("Waiting vehicles")
        self.ax_bar.set_title("Current count per lane")

        # Right panel: count history over time, one line per lane
        self.lines = {
            name: self.ax_line.plot([], [], label=name)[0] for name in lane_names
        }
        self.ax_line.set_xlim(0, history_seconds)
        self.ax_line.set_ylim(0, 10)
        self.ax_line.set_xlabel("seconds ago" if False else "time (s)")
        self.ax_line.set_ylabel("Waiting vehicles")
        self.ax_line.set_title(f"Last {history_seconds}s per lane")
        self.ax_line.legend(loc="upper left", fontsize=8)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.show()

    def update(self, counts):
        if not self.enabled:
            return

        now = time.time() - self.start_time
        self.times.append(now)
        for name in self.lane_names:
            self.history[name].append(counts.get(name, 0))

        # Drop points older than the rolling window
        while self.times and (now - self.times[0]) > self.history_seconds:
            self.times.popleft()
            for name in self.lane_names:
                self.history[name].popleft()

        values = [counts.get(name, 0) for name in self.lane_names]

        # -- update bar panel --
        for bar, v in zip(self.bars, values):
            bar.set_height(v)
        max_v = max(values) if values else 0
        self.ax_bar.set_ylim(0, max(5, max_v + 2))

        # -- update line panel --
        max_hist = 0
        for name in self.lane_names:
            xs = list(self.times)
            ys = list(self.history[name])
            self.lines[name].set_data(xs, ys)
            if ys:
                max_hist = max(max_hist, max(ys))
        self.ax_line.set_xlim(max(0, now - self.history_seconds), max(now, 1))
        self.ax_line.set_ylim(0, max(5, max_hist + 2))

        self.fig.canvas.draw_idle()
        # plt.pause lets the GUI event loop breathe without blocking the
        # OpenCV window - both run cooperatively in the same thread.
        plt.pause(0.001)


def main(source, show_window=True, post_url=None, refresh_seconds=None):
    if cv2 is None:
        raise RuntimeError("OpenCV is required for webcam/video counting") from _cv2_IMPORT_ERROR

    lane_rois = load_lane_rois()
    lane_names = list(lane_rois.keys())

    cap = open_video_source(source)
    reference = build_static_reference(cap)

    chart = LiveChart(lane_names)
    last_refresh = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended / camera disconnected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        diff = cv2.absdiff(gray, reference)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=2)

        lane_counts = {}
        for lane_name, roi in lane_rois.items():
            x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
            lane_mask = mask[y1:y2, x1:x2]
            count, _ = count_contours_in_mask(lane_mask)
            lane_counts[lane_name] = count

            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"{lane_name}: {count}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ---- This dict is exactly what a greedy scheduler needs ----
        print(f"[COUNT] {lane_counts}")

        chart.update(lane_counts)

        if post_url:
            try:
                import requests
                requests.post(post_url, json=lane_counts, timeout=1)
            except Exception as e:
                print("Could not send counts to controller:", e)

        if show_window:
            cv2.imshow("Lane vehicle counting (OpenCV, static reference)", frame)
            cv2.imshow("Foreground mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                print("Refreshing static reference frame (assuming road is empty now)...")
                reference = build_static_reference(cap)

        # Optional automatic periodic refresh (e.g. every N seconds), useful
        # for a real camera where lighting drifts slowly over the day.
        if refresh_seconds and (time.time() - last_refresh) > refresh_seconds:
            print("Auto-refreshing static reference frame...")
            reference = build_static_reference(cap)
            last_refresh = time.time()

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
    parser.add_argument("--refresh-seconds", type=float, default=None,
                         help="optional: automatically re-capture the empty-road reference "
                              "every N seconds (only sensible for a real, live camera)")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    main(src, show_window=not args.no_window, post_url=args.post_url,
         refresh_seconds=args.refresh_seconds)
