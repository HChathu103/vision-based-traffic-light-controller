import argparse
import json
import os
import time
from collections import deque
import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ---- Tunable parameters -------------------------------------------------
MIN_CONTOUR_AREA = 250        # pixels; ignore blobs smaller than this
DIFF_THRESHOLD = 35           # 0-255; absolute difference threshold
REFERENCE_WARMUP_FRAMES = 30  # frames used to build background
ROI_CONFIG_FILE = "lane_rois.json"

DEFAULT_LANE_ROIS = {
    "West":  (0.00, 0.46, 0.2, 0.5),
    "South": (0.42, 0.55, 0.58, 1.00),
}

def load_lane_rois(path=ROI_CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    print(f"No {path} found - using default ROIs.")
    return DEFAULT_LANE_ROIS

def roi_pixels(frame_shape, roi_fraction):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = roi_fraction
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

def build_static_reference(cap, warmup_frames=REFERENCE_WARMUP_FRAMES):
    frames = []
    print(f"Building background reference ({warmup_frames} frames)...")
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if not frames:
        raise RuntimeError("Could not read frames for background calibration.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)

def count_contours_in_mask(mask, min_area=MIN_CONTOUR_AREA):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = sum(1 for c in contours if cv2.contourArea(c) >= min_area)
    return count

class LiveChart:
    def __init__(self, lane_names, history_seconds=60):
        self.enabled = plt is not None
        if not self.enabled:
            return

        self.lane_names = lane_names
        self.history_seconds = history_seconds
        self.start_time = time.time()
        self.times = deque()
        self.history = {name: deque() for name in lane_names}

        plt.ion()
        self.fig, (self.ax_bar, self.ax_line) = plt.subplots(1, 2, figsize=(10, 4))

        self.bars = self.ax_bar.bar(lane_names, [0] * len(lane_names), color="tab:orange")
        self.ax_bar.set_ylim(0, 10)
        self.ax_bar.set_ylabel("Waiting vehicles")
        self.ax_bar.set_title("Current count per lane")

        self.lines = {name: self.ax_line.plot([], [], label=name)[0] for name in lane_names}
        self.ax_line.set_xlim(0, history_seconds)
        self.ax_line.set_ylim(0, 10)
        self.ax_line.set_xlabel("time (s)")
        self.ax_line.set_ylabel("Waiting vehicles")
        self.ax_line.set_title(f"Last {history_seconds}s per lane")
        self.ax_line.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout()

    def update(self, counts):
        if not self.enabled:
            return

        now = time.time() - self.start_time
        self.times.append(now)
        for name in self.lane_names:
            self.history[name].append(counts.get(name, 0))

        while self.times and (now - self.times[0]) > self.history_seconds:
            self.times.popleft()
            for name in self.lane_names:
                self.history[name].popleft()

        values = [counts.get(name, 0) for name in self.lane_names]

        for bar, v in zip(self.bars, values):
            bar.set_height(v)
        self.ax_bar.set_ylim(0, max(5, max(values) + 2) if values else 5)

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
        plt.pause(0.001)

def main(video_path, show_window=True, post_url=None):
    lane_rois = load_lane_rois()
    lane_names = list(lane_rois.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    reference = build_static_reference(cap)
    chart = LiveChart(lane_names)
    
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Image preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Difference masking
        diff = cv2.absdiff(gray, reference)
        _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=2)

        lane_counts = {}
        for lane_name, roi in lane_rois.items():
            x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
            count = count_contours_in_mask(mask[y1:y2, x1:x2])
            lane_counts[lane_name] = count

            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"{lane_name}: {count}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"[COUNT] {lane_counts}")

        # PERFORMANCE BOOST: Only update the slow chart window every 30 frames (~once a second)
        if frame_count % 30 == 0:
            chart.update(lane_counts)

        if post_url:
            try:
                import requests
                requests.post(post_url, json=lane_counts, timeout=0.2) # Fast timeout
            except Exception:
                pass

        if show_window:
            cv2.imshow("Tracking View", frame)
            cv2.imshow("Foreground Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to traffic video file")
    parser.add_argument("--no-window", action="store_true", help="Run in headless mode")
    parser.add_argument("--post-url", default=None, help="Server endpoint for lane metrics")
    args = parser.parse_args()

    main(args.source, show_window=not args.no_window, post_url=args.post_url)