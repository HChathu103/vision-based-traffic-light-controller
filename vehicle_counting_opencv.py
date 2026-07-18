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
MIN_CONTOUR_AREA = 200        # pixels; ignore blobs smaller than this
REFERENCE_IMAGE_PATH = "intersection1.png"  # Path to your static reference image
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

def load_static_reference(image_path, target_shape=None):
    """
    Loads the background reference image as grayscale and auto-scales it 
    to match the video resolution to prevent dimensions mismatch errors.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image file not found at: {image_path}")
        
    print(f"Loading static background reference from image: {image_path}...")
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        raise RuntimeError(f"Could not read or decode the image file: {image_path}")
        
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Auto-adjust shape if video size differs from the image template size
    if target_shape is not None and gray_ref.shape[:2] != target_shape[:2]:
        print(f"[!] Auto-resizing reference image from {gray_ref.shape[1]}x{gray_ref.shape[0]} "
              f"to match video frame dimensions {target_shape[1]}x{target_shape[0]}")
        gray_ref = cv2.resize(gray_ref, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        
    return gray_ref

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
        self.fig, (self.ax_bar, self.ax_line) = plt.subplots(1, 2, figsize=(8, 4))

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

    # Warmup check: read the first frame to safely query the video size dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Video source file is empty or corrupt.")
        cap.release()
        return

    # Pass the shape parameters to ensure our template matches perfectly
    reference = load_static_reference(REFERENCE_IMAGE_PATH, target_shape=first_frame.shape)
    
    # Pre-blur the reference frame once to improve pixel background subtraction speed
    blurred_reference = cv2.GaussianBlur(reference, (5, 5), 0)
    
    chart = LiveChart(lane_names)
    frame_count = 0

    while True:
        # Step handling to reuse our pre-read first frame correctly
        if frame_count == 0:
            frame = first_frame
            ret = True
        else:
            ret, frame = cap.read()
            
        if not ret:
            break

        frame_count += 1

        # Step 1: Image preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Step 2: Adaptive masking logic using dynamic Otsu threshold tracking
        diff = cv2.absdiff(gray, blurred_reference)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological noise-cleaning operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=2)

        # Step 3: Count tracking inside segmented lane regions
        lane_counts = {}
        for lane_name, roi in lane_rois.items():
            x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
            roi_mask = mask[y1:y2, x1:x2]
            
            contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            count = 0
            for c in contours:
                if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                    x, y, w, h = cv2.boundingRect(c)
                    # Filter out skinny lightning artifacts, dust lines, or cross-lane boundary errors
                    if w > 12 and h > 12: 
                        count += 1
                        
            lane_counts[lane_name] = count

            if show_window:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                cv2.putText(frame, f"{lane_name}: {count}", (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print(f"[COUNT] {lane_counts}")

        # Performance-safe chart data update steps (once every ~1 second)
        if frame_count % 30 == 0:
            chart.update(lane_counts)

        if post_url:
            try:
                import requests
                requests.post(post_url, json=lane_counts, timeout=0.2)
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