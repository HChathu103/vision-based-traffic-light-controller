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

# File output targets
OUTPUT_LINE_PLOT = "traffic_waiting_history.png"
OUTPUT_SUMMARY_TXT = "traffic_summary.txt"

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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image file not found at: {image_path}")
        
    print(f"Loading static background reference from image: {image_path}...")
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        raise RuntimeError(f"Could not read or decode the image file: {image_path}")
        
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
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
        
        # History limits rolling display queues
        self.times = deque()
        self.history = {name: deque() for name in lane_names}
        
        # Cumulative structural logs for the final summary report table
        self.full_times_log = []
        self.full_history_log = {name: [] for name in lane_names}

        plt.ion()
        
        # SEPARATE PLOT 1: Bar Chart Window
        self.fig_bar, self.ax_bar = plt.subplots(figsize=(5, 4))
        self.bars = self.ax_bar.bar(lane_names, [0] * len(lane_names), color="tab:orange")
        self.ax_bar.set_ylim(0, 10)
        self.ax_bar.set_ylabel("Waiting Vehicles")
        self.ax_bar.set_title("Current Count Per Lane")
        self.fig_bar.tight_layout()

        # SEPARATE PLOT 2: Continuous Timeline Window
        self.fig_line, self.ax_line = plt.subplots(figsize=(6, 4))
        self.lines = {name: self.ax_line.plot([], [], label=name)[0] for name in lane_names}
        self.ax_line.set_xlim(0, history_seconds)
        self.ax_line.set_ylim(0, 10)
        self.ax_line.set_xlabel("Time (s)")
        self.ax_line.set_ylabel("Waiting Vehicles")
        self.ax_line.set_title(f"Live Queue History (Last {history_seconds}s)")
        self.ax_line.legend(loc="upper left", fontsize=8)
        self.fig_line.tight_layout()

    def update(self, counts):
        if not self.enabled:
            return

        now = time.time() - self.start_time
        self.times.append(now)
        self.full_times_log.append(now)
        
        for name in self.lane_names:
            val = counts.get(name, 0)
            self.history[name].append(val)
            self.full_history_log[name].append(val)

        # Slide visual rolling frame window
        while self.times and (now - self.times[0]) > self.history_seconds:
            self.times.popleft()
            for name in self.lane_names:
                self.history[name].popleft()

        # Render separated Windows
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

        self.fig_bar.canvas.draw_idle()
        self.fig_line.canvas.draw_idle()
        plt.pause(0.001)

    def save_and_summarize(self):
        """Generates separate image assets and compile analytics metrics into text files."""
        if not self.enabled or not self.full_times_log:
            print("[!] Processing chart data is unavailable.")
            return
            
        try:
            plt.ioff()
        
            # 1. Render and save the FULL contextual history line plot
            self.ax_line.cla() # Clean standard rolling lines framework axes
            for name in self.lane_names:
                self.ax_line.plot(self.full_times_log, self.full_history_log[name], label=name)
            
            self.ax_line.set_xlim(0, max(self.full_times_log))
            
            # Find global peak metrics to scale visual range accurately
            all_vals = [v for lst in self.full_history_log.values() for v in lst]
            max_global = max(all_vals) if all_vals else 5
            self.ax_line.set_ylim(0, max(5, max_global + 2))
            
            self.ax_line.set_xlabel("Total Session Duration (seconds)")
            self.ax_line.set_ylabel("Waiting Vehicles")
            self.ax_line.set_title("Complete Structural Execution Log (Full Plot)")
            self.ax_line.legend(loc="upper left")
            self.fig_line.tight_layout()
            
            self.fig_line.savefig(OUTPUT_LINE_PLOT, dpi=150)
            print(f"[SUCCESS] Complete time history path plot saved to: {OUTPUT_LINE_PLOT}")
            
            # 2. Compute Traffic Statistics
            summary_lines = [
                "=" * 60,
                "                   TRAFFIC ANALYSIS METRICS                     ",
                "=" * 60,
                f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Total Monitoring Time: {max(self.full_times_log):.2f} seconds\n",
                "{:<12} | {:<15} | {:<15} | {:<15}".format("Lane Name", "Avg Vehicles", "Max Vehicles", "Congestion Ratio*"),
                "-" * 60
            ]
            
            busy_lane = None
            highest_average = -1
            highest_peak = -1
            peak_lane = None

            for name in self.lane_names:
                data = self.full_history_log[name]
                avg_count = np.mean(data) if data else 0
                max_count = np.max(data) if data else 0
                
                # Congestion Ratio = % of time lane has vehicles waiting
                busy_frames = sum(1 for x in data if x > 0)
                congestion_ratio = (busy_frames / len(data)) * 100 if data else 0
                
                summary_lines.append("{:<12} | {:<15.2f} | {:<15} | {:<14.1f}%".format(
                    name, avg_count, max_count, congestion_ratio
                ))
                
                if avg_count > highest_average:
                    highest_average = avg_count
                    busy_lane = name
                if max_count > highest_peak:
                    highest_peak = max_count
                    peak_lane = name
            
            summary_lines.extend([
                "-" * 60,
                "* Congestion Ratio represents percentage of frames where cars were actively waiting.",
                "\n>>> INSIGHT BREAKDOWN:",
                f" - Most Consistently Busy Lane (Highest Avg Load): '{busy_lane}' ({highest_average:.2f} vehicles avg)",
                f" - Absolute Peak Congestion Lane (Max Bottleneck): '{peak_lane}' ({highest_peak} vehicles max)",
                "=" * 60
            ])
            
            # Write summary log contents to local disk
            report_text = "\n".join(summary_lines)
            with open(OUTPUT_SUMMARY_TXT, "w") as f:
                f.write(report_text)
                
            print(f"[SUCCESS] Analytical summary metrics compiled inside: {OUTPUT_SUMMARY_TXT}")
            print("\n" + report_text) # Print to console for immediate visibility
            
        except Exception as e:
            print(f"[ERROR] Run failure compiling metrics layouts: {e}")

def main(video_path, show_window=True, post_url=None):
    lane_rois = load_lane_rois()
    lane_names = list(lane_rois.keys())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Video source file is empty or corrupt.")
        cap.release()
        return

    reference = load_static_reference(REFERENCE_IMAGE_PATH, target_shape=first_frame.shape)
    blurred_reference = cv2.GaussianBlur(reference, (5, 5), 0)
    
    chart = LiveChart(lane_names)
    frame_count = 0

    try:
        while True:
            if frame_count == 0:
                frame = first_frame
                ret = True
            else:
                ret, frame = cap.read()
                
            if not ret:
                break

            frame_count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            diff = cv2.absdiff(gray, blurred_reference)
            _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=2)

            lane_counts = {}
            for lane_name, roi in lane_rois.items():
                x1, y1, x2, y2 = roi_pixels(frame.shape, roi)
                roi_mask = mask[y1:y2, x1:x2]
                
                contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                count = 0
                for c in contours:
                    if cv2.contourArea(c) >= MIN_CONTOUR_AREA:
                        x, y, w, h = cv2.boundingRect(c)
                        if w > 12 and h > 12: 
                            count += 1
                            
                lane_counts[lane_name] = count

                if show_window:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(frame, f"{lane_name}: {count}", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(f"[COUNT] {lane_counts}")

            # Keep background arrays updating steadily every 30 frames 
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
    finally:
        print("\nClosing video streams and processing final metrics report charts...")
        chart.save_and_summarize()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to traffic video file")
    parser.add_argument("--no-window", action="store_true", help="Run in headless mode")
    parser.add_argument("--post-url", default=None, help="Server endpoint for lane metrics")
    args = parser.parse_args()

    main(args.source, show_window=not args.no_window, post_url=args.post_url)