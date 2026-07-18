# 🚦 Traffic Monitoring and Vehicle Counting System using OpenCV 
==================================================================

# Overview
--------
This project is a small end-to-end prototype that simulates an intersection, counts vehicles per lane, and computes dynamic green-times using a greedy + max-heap allocation. The system detects and counts waiting vehicles in predefined traffic lanes using background subtraction techniques and Region of Interest (ROI) analysis. It also includes a traffic simulation environment, ROI calibration tool, and traffic analytics generation.

-----
## 📌 Project Features

- 🚗 Vehicle detection using OpenCV
- 📊 Vehicle counting for multiple lanes
- 📍 Interactive ROI calibration tool
- 🎮 Traffic intersection simulation using Pygame
- 📈 Live traffic monitoring charts
- 📉 Traffic history graph generation
- 📝 Automatic traffic summary report
- ⚙️ Resolution-independent ROI configuration
- 🖼️ Background subtraction with static reference image

---
## 📂 Project Structure

```
.
├── vehicle_counting_opencv.py     # Main vehicle counting application
├── simulation.py                  # Pygame visual simulation of vehicles and traffic lights. It gathers lane counts from simulated                                          vehicles and asks the controller for a timing plan.
├── calibrate_rois.py              # ROI calibration tool
├── traffic_light_controller.py    # Decision logic: implements the 90s cycle, 10s safety minimum per lane, and distributes remaining                                        time proportionally using a greedy / max-heap approach.
├── lane_rois.json                 # Saved lane regions
├── intersection1.png              # Reference background image
├── traffic_video.mp4              # Input traffic video (user supplied)
├── traffic_waiting_history.png    # Generated history graph
├── traffic_summary.txt            # Generated statistics
├── images/                        # Vehicle and signal images

└── README.md
```
---

## 🛠 Technologies Used

- Python 3.x
- OpenCV
- NumPy
- Matplotlib
- Pygame
- JSON

---

## 📦 Required Python Libraries

Install the required packages:

```bash
pip install opencv-python numpy matplotlib pygame requests
```

---

# ROI Calibration

Before running the vehicle counter, create lane regions.

Run:

```bash
python calibrate_rois.py --source traffic_video.mp4
```

or use a webcam:

```bash
python calibrate_rois.py --source 0
```

### Controls

| Key | Function |
|------|----------|
| Left Mouse | Draw ROI |
| C | Clear all ROIs |
| S | Save ROIs |
| Q | Quit without saving |

The ROIs will be saved in:

```
lane_rois.json
```

---

# Run Vehicle Counting

```bash
python vehicle_counting_opencv.py --source traffic_video.mp4
```

Run without display windows:

```bash
python vehicle_counting_opencv.py --source traffic_video.mp4 --no-window
```

Send lane counts to a server:

```bash
python vehicle_counting_opencv.py --source traffic_video.mp4 --post-url http://localhost:5000/update
```

---

# Traffic Simulation

Run the traffic simulation:

```bash
python simulation.py
```

The simulation demonstrates:

- Vehicle generation
- Signal timing
- Vehicle movement
- Queue management
- Traffic light control

---

# Output Files

The system automatically generates:

## Traffic History

```
traffic_waiting_history.png
```

Contains the waiting vehicle history over time.

---

## Traffic Summary

```
traffic_summary.txt
```

Contains:

- Average vehicles
- Maximum vehicles
- Congestion ratio
- Most congested lane
- Peak traffic statistics

---

# Detection Method

The vehicle counting algorithm performs:

1. Load static background image
2. Convert to grayscale
3. Gaussian Blur
4. Background subtraction
5. Otsu thresholding
6. Morphological filtering
7. Contour detection
8. ROI-based vehicle counting
9. Live statistics generation

---

# Project Workflow

```
Traffic Video
      │
      ▼
Load Reference Image
      │
      ▼
Background Subtraction
      │
      ▼
Thresholding
      │
      ▼
Morphological Operations
      │
      ▼
Contour Detection
      │
      ▼
ROI Vehicle Counting
      │
      ▼
Traffic Statistics
      │
      ▼
Graphs + Summary Report
```

---

# Example Results

The application provides:

- Live vehicle counts
- Lane-by-lane traffic analysis
- Traffic history graph
- Congestion statistics
- Waiting vehicle trends

---

# Future Improvements

The current system focuses on monitoring and analyzing traffic conditions through vehicle detection and counting. In future work, the following enhancements can be implemented:

- 🚦 **Dynamic Traffic Light Control System**
  - Integrate the vehicle counting module with an adaptive traffic signal controller.
  - Automatically adjust green light durations based on the number of waiting vehicles in each lane.
  - Reduce unnecessary waiting times and improve overall traffic flow.
  - Allocate longer green phases to heavily congested lanes while reducing green time for low-traffic lanes.
  - Continuously update signal timings using real-time traffic data instead of fixed signal schedules.

- 🤖 Deep learning-based vehicle detection (YOLOv8, YOLOv11, etc.) for improved accuracy.

- 🎥 Multi-camera traffic monitoring to cover larger intersections.

- 📍 Vehicle tracking across multiple frames for more accurate counting.

- 🚗 Vehicle classification (car, bus, truck, motorcycle, bicycle).

- 🚨 Emergency vehicle priority detection and automatic signal preemption.

- ☁️ Cloud database integration for long-term traffic data storage and analytics.

- 🌐 Web dashboard for real-time traffic visualization and remote monitoring.

- 📱 Mobile application for live traffic monitoring and traffic statistics.

- 🛣️ Support for multiple intersections with centralized traffic management.

- 📊 AI-based traffic prediction using historical traffic patterns and machine learning models.

- 🚔 Automatic incident detection, including traffic congestion, accidents, and road blockages.

---

# Author

**Harsha Chathuranga**

UNdergraduate student of Electrical and Electronic Engineering

Faculty of Engineering

University of Sri Jayewardenepura

---

# License

This project was developed for educational purposes as part of the Image Processing.

Feel free to fork, modify, and improve the project for learning and research.
