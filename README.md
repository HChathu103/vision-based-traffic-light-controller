Smart Traffic Light (Simulation + Counting + Controller)
======================================================

Overview
--------
This project is a small end-to-end prototype that simulates an intersection, counts vehicles per lane, and computes dynamic green-times using a greedy + max-heap allocation.

Files
-----
- `simulation.py` — Pygame visual simulation of vehicles and traffic lights. It gathers lane counts from simulated vehicles and asks the controller for a timing plan.
- `vehicle_counting_opencv.py` — OpenCV-based vehicle counting utilities. Provides helpers to normalize counts and (optionally) run live video counting.
- `traffic_light_controller.py` — Decision logic: implements the 90s cycle, 10s safety minimum per lane, and distributes remaining time proportionally using a greedy / max-heap approach.
- `test_alloc.py` — Small helper that demonstrates the allocation math for example counts.

Requirements
------------
- Python 3.8+ recommended
- Packages: `pygame`, `opencv-python`, `numpy`, `flask`, `requests` (only needed if you run the Flask API)

Install (PowerShell)
--------------------
py -3 -m pip install --upgrade pip
py -3 -m pip install pygame opencv-python numpy flask requests

Run
---
- Run the visual simulation (uses controller logic in-process):

```powershell
py -3 simulation.py
```

- Run the controller server (optional; used for POST/GET debug):

```powershell
py -3 traffic_light_controller.py
```

- Run the OpenCV counting (webcam/video):

```powershell
py -3 vehicle_counting_opencv.py --source 0
```

- Run the allocation demo (example math):

```powershell
py -3 test_alloc.py
```

Allocation math (90s cycle, 10s safety minimum)
------------------------------------------------
1. Total cycle = 90 seconds.
2. Safety minimum per lane = 10 seconds. With 4 lanes -> reserved = 4 * 10 = 40s.
3. Remaining pool = 90 - 40 = 50s.
4. Let counts = {North, West, South, East} (raw integer counts). Let total = sum(counts).
5. For each lane:
   - Extra time = remaining_pool * (lane_count / total)
   - Green time = MIN_GREEN + Extra time

Example (your provided values):
- Counts: North=10, West=7, South=5, East=3 → total = 25
- Remaining pool = 50s
- North extra = 50 * 10/25 = 20 → Green = 10 + 20 = 30s
- West extra  = 50 * 7/25  = 14 → Green = 10 + 14 = 24s
- South extra = 50 * 5/25  = 10 → Green = 10 + 10 = 20s
- East extra  = 50 * 3/25  = 6  → Green = 10 + 6  = 16s
- Sum: 30 + 24 + 20 + 16 = 90s (valid)

Notes
-----
- The project currently runs in-process: `simulation.py` calls controller functions directly.
- `vehicle_counting_opencv.py` exposes normalization helpers so you can use the same counting logic for simulation-sourced counts or webcam frames.
- If you want the OpenCV detector to run on frames produced by the simulation, I can add a small frame-export hook (next step).

If you want a short handout or slide-ready one-page description of the math and flow, I can make that next.