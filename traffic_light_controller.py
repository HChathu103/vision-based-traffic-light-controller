

import heapq
import time
import threading
from flask import Flask, request, jsonify

try:
    from vehicle_counting_opencv import normalize_lane_counts
except Exception:  # pragma: no cover - fallback if module import fails
    def normalize_lane_counts(counts, lane_names=None):
        return counts

# ----------------------------- CONFIG -----------------------------------
LANES = ["North", "South", "East", "West"]
CYCLE_TIME = 90         # total seconds shared per cycle
MIN_GREEN = 10          # safety minimum per lane (4 lanes -> 40 seconds)
# No artificial cap on lane counts; use raw counts from detector/simulation
MAX_LANE_COUNT = None
MAX_GREEN = CYCLE_TIME - MIN_GREEN * (len(LANES) - 1)

app = Flask(__name__)
latest_counts = {lane: 0 for lane in LANES}
lock = threading.Lock()


# ------------------- CORE ALGORITHM: GREEDY + MAX-HEAP -------------------
def compute_green_times(counts: dict) -> dict:
    """Return a dynamic green-time plan for all lanes.

    The cycle allocates the required 40 seconds as safety minimum time
    across 4 lanes and then distributes the remaining 50 seconds using a
    greedy max-heap ranking based on current vehicle counts (capped at 8).
    """
    normalized_counts = normalize_lane_counts(counts, LANES)
    # use raw, non-capped counts
    clipped_counts = {
        lane: max(int(normalized_counts.get(lane, 0)), 0)
        for lane in LANES
    }
    total_weight = sum(clipped_counts.values())

    if total_weight == 0:
        equal = CYCLE_TIME / len(LANES)
        return {lane: int(equal) for lane in LANES}

    heap = [(-clipped_counts[lane], lane) for lane in LANES]
    heapq.heapify(heap)

    remaining = CYCLE_TIME - MIN_GREEN * len(LANES)
    raw_times = {}

    while heap:
        neg_count, lane = heapq.heappop(heap)
        weight = clipped_counts[lane]
        raw_times[lane] = MIN_GREEN + remaining * (weight / total_weight)

    rounded_times = {lane: int(raw_times[lane]) for lane in LANES}
    delta = CYCLE_TIME - sum(rounded_times.values())

    for lane in sorted(LANES, key=lambda item: raw_times[item] - rounded_times[item], reverse=True):
        if delta == 0:
            break
        rounded_times[lane] += 1
        delta -= 1

    return {lane: min(MAX_GREEN, rounded_times[lane]) for lane in LANES}


def log_plan_summary(counts: dict):
    """Print the counts, capped weights, and green-time calculation to the terminal."""
    normalized_counts = normalize_lane_counts(counts, LANES)
    clipped_counts = {
        lane: max(int(normalized_counts.get(lane, 0)), 0)
        for lane in LANES
    }
    total_weight = sum(clipped_counts.values())
    remaining_dynamic_time = CYCLE_TIME - MIN_GREEN * len(LANES)

    print("\n[COUNT UPDATE]")
    print(f"Raw vehicle counts: {counts}")
    print(f"Capped counts used for allocation: {clipped_counts}")
    print(f"Total weighted vehicles: {total_weight}")
    print(f"Safety minimum per lane: {MIN_GREEN}s")
    print(f"Remaining dynamic time to distribute: {remaining_dynamic_time}s")

    plan = compute_green_times(counts)
    print(f"Computed green times: {plan}")
    return plan


# ------------------------------ HTTP API ---------------------------------
@app.route("/update_counts", methods=["POST"])
def update_counts():
    """vehicle_counting_opencv.py calls this every cycle."""
    global latest_counts
    data = request.get_json(force=True)
    with lock:
        latest_counts.update(data)
    log_plan_summary(data)
    return jsonify({"status": "ok", "received": data})


@app.route("/current_plan", methods=["GET"])
def current_plan():
    with lock:
        counts_snapshot = dict(latest_counts)
    plan = log_plan_summary(counts_snapshot)
    return jsonify({"counts": counts_snapshot, "green_times_seconds": plan})


def control_loop():
    """Runs forever: every CYCLE_TIME seconds, recompute and push a plan."""
    while True:
        with lock:
            counts_snapshot = dict(latest_counts)
        plan = log_plan_summary(counts_snapshot)
        time.sleep(CYCLE_TIME)


def get_current_plan(counts: dict = None):
    """Return the green-time plan for the supplied counts without requiring Flask."""
    if counts is None:
        with lock:
            counts = dict(latest_counts)
    return compute_green_times(counts)


if __name__ == "__main__":
    t = threading.Thread(target=control_loop, daemon=True)
    t.start()
    print("Traffic controller running: POST vehicle counts to /update_counts")
    print("GET /current_plan to see the latest computed plan")
    app.run(host="0.0.0.0", port=5000)
