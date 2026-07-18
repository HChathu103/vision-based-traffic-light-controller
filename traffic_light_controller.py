"""
traffic_light_controller.py
----------------------------
Small Flask server that sits between the OpenCV vehicle counter and the
traffic-light simulation/hardware, and decides how much GREEN TIME each
lane should get, based on how many vehicles are currently waiting there.

Flow:
    vehicle_counting_opencv.py  --(POST /update_counts, JSON per-lane counts)-->
        traffic_light_controller.py  --(computes green times)-->
            GET /green_times  <-- simulation / hardware polls this

Scheduling method: GREEDY PROPORTIONAL ALLOCATION
--------------------------------------------------
No ML, easy to explain in a viva:

  1. Every lane gets a guaranteed MIN_GREEN (fairness - nobody starves).
  2. Whatever cycle time is left over (BASE_CYCLE_SECONDS - all the
     MIN_GREEN's already handed out) is split between lanes IN
     PROPORTION to how many vehicles are waiting there right now.
  3. Nothing is allowed to go below MIN_GREEN or above MAX_GREEN.
  4. If every lane is empty (or no counts have arrived yet), everyone
     just gets a default, equal share - no reason to favor one lane.

This is "greedy" in the sense that the busiest lane right now gets the
biggest slice of the extra time - it doesn't plan multiple cycles
ahead, it just reacts to the latest counts.

Run it:
    python traffic_light_controller.py
    (defaults to http://127.0.0.1:5000)

Then point the counter at it:
    python vehicle_counting_opencv.py --source 2026final.mp4 \
        --post-url http://127.0.0.1:5000/update_counts

Check what it decided:
    curl http://127.0.0.1:5000/green_times
    curl http://127.0.0.1:5000/status        # counts + green times + age
"""

import threading
import time

from flask import Flask, jsonify, request

# ---- Tunable scheduling parameters --------------------------------------
MIN_GREEN = 5          # seconds - every lane gets at least this much
MAX_GREEN = 30          # seconds - cap, so one lane can't hog forever
BASE_CYCLE_SECONDS = 40  # total green seconds to split across all lanes
                         # (must be >= MIN_GREEN * number_of_lanes)
STALE_AFTER_SECONDS = 15  # if no new counts arrive for this long, counts
                           # are considered stale and we fall back to
                           # equal green times (fail-safe, not "frozen"
                           # on old data)

app = Flask(__name__)

_lock = threading.Lock()
_state = {
    "counts": {},          # last received {lane: count}
    "green_times": {},     # last computed {lane: seconds}
    "last_updated": 0.0,   # time.time() of last /update_counts call
}


def compute_green_times(counts, min_green=MIN_GREEN, max_green=MAX_GREEN,
                         base_cycle=BASE_CYCLE_SECONDS):
    """Greedy proportional allocation - see module docstring."""
    lanes = list(counts.keys())
    if not lanes:
        return {}

    guaranteed_total = min_green * len(lanes)
    extra_pool = max(0, base_cycle - guaranteed_total)

    total_waiting = sum(max(0, c) for c in counts.values())

    green_times = {}
    if total_waiting == 0:
        # Nobody waiting anywhere - split evenly, no reason to favor a lane
        equal_share = min_green + extra_pool / len(lanes)
        for lane in lanes:
            green_times[lane] = int(round(min(max_green, equal_share)))
        return green_times

    for lane in lanes:
        share = max(0, counts[lane]) / total_waiting
        raw_time = min_green + share * extra_pool
        green_times[lane] = int(round(min(max_green, max(min_green, raw_time))))

    return green_times


@app.route("/update_counts", methods=["POST"])
def update_counts():
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected a JSON object of {lane: count}"}), 400

    counts = {str(lane): int(count) for lane, count in payload.items()}
    green_times = compute_green_times(counts)

    with _lock:
        _state["counts"] = counts
        _state["green_times"] = green_times
        _state["last_updated"] = time.time()

    return jsonify({"received": counts, "green_times": green_times})


@app.route("/green_times", methods=["GET"])
def green_times():
    with _lock:
        age = time.time() - _state["last_updated"] if _state["last_updated"] else None
        stale = age is None or age > STALE_AFTER_SECONDS

        if stale:
            # Fail-safe: don't act on old/no data, fall back to a neutral
            # equal split among whatever lanes we last knew about.
            lanes = list(_state["counts"].keys())
            if lanes:
                fallback = compute_green_times({lane: 0 for lane in lanes})
            else:
                fallback = {}
            return jsonify({"green_times": fallback, "stale": True, "age_seconds": age})

        return jsonify({"green_times": _state["green_times"], "stale": False, "age_seconds": age})


@app.route("/status", methods=["GET"])
def status():
    with _lock:
        age = time.time() - _state["last_updated"] if _state["last_updated"] else None
        return jsonify({
            "counts": _state["counts"],
            "green_times": _state["green_times"],
            "age_seconds": age,
            "min_green": MIN_GREEN,
            "max_green": MAX_GREEN,
            "base_cycle_seconds": BASE_CYCLE_SECONDS,
        })


if __name__ == "__main__":
    print(f"Scheduler config: MIN_GREEN={MIN_GREEN}s  MAX_GREEN={MAX_GREEN}s  "
          f"BASE_CYCLE_SECONDS={BASE_CYCLE_SECONDS}s")
    print("POST counts to      http://127.0.0.1:5000/update_counts")
    print("Poll decisions at   http://127.0.0.1:5000/green_times")
    print("Full status at      http://127.0.0.1:5000/status")
    app.run(host="0.0.0.0", port=5000, debug=False)
