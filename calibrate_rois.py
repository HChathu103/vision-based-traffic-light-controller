"""
calibrate_rois.py
------------------
One-time helper to draw the "waiting zone" box for each lane, by hand,
on a real frame from your video/camera - instead of guessing pixel
coordinates in code.

Why this matters:
Hardcoded pixel boxes (like the North/South/East/West fractions in the
original script) only line up by luck. The camera angle, resolution,
or window position will be different every time you record. This tool
lets you click-drag a rectangle directly on a real frame so the boxes
always match what the camera actually sees.

Usage:
    python calibrate_rois.py --source Traffic.mp4
    python calibrate_rois.py --source 0                     # webcam
    python calibrate_rois.py --source http://<esp32-ip>:81/stream

Controls:
    - Click and drag to draw a rectangle for one lane's waiting zone.
    - Press a letter/number key to LABEL and SAVE that rectangle
      (e.g. press 'w' for "West", 's' for "South"...). You'll be
      prompted in the terminal for the lane name after each drag.
    - Press 'u' to undo the last saved ROI.
    - Press 'q' to quit and write lane_rois.json.


"""

import argparse
import json
import sys

import cv2

drawing = False
ix, iy = -1, -1
current_box = None
saved_rois = {}


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, current_box

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_box = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))


def main(source):
    # NOTE: current_box and saved_rois are assigned to later in this
    # function (e.g. `current_box = None`, `del saved_rois[...]`). In
    # Python, assigning to a name ANYWHERE inside a function makes that
    # name local to the WHOLE function - including lines above the
    # assignment that only read it. Without this `global` declaration,
    # the earlier `if current_box:` read raises UnboundLocalError.
    global current_box, saved_rois

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)

    # Grab a representative frame (skip the first few, sometimes blank/dark)
    frame = None
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame from the source.")
            sys.exit(1)
    cap.release()

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")
    print("Draw a rectangle over each lane's waiting/queue zone, then")
    print("type the lane name in this terminal when prompted.\n")

    cv2.namedWindow("Calibrate ROIs")
    cv2.setMouseCallback("Calibrate ROIs", mouse_callback)

    while True:
        display = frame.copy()

        # Draw already-saved ROIs
        for name, (x1, y1, x2, y2) in saved_rois.items():
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, name, (x1 + 4, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw the box currently being dragged
        if current_box:
            cv2.rectangle(display, current_box[:2], current_box[2:], (0, 200, 255), 2)

        cv2.putText(display, "drag=box  s=save last box  u=undo  q=quit+write json",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Calibrate ROIs", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('s') and current_box:
            name = input("Lane name for this box (e.g. West, South): ").strip()
            if name:
                # Store as fractions of width/height so it survives resolution changes
                x1, y1, x2, y2 = current_box
                saved_rois[name] = (x1, y1, x2, y2)
                print(f"Saved ROI for '{name}': {current_box}")
            current_box = None

        elif key == ord('u') and saved_rois:
            last_key = list(saved_rois.keys())[-1]
            del saved_rois[last_key]
            print(f"Removed last ROI: {last_key}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Convert pixel boxes -> fractions of frame size (resolution-independent)
    fractional = {
        name: (x1 / w, y1 / h, x2 / w, y2 / h)
        for name, (x1, y1, x2, y2) in saved_rois.items()
    }

    with open("lane_rois.json", "w") as f:
        json.dump(fractional, f, indent=2)

    print("\nSaved lane_rois.json:")
    print(json.dumps(fractional, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="video file, webcam index, or stream URL")
    args = parser.parse_args()
    main(args.source)