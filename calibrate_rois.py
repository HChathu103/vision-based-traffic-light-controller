import argparse
import json
import os
import cv2
import numpy as np

# Global variables to handle mouse interaction
drawing = False
ix, iy = -1, -1
current_box = None
rois = {}
current_lane_index = 1

def select_lane_name():
    global current_lane_index
    name = f"Lane_{current_lane_index}"
    current_lane_index += 1
    return name

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_box, rois, frame_w, frame_h

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        
        # Ensure the box actually has a width and height
        if (x2 - x1) > 10 and (y2 - y1) > 10:
            lane_name = select_lane_name()
            # Save as relative fractions (0.0 to 1.0) so it's resolution-independent
            rois[lane_name] = (
                round(x1 / frame_w, 4),
                round(y1 / frame_h, 4),
                round(x2 / frame_w, 4),
                round(y2 / frame_h, 4)
            )
            print(f"Added {lane_name}: {rois[lane_name]}")
        current_box = None

def main():
    global frame_w, frame_h, current_box, rois
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video file path, '0' for webcam, or stream URL")
    parser.add_argument("--output", default="lane_rois.json", help="JSON file destination")
    args = parser.parse_args()

    # Open video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Error: Could not open video source {src}")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame from the video source.")
        cap.release()
        return

    frame_h, frame_w = frame.shape[:2]
    
    cv2.namedWindow("ROI Calibration Tool")
    cv2.setMouseCallback("ROI Calibration Tool", mouse_callback)

    print("\n--- ROI CALIBRATION INSTRUCTIONS ---")
    print("1. Click and drag left mouse button to draw a box around a lane.")
    print("2. Press 'c' to clear all defined zones and start fresh.")
    print("3. Press 's' to save the current boxes to JSON and exit.")
    print("4. Press 'q' to quit without saving.\n")

    while True:
        # Clone the pristine frame copy to draw fresh graphics onto
        display_frame = frame.copy()

        # Draw already saved ROIs
        for name, coords in rois.items():
            x1 = int(coords[0] * frame_w)
            y1 = int(coords[1] * frame_h)
            x2 = int(coords[2] * frame_w)
            y2 = int(coords[3] * frame_h)
            
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, name, (x1 + 5, y1 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the active live-dragged box if it exists
        if current_box:
            cv2.rectangle(display_frame, (current_box[0], current_box[1]), 
                          (current_box[2], current_box[3]), (255, 0, 0), 2)

        cv2.imshow("ROI Calibration Tool", display_frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("Exited calibration without saving changes.")
            break
        elif key == ord('c'):
            rois.clear()
            global current_lane_index
            current_lane_index = 1
            print("Cleared all calibrated regions.")
        elif key == ord('s'):
            if not rois:
                print("No ROIs calibrated. Nothing to save.")
                continue
            with open(args.output, 'w') as f:
                json.dump(rois, f, indent=4)
            print(f"\nSuccessfully saved configuration mapping to {args.output}!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()