# Importing Libraries
import cv2 as cv
import numpy as np
import argparse
import sys
import os
import time
import json
from pathlib import Path


def save_cords(spots):
    # Parameters
    save = False
    path = Path("Spot_Cords.json")
    cords_dict = {}

    for idx, pt in enumerate(spots):
        value = [[int(coord) for coord in pt[i]]
                 for i in range(spots.shape[1])]

        if np.all(pt != 0):
            cords_dict.update({idx: value})

    if path.exists():
        with path.open("r") as f:
            data = json.load(f)
            new_idx = len(data.keys())

            # Adding values for new indices, and merging two dicts
            for val in cords_dict.values():
                data.update({new_idx: val})
                new_idx += 1

            with path.open("w") as f:
                json.dump(data, f, indent=4)

        save = True

    else:
        with path.open("w") as f:
            json.dump(cords_dict, f, indent=4)

        data = cords_dict
        save = True

    return save, len(data.keys())


def final_process(cap, pts, RES):
    # Parameters
    cond = int(len(pts) / 4)
    spots = np.zeros((10, 4, 2), dtype=int)
    save_msg = False
    save_msg_start_t = 0
    save_msg_total_t = 1  # 1 second to show the message

    for j in range(cond):
        for k in range(spots.shape[1]):
            idx = j * 4 + k
            spots[j][k] = pts[idx]

    save, num = save_cords(spots)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("No More Frames!")
            break

        frame = cv.resize(frame, RES)

        color_idx = 0
        for poly in range(cond):
            color_idx += 15
            color_idx = min(color_idx, 255)

            # Depicting Polygons
            cv.fillPoly(frame, pts=[spots[poly]],
                        color=(color_idx, 50, color_idx))

        # Saving parking spots
        cv.putText(frame, "To save coordinates, press s", (7, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv.LINE_AA)

        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break

        elif key == ord('s') and save and num > 0:
            save_msg = True
            save_msg_start_t = time.time()
            save = False  # Pressing 's' is allowed only once in the exceution time

        if save_msg:
            if time.time() - save_msg_start_t < save_msg_total_t:
                cv.putText(frame, "Coordinates Saved ...", (7, 40),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv.LINE_AA)
                cv.putText(frame, f"Total Coordinated: {num}", (7, 62),
                           cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 1, cv.LINE_AA)
        else:
            save_msg = False

        cv.imshow("Parking Spots", frame)


def get_parking_coordinates(event, x, y, flags, param):
    cordPts = param

    if event == cv.EVENT_LBUTTONDBLCLK:
        cordPts.append((x, y))


def specify_parking_spots(cap):
    # Parameters
    cordPts = list()
    RES = (1280, 720)
    fixed_fps = 60
    frame_delay = 1.0 / fixed_fps
    winName = 'image'

    cv.namedWindow(winName)
    cv.setMouseCallback(winName, get_parking_coordinates, cordPts)

    while True:
        start_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("No More Frames!")
            break

        frame = cv.resize(frame, RES)
        for pt in cordPts:
            cv.circle(frame, pt, 5, (0, 255, 0), -1)
        cv.putText(frame, "Select Coordinates", (7, 20),
                   cv.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1, cv.LINE_AA)
        cv.imshow(winName, frame)

        # fixating the fps at 60
        elapsed = time.time() - start_time
        wait_time = max(1, int((frame_delay - elapsed) * 1000))

        # Key operations
        key = cv.waitKey(wait_time) & 0xFF
        # Quit:
        if key == ord('q'):
            break

        # Pause:
        elif key == ord("p"):
            cv.waitKey()

    cv.destroyWindow(winName)

    final_process(cap, cordPts, RES)


if __name__ == "__main__":

    # user argument for initial assessment
    parser = argparse.ArgumentParser(
        description="Specify the parking spots. (Determine the number of parking spots visible to the static camera)"
    )
    parser.add_argument(
        "--src", help="Path to the video file containing parking spots within the proper visible range", required=True
    )
    args = parser.parse_args()
    src_vid = args.src

    if not os.path.exists(src_vid):
        err_msg = f"Entered Path doesn't exist:\n{src_vid}"
        sys.exit(err_msg)

    capture = cv.VideoCapture(src_vid)
    if not capture.isOpened():
        sys.exit("Capture Not Opened!")

    specify_parking_spots(capture)

    capture.release()
    cv.destroyAllWindows()
