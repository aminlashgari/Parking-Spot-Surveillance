# Importing libraries
import cv2 as cv
import argparse
import sys
import json
import os
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import time
import math
from shapely import Polygon, Point


class Utilops():
    # This class is made for timing the code, and dislaying the runtime on the screen
    def __init__(self):
        self.runtime_initial = time.perf_counter()
        # self.runtime = None

    def get_runtime(self, frame, total_time):
        self.frame = frame
        self.total_time = total_time
        self.runtime = math.floor(
            time.perf_counter() - (self.runtime_initial + self.total_time))
        cv.putText(self.frame, "Runtime: {}".format(self.runtime), (7, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv.LINE_AA)
        return

    def util_timer(self):
        self.timer = time.perf_counter()

        return self.timer

    def get_fps(self, frame, cap):
        self.cap = cap
        self.frame = frame
        fps = int(cap.get(cv.CAP_PROP_FPS))
        cv.putText(self.frame, "FPS: {}".format(fps), (7, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv.LINE_AA)

        return


class Keyops():
    __RESOLUTION = (640, 480)  # meant to be used internally
    # A class for keyboard operations in which different scenarios can be handled

    def __init__(self):
        self.quit = False
        self.verified = False
        self.pause = False
        self.save = False
        self.video_writer = True

        # Output video parameters
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out_path = os.path.join(os.getcwd(), 'demo')
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        self.out_save = cv.VideoWriter(
            os.path.join(out_path, 'output.avi'), fourcc, 30.0, self.__RESOLUTION)
        if not self.out_save.isOpened():
            self.video_writer = False
            print("VideoWriter: N/A")

    def keyboard_interrupt(self, key):
        # Pause
        if key == ord('p'):
            self.pause = True

        # Quit
        elif key == ord('q'):
            self.quit = True

    def user_verified(self, key):
        # Verifying
        if key == ord('t'):
            self.verified = True

    def save_video(self, key):
        # Saving
        if key == ord('s'):
            self.save = True

    def write_video(self, frame):
        # Writing video frames
        self.frame = frame
        if self.save and self.video_writer:
            frame_resized = cv.resize(self.frame, self.__RESOLUTION)
            self.out_save.write(frame_resized)

        return


def load_cords() -> dict:
    # loading json data file
    path = Path("Spot_Cords.json")

    if path.exists():
        with path.open("r") as f:
            # Coordinates of unoccupied parking spots
            cords = json.load(f)

    else:
        sys.exit("No such path!")

    return dict(cords)


def display_parking_msg(frame, parking_ids, parking_numbers):

    cv.putText(frame, "# of occupied spots: {}".format(len(
        parking_ids)), (7, 60), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv.LINE_AA)

    cv.putText(frame, "# of unoccupied spots: {}".format(parking_numbers - len(parking_ids)),
               (7, 80), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1, cv.LINE_AA)

    return frame


def spots_process(cords, frame, model, inference_cnt, parking_ids) -> np.array:

    # Runing inference on each frame
    results = model(frame, stream=True, verbose=False)
    for result in results:
        xyxys = result.boxes.xyxy.cpu().numpy()
        for xyxy in xyxys:
            # x1: top-left, y1: top-left, x2: bottom-right, y2: bottom-right
            x1, y1, x2, y2 = xyxy[:4]
            pt1 = (int(x1), int(y1))
            pt2 = (int(x2), int(y2))
            # center of each box
            center_x = (pt1[0] + pt2[0]) // 2
            center_y = (pt1[1] + pt2[1]) // 2
            center_pt = (center_x, center_y)
            box_pt = Point(center_pt)

            if inference_cnt % 120 == 0:  # Every two seconds, we have inference.

                for id, pts in cords.items():
                    if Polygon(pts).contains(box_pt):
                        if id not in parking_ids:
                            parking_ids.append(id)

            frame = display_parking_msg(frame, parking_ids, len(cords))

            cv.rectangle(frame, pt1, pt2, (255, 0, 0), 2)

    return frame


def test_process(cords, vid_src) -> None:

    # Loading a sample video from static camera to verfiy the previously saved parking coordinates(spots) on live feed
    RES: tuple = (1280, 720)
    cap = cv.VideoCapture(vid_src)
    if not cap.isOpened():
        sys.exit("Capture Not Opened!")

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    total_time: int = 0
    cnt: int = 0
    # Class instances
    key_handler = Keyops()
    runtime_counter = Utilops()
    while True:

        # runtime_counter = Utilops()
        isTrue, frame = cap.read()
        if not isTrue:
            print("No frame available!!")
            break

        frame = cv.resize(frame, RES)
        runtime_counter.get_fps(frame, cap)
        # Key operations
        key = cv.waitKey(10) & 0xFF
        key_handler.keyboard_interrupt(key)
        key_handler.user_verified(key)
        if key_handler.quit:
            break

        if key_handler.verified:
            cnt += 1
            # show the verification message for 120 frames(2 seconds)
            if cnt <= 120:
                cv.putText(frame, "Coordinates have been verified", (7, frame.shape[0] - 10),
                           cv.FONT_HERSHEY_COMPLEX, 0.75, (255, 0, 0), 1, cv.LINE_AA)

        if not key_handler.pause:
            runtime_counter.get_runtime(frame, total_time)

        if key_handler.pause:
            pause_start = runtime_counter.util_timer()
            cv.waitKey()
            pause_end = runtime_counter.util_timer()
            pause_time = (pause_end - pause_start)
            total_time += pause_time
            key_handler.pause = False

        color_idx = 0
        for k, _ in cords.items():
            color_idx += 15
            color_idx = min(color_idx, 255)
            # Depicting Polygons
            cv.fillPoly(frame, pts=np.array([cords[k]], dtype=np.int32),
                        color=(color_idx, 50, color_idx))

        cv.imshow("Test Prompt", frame)

    cap.release()
    cv.destroyAllWindows()

    return


def main_process(cords, vid_src, model_path) -> None:

    # loading the model(best.pt from my_model)
    model = YOLO(model_path, task='detect', verbose=False)

    # parameters
    RES: tuple = (1280, 720)
    total_time: int = 0
    cap = cv.VideoCapture(vid_src)
    if not cap.isOpened():
        sys.exit("Capture Not Opened!")

    inference_cnt: int = 0
    parking_ids: list = []
    # Class Instances
    key_handler = Keyops()
    runtime_counter = Utilops()
    while True:
        isTrue, frame = cap.read()
        if not isTrue:
            print("No frame available!!")
            break

        frame = cv.resize(frame, RES)
        runtime_counter.get_fps(frame, cap)

        # Key operations
        key = cv.waitKey(10) & 0xFF
        key_handler.keyboard_interrupt(key)
        key_handler.save_video(key)
        if key_handler.quit:
            break

        if not key_handler.pause:
            runtime_counter.get_runtime(frame, total_time)

        if key_handler.pause:
            pause_start = runtime_counter.util_timer()
            cv.waitKey()
            pause_end = runtime_counter.util_timer()
            pause_time = (pause_end - pause_start)
            total_time += pause_time
            key_handler.pause = False

        # processing parking spots
        inference_cnt += 1
        frame = spots_process(cords, frame, model, inference_cnt, parking_ids)

        if key_handler.save:
            key_handler.write_video(frame)

        cv.imshow("Main Window", frame)

    cap.release()
    cv.destroyAllWindows()


def main(vid_src, model_path) -> None:
    # Calling Functions
    cords = load_cords()
    for id, pts in cords.items():
        id = int(id)
        pts = np.array(pts, dtype=np.uint16)

    test_process(cords, vid_src)
    main_process(cords, vid_src, model_path)


if __name__ == "__main__":
    # comment the line below if the system is not optimized to run the program
    cv.setUseOptimized(onoff=True)

    # parsing user arugments
    parser = argparse.ArgumentParser(
        description="The main program in which the status of predefined parking spots will be updated."
    )
    parser.add_argument(
        "-s", "--src", help="Enter the path to a video file which is ought to be used soley for testing purporses.", required=True
    )
    parser.add_argument(
        "--model_path", help="Enter the path to a video file which is ought to be used soley for testing purporses.", required=True
    )
    args = parser.parse_args()
    vid_src, model_path = args.src, args.model_path

    # Checking wether an entered path is valid or not
    if not os.path.exists(vid_src) or not os.path.exists(model_path):
        sys.exit("Not a valid path!")

    main(vid_src, model_path)
