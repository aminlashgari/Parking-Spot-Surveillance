# Importing Libraries
import os
import sys
import argparse
import cv2 as cv


def main(dir_path: str, dir_save: str) -> None:
    # Parameters
    frame_cnt: int = 0
    dir_path: str = dir_path
    vid_std_format: list[str] = ['.mp4', '.mkv',
                                 '.wmv', '.avi', '.mov', '.GIF' '.gif', '.webm']

    for idx, vid_path in enumerate(os.listdir(dir_path)):
        vid = os.path.join(dir_path, vid_path)
        cap = cv.VideoCapture(vid)
        winName: str = f"Vid_{idx}"

        if not cap.isOpened():
            err_msg: str = f"Corrupt video file!\nor Not a valid format:\n{vid_std_format}"
            sys.exit(err_msg)

        while True:
            isTrue, frame = cap.read()

            if not isTrue:
                print("Can't read frames/or no more frames!")
                break

            # Key Operations
            key = cv.waitKey(10) & 0xFF
            if key == ord('q'):
                break

            # take a snapshot by pressing p
            if key == ord('p'):
                img_name = os.path.join(dir_save, f"img_{frame_cnt}.jpg")
                cv.imwrite(img_name, frame)
                frame_cnt += 1

            cv.imshow(winName, frame)

        cv.destroyWindow(winName)


if __name__ == "__main__":

    # argument parser
    parser = argparse.ArgumentParser(
        description="Enter the path to the folder containing your video files"
    )
    parser.add_argument(
        "-p", "--path", help="A valid path to your directory containing video files should be provided", required=True
    )
    parser.add_argument(
        "-s", "--save", help="A valid path to your directory containing your images ready to be used for labeling", required=False
    )
    args = parser.parse_args()
    dir_path, dir_save = args.path, str(args.save)

    # Checking for directory path of video files
    if not os.path.isdir(dir_path):
        sys.exit("Path: Non-existent!!")

    # Checking for directory path of image files, receiving user argument
    if not os.path.exists(dir_save) and not os.path.isdir(dir_save):
        new_dir_save = os.path.join(os.getcwd(), "raw_data")
        if not os.path.exists(new_dir_save):
            os.mkdir(new_dir_save)

    main(dir_path, dir_save=new_dir_save)
