# Import necessary libraries
import os
import argparse
import sys
import random
import shutil
import math


def report_output(data_length, dirs, train_ratio) -> None:
    # Parameters
    total_data_num: int = data_length
    total_train_data: int = int(total_data_num * train_ratio)
    total_valid_data: int = math.ceil(total_data_num - total_train_data)
    # Total number of data
    print("Total Number of Images and Labels: {}".format(total_data_num))
    print("Total Number of Train Images and Labels: {}".format(total_train_data))
    print("Total Number of Validation Images and Labels: {}\n".format(total_valid_data))
    # displaying the (directory)folder structure
    [print("{}) Created Directory at: {}".format(i, dirs[i]))
     for i in range(len(dirs))]

    return None


def split_data(data_path: str, train_ratio: float) -> list:

    valid_range: str = "(0.05<= train_ratio<= 0.95)"
    if train_ratio > 0.95:
        sys.exit(
            "Not a valid train_ratio, should be lower than: {}\nValid Range: {}".format(train_ratio, valid_range))

    elif train_ratio < 0.05:
        sys.exit(
            "Not a valid train_ratio, should be higher than: {}\nValid Range: {}".format(train_ratio, valid_range))

    else:
        out_data_dir = os.path.join(os.getcwd(), "data")
        shutil.rmtree(out_data_dir, ignore_errors=True)

        # ignoring classes.txt
        data_folders = [os.path.join(data_path, file) for file in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, file))]

        images_dir = [os.path.join(data_folders[0], i)
                      for i in os.listdir(data_folders[0])]
        labels_dir = [os.path.join(data_folders[1], i)
                      for i in os.listdir(data_folders[1])]
        assert len(images_dir) == len(labels_dir)
        data = list(zip(images_dir, labels_dir))
        random.shuffle(data)
        images_dir_shuffled, labels_dir_shuffled = zip(*data)
        images_dir_shuffled, labels_dir_shuffled = list(
            images_dir_shuffled), list(labels_dir_shuffled)

        # folder structure
        # train: images(0), labels(1), validation: images(2), labels(3)
        dir_addresses: list[str] = []
        output_data = {"train": ["images", "labels"],
                       "validation": ["images", "labels"]}
        for key, value in output_data.items():
            sub_dir = os.path.join(out_data_dir, key)
            os.makedirs(sub_dir, exist_ok=True)

            for i in value:
                sub_sub_dir = os.path.join(sub_dir, i)
                os.makedirs(sub_sub_dir, exist_ok=True)
                dir_addresses.append(sub_sub_dir)

        # copying images and labels into the train and validation folders based on the train_ratio(split_ratio)
        for i in range(int(train_ratio * 150)):
            shutil.copy(src=images_dir_shuffled[i], dst=dir_addresses[0])
            shutil.copy(src=labels_dir_shuffled[i], dst=dir_addresses[1])

        valid_ratio = int(train_ratio * len(images_dir_shuffled))
        images_dir_shuffled = images_dir_shuffled[valid_ratio:]
        labels_dir_shuffled = labels_dir_shuffled[valid_ratio:]
        for i in range(len(images_dir_shuffled)):
            shutil.copy(src=images_dir_shuffled[i], dst=dir_addresses[2])
            shutil.copy(src=labels_dir_shuffled[i], dst=dir_addresses[3])

    return len(images_dir), dir_addresses


def main(data_path, train_ratio):
    try:
        train_ratio = float(train_ratio)
    except (TypeError, ValueError):
        sys.exit("Not a valid train_ratio")
    else:
        data_length, dirs = split_data(data_path, train_ratio)
        report_output(data_length, dirs, train_ratio)


if __name__ == "__main__":

    # user arguments
    parser = argparse.ArgumentParser(
        description="This program receives the ratio of train/validation data from a user argument, then splits the data in folders appropriate for yolo based models"
    )
    parser.add_argument(
        "-p", "--path", help="Path to the folder which contains the data(whole images)", required=True
    )
    parser.add_argument(
        "--train_ratio", help="The ratio for trainable images and labels (1- train_ratio) will be used for validation data", required=True
    )
    args = parser.parse_args()
    data_path, train_ratio = args.path, args.train_ratio

    if not os.path.exists(data_path):
        sys.exit("Folder doesn't exist")

    main(data_path, train_ratio)
