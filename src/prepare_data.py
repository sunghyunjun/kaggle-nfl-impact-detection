from argparse import ArgumentParser
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def make_images_from_single_video(video_name, video_dir, out_dir):
    video_path = os.path.join(video_dir, video_name)
    vidcap = cv2.VideoCapture(video_path)

    video_name = video_name.replace(".mp4", "")
    out_dir = os.path.join(out_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    frame = 0
    while True:
        ret, image = vidcap.read()
        if not ret:
            break
        frame += 1
        image_name = f"{video_name}_{frame}.png"
        image_path = os.path.join(out_dir, image_name)
        _ = cv2.imwrite(image_path, image)


def make_images(video_dir, out_dir):
    filenames = os.listdir(video_dir)
    pbar = tqdm(filenames)

    for filename in pbar:
        pbar.set_description(f"Video: {filename:30}")
        make_images_from_single_video(filename, video_dir, out_dir)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    args = parser.parse_args()

    # train video
    train_dir = os.path.join(args.dataset_dir, "train")
    train_out_dir = os.path.join(args.dataset_dir, "images-from-video", "train")
    make_images(train_dir, train_out_dir)

    # test video
    test_dir = os.path.join(args.dataset_dir, "test")
    test_out_dir = os.path.join(args.dataset_dir, "images-from-video", "test")
    make_images(test_dir, test_out_dir)


if __name__ == "__main__":
    main()