from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        print(f"Save to {name}.pkl")
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        print(f"Load {name}.pkl")
        return pickle.load(f)


def make_overlap_array(overlap):
    overlap_array_after = np.array(range(overlap)) + 1
    overlap_array_before = overlap_array_after[::-1] * -1
    overlap_array = np.concatenate([overlap_array_before, overlap_array_after], axis=0)
    return overlap_array


def make_pickle_from_labels(
    dataset_dir="../dataset", debug=False, impactonly=False, overlap=None
):
    train_labels_path = os.path.join(dataset_dir, "train_labels.csv")
    # train_labels = pd.read_csv("../dataset/train_labels.csv").fillna(0)
    train_labels = pd.read_csv(train_labels_path).fillna(0)

    train_labels["impact"] = train_labels["impact"].astype("int64")
    train_labels["confidence"] = train_labels["confidence"].astype("int64")
    train_labels["visibility"] = train_labels["visibility"].astype("int64")
    train_labels["image"] = (
        train_labels.video.str.replace(".mp4", "")
        + "_"
        + train_labels.frame.astype(str)
        + ".jpg"
    )
    train_labels.drop(train_labels[train_labels.frame == 0].index, inplace=True)

    if overlap is not None:
        overlap_array = make_overlap_array(overlap)
        train_labels_with_impact = train_labels[train_labels["impact"] > 0]
        pbar = tqdm(train_labels_with_impact[["video", "frame", "label"]].values)
        for row in pbar:
            pbar.set_description("Processing overlap array")
            frames = overlap_array + row[1]
            train_labels.loc[
                (train_labels["video"] == row[0])
                & (train_labels["frame"].isin(frames))
                & (train_labels["label"] == row[2]),
                "impact",
            ] = 1

    train_labels["weight_0.1"] = np.where(train_labels["impact"] == 1, 1, 0.1)
    train_labels["weight_0.05"] = np.where(train_labels["impact"] == 1, 1, 0.05)

    if impactonly:
        train_labels = train_labels[train_labels.impact == 1]

    image_ids = train_labels.image.unique()

    data = {}

    if debug:
        max_index = 100
    else:
        max_index = len(image_ids)

    pbar = tqdm(range(max_index))

    for index in pbar:
        image_id = image_ids[index]
        records = train_labels.loc[train_labels["image"] == image_id].copy()
        records["xmin"] = records["left"]
        records["ymin"] = records["top"]
        records["xmax"] = records["left"] + records["width"]
        records["ymax"] = records["top"] + records["height"]

        # exclude impactType. it has object dtype.
        boxes_labels = records[
            [
                "xmin",
                "ymin",
                "xmax",
                "ymax",
                "impact",
                "confidence",
                "visibility",
                "weight_0.1",
                "weight_0.05",
            ]
        ].values
        data[image_id] = boxes_labels

    filename = "train_labels"
    if impactonly:
        filename += "_impactonly"

    if overlap is not None:
        filename += "_overlap" + str(overlap)

    filename = os.path.join(dataset_dir, filename)
    save_obj(data, filename)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    # parser.add_argument("--debug", type=str, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--impactonly", action="store_true")
    parser.add_argument("--overlap", type=int)
    args = parser.parse_args()

    args_dict = vars(args)
    # make_pickle_from_labels(args.dataset_dir, args.debug, args.impactonly, args)
    make_pickle_from_labels(**args_dict)


if __name__ == "__main__":
    main()