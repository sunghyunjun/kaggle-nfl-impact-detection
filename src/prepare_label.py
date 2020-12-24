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


def make_pickle_from_labels(data_dir="../dataset", debug=False, impact_only=False):
    train_labels = pd.read_csv("../dataset/train_labels.csv").fillna(0)

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

    if impact_only:
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
            ["xmin", "ymin", "xmax", "ymax", "impact", "confidence", "visibility"]
        ].values
        data[image_id] = boxes_labels

    if impact_only:
        filename = os.path.join(data_dir, "train_labels_impact_only")
    else:
        filename = os.path.join(data_dir, "train_labels")
    save_obj(data, filename)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../dataset")
    # parser.add_argument("--debug", type=str, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--impactonly", action="store_true")
    args = parser.parse_args()

    make_pickle_from_labels(args.dataset_dir, args.debug, args.impactonly)


if __name__ == "__main__":
    main()