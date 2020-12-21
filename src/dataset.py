import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

import torch
from torch.utils.data import Dataset

from utils import *


class ImpactDataset(Dataset):
    def __init__(
        self,
        data_dir="../dataset",
        image_ids=None,
        loader=pil_loader,
        transform=None,
        bboxes_yxyx=True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.loader = loader
        self.transform = transform
        self.bboxes_yxyx = bboxes_yxyx
        self.filepath = os.path.join(self.data_dir, "train_labels.pkl")
        self.load_train_pickle()
        self.image_ids = image_ids
        if self.image_ids is None:
            self.image_ids = list(self.train_labels.keys())
        self.length = len(self.image_ids)

    def __getitem__(self, index: int):
        image, bboxes, labels = self.load_image_boxes_labels(index)
        sample = {"image": image, "bboxes": bboxes, "labels": labels}

        if self.transform:
            sample = self.transform(
                image=sample["image"], bboxes=sample["bboxes"], labels=sample["labels"]
            )

            sample["bboxes"] = torch.Tensor(sample["bboxes"])
            sample["labels"] = torch.IntTensor(sample["labels"])
        else:
            sample = A.Compose([ToTensorV2()])(
                image=sample["image"], bboxes=sample["bboxes"], labels=sample["labels"]
            )

            sample["bboxes"] = torch.Tensor(sample["bboxes"])
            sample["labels"] = torch.IntTensor(sample["labels"])

        if self.bboxes_yxyx:
            # yxyx: for efficientdet training
            sample["bboxes"][:, [0, 1, 2, 3]] = sample["bboxes"][:, [1, 0, 3, 2]]

        return sample

    def __len__(self) -> int:
        return self.length

    def load_image_boxes_labels(self, index):
        image_id = self.image_ids[index]
        video_name = image_id[0 : image_id.rfind("_")]
        image_path = os.path.join(
            self.data_dir, "images-from-video", "train", video_name, image_id
        )
        image = self.loader(image_path)
        image = np.array(image, dtype=np.float32)
        image /= 255.0

        records = self.train_labels[image_id]
        # xyxy: xmin, ymin, xmax, ymax
        bboxes = records[:, :4]

        # TODO: Temporary fix. Set labels 0, 1 to 1, 2
        labels = records[:, 4] + 1

        return image, bboxes, labels

    def load_train_pickle(self):
        if "gs://" in self.data_dir:
            self.train_labels = gcs_load_obj(self.filepath)
        else:
            self.train_labels = load_obj(self.filepath)

    def load_train_csv(self):
        self.train_labels = pd.read_csv(self.filepath).fillna(0)
        self.train_labels["impact"] = self.train_labels["impact"].astype("int64")
        self.train_labels["confidence"] = self.train_labels["confidence"].astype(
            "int64"
        )
        self.train_labels["visibility"] = self.train_labels["visibility"].astype(
            "int64"
        )
        self.train_labels["image"] = (
            self.train_labels.video.str.replace(".mp4", "")
            + "_"
            + self.train_labels.frame.astype(str)
            + ".jpg"
        )
        # drop data with frame=0
        self.train_labels.drop(
            self.train_labels[self.train_labels.frame == 0].index, inplace=True
        )