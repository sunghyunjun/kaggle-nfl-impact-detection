import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import ImpactDataset, ImpactSeqDataset
from utils import *


class ImpactDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../dataset",
        batch_size=32,
        num_workers=2,
        impactonly=False,
        oversample=False,
        seq_mode=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.impactonly = impactonly
        self.oversample = oversample
        self.seq_mode = seq_mode
        self.filepath = os.path.join(self.data_dir, "train_labels.csv")
        self.load_train_csv()

    def setup(self, stage=None):
        if "gs://" in self.data_dir:
            loader = gcs_pil_loader
        else:
            loader = pil_loader

        self.train_image_ids, self.valid_image_ids = self.make_image_ids_fold(
            n_splits=10
        )

        if self.seq_mode:
            self.train_dataset = ImpactSeqDataset(
                data_dir=self.data_dir,
                image_ids=self.train_image_ids,
                loader=loader,
                impactonly=self.impactonly,
                transform=self.get_train_transform(),
            )
            self.valid_dataset = ImpactSeqDataset(
                data_dir=self.data_dir,
                image_ids=self.valid_image_ids,
                loader=loader,
                impactonly=self.impactonly,
                transform=self.get_valid_transform(),
            )
        else:
            self.train_dataset = ImpactDataset(
                data_dir=self.data_dir,
                image_ids=self.train_image_ids,
                loader=loader,
                impactonly=self.impactonly,
                transform=self.get_train_transform(),
            )
            self.valid_dataset = ImpactDataset(
                data_dir=self.data_dir,
                image_ids=self.valid_image_ids,
                loader=loader,
                impactonly=self.impactonly,
                transform=self.get_valid_transform(),
            )

        if self.oversample:
            self.weight = self.make_sample_weight(self.train_image_ids)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                self.weight, len(self.weight)
            )

    def train_dataloader(self):
        if self.oversample:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                # shuffle=True,
                sampler=self.sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.make_batch,
            )
        else:
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.make_batch,
            )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.make_batch,
        )
        return valid_loader

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

        if self.impactonly:
            self.train_labels = self.train_labels[self.train_labels.impact == 1]

    def make_image_ids_fold(self, n_splits=10):
        train_video_list, valid_video_list = self.make_video_fold(n_splits=n_splits)
        train_image = self.train_labels[
            self.train_labels.video.isin(train_video_list)
        ].image
        train_image_ids = train_image.unique()
        valid_image = self.train_labels[
            self.train_labels.video.isin(valid_video_list)
        ].image
        valid_image_ids = valid_image.unique()
        return train_image_ids, valid_image_ids

    def make_video_fold(self, n_splits=10):
        def get_class(x):
            if x >= 22:
                y = 3
            elif x >= 19:
                y = 2
            elif x >= 16:
                y = 1
            else:
                y = 0
            return y

        df_video = (
            self.train_labels.groupby(["video"], as_index=False)
            .impact.sum()
            .rename(columns={"impact": "impact_sum"})
        )
        df_video["impact_sum_class"] = df_video.impact_sum.apply(get_class)
        skf = StratifiedKFold(n_splits=n_splits)
        train_fold = []
        valid_fold = []
        for train_index, valid_index in skf.split(
            df_video.video, df_video.impact_sum_class
        ):
            train_fold.append(train_index)
            valid_fold.append(valid_index)

        train_video_list = df_video.iloc[train_fold[0], :].video.tolist()
        valid_video_list = df_video.iloc[valid_fold[0], :].video.tolist()
        return train_video_list, valid_video_list

    def make_batch(self, samples):
        image = torch.stack([sample["image"] for sample in samples])
        bboxes = [sample["bboxes"] for sample in samples]
        labels = [sample["labels"] for sample in samples]
        image_id = [sample["image_id"] for sample in samples]

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "image_id": image_id,
        }

    def make_sample_weight(self, image_ids):
        # weight = [None] * len(image_ids)
        # self.weight = self.train_dataset.train_labels["weight_0.05"]
        weight = []
        # for index, image_id in enumerate(image_ids):
        #     if (
        #         1
        #         in self.train_labels[
        #             self.train_labels.image == image_id
        #         ].impact.to_numpy()
        #     ):
        #         weight.append(1)
        #     else:
        #         weight.append(0.05)
        for index, image_id in enumerate(image_ids):
            # weight_0.1
            # weight.append(self.train_dataset.train_labels[image_id][:, 7].max())

            # weight_0.05
            weight.append(self.train_dataset.train_labels[image_id][:, 8].max())
        return weight

    def get_train_transform(self):
        return A.Compose(
            [
                # A.RandomSizedCrop(
                #     min_max_height=(700, 700), height=1024, width=1024, p=0.5
                # ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.9
                ),
                # A.OneOf(
                #     [
                #         A.Blur(p=0.3),
                #         A.GaussNoise(p=0.3),
                #         A.IAASharpen(p=0.3),
                #     ],
                #     p=0.9,
                # ),
                A.OneOf(
                    [
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                    ],
                    p=0.5,
                ),
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )

    def get_valid_transform(self):
        return A.Compose(
            [
                A.Resize(height=512, width=512, p=1.0),
                ToTensorV2(p=1.0),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                min_area=0,
                min_visibility=0,
                label_fields=["labels"],
            ),
        )