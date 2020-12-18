from argparse import ArgumentParser
import io
import os

# import pickle
import pickle5 as pickle
import random
from urllib.parse import urlparse

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from effdet import create_model
from effdet.bench import DetBenchPredict

from PIL import Image
from google.cloud import storage
from google.api_core.retry import Retry


def load_obj(path):
    with open(path, "rb") as f:
        print(f"Load {path}")
        return pickle.load(f)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


# def download_blob_byte(bucket_name, source_blob_name):
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(bucket_name)
#     blob = bucket.blob(source_blob_name)
#     result = blob.download_as_bytes()
#     return result


@Retry()
def gcs_pil_loader(uri):
    uri = urlparse(uri)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(uri.netloc)
    b = bucket.blob(uri.path[1:], chunk_size=None)
    image = Image.open(io.BytesIO(b.download_as_string()))
    return image.convert("RGB")


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class ImpactDataset(Dataset):
    def __init__(
        self, data_dir="../dataset", image_ids=None, loader=pil_loader, transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.loader = loader
        self.transform = transform
        self.filepath = os.path.join(self.data_dir, "train_labels.csv")
        self.load_train_csv()
        self.image_ids = image_ids
        if self.image_ids is None:
            self.image_ids = self.train_labels.image.unique()
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

        records = self.train_labels.loc[self.train_labels["image"] == image_id]
        boxes = records[["left", "width", "top", "height"]].values
        bboxes_xmin = boxes[:, 0]  # left
        bboxes_ymin = boxes[:, 2]  # top
        bboxes_xmax = boxes[:, 0] + boxes[:, 1]  # left + width
        bboxes_ymax = boxes[:, 2] + boxes[:, 3]  # top + height
        bboxes = np.array([bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax]).T
        labels = records["impact"].values

        return image, bboxes, labels

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


class ImpactDataset_V2(Dataset):
    def __init__(
        self, data_dir="../dataset", image_ids=None, loader=pil_loader, transform=None
    ):
        super().__init__()
        self.data_dir = data_dir
        self.loader = loader
        self.transform = transform
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
        # bboxes_xmin = records[:, 0]
        # bboxes_ymin = records[:, 1]
        # bboxes_xmax = records[:, 2]
        # bboxes_ymax = records[:, 3]
        # bboxes = np.array([bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax]).T
        bboxes = records[:, :4]

        # TODO: Temporary fix. Set labels 0, 1 to 1, 2
        labels = records[:, 4] + 1

        return image, bboxes, labels

    def load_train_pickle(self):
        self.train_labels = load_obj(self.filepath)


class ImpactDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../dataset", batch_size=32, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
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

        self.train_dataset = ImpactDataset_V2(
            data_dir=self.data_dir,
            image_ids=self.train_image_ids,
            loader=loader,
            transform=self.get_train_transform(),
        )
        self.valid_dataset = ImpactDataset_V2(
            data_dir=self.data_dir,
            image_ids=self.valid_image_ids,
            loader=loader,
            transform=self.get_valid_transform(),
        )

    def train_dataloader(self):
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

        return {"image": image, "bboxes": bboxes, "labels": labels}

    def get_train_transform(self):
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


class ImpactDetector(pl.LightningModule):
    def __init__(self, init_lr=1e-4, weight_decay=1e-5, **kwargs):
        super().__init__()
        self.model = self.get_model()
        self.predictor = DetBenchPredict(self.model.model)
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()

    def forward(self, x):
        return self.predictor.forward(x)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        boxes = batch["bboxes"]
        labels = batch["labels"]
        output = self.model(image, {"bbox": boxes, "cls": labels})
        loss = output["loss"]
        class_loss = output["class_loss"]
        box_loss = output["box_loss"]

        self.log("tr_loss", loss)
        self.log("tr_cls_loss", class_loss)
        self.log("tr_box_loss", box_loss)

        return loss

    # TODO: check target['img_scale'], target['img_size']
    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        boxes = batch["bboxes"]
        labels = batch["labels"]
        output = self.model(
            image, {"bbox": boxes, "cls": labels, "img_scale": None, "img_size": None}
        )
        loss = output["loss"]
        class_loss = output["class_loss"]
        box_loss = output["box_loss"]

        self.log("val_loss", loss)
        self.log("val_cls_loss", class_loss)
        self.log("val_box_loss", box_loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )
        return optimizer

    def get_model(self):
        model = create_model(
            model_name="tf_efficientdet_d0",
            bench_task="train",
            num_classes=2,
            pretrained=True,
            checkpoint_path="",
            checkpoint_ema=False,
            bench_labeler=True,
        )
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser


def visualize_single_sample(sample):
    sample_image = np.transpose(sample["image"][0], (1, 2, 0))
    img = sample_image.numpy().copy()
    sample_boxes = sample["bboxes"][0].numpy().astype(np.int16)
    sample_labels = sample["labels"][0].numpy()
    for i in range(len(sample_boxes)):
        cv2.rectangle(
            img,
            (sample_boxes[i, 0], sample_boxes[i, 1]),
            (sample_boxes[i, 2], sample_boxes[i, 3]),
            (0, 0, 0),
            thickness=3,
        )
        cv2.putText(
            img,
            str(sample_labels[i]),
            (sample_boxes[i, 0], max(0, sample_boxes[i, 1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness=1,
        )
    plt.imshow(img)
    plt.savefig("test_fig.png")


def test_datamodule(dm):
    dm.prepare_data()
    dm.setup()
    tr_it = iter(dm.train_dataloader())
    data = next(tr_it)
    visualize_single_sample(data)


def main():
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="Test")
    parser.add_argument(
        "--dataset_dir", default="../dataset", metavar="DIR", help="path to dataset"
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser = ImpactDetector.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ----------
    # logger
    # ----------
    neptune_logger = NeptuneLogger(
        api_key=os.environ["NEPTUNE_API_TOKEN"],
        project_name="sunghyun.jun/sandbox",
        experiment_name=args.exp_name,
        params={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "init_lr": args.init_lr,
            "weight_decay": args.weight_decay,
        },
        tags=["pytorch-lightning"],
    )

    # ----------
    # data
    # ----------
    dm = ImpactDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    # test_datamodule(dm)

    # ----------
    # model
    # ----------
    dict_args = vars(args)
    impact_detector = ImpactDetector(**dict_args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="impact-detector-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # ----------
    # training
    # ----------
    trainer = pl.Trainer.from_argparse_args(
        args, logger=neptune_logger, callbacks=[checkpoint_callback]
    )

    # ----------
    # for debug
    # ----------
    # args.max_epochs = 1
    # args.limit_train_batches = 10
    # args.limit_val_batches = 10
    # trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(impact_detector, dm)

    # ----------
    # cli example
    # ----------
    # python train.py --exp_name=Test --dataset_dir=../dataset --batch_size=4 --num_workers=2 --max_epochs=1 --init_lr=1e-3 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10
    # python train.py --exp_name=Test --dataset_dir=gs://amiere-nfl-asia/dataset --batch_size=4 --num_workers=2 --max_epochs=1 --init_lr=1e-3 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10


if __name__ == "__main__":
    main()