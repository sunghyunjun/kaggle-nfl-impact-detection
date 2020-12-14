from argparse import ArgumentParser
import io
import os
import random
from urllib.parse import urlparse

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd

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


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


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


class HelmetDataset(Dataset):
    def __init__(
        self,
        data_dir="../dataset",
        dataset_type="train",
        valid_split=0.1,
        loader=pil_loader,
        transform=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.valid_split = valid_split
        self.loader = loader
        self.transform = transform
        self.filepath = os.path.join(self.data_dir, "image_labels.csv")
        self.image_labels = pd.read_csv(self.filepath)
        self.image_ids = self.get_image_ids()
        self.length = len(self.image_ids)

    def get_image_ids(self):
        image_ids = self.image_labels.image.unique()
        valid_index = int(len(image_ids) * self.valid_split)
        train_image_ids = image_ids[:-valid_index]
        valid_image_ids = image_ids[-valid_index:]

        if self.dataset_type == "train":
            return train_image_ids
        elif self.dataset_type == "valid":
            return valid_image_ids

    def __getitem__(self, index: int):
        image, bboxes = self.load_image_boxes(index)
        labels = np.ones(bboxes.shape[0], dtype=np.int32)

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

    def load_image_boxes(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.data_dir, "images", image_id)
        image = self.loader(image_path)
        image = np.array(image, dtype=np.float32)
        image /= 255.0

        records = self.image_labels.loc[self.image_labels["image"] == image_id]
        boxes = records[["left", "width", "top", "height"]].values
        bboxes_xmin = boxes[:, 0]  # left
        bboxes_ymin = boxes[:, 2]  # top
        bboxes_xmax = boxes[:, 0] + boxes[:, 1]  # left + width
        bboxes_ymax = boxes[:, 2] + boxes[:, 3]  # top + height
        bboxes = np.array([bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax]).T

        return image, bboxes


class HelmetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../dataset", batch_size=32, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        valid_split = 0.1

        if "gs://" in self.data_dir:
            loader = gcs_pil_loader
        else:
            loader = pil_loader

        self.train_dataset = HelmetDataset(
            data_dir=self.data_dir,
            dataset_type="train",
            valid_split=valid_split,
            loader=loader,
            transform=self.get_train_transform(),
        )
        self.valid_dataset = HelmetDataset(
            data_dir=self.data_dir,
            dataset_type="valid",
            valid_split=valid_split,
            loader=loader,
            transform=self.get_valid_transform(),
        )

    # TODO: for custom batch, define pin_memory()
    # https://pytorch.org/docs/stable/data.html#memory-pinning
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


class HelmetDetector(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.model = self.get_model()
        self.predictor = DetBenchPredict(self.model.model)
        self.init_lr = args.init_lr
        self.weight_decay = args.weight_decay

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
            num_classes=1,
            pretrained=False,
            checkpoint_path="",
            checkpoint_ema=False,
            bench_labeler=True,
        )
        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-3)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser


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
    parser = HelmetDetector.add_model_specific_args(parser)
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
    dm = HelmetDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ----------
    # model
    # ----------
    helmet_detector = HelmetDetector(args)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="helmet-detector-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # ----------
    # training
    # ----------
    trainer = pl.Trainer.from_argparse_args(
        args, logger=neptune_logger, callbacks=[checkpoint_callback]
    )
    trainer.fit(helmet_detector, dm)

    # ----------
    # cli example
    # ----------
    # python train-helmet.py --exp_name=Test --dataset_dir=../dataset --batch_size=4 --num_workers=4 --max_epochs=1 --init_lr=1e-3 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10
    # python train-helmet.py --exp_name=Test --dataset_dir=gs://amiere-nfl-asia/dataset --batch_size=4 --num_workers=4 --max_epochs=1 --init_lr=1e-3 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10


if __name__ == "__main__":
    main()