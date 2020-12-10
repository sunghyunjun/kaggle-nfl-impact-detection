from argparse import ArgumentParser
import os
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
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


class HelmetDataset(Dataset):
    def __init__(self, df, image_ids, transform=None, data_dir="./dataset"):
        super().__init__()
        self.transform = transform
        self.df = df
        self.data_dir = data_dir
        self.image_ids = image_ids
        self.length = len(self.image_ids)

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
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).copy().astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        records = self.df.loc[self.df["image"] == image_id]
        boxes = records[["left", "width", "top", "height"]].values
        bboxes_xmin = boxes[:, 0]  # left
        bboxes_ymin = boxes[:, 2]  # top
        bboxes_xmax = boxes[:, 0] + boxes[:, 1]  # left + width
        bboxes_ymax = boxes[:, 2] + boxes[:, 3]  # top + height
        bboxes = np.array([bboxes_xmin, bboxes_ymin, bboxes_xmax, bboxes_ymax]).T

        return image, bboxes


class HelmetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./dataset", batch_size=32, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filepath = os.path.join(self.data_dir, "image_labels.csv")
        self.image_labels = pd.read_csv(self.filepath)
        self.image_ids = self.image_labels.image.unique()

    def setup(self, stage=None):
        valid_split = 0.1
        valid_index = int(len(self.image_ids) * valid_split)
        train_image_ids = self.image_ids[:-valid_index]
        valid_image_ids = self.image_ids[-valid_index:]

        self.train_dataset = HelmetDataset(
            self.image_labels,
            train_image_ids,
            transform=self.get_train_transform(),
            data_dir=self.data_dir,
        )
        self.valid_dataset = HelmetDataset(
            self.image_labels,
            valid_image_ids,
            transform=self.get_valid_transform(),
            data_dir=self.data_dir,
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
    def __init__(self, init_lr: float = 1e-3, weight_decay: float = 1e-5):
        super().__init__()
        self.model = self.get_model()
        self.init_lr = init_lr
        self.weight_decay = weight_decay

    def forward(self, x):
        pass

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


def main():
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--root", metavar="DIR", help="path to dataset")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--init_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--exp_name", default="Test")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ----------
    # logger
    # ----------
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMzdiNWQ2YWItZmYwZi00NTMyLWI0MTAtNDc3MTYxMTg0ZWMzIn0=",
        project_name="sunghyun.jun/sandbox",
        experiment_name=args.exp_name,
        params={"init_lr": args.init_lr},
        tags=["pytorch-lightning"],
    )

    # ----------
    # data
    # ----------
    dm = HelmetDataModule(
        data_dir=args.root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ----------
    # model
    # ----------
    helmet_detector = HelmetDetector(
        init_lr=args.init_lr, weight_decay=args.weight_decay
    )
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
    # python train-helmet.py --root=../dataset --batch_size=4 --num_workers=4 --max_epochs=1 --limit_train_batches=10 --limit_val_batches=10


if __name__ == "__main__":
    main()