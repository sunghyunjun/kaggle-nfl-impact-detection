import argparse
import os
import random

import cv2

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from effdet import create_model

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pytorch_lightning as pl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", metavar="DIR", help="path to dataset")
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()
    return args


def set_seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def make_batch(samples):
    image = torch.stack([sample["image"] for sample in samples])
    bboxes = [sample["bboxes"] for sample in samples]
    labels = [sample["labels"] for sample in samples]

    return {"image": image, "bboxes": bboxes, "labels": labels}


def get_train_transform():
    return A.Compose(
        [
            A.Resize(height=512, width=512, p=1.0),
            ToTensorV2(p=1.0),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]
        ),
    )


class HelmetDataset(Dataset):
    def __init__(self, df, image_ids, transform=None, root_dir="../dataset"):
        super().__init__()
        self.transform = transform
        self.df = df
        self.root_dir = root_dir
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
        # image_path = os.path.join("../dataset/images", image_id)
        image_path = os.path.join(self.root_dir, "images", image_id)
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


class HelmetDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = self.get_model()

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

        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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


def main(hparams):
    set_seed_everything(0)

    ROOT_DIR = hparams.root

    filepath = os.path.join(ROOT_DIR, "image_labels.csv")

    # image_labels = pd.read_csv("../dataset/image_labels.csv")
    image_labels = pd.read_csv(filepath)
    image_ids = image_labels.image.unique()

    valid_index = int(len(image_ids) * 0.1)
    train_image_ids = image_ids[:-valid_index]
    valid_image_ids = image_ids[-valid_index:]

    train_dataset = HelmetDataset(
        image_labels,
        train_image_ids,
        transform=get_train_transform(),
        root_dir=ROOT_DIR,
    )
    valid_dataset = HelmetDataset(
        image_labels, valid_image_ids, transform=None, root_dir=ROOT_DIR
    )

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=make_batch
    )

    helmet_detector = HelmetDetector()

    # trainer = pl.Trainer(max_epochs=1, gpus=hparams.gpus)
    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(helmet_detector, train_loader)


if __name__ == "__main__":
    args = get_args()
    main(args)