from argparse import ArgumentParser

import torch
import pytorch_lightning as pl

from effdet import create_model, create_model_from_config
from effdet.config import get_efficientdet_config
from effdet.bench import DetBenchPredict


class ImpactDetector(pl.LightningModule):
    def __init__(
        self, model_name="tf_efficientdet_d0", init_lr=1e-4, weight_decay=1e-5, **kwargs
    ):
        super().__init__()
        self.model_name = model_name
        self.model = self.get_model(self.model_name)
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

    def get_model(self, model_name="tf_efficientdet_d0"):
        model_name = model_name
        config = get_efficientdet_config(model_name)
        config.image_size = (512, 512)
        # config.anchor_scale = 1
        # config.norm_kwargs = dict(eps=0.001, momentum=0.01)
        model = create_model_from_config(
            config=config,
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