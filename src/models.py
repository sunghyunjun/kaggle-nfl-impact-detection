from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import pytorch_lightning as pl

from effdet import create_model, create_model_from_config
from effdet.config import get_efficientdet_config
from effdet.bench import DetBenchPredict


class ImpactDetector(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientdet_d0",
        init_lr=1e-4,
        weight_decay=1e-5,
        max_epochs=10,
        impactonly=False,
        seqmode=False,
        fullsizeimage=False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.impactonly = impactonly
        self.seqmode = seqmode
        self.fullsizeimage = fullsizeimage
        self.model = self.get_model(
            model_name=self.model_name,
            impactonly=self.impactonly,
            seqmode=self.seqmode,
            fullsizeimage=self.fullsizeimage,
        )
        self.predictor = DetBenchPredict(self.model.model)
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
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

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(
        self,
        model_name="tf_efficientdet_d0",
        impactonly=False,
        seqmode=False,
        fullsizeimage=False,
    ):
        model_name = model_name
        config = get_efficientdet_config(model_name)

        if fullsizeimage:
            # (H, W) size % 128 = 0 on each dim
            config.image_size = (640, 1280)
            # config.image_size = (1280, 1280)
        else:
            config.image_size = (512, 512)
        # config.anchor_scale = 1
        # config.norm_kwargs = dict(eps=0.001, momentum=0.01)

        if impactonly:
            num_classes = 1
        else:
            num_classes = 2

        model = create_model_from_config(
            config=config,
            bench_task="train",
            num_classes=num_classes,
            pretrained=True,
            checkpoint_path="",
            checkpoint_ema=False,
            bench_labeler=True,
        )

        if seqmode == True:
            model.model.backbone.conv_stem = timm.models.layers.Conv2dSame(
                9, 32, kernel_size=(3, 3), stride=(2, 2), bias=False
            )

        return model

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parser