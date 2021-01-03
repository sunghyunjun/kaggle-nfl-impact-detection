from argparse import ArgumentParser
import os
import random

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import ImpactDataModule
from models import ImpactDetector


def main():
    # ----------
    # seed
    # ----------
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--exp_name", default="Test")
    parser.add_argument("--model_name", default="tf_efficientdet_d0")
    parser.add_argument(
        "--dataset_dir", default="../dataset", metavar="DIR", help="path to dataset"
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--impactonly", action="store_true")
    parser.add_argument("--impactdefinitive", action="store_true")
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--seqmode", action="store_true")
    parser.add_argument("--fullsizeimage", action="store_true")
    parser.add_argument("--anchor_scale", default=4, type=int)
    parser = ImpactDetector.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ----------
    # data
    # ----------
    dm = ImpactDataModule(
        data_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        impactonly=args.impactonly,
        impactdefinitive=args.impactdefinitive,
        overlap=args.overlap,
        oversample=args.oversample,
        seqmode=args.seqmode,
        fullsizeimage=args.fullsizeimage,
    )

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
    # logger
    # ----------
    if not args.debug:
        neptune_logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project_name="sunghyun.jun/nfl-impact",
            experiment_name=args.exp_name,
            params={
                "model_name": args.model_name,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "init_lr": args.init_lr,
                "weight_decay": args.weight_decay,
            },
            tags=["pytorch-lightning"],
        )

        lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------
    # training
    # ----------
    if not args.debug:
        trainer = pl.Trainer.from_argparse_args(
            args, logger=neptune_logger, callbacks=[checkpoint_callback, lr_monitor]
        )
    else:
        args.max_epochs = 1
        args.limit_train_batches = 10
        args.limit_val_batches = 10
        trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(impact_detector, dm)

    # ----------
    # cli example
    # ----------
    # python train.py --exp_name=Test --dataset_dir=../dataset --batch_size=4 --num_workers=2 --max_epochs=1 --init_lr=1e-5 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10
    # python train.py --exp_name=Test --dataset_dir=gs://amiere-nfl-asia/dataset-jpg --batch_size=4 --num_workers=2 --max_epochs=1 --init_lr=1e-5 --weight_decay=1e-5 --limit_train_batches=10 --limit_val_batches=10


if __name__ == "__main__":
    main()