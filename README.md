# kaggle-nfl-impact-detection

Code for 56th place solution in [Kaggle NFL 1st and Future - Impact Detection Competition][kaggle_link].

[kaggle_link]: https://www.kaggle.com/c/nfl-impact-detection

## Overview

This solution is One-Stage model using [EfficientDet][efficientdet_link].

[efficientdet_link]: https://github.com/rwightman/efficientdet-pytorch

The model detect Impact of Helmet. (Red box at [413, 340])

Using tf_efficientdet_d6 and post-processing, private score is 0.2117

![helmetimpact](https://user-images.githubusercontent.com/61550593/104441857-991bd600-55d7-11eb-8866-8fb133c0621d.png)

## Prepare dataset

Download Competition data [on the Kaggle competition page][kaggle_dataset_link]

[kaggle_dataset_link]: https://www.kaggle.com/c/nfl-impact-detection/data

```bash
kaggle competitions download -c nfl-impact-detection
```

Create image files from train and test video files.

```bash
python prepare_data.py --dataset_dir=./dataset
```

Create train label object file.

```bash
python prepare_label.py --dataset_dir=./dataset \
                        --impactonly \
```

- Flags

--impactonly

--impactdefinitive

--overlap

## Train

You can use Pytorch-Lightnig Trainer flags

```bash
python train.py --dataset_dir=./dataset \
                --exp_name=your_exp_name \
                --model_name=tf_efficientdet_d6 \
                --impactonly \
                --fullsizeimage \
                --anchor_scale=1 \
                --fold_index=1 \
                --precision=16 \
                --gpus=1 \
                --batch_size=2 \
                --num_workers=4 \
                --init_lr=4.0e-5 \
                --weight_decay=1.0e-4 \
                --max_epochs=30 \
                --default_root_dir=./mount_checkpoint \
                --progress_bar_refresh_rate=30 \
```

- Flags

--impactonly

--impactdefinitive

--overlap

--seqmode

--fullsizeimage

--fold_index

--anchor_scale

## Evaluate

```bash
python evaluate.py --dataset_dir={DATA_DIR} \
                   --checkpoint={CHECKPOINT} \
                   --exp_name={EXP_NAME} \
                   --batch_size=4 \
                   --num_workers=2 \
                   --impactonly \
                   --fullsizeimage \
                   --impactdefinitive \
                   --checkpoint2={CHECKPOINT2} \
                   --checkpoint3={CHECKPOINT3} \
```

- Flags

--impactonly

--impactdefinitive

--overlap

--seqmode

--fullsizeimage

--tta

--fold_index
