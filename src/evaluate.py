from argparse import ArgumentParser

import copy
import os
import random

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import torch
from torchvision.ops.boxes import batched_nms, nms

from tqdm import tqdm

import pytorch_lightning as pl
import neptune

from dataset import *
from datamodule import *
from models import *
from utils import *


def prediction_filtered(boxes, scores, labels, threshold=0.2):
    len_pred = boxes[0].shape[0]
    batch_size = len(boxes)

    boxes_batch = []
    labels_batch = []
    count = 0
    for i in range(batch_size):
        boxes_filtered = []
        labels_filtered = []
        for j in range(len_pred):
            if scores[i][j] >= threshold:
                count += 1
                boxes_filtered.append(boxes[i][j, :4])
                labels_filtered.append(labels[i][j])

        boxes_filtered = np.asarray(boxes_filtered)
        labels_filtered = np.asarray(labels_filtered)
        boxes_batch.append(boxes_filtered)
        labels_batch.append(labels_filtered)
    print(f"{count} of {batch_size * len_pred}")
    return boxes_batch, labels_batch


def iou(bbox1, bbox2):
    """
    Competition metric, F1 score.
    Original code from https://www.kaggle.com/nvnnghia/evaluation-metrics
    For more information about it:
    https://www.kaggle.com/c/nfl-impact-detection/discussion/197672#1095339
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes):
    """
    Competition metric, F1 score.
    Original code from https://www.kaggle.com/nvnnghia/evaluation-metrics
    For more information about it:
    https://www.kaggle.com/c/nfl-impact-detection/discussion/197672#1095339
    """
    cost_matix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0] - box2[0])
            if dist > 4:
                continue
            iou_score = iou(box1[1:], box2[1:])

            if iou_score < 0.35:
                continue
            else:
                cost_matix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matix)
    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1
    return tp, fp, fn


def make_prediction(model, valid_loader, device, debug=False, fullsizeimage=False):
    if fullsizeimage:
        resize_height = 640
        resize_width = 1280
    else:
        resize_height = 512
        resize_width = 512

    image_ids = []
    boxes_gt = []
    labels_gt = []
    boxes_pred = []
    scores_pred = []
    labels_pred = []

    for index, batch in enumerate(tqdm(valid_loader)):
        if debug and index > 3:
            break

        image = batch["image"]
        boxes_ = batch["bboxes"]
        labels_ = batch["labels"]
        image_id = batch["image_id"]

        for i in range(len(image_id)):
            boxes_[i] = boxes_[i][:, [1, 0, 3, 2]].numpy()

            boxes_[i][:, 0] = boxes_[i][:, 0] * 1280 / resize_width
            boxes_[i][:, 1] = boxes_[i][:, 1] * 720 / resize_height
            boxes_[i][:, 2] = boxes_[i][:, 2] * 1280 / resize_width
            boxes_[i][:, 3] = boxes_[i][:, 3] * 720 / resize_height

            boxes_[i] = boxes_[i].astype(np.int32)

            boxes_[i][:, 0] = boxes_[i][:, 0].clip(min=0, max=1280 - 1)
            boxes_[i][:, 1] = boxes_[i][:, 1].clip(min=0, max=720 - 1)
            boxes_[i][:, 2] = boxes_[i][:, 2].clip(min=0, max=1280 - 1)
            boxes_[i][:, 3] = boxes_[i][:, 3].clip(min=0, max=720 - 1)

            labels_[i] = labels_[i].numpy().astype(np.int32)

        prediction = model(image.to(device))
        boxes = prediction[:, :, :4].detach().cpu().numpy()
        scores = prediction[:, :, 4].detach().cpu().numpy()
        labels = prediction[:, :, 5].detach().cpu().numpy().astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0] * 1280 / resize_width
        boxes[:, :, 1] = boxes[:, :, 1] * 720 / resize_height
        boxes[:, :, 2] = boxes[:, :, 2] * 1280 / resize_width
        boxes[:, :, 3] = boxes[:, :, 3] * 720 / resize_height

        boxes = boxes.astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0].clip(min=0, max=1280 - 1)
        boxes[:, :, 1] = boxes[:, :, 1].clip(min=0, max=720 - 1)
        boxes[:, :, 2] = boxes[:, :, 2].clip(min=0, max=1280 - 1)
        boxes[:, :, 3] = boxes[:, :, 3].clip(min=0, max=720 - 1)

        image_ids.extend(image_id)
        boxes_gt.extend(boxes_)
        labels_gt.extend(labels_)
        boxes_pred.extend(boxes)
        scores_pred.extend(scores)
        labels_pred.extend(labels)

    return image_ids, boxes_gt, labels_gt, boxes_pred, scores_pred, labels_pred


def tensor_random_brightness_contrast(image, brightness_limit=0.2, contrast_limit=0.2):
    alpha = 1.0 + random.uniform(contrast_limit, contrast_limit)
    beta = 0.0 + random.uniform(brightness_limit, brightness_limit)

    image *= alpha
    image += beta * torch.mean(image)
    image_clipped = torch.clip(image, 0, 1)

    return image_clipped


def make_prediction_tta(model, valid_loader, device, debug=False, fullsizeimage=False):
    if fullsizeimage:
        resize_height = 640
        resize_width = 1280
    else:
        resize_height = 512
        resize_width = 512

    image_ids = []
    boxes_gt = []
    labels_gt = []
    boxes_pred = []
    scores_pred = []
    labels_pred = []

    for index, batch in enumerate(tqdm(valid_loader)):
        if debug and index > 3:
            break

        image = batch["image"]
        boxes_ = batch["bboxes"]
        labels_ = batch["labels"]
        image_id = batch["image_id"]

        batch_size = len(image_id)

        for i in range(batch_size):
            boxes_[i] = boxes_[i][:, [1, 0, 3, 2]].numpy()

            boxes_[i][:, 0] = boxes_[i][:, 0] * 1280 / resize_width
            boxes_[i][:, 1] = boxes_[i][:, 1] * 720 / resize_height
            boxes_[i][:, 2] = boxes_[i][:, 2] * 1280 / resize_width
            boxes_[i][:, 3] = boxes_[i][:, 3] * 720 / resize_height

            boxes_[i] = boxes_[i].astype(np.int32)

            boxes_[i][:, 0] = boxes_[i][:, 0].clip(min=0, max=1280 - 1)
            boxes_[i][:, 1] = boxes_[i][:, 1].clip(min=0, max=720 - 1)
            boxes_[i][:, 2] = boxes_[i][:, 2].clip(min=0, max=1280 - 1)
            boxes_[i][:, 3] = boxes_[i][:, 3].clip(min=0, max=720 - 1)

            labels_[i] = labels_[i].numpy().astype(np.int32)

        prediction = model(image.to(device))

        # image_aug = image + torch.rand(image.shape) * 0.01
        image_aug = tensor_random_brightness_contrast(image)

        prediction_aug = model(image_aug.to(device))

        boxes_org = prediction[:, :, :4]
        scores_org = prediction[:, :, 4]
        labels_org = prediction[:, :, 5]

        boxes_aug = prediction_aug[:, :, :4]
        scores_aug = prediction_aug[:, :, 4]
        labels_aug = prediction_aug[:, :, 5]

        boxes_cat = torch.cat((boxes_org, boxes_aug), axis=1)
        scores_cat = torch.cat((scores_org, scores_aug), axis=1)
        labels_cat = torch.cat((labels_org, labels_aug), axis=1)

        boxes_nms = torch.zeros((batch_size, 100, 4)).to(device)
        scores_nms = torch.zeros((batch_size, 100)).to(device)
        labels_nms = torch.zeros((batch_size, 100)).to(device)

        for j in range(batch_size):
            top_detection_idx = batched_nms(
                boxes_cat[j, :, :],
                scores_cat[j, :],
                labels_cat[j, :],
                iou_threshold=0.5,
            )
            # print(scores_cat[j, :].shape, labels_cat[j, :].shape)
            # print(top_detection_idx.shape)
            top_detection_idx = top_detection_idx[:100]
            # print(top_detection_idx.shape)
            # print(top_detection_idx)

            boxes_nms[j, :, :] = torch.cat(
                (
                    boxes_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx), 4)).to(device),
                ),
            )
            scores_nms[j, :] = torch.cat(
                (
                    scores_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )
            labels_nms[j, :] = torch.cat(
                (
                    labels_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )

        boxes = boxes_nms.detach().cpu().numpy()
        scores = scores_nms.detach().cpu().numpy()
        labels = labels_nms.detach().cpu().numpy().astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0] * 1280 / resize_width
        boxes[:, :, 1] = boxes[:, :, 1] * 720 / resize_height
        boxes[:, :, 2] = boxes[:, :, 2] * 1280 / resize_width
        boxes[:, :, 3] = boxes[:, :, 3] * 720 / resize_height

        boxes = boxes.astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0].clip(min=0, max=1280 - 1)
        boxes[:, :, 1] = boxes[:, :, 1].clip(min=0, max=720 - 1)
        boxes[:, :, 2] = boxes[:, :, 2].clip(min=0, max=1280 - 1)
        boxes[:, :, 3] = boxes[:, :, 3].clip(min=0, max=720 - 1)

        image_ids.extend(image_id)
        boxes_gt.extend(boxes_)
        labels_gt.extend(labels_)
        boxes_pred.extend(boxes)
        scores_pred.extend(scores)
        labels_pred.extend(labels)

    return image_ids, boxes_gt, labels_gt, boxes_pred, scores_pred, labels_pred


def make_prediction_ensemble(
    model_1, model_2, valid_loader, device, debug=False, fullsizeimage=False
):
    if fullsizeimage:
        resize_height = 640
        resize_width = 1280
    else:
        resize_height = 512
        resize_width = 512

    image_ids = []
    boxes_gt = []
    labels_gt = []
    boxes_pred = []
    scores_pred = []
    labels_pred = []

    for index, batch in enumerate(tqdm(valid_loader)):
        if debug and index > 3:
            break

        image = batch["image"]
        boxes_ = batch["bboxes"]
        labels_ = batch["labels"]
        image_id = batch["image_id"]

        batch_size = len(image_id)

        for i in range(batch_size):
            boxes_[i] = boxes_[i][:, [1, 0, 3, 2]].numpy()

            boxes_[i][:, 0] = boxes_[i][:, 0] * 1280 / resize_width
            boxes_[i][:, 1] = boxes_[i][:, 1] * 720 / resize_height
            boxes_[i][:, 2] = boxes_[i][:, 2] * 1280 / resize_width
            boxes_[i][:, 3] = boxes_[i][:, 3] * 720 / resize_height

            boxes_[i] = boxes_[i].astype(np.int32)

            boxes_[i][:, 0] = boxes_[i][:, 0].clip(min=0, max=1280 - 1)
            boxes_[i][:, 1] = boxes_[i][:, 1].clip(min=0, max=720 - 1)
            boxes_[i][:, 2] = boxes_[i][:, 2].clip(min=0, max=1280 - 1)
            boxes_[i][:, 3] = boxes_[i][:, 3].clip(min=0, max=720 - 1)

            labels_[i] = labels_[i].numpy().astype(np.int32)

        prediction = model_1(image.to(device))

        prediction_2 = model_2(image.to(device))

        boxes_org = prediction[:, :, :4]
        scores_org = prediction[:, :, 4]
        labels_org = prediction[:, :, 5]

        boxes_2 = prediction_2[:, :, :4]
        scores_2 = prediction_2[:, :, 4]
        labels_2 = prediction_2[:, :, 5]

        boxes_cat = torch.cat((boxes_org, boxes_2), axis=1)
        scores_cat = torch.cat((scores_org, scores_2), axis=1)
        labels_cat = torch.cat((labels_org, labels_2), axis=1)

        boxes_nms = torch.zeros((batch_size, 100, 4)).to(device)
        scores_nms = torch.zeros((batch_size, 100)).to(device)
        labels_nms = torch.zeros((batch_size, 100)).to(device)

        for j in range(batch_size):
            top_detection_idx = batched_nms(
                boxes_cat[j, :, :],
                scores_cat[j, :],
                labels_cat[j, :],
                iou_threshold=0.5,
            )
            # print(scores_cat[j, :].shape, labels_cat[j, :].shape)
            # print(top_detection_idx.shape)
            top_detection_idx = top_detection_idx[:100]
            # print(top_detection_idx.shape)
            # print(top_detection_idx)

            boxes_nms[j, :, :] = torch.cat(
                (
                    boxes_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx), 4)).to(device),
                ),
            )
            scores_nms[j, :] = torch.cat(
                (
                    scores_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )
            labels_nms[j, :] = torch.cat(
                (
                    labels_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )

        boxes = boxes_nms.detach().cpu().numpy()
        scores = scores_nms.detach().cpu().numpy()
        labels = labels_nms.detach().cpu().numpy().astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0] * 1280 / resize_width
        boxes[:, :, 1] = boxes[:, :, 1] * 720 / resize_height
        boxes[:, :, 2] = boxes[:, :, 2] * 1280 / resize_width
        boxes[:, :, 3] = boxes[:, :, 3] * 720 / resize_height

        boxes = boxes.astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0].clip(min=0, max=1280 - 1)
        boxes[:, :, 1] = boxes[:, :, 1].clip(min=0, max=720 - 1)
        boxes[:, :, 2] = boxes[:, :, 2].clip(min=0, max=1280 - 1)
        boxes[:, :, 3] = boxes[:, :, 3].clip(min=0, max=720 - 1)

        image_ids.extend(image_id)
        boxes_gt.extend(boxes_)
        labels_gt.extend(labels_)
        boxes_pred.extend(boxes)
        scores_pred.extend(scores)
        labels_pred.extend(labels)

    return image_ids, boxes_gt, labels_gt, boxes_pred, scores_pred, labels_pred


def make_prediction_ensemble_v2(
    model_1, model_2, model_3, valid_loader, device, debug=False, fullsizeimage=False
):
    if fullsizeimage:
        resize_height = 640
        resize_width = 1280
    else:
        resize_height = 512
        resize_width = 512

    image_ids = []
    boxes_gt = []
    labels_gt = []
    boxes_pred = []
    scores_pred = []
    labels_pred = []

    for index, batch in enumerate(tqdm(valid_loader)):
        if debug and index > 3:
            break

        image = batch["image"]
        boxes_ = batch["bboxes"]
        labels_ = batch["labels"]
        image_id = batch["image_id"]

        batch_size = len(image_id)

        for i in range(batch_size):
            boxes_[i] = boxes_[i][:, [1, 0, 3, 2]].numpy()

            boxes_[i][:, 0] = boxes_[i][:, 0] * 1280 / resize_width
            boxes_[i][:, 1] = boxes_[i][:, 1] * 720 / resize_height
            boxes_[i][:, 2] = boxes_[i][:, 2] * 1280 / resize_width
            boxes_[i][:, 3] = boxes_[i][:, 3] * 720 / resize_height

            boxes_[i] = boxes_[i].astype(np.int32)

            boxes_[i][:, 0] = boxes_[i][:, 0].clip(min=0, max=1280 - 1)
            boxes_[i][:, 1] = boxes_[i][:, 1].clip(min=0, max=720 - 1)
            boxes_[i][:, 2] = boxes_[i][:, 2].clip(min=0, max=1280 - 1)
            boxes_[i][:, 3] = boxes_[i][:, 3].clip(min=0, max=720 - 1)

            labels_[i] = labels_[i].numpy().astype(np.int32)

        prediction = model_1(image.to(device))

        prediction_2 = model_2(image.to(device))

        prediction_3 = model_3(image.to(device))

        boxes_org = prediction[:, :, :4]
        scores_org = prediction[:, :, 4]
        labels_org = prediction[:, :, 5]

        boxes_2 = prediction_2[:, :, :4]
        scores_2 = prediction_2[:, :, 4]
        labels_2 = prediction_2[:, :, 5]

        boxes_3 = prediction_3[:, :, :4]
        scores_3 = prediction_3[:, :, 4]
        labels_3 = prediction_3[:, :, 5]

        boxes_cat = torch.cat((boxes_org, boxes_2, boxes_3), axis=1)
        scores_cat = torch.cat((scores_org, scores_2, scores_3), axis=1)
        labels_cat = torch.cat((labels_org, labels_2, labels_3), axis=1)

        boxes_nms = torch.zeros((batch_size, 100, 4)).to(device)
        scores_nms = torch.zeros((batch_size, 100)).to(device)
        labels_nms = torch.zeros((batch_size, 100)).to(device)

        for j in range(batch_size):
            top_detection_idx = batched_nms(
                boxes_cat[j, :, :],
                scores_cat[j, :],
                labels_cat[j, :],
                iou_threshold=0.5,
            )
            # print(scores_cat[j, :].shape, labels_cat[j, :].shape)
            # print(top_detection_idx.shape)
            top_detection_idx = top_detection_idx[:100]
            # print(top_detection_idx.shape)
            # print(top_detection_idx)

            boxes_nms[j, :, :] = torch.cat(
                (
                    boxes_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx), 4)).to(device),
                ),
            )
            scores_nms[j, :] = torch.cat(
                (
                    scores_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )
            labels_nms[j, :] = torch.cat(
                (
                    labels_cat[j, top_detection_idx],
                    torch.zeros((100 - len(top_detection_idx))).to(device),
                )
            )

        boxes = boxes_nms.detach().cpu().numpy()
        scores = scores_nms.detach().cpu().numpy()
        labels = labels_nms.detach().cpu().numpy().astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0] * 1280 / resize_width
        boxes[:, :, 1] = boxes[:, :, 1] * 720 / resize_height
        boxes[:, :, 2] = boxes[:, :, 2] * 1280 / resize_width
        boxes[:, :, 3] = boxes[:, :, 3] * 720 / resize_height

        boxes = boxes.astype(np.int32)

        boxes[:, :, 0] = boxes[:, :, 0].clip(min=0, max=1280 - 1)
        boxes[:, :, 1] = boxes[:, :, 1].clip(min=0, max=720 - 1)
        boxes[:, :, 2] = boxes[:, :, 2].clip(min=0, max=1280 - 1)
        boxes[:, :, 3] = boxes[:, :, 3].clip(min=0, max=720 - 1)

        image_ids.extend(image_id)
        boxes_gt.extend(boxes_)
        labels_gt.extend(labels_)
        boxes_pred.extend(boxes)
        scores_pred.extend(scores)
        labels_pred.extend(labels)

    return image_ids, boxes_gt, labels_gt, boxes_pred, scores_pred, labels_pred


def make_gt_df(image_ids, boxes_gt, labels_gt):
    boxes_gt_df = pd.DataFrame(
        np.concatenate(boxes_gt), columns=["xmin", "ymin", "xmax", "ymax"]
    )
    labels_gt_df = pd.DataFrame(np.concatenate(labels_gt), columns=["label"])

    image_ids_gt = []
    for i in range(len(image_ids)):
        image_ids_gt += [image_ids[i]] * len(boxes_gt[i])

    image_ids_gt_df = pd.DataFrame({"image_name": image_ids_gt})
    gt_df = pd.concat([image_ids_gt_df, boxes_gt_df, labels_gt_df], axis=1)

    gt_df["gameKey"] = gt_df.image_name.str.split("_").str[0].astype(int)
    gt_df["playID"] = gt_df.image_name.str.split("_").str[1].astype(int)
    gt_df["view"] = gt_df.image_name.str.split("_").str[2]
    gt_df["frame"] = (
        gt_df.image_name.str.split("_").str[3].str.replace(".jpg", "").astype(int)
    )
    gt_df["video"] = gt_df.image_name.str.rsplit("_", 1).str[0] + ".mp4"
    gt_df = gt_df[
        [
            "gameKey",
            "playID",
            "view",
            "video",
            "frame",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "label",
            "image_name",
        ]
    ]

    gt_df = gt_df[gt_df.label == 2]

    return gt_df


def make_pred_df(image_ids, boxes_pred, scores_pred, labels_pred, impact_class):
    boxes_pred_df = pd.DataFrame(
        np.concatenate(boxes_pred), columns=["xmin", "ymin", "xmax", "ymax"]
    )
    scores_pred_df = pd.DataFrame(np.concatenate(scores_pred), columns=["score"])
    labels_pred_df = pd.DataFrame(np.concatenate(labels_pred), columns=["label"])

    image_ids_pred = []
    for i in range(len(image_ids)):
        image_ids_pred += [image_ids[i]] * len(boxes_pred[i])

    image_ids_pred_df = pd.DataFrame({"image_name": image_ids_pred})
    pred_df = pd.concat(
        [image_ids_pred_df, boxes_pred_df, scores_pred_df, labels_pred_df], axis=1
    )

    pred_df["gameKey"] = pred_df.image_name.str.split("_").str[0].astype(int)
    pred_df["playID"] = pred_df.image_name.str.split("_").str[1].astype(int)
    pred_df["view"] = pred_df.image_name.str.split("_").str[2]
    pred_df["frame"] = (
        pred_df.image_name.str.split("_").str[3].str.replace(".jpg", "").astype(int)
    )
    pred_df["video"] = pred_df.image_name.str.rsplit("_", 1).str[0] + ".mp4"
    pred_df = pred_df[
        [
            "gameKey",
            "playID",
            "view",
            "video",
            "frame",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "score",
            "label",
            "image_name",
        ]
    ]

    pred_df_raw = pred_df[pred_df.label == impact_class]
    return pred_df_raw


def calc_metric(gt_df, pred_df):
    tp_list, fp_list, fn_list = [], [], []
    for video in set(gt_df["video"]):
        pred_boxes = pred_df[pred_df.video == video][
            ["frame", "xmin", "ymin", "xmax", "ymax"]
        ].to_numpy()
        gt_boxes = gt_df[gt_df.video == video][
            ["frame", "xmin", "ymin", "xmax", "ymax"]
        ].to_numpy()
        tp, fp, fn = precision_calc(gt_boxes, pred_boxes)
        tp_list.append(tp)
        fp_list.append(fp)
        fn_list.append(fn)

    tp = np.sum(tp_list)
    fp = np.sum(fp_list)
    fn = np.sum(fn_list)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print(
        f"TP: {tp}, FP: {fp}, FN: {fn}, PRECISION: {precision:.4f}, RECALL: {recall:.4f}, F1 SCORE: {f1_score:.4f}"
    )
    return tp, fp, fn, precision, recall, f1_score


def make_pred_filtered_df(pred_df, iou_threshold=0.5):
    """
    Post-Processing. Filter bbox over +- 4 frames.
    """
    dropIDX = []
    for keys in pred_df.groupby("video").size().to_dict().keys():
        tmp_df = pred_df.query("video == @keys")
        for base_index, base_row in tmp_df.iterrows():
            base_bbox = base_row[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
            base_video = base_row["video"]
            base_frame = base_row["frame"]
            for index, row in tmp_df.iterrows():
                frame = row["frame"]
                # if frame > base_frame:
                if base_frame < frame <= base_frame + 4:
                    bbox = row[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
                    iou_score = iou(base_bbox, bbox)
                    if iou_score > iou_threshold:
                        dropIDX.append(index)
                        # print(base_video, base_frame, frame, iou_score, len(set(dropIDX)))
    dropIDX = set(dropIDX)

    pred_filtered_df = pred_df.drop(index=dropIDX).reset_index(drop=True)
    return pred_filtered_df


def make_pred_bothzone_filtered_df(pred_df):
    """
    Post-Processing.
    Original code from https://www.kaggle.com/artkulak/2class-object-detection-inference-with-filtering
    Remove all boxes which are not present in both Sidezone and Endzone views.

    """
    dropIDX = []
    for keys in pred_df.groupby(["gameKey", "playID"]).size().to_dict().keys():
        tmp_df = pred_df.query("gameKey == @keys[0] and playID == @keys[1]")

        for index, row in tmp_df.iterrows():
            if row["view"] == "Endzone":
                check_df = tmp_df.query('view == "Sideline"')
                if (
                    check_df["frame"]
                    .apply(lambda x: np.abs(x - row["frame"]) <= 4)
                    .sum()
                    == 0
                ):
                    dropIDX.append(index)

            if row["view"] == "Sideline":
                check_df = tmp_df.query('view == "Endzone"')
                if (
                    check_df["frame"]
                    .apply(lambda x: np.abs(x - row["frame"]) <= 4)
                    .sum()
                    == 0
                ):
                    dropIDX.append(index)

    pred_filtered_df = pred_df.drop(index=dropIDX).reset_index(drop=True)
    return pred_filtered_df


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
    parser.add_argument(
        "--dataset_dir", default="../dataset", metavar="DIR", help="path to dataset"
    )
    parser.add_argument(
        "--checkpoint",
        default="../notebook/d3-test-0.93.ckpt",
        help="path to checkpoint",
    )
    parser.add_argument(
        "--checkpoint2", default=None, help="path to checkpoint of second model"
    )
    parser.add_argument(
        "--checkpoint3", default=None, help="path to checkpoint of third model"
    )
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--impactonly", action="store_true")
    parser.add_argument("--impactdefinitive", action="store_true")
    parser.add_argument("--seqmode", action="store_true")
    parser.add_argument("--fullsizeimage", action="store_true")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--fold_index", default=0, type=int)
    args = parser.parse_args()

    EXP_NAME = args.exp_name
    DATA_DIR = args.dataset_dir
    CHECKPOINT = args.checkpoint
    CHECKPOINT2 = args.checkpoint2
    CHECKPOINT3 = args.checkpoint3
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers

    DEBUG = args.debug

    IMPACTONLY = args.impactonly
    IMPACTDEFINITIVE = args.impactdefinitive
    SEQMODE = args.seqmode
    FULLSIZEIMAGE = args.fullsizeimage
    TTA = args.tta

    FOLD_INDEX = args.fold_index

    if IMPACTONLY:
        IMPACT_CLASS = 1
    else:
        IMPACT_CLASS = 2

    # ----------
    # logger
    # ----------
    if not DEBUG:
        neptune.init(
            project_qualified_name="sunghyun.jun/nfl-eval",
            api_token=os.environ["NEPTUNE_API_TOKEN"],
        )
        neptune.create_experiment(
            name=EXP_NAME,
            params={
                "batch_size": BATCH_SIZE,
                "num_workers": NUM_WORKERS,
                "impactonly": IMPACTONLY,
                "definitive": IMPACTDEFINITIVE,
                "seqmode": SEQMODE,
                "fullsizeimage": FULLSIZEIMAGE,
            },
            upload_source_files=["*.py", "requirements.txt"],
            tags=["pytorch-lightning", "evaluate"],
        )

    # ----------
    # data
    # ----------
    dm = ImpactDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        impactdefinitive=IMPACTDEFINITIVE,
        seqmode=SEQMODE,
        fullsizeimage=FULLSIZEIMAGE,
        fold_index=FOLD_INDEX,
    )

    dm.prepare_data()
    dm.setup()

    # ----------
    # model
    # ----------
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device {device}")
    impact_detector = ImpactDetector.load_from_checkpoint(CHECKPOINT)
    impact_detector.to(device)
    impact_detector.eval()

    if CHECKPOINT2 is not None:
        impact_detector2 = ImpactDetector.load_from_checkpoint(CHECKPOINT2)
        impact_detector2.to(device)
        impact_detector2.eval()

    if CHECKPOINT3 is not None:
        impact_detector3 = ImpactDetector.load_from_checkpoint(CHECKPOINT3)
        impact_detector3.to(device)
        impact_detector3.eval()

    torch.set_grad_enabled(False)

    # ----------
    # evaluation
    # ----------
    if CHECKPOINT2 is not None:
        if CHECKPOINT3 is not None:
            (
                image_ids,
                boxes_gt,
                labels_gt,
                boxes_pred,
                scores_pred,
                labels_pred,
            ) = make_prediction_ensemble_v2(
                impact_detector,
                impact_detector2,
                impact_detector3,
                dm.val_dataloader(),
                device,
                DEBUG,
                FULLSIZEIMAGE,
            )

        else:
            (
                image_ids,
                boxes_gt,
                labels_gt,
                boxes_pred,
                scores_pred,
                labels_pred,
            ) = make_prediction_ensemble(
                impact_detector,
                impact_detector2,
                dm.val_dataloader(),
                device,
                DEBUG,
                FULLSIZEIMAGE,
            )
    else:
        if not TTA:
            (
                image_ids,
                boxes_gt,
                labels_gt,
                boxes_pred,
                scores_pred,
                labels_pred,
            ) = make_prediction(
                impact_detector, dm.val_dataloader(), device, DEBUG, FULLSIZEIMAGE
            )
        else:
            (
                image_ids,
                boxes_gt,
                labels_gt,
                boxes_pred,
                scores_pred,
                labels_pred,
            ) = make_prediction_tta(
                impact_detector, dm.val_dataloader(), device, DEBUG, FULLSIZEIMAGE
            )

    gt_df = make_gt_df(image_ids, boxes_gt, labels_gt)

    pred_df_raw = make_pred_df(
        image_ids, boxes_pred, scores_pred, labels_pred, IMPACT_CLASS
    )

    SCORE_THRESHOLD = [
        0.26,
        0.28,
        0.30,
        0.32,
        0.34,
        0.36,
        0.38,
        0.40,
        0.42,
        0.44,
        0.46,
        0.48,
        0.50,
    ]
    IOU_THRESHOLD = [0.4, 0.5, 0.6, 0.7]

    for thres in SCORE_THRESHOLD:
        pred_df = pred_df_raw[pred_df_raw.score > thres]
        print("-" * 120)
        print(f"Raw - Score: {thres:.2f}             -    -", end="")
        tp, fp, fn, precision, recall, f1_score = calc_metric(gt_df, pred_df)
        if not DEBUG:
            neptune.log_metric(f"{thres:.2f}-RAW", f1_score)

        for iou_thres in IOU_THRESHOLD:
            pred_filtered_df = make_pred_filtered_df(pred_df, iou_thres)
            print(f"PP  - Score: {thres:.2f} - iou: {iou_thres:.2f} -    -", end="")
            tp, fp, fn, precision, recall, f1_score = calc_metric(
                gt_df, pred_filtered_df
            )
            if not DEBUG:
                neptune.log_metric(f"{thres:.2f}-{iou_thres:.2f}", f1_score)

            pred_bothzone_filtered_df = make_pred_bothzone_filtered_df(pred_filtered_df)
            print(f"PP  - Score: {thres:.2f} - iou: {iou_thres:.2f} - PP -", end="")
            tp, fp, fn, precision, recall, f1_score = calc_metric(
                gt_df, pred_bothzone_filtered_df
            )
            if not DEBUG:
                neptune.log_metric(f"{thres:.2f}-{iou_thres:.2f}-", f1_score)


if __name__ == "__main__":
    main()