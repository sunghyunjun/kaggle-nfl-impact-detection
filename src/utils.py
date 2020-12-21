import io
import pickle5 as pickle
from urllib.parse import urlparse

import cv2
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
from google.cloud import storage
from google.api_core.retry import Retry


def gcs_load_obj(uri):
    uri = urlparse(uri)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(uri.netloc)
    b = bucket.blob(uri.path[1:], chunk_size=None)
    obj = pickle.load(io.BytesIO(b.download_as_string()))
    return obj


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


def visualize_single_sample(sample, bboxes_yxyx=True):
    sample_image = np.transpose(sample["image"], (1, 2, 0))
    img = sample_image.numpy().copy()
    sample_boxes = sample["bboxes"].numpy().astype(np.int16)

    if bboxes_yxyx:
        # set bboxes xyxy for plot
        sample_boxes[:, [0, 1, 2, 3]] = sample_boxes[:, [1, 0, 3, 2]]

    sample_labels = sample["labels"].numpy()
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
    plt.show()
    # plt.savefig("test_fig.png")


def test_datamodule(dm):
    dm.prepare_data()
    dm.setup()
    tr_it = iter(dm.train_dataloader())
    data = next(tr_it)
    visualize_single_sample(data)