# -*- coding: utf-8 -*-

import os
import attr
import h5py
import numpy as np

from typing import List
from argparse import ArgumentParser, Namespace
from PIL import Image

__all__ = ["BoundingBox", "ImageAnnotation", "read_image", "create_index", "create_h5", "get_args"]


@attr.s
class BoundingBox:
    xmin = attr.ib(type=int)
    xmax = attr.ib(type=int)
    ymin = attr.ib(type=int)
    ymax = attr.ib(type=int)


@attr.s
class ImageAnnotation:
    filepath = attr.ib(type=str)  # path to image file
    bbox = attr.ib(type=BoundingBox)  # bounding box


def get_args(categories: List[str]) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="Directory to dataset", required=True, type=str)
    parser.add_argument("--train_rate", help="Rate of train images", type=float, default=0.9)
    parser.add_argument('--category', type=str, help='Category', default="all",
                        choices=[*categories, "all"])

    return parser.parse_args()


def read_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        matrix = np.asarray(image, dtype=np.uint8)
        if matrix.ndim == 2:
            matrix = np.stack((matrix,) * 3, axis=-1)  # [H, W] -> [H, W, C]
        matrix = matrix.transpose((2, 0, 1))  # [H, W, C] -> [C, H, W]
    return matrix


def create_index(index_path: str, paths: List[str]) -> None:
    index_dir = os.path.dirname(index_path)
    if index_dir and not os.path.isdir(index_dir):
        os.makedirs(index_dir, exist_ok=True)

    with open(index_path, mode='w') as __fout:
        for path in paths:
            rel_path = os.path.relpath(path, index_dir)
            __fout.write(f"{rel_path}\n")


def create_h5(h5_path: str, pixels: np.ndarray, bbox: BoundingBox) -> None:
    dir_path = os.path.dirname(h5_path)
    if dir_path and not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with h5py.File(h5_path, mode='w') as data:
        data.create_dataset("pixels", data=pixels, dtype=np.uint8)
        bbox_data = data.create_group("bbox")
        bbox_data.create_dataset("xmin", data=bbox.xmin)
        bbox_data.create_dataset("xmax", data=bbox.xmax)
        bbox_data.create_dataset("ymin", data=bbox.ymin)
        bbox_data.create_dataset("ymax", data=bbox.ymax)
