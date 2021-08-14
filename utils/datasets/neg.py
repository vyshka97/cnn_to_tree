# -*- coding: utf-8 -*-

import os
import random

from argparse import Namespace, ArgumentParser
from tqdm import tqdm

from common import *


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="Directory to dataset", required=True, type=str)
    parser.add_argument("--train_rate", help="Rate of train images", type=float, default=0.9)

    return parser.parse_args()


def convert_to_hdf5(dataset_dir: str, train_rate: float = 0.9) -> None:
    source_dir = os.path.join(dataset_dir, "source")
    h5_dir = os.path.join(dataset_dir, "h5")

    index_dir = os.path.join(dataset_dir, "index_files")
    train_index_path = os.path.join(index_dir, "train", "all.lst")
    val_index_path = os.path.join(index_dir, "validation", "all.lst")

    train_paths = []
    val_paths = []

    for filename in tqdm(os.listdir(source_dir)):
        image_path = os.path.join(source_dir, filename)
        image_name = filename.split(".")[0]
        pixels = read_image(image_path)
        _, h, w = pixels.shape
        target_path = os.path.join(h5_dir, f"{image_name}.h5")
        bbox = BoundingBox(xmin=0, xmax=w - 1, ymin=0, ymax=h - 1)
        create_h5(target_path, pixels, bbox)

        if random.random() < train_rate:
            train_paths.append(target_path)
        else:
            val_paths.append(target_path)

    create_index(train_index_path, train_paths)
    create_index(val_index_path, val_paths)


def main() -> None:
    args = get_args()
    convert_to_hdf5(args.dataset_dir)


if __name__ == "__main__":
    random.seed(122)
    main()
