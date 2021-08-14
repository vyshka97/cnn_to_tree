# -*- coding: utf-8 -*-

import os

from typing import Set, Dict
from collections import defaultdict
from argparse import Namespace, ArgumentParser
from tqdm import tqdm

from common import *

MIN_AREA: int = 2500

"""
    Run command: python utils/datasets/cub.py --dataset_dir 'dataset_dir' ...

    Expected directory structure in 'dataset_dir'/source
    - images.txt - Images are contained in the directory images/ : <image_id> <image_name>
    - train_test_split.txt - Train/test split: <image_id> <is_training_image>
    - image_class_labels.txt - Image class labels: <image_id> <class_id>
    - classes.txt - The list of class names (bird species): <class_id> <class_name>
    - bounding_boxes.txt - Bounding box labels: <image_id> <x> <y> <width> <height>

    Write all data to 'dataset_dir'/h5/ directory
"""


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", help="Directory to dataset", required=True, type=str)

    return parser.parse_args()


def convert_to_hdf5(dataset_dir: str) -> None:
    source_dir = os.path.join(dataset_dir, "source")
    h5_dir = os.path.join(dataset_dir, "h5")

    index_dir = os.path.join(dataset_dir, "index_files")
    train_index_dir = os.path.join(index_dir, "train")
    val_index_dir = os.path.join(index_dir, "validation")

    image_dir = os.path.join(source_dir, "images")
    image_labels_path = os.path.join(source_dir, "image_class_labels.txt")
    label_class_path = os.path.join(source_dir, "classes.txt")
    image_paths_path = os.path.join(source_dir, "images.txt")
    bboxes_path = os.path.join(source_dir, "bounding_boxes.txt")
    splitting_path = os.path.join(source_dir, "train_test_split.txt")

    train_class2paths = defaultdict(list)
    val_class2paths = defaultdict(list)

    img2label = read_image_class_labels(image_labels_path)
    label2class = read_class_labels_mapping(label_class_path)
    train_image_ids = read_train_test_splitting(splitting_path)

    img2path = read_image_paths(image_paths_path, image_dir)
    img2bbox = read_bounding_boxes(bboxes_path)

    for image_id, image_path in tqdm(img2path.items()):
        label = img2label[image_id]
        class_name = label2class[label]
        bbox = img2bbox.get(image_id)
        if bbox is None:
            continue

        annot = ImageAnnotation(filepath=image_path, bbox=bbox)
        pixels = read_image(image_path)
        image_basename = os.path.basename(image_path).split(".")[0] + ".h5"
        image_last_dir = os.path.basename(os.path.dirname(image_path))
        h5_path = os.path.join(h5_dir, image_last_dir, image_basename)
        create_h5(h5_path, pixels, annot.bbox)

        if image_id in train_image_ids:
            train_class2paths[class_name].append(h5_path)
        else:
            val_class2paths[class_name].append(h5_path)

    all_train_paths = []
    for class_name, paths in train_class2paths.items():
        all_train_paths.extend(paths)
        index_path = os.path.join(train_index_dir, f"{class_name}.lst")
        create_index(index_path, paths)

    all_val_paths = []
    for class_name, paths in val_class2paths.items():
        all_val_paths.extend(paths)
        index_path = os.path.join(val_index_dir, f"{class_name}.lst")
        create_index(index_path, paths)

    train_index_path = os.path.join(train_index_dir, "all.lst")
    val_index_path = os.path.join(val_index_dir, "all.lst")
    create_index(train_index_path, all_train_paths)
    create_index(val_index_path, all_val_paths)


def read_train_test_splitting(filepath: str) -> Set[int]:
    # return train images
    train_set = set()
    with open(filepath) as __fin:
        for line in __fin:
            image_id, is_train = line.strip().split()
            image_id = int(image_id)
            if is_train == "1":
                train_set.add(image_id)
    return train_set


def read_class_labels_mapping(filepath: str) -> Dict[int, str]:
    result = dict()
    with open(filepath) as __fin:
        for line in __fin:
            label, class_name = line.strip().split()
            class_name = class_name.split(".")[1]
            label = int(label)
            result[label] = class_name
    return result


def read_image_class_labels(filepath: str) -> Dict[int, int]:
    result = dict()
    with open(filepath) as __fin:
        for line in __fin:
            image_id, label = line.strip().split()
            image_id = int(image_id)
            label = int(label)
            result[image_id] = label
    return result


def read_image_paths(filepath: str, image_dir: str) -> Dict[int, str]:
    result = dict()
    with open(filepath) as __fin:
        for line in __fin:
            image_id, path = line.strip().split()
            path = os.path.join(image_dir, path)
            if not os.path.exists(path):
                continue
            image_id = int(image_id)
            result[image_id] = path
    return result


def read_bounding_boxes(filepath: str) -> Dict[int, BoundingBox]:
    result = dict()
    with open(filepath) as __fin:
        for line in __fin:
            image_id, x, y, width, height = line.strip().split()
            image_id = int(image_id)
            xmin = int(float(x))
            ymin = int(float(y))
            xmax = xmin + int(float(width))
            ymax = ymin + int(float(height))
            if (xmax - xmin + 1) * (ymax - ymin + 1) < MIN_AREA:
                continue
            result[image_id] = BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    return result


def main() -> None:
    args = get_args()
    convert_to_hdf5(args.dataset_dir)


if __name__ == "__main__":
    main()
