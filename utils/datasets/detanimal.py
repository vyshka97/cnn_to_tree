# -*- coding: utf-8 -*-

import os
import h5py
import random

from typing import List, Dict
from tqdm import tqdm

from common import *

MIN_AREA: int = 2500
CATEGORIES: List[str] = [
    "monkey", "domestic cat", "bird", "camel", "turtle", "giant panda", "hamster", "rabbit", "lesser panda",
    "hippopotamus", "antelope", "sheep", "frog", "bear", "horse", "lizard", "lion", "squirrel", "koala", "cattle",
    "goldfish", "elephant", "swine", "armadillo", "tiger", "fox", "otter", "zebra", "dog", "lobster",
]


"""
    Run command: python utils/datasets/detanimal.py --dataset_dir 'dataset_dir'

    Expected directory structure in 'dataset_dir'/source
    - *_obj/img/img/*.jpg - jpeg images
    - *_obj/img/data.mat - annotation data
    
    Expected file 'dataset_dir'/imagenet_synset_to_human_label_map.txt - mapping between label and classes
    
    All data are from https://github.com/zqs1022/detanimalpart

    Write all data to 'dataset_dir'/h5/ directory
"""


def read_label_category_mapping(filepath: str) -> Dict[str, str]:
    cat2label = dict()
    with open(filepath) as __fin:
        for line in __fin:
            label, category = line.strip().split("\t")
            category = category.split(", ")[0]
            cat2label[category] = label
    return cat2label


def convert_to_hdf5(dataset_dir: str, category: str = "all", train_rate: float = 0.9) -> None:
    source_dir = os.path.join(dataset_dir, "source")
    target_dir = os.path.join(dataset_dir, "h5")

    index_dir = os.path.join(dataset_dir, "index_files")
    train_index_dir = os.path.join(index_dir, "train")
    val_index_dir = os.path.join(index_dir, "validation")

    label_category_mapping_path = os.path.join(dataset_dir, "imagenet_synset_to_human_label_map.txt")
    cat2label = read_label_category_mapping(label_category_mapping_path)

    if category == "all":
        categories = CATEGORIES
    else:
        categories = [category]

    all_train_paths = []
    all_val_paths = []

    for category in categories:
        print(f"Process {category}")

        label = cat2label[category]

        dir_path = os.path.join(source_dir, f"{label}_obj")
        annot_path = os.path.join(dir_path, "img", "data.mat")
        img_dir = os.path.join(dir_path, "img", "img")

        annotations = read_annotation(annot_path, img_dir)

        category = category.replace(" ", "_")
        train_index_path = os.path.join(train_index_dir, f"{category}.lst")
        val_index_path = os.path.join(val_index_dir, f"{category}.lst")
        train_paths = []
        val_paths = []

        for idx, annot in tqdm(enumerate(annotations), total=len(annotations)):
            dirname = os.path.relpath(annot.filepath, source_dir).split("/")[0]
            basename = os.path.basename(annot.filepath).split(".")[0]
            target_path = os.path.join(target_dir, dirname, basename, f"{idx}.h5")

            pixels = read_image(annot.filepath)
            create_h5(target_path, pixels, annot.bbox)

            if random.random() < train_rate:
                train_paths.append(target_path)
                all_train_paths.append(target_path)
            else:
                val_paths.append(target_path)
                all_val_paths.append(target_path)

        create_index(train_index_path, train_paths)
        create_index(val_index_path, val_paths)

    train_index_path = os.path.join(train_index_dir, "all.lst")
    val_index_path = os.path.join(val_index_dir, "all.lst")
    create_index(train_index_path, all_train_paths)
    create_index(val_index_path, all_val_paths)


def read_annotation(data_path: str, img_dir: str) -> List[ImageAnnotation]:
    result = []
    with h5py.File(data_path, mode='r') as annot:
        objects = annot["samples"]["obj"]
        for idx, obj in enumerate(objects, 1):
            ref = obj.item()
            bbox_data = annot[ref]["bndbox"]
            xmin = int(bbox_data["xmin"][()].item()) - 1
            xmax = int(bbox_data["xmax"][()].item()) - 1
            ymin = int(bbox_data["ymin"][()].item()) - 1
            ymax = int(bbox_data["ymax"][()].item()) - 1
            if (xmax - xmin + 1) * (ymax - ymin + 1) < MIN_AREA:
                continue
            bbox = BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            img_annot = ImageAnnotation(filepath=os.path.join(img_dir, "%05d.jpg" % idx), bbox=bbox)
            result.append(img_annot)

    return result


def main() -> None:
    args = get_args(CATEGORIES)
    convert_to_hdf5(args.dataset_dir, category=args.category, train_rate=args.train_rate)


if __name__ == "__main__":
    random.seed(122)
    main()
