# -*- coding: utf-8 -*-

import os
import h5py
import random

from typing import List
from tqdm import tqdm

from common import *

MIN_AREA: int = 2500
CATEGORIES: List[str] = ["bird", "cat", "cow", "dog", "horse", "sheep"]

"""
    Run command: python utils/datasets/voc.py --dataset_dir 'dataset_dir' ...
    
    Expected directory structure in 'dataset_dir'/source
    - JPEGImages/ - Directory with jpeg original images
    - truth_annotations/ - Directory with image annotations for 6 categories from
    https://github.com/zqs1022/interpretableCNN/tree/master/code/data_input/data_input_VOC

    Write all data to 'dataset_dir'/h5/ directory
    Write index files of h5 paths to 'dataset_dir'/index_files/ directory
"""


def convert_to_hdf5(dataset_dir: str, train_rate: float = 0.9, category: str = "all") -> None:
    source_dir = os.path.join(dataset_dir, "source")
    h5_dir = os.path.join(dataset_dir, "h5")
    image_dir = os.path.join(source_dir, "JPEGImages")

    index_dir = os.path.join(dataset_dir, "index_files")
    train_index_dir = os.path.join(index_dir, "train")
    val_index_dir = os.path.join(index_dir, "validation")
    all_train_paths = []
    all_val_paths = []

    if category == "all":
        categories = CATEGORIES
    else:
        categories = [category]

    for category in categories:
        print(f"Process {category}")

        train_index_path = os.path.join(train_index_dir, f"{category}.lst")
        val_index_path = os.path.join(val_index_dir, f"{category}.lst")
        train_paths = []
        val_paths = []

        annotations = read_annotation(source_dir, category, image_dir)
        for idx, annot in tqdm(enumerate(annotations), total=len(annotations)):
            pixels = read_image(annot.filepath)
            image_name = os.path.basename(annot.filepath).split(".")[0]
            h5_path = os.path.join(h5_dir, image_name, f"{category}_{idx}.h5")
            create_h5(h5_path, pixels, annot.bbox)

            if random.random() < train_rate:
                train_paths.append(h5_path)
                all_train_paths.append(h5_path)
            else:
                val_paths.append(h5_path)
                all_val_paths.append(h5_path)

        create_index(train_index_path, train_paths)
        create_index(val_index_path, val_paths)

    train_index_path = os.path.join(train_index_dir, "all.lst")
    val_index_path = os.path.join(val_index_dir, "all.lst")
    create_index(train_index_path, all_train_paths)
    create_index(val_index_path, all_val_paths)


def read_annotation(data_path: str, category: str, image_dir: str) -> List[ImageAnnotation]:
    annot_path = os.path.join(data_path, "truth_annotations", f'truth_{category}.mat')
    annot_list = []
    with h5py.File(annot_path, mode='r') as h5_file:
        truth = h5_file['truth']["obj"]
        for i, t in enumerate(truth):
            data = h5_file[(t[0])]
            bndbox = data['bndbox']
            xmin = int(bndbox['Wmin'][()].item()) - 1
            ymin = int(bndbox['Hmin'][()].item()) - 1
            xmax = int(bndbox['Wmax'][()].item()) - 1
            ymax = int(bndbox['Hmax'][()].item()) - 1
            if (xmax - xmin + 1) * (ymax - ymin + 1) < MIN_AREA:
                continue
            filename = data['filename'][()].squeeze()
            filename = "".join(map(chr, filename))
            filepath = os.path.join(image_dir, filename)
            bbox = BoundingBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            annot_list.append(ImageAnnotation(bbox=bbox, filepath=filepath))
    return annot_list


def main() -> None:
    random.seed(122)
    args = get_args(CATEGORIES)
    convert_to_hdf5(args.dataset_dir, train_rate=args.train_rate)


if __name__ == "__main__":
    main()
