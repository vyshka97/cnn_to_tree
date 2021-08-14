# -*- coding: utf-8 -*-

import os
import h5py
import logging
import torch
import math
import random
import torchvision.transforms.functional as F

from typing import Dict, Any, Optional, List, Tuple
from torch.utils.data import Dataset

from ..hyperparams import Hyperparams
from .processor import Processor

logger = logging.getLogger()

__all__ = ["BaseDataset", "ResizedDataset", "BalancedDataset"]


class BaseDataset(Dataset):

    def __init__(self, index_path: str, hparams: Hyperparams, label: int, processor: Optional[Processor] = None):
        self.index_path = index_path
        self.hparams = hparams
        self.label = label
        self.processor = processor

        index_dir = os.path.dirname(self.index_path)

        with open(self.index_path) as __fin:
            files_paths = __fin.read().splitlines()

        self.paths = [p if os.path.isabs(p) else os.path.join(index_dir, p) for p in files_paths]

    def __len__(self) -> int:
        return len(self.paths)

    @staticmethod
    def read(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            logger.error(f"No such image on the given path : {path}")
            raise IOError

        instance = {'path': path}
        with h5py.File(path, 'r', driver='core') as data:
            pixels = torch.from_numpy(data["pixels"][()])
            assert len(pixels.shape) == 3 and pixels.shape[0] == 3
            if pixels.dtype is torch.uint8:
                pixels = pixels.float() / 255  # scale to range [0, 1]
            instance['pixels'] = pixels
            instance["xmin"] = int(data["bbox/xmin"][()])
            instance["xmax"] = int(data["bbox/xmax"][()])
            instance["ymin"] = int(data["bbox/ymin"][()])
            instance["ymax"] = int(data["bbox/ymax"][()])

        return instance

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Любой наследник должен возвращать dict, где как минимум есть ключи `x`, `y`, `path`
        """
        raise NotImplementedError()


class ResizedDataset(BaseDataset):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.paths[index]
        instance = self.read(path)

        instance = {
            "x": self.resize(instance),
            "y": torch.LongTensor([self.label]),
            "path": instance["path"],
        }
        if self.processor:
            instance = self.processor(instance)

        return instance

    def resize(self, instance: Dict[str, Any]) -> torch.Tensor:
        xmin, ymin, xmax, ymax = instance["xmin"], instance["ymin"], instance["xmax"], instance["ymax"]
        x = instance["pixels"][:, ymin: ymax + 1, xmin: xmax + 1]  # [3, H, W]
        x = F.resize(x, self.hparams.image_size)  # [3, H, W] -> [3, I1, I2], where [I1, I2] is self.hparams.image_size
        return x


class BalancedDataset(Dataset):

    def __init__(self, datasets_with_probs: List[Tuple[Dataset, float]], size: int):
        assert len(datasets_with_probs) > 0
        assert size > 0

        self.datasets, probs = zip(*datasets_with_probs)
        sum_probs = sum(probs)
        probs = list(map(lambda x: x / sum_probs, probs))

        self.items = []
        for didx, ds in enumerate(self.datasets):
            tmp_items = [(didx, x) for x in range(min(len(ds), math.floor(size * probs[didx])))]
            if len(tmp_items) < size * probs[didx]:
                choices = random.choices(range(len(ds)), k=math.floor((size * probs[didx]) - len(tmp_items)))
                tmp_items.extend((didx, x) for x in choices)
            self.items.extend(tmp_items)
        random.shuffle(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d, di = self.items[index]
        return self.datasets[d][di]

    def __len__(self) -> int:
        return len(self.items)

    def on_post_epoch_callback(self) -> None:
        for d in self.datasets:
            if hasattr(d, 'on_post_epoch_callback'):
                d.on_post_epoch_callback()
