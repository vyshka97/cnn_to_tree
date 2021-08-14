# -*- coding: utf-8 -*-

from typing import List, Dict, Any
from torch.utils.data._utils.collate import default_collate

from .batch import Batch

__all__ = ["Collator"]


class Collator:
    """
    Дефолтный collator, который оборачивает данные в Batch
    """

    def __call__(self, instances: List[Dict[str, Any]]) -> Batch:
        instances = default_collate(instances)
        batch = Batch(instances['x'], instances['y'].squeeze(1), paths=instances['path'])
        return batch
