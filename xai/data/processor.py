# -*- coding: utf-8 -*-

import random
import torchvision.transforms.functional as F

from typing import Dict, Any, List

__all__ = ["Processor", "ChainProcessor", "RandomApplier", "HorizontalFlip", "VerticalFlip", "Normalize"]


class Processor:
    """
    Все процессоры, которые изменяют картинку -- ожидают словарь с полем 'x'
    """

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        return instance


class ChainProcessor(Processor):

    def __init__(self, chain: List[Processor]):
        self.chain = chain

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        for processor in self.chain:
            instance = processor(instance)

        return instance


class RandomApplier(Processor):

    def __init__(self, processor: Processor, prob: float = 0.5):
        self.processor = processor
        self.prob = prob

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        if self.prob < random.random():
            return instance
        return self.processor(instance)


class HorizontalFlip(Processor):

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        instance["x"] = F.hflip(instance["x"])
        return instance


class VerticalFlip(Processor):

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        instance["x"] = F.vflip(instance["x"])
        return instance


class Normalize(Processor):

    def __init__(self, mean: List[float], std: List[float], inplace: bool = False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        instance["x"] = F.normalize(instance["x"], self.mean, self.std, self.inplace)
        return instance
