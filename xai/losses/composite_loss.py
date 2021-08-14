# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from enum import Enum
from torch import device
from typing import Tuple, Dict, Union, Optional

__all__ = ["LossType", "CompositeLoss"]


class LossType(Enum):
    CROSS_ENTROPY = 'CrossEntropy'
    BASE_FILTER = "BaseFilter"
    NEW_FILTER = "NewFilter"


class CompositeLoss:
    def __init__(self):
        self.loss_dict = dict()
        self.loss_coef = dict()

    def add_loss(self, name: LossType, loss: nn.Module, coef: float) -> None:
        self.loss_dict[name] = loss
        self.loss_coef[name] = coef

    def has_loss(self, name: str) -> bool:
        return name in self.loss_dict

    def cpu(self) -> "CompositeLoss":
        for loss_key in self.loss_dict:
            self.loss_dict[loss_key] = self.loss_dict[loss_key].cpu()
        return self

    def cuda(self, device_id: Optional[Union[int, device]] = None) -> "CompositeLoss":
        for loss_key in self.loss_dict:
            self.loss_dict[loss_key] = self.loss_dict[loss_key].cuda(device_id)
        return self

    def calculate(self, data: Dict[LossType, dict], epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        result = None
        for key in self.loss_dict:
            coef = self.loss_coef[key]
            if callable(coef):
                coef = coef(epoch)

            if abs(coef) < 1e-10:
                continue

            if key in data:
                loss = self.loss_dict[key]
                loss_val = loss(**data[key]) * coef
                losses[key.value] = loss_val.cpu().item()

                if result is None:
                    result = loss_val
                else:
                    result += loss_val

            else:
                print('Error no data for ' + key + ' loss')

        losses['loss'] = result.cpu().item()
        return result, losses

    def __call__(self, data: Dict[LossType, dict], epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        return self.calculate(data, epoch)
