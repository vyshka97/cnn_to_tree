# -*- coding: utf-8 -*-

import logging
import torch

from enum import Enum
from typing import Dict, Tuple, Any

from xai.data import Batch
from xai.hyperparams import Hyperparams

logger = logging.getLogger(__name__)

__all__ = ["OutputType", "ForwardType", "XaiCnn", "Forwarder", "LossCalculator"]


class OutputType(Enum):
    MODEL_OUTPUT = "model_output"
    BASE_INTER_MAP = "base_interpretable_map"
    NEW_INTER_MAP = "new_interpretable_map"


class ForwardType(Enum):
    DEFAULT_FORWARD = 0


class XaiCnn(torch.nn.Module):

    def __init__(self, hparams: Hyperparams):
        super().__init__()
        self.hparams = hparams

    def forward(self, forward_type: ForwardType, *args, **kwargs) -> Dict[OutputType, Any]:
        if forward_type == ForwardType.DEFAULT_FORWARD:
            return self.default_forward(*args, **kwargs)
        raise RuntimeError(f"No such forward type : {str(forward_type)}")

    def default_forward(self, x: torch.Tensor) -> Dict[OutputType, Any]:
        raise NotImplementedError("No default_forward implementation")

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_mem_usage(self) -> int:
        """
        :return: model memory usage in bytes
        """
        mem_params = sum([param.nelement() * param.element_size() for param in self.parameters()])
        mem_bufs = sum([buf.nelement() * buf.element_size() for buf in self.buffers()])
        return mem_params + mem_bufs

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        # Добавим в state_dict все ключи модели, которых там нет.
        self_state_dict = self.state_dict()
        for k, v in self_state_dict.items():
            if k not in state_dict:
                state_dict[k] = v
                logger.info(f'use default initialization for module {k}')
        # И выкинем все те ключи, которых нет в модели.
        keys_to_remove = []
        for k, v in state_dict.items():
            if k not in self_state_dict:
                keys_to_remove.append(k)
        for k in keys_to_remove:
            state_dict.pop(k)
            logger.info(f'module {k} dropped from loaded checkpoint')

        super().load_state_dict(state_dict, strict)


class Forwarder:
    def __init__(self, hparams: Hyperparams):
        self.hparams = hparams

    def forward_train(self, model: XaiCnn, batch: Batch) -> Dict[OutputType, Any]:
        raise NotImplementedError("No train forward implementation for training")

    def forward_val(self, model: XaiCnn, batch: Batch) -> Dict[OutputType, Any]:
        raise NotImplementedError("No val forward implementation for validation")


class LossCalculator:
    def __init__(self, hparams: Hyperparams):
        self.hparams = hparams

    def calc_train(self, fwd_res: Dict[OutputType, Any], b: Batch, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError("No loss calculation implementation for training")

    def calc_val(self, fwd_res: Dict[OutputType, Any], b: Batch, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        raise NotImplementedError("No loss calculation implementation for validation")

    def cpu(self) -> "LossCalculator":
        raise NotImplementedError("No moving to cpu implementation")

    def cuda(self) -> "LossCalculator":
        raise NotImplementedError("No moving to cuda implementation")
