# -*- coding: utf-8 -*-

import logging
import torch
import torch.nn as nn

from typing import Optional, Union

from xai.modules import Template

__all__ = ["FilterLoss"]

logger = logging.getLogger(__file__)


class FilterLoss(nn.Module):
    MEAN = "mean"
    SUM = "sum"
    NONE = "none"

    def __init__(self, template: Template, alpha: Optional[float] = None, weight_coef: float = 5e-6,
                 reduction: str = "mean", update_weights: bool = True):
        super().__init__()
        if alpha is not None and (alpha < 0 or alpha > 1):
            raise ValueError("alpha should be from 0 to 1")

        if reduction not in {self.MEAN, self.SUM, self.NONE}:
            raise ValueError(f"{reduction} is not a valid value for reduction")
        self.reduction = reduction

        self.weight_coef = weight_coef
        self.update_weights = update_weights
        if update_weights:
            self.__cnt = 0
            self.weights = None

        self.all_templates = torch.cat([template.pos_templates, template.neg_template], dim=0)  # [M^2 + 1, M, M]

        n_square = template.map_size ** 2
        self.tau = 0.5 / n_square

        if alpha is not None:
            pos_prob = alpha / n_square
            neg_prob = 1 - alpha
        else:
            pos_prob = 1 / (1 + n_square)
            neg_prob = pos_prob

        # [M^2 + 1]
        self.p_t = torch.FloatTensor([pos_prob] * n_square + [neg_prob]).to(device=self.all_templates.device)

    # we expect that input with dimensions [B, C, M, M]
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.permute(1, 0, 2, 3)  # [B, C, M, M] -> [C, B, M, M]

        # [C, B, 1, M, M] * [1, 1, M^2 + 1, M, M] -> [C, B, M^2 + 1]
        p_x_t = (x.unsqueeze(2) * self.tau * self.all_templates[(None,) * 2]).sum(-1).sum(-1).softmax(dim=1)
        p_x = (self.p_t[(None,) * 2] * p_x_t).sum(-1)  # [1, 1, M^2 + 1] * [C, B, M^2 + 1] -> [C, B, M^2 + 1] -> [C, B]

        # [C, B, M^2 + 1] / [C, B, 1] -> [C, B, M^2 + 1] -> [C, M^2 + 1]
        entropy = (p_x_t * (p_x_t / p_x.unsqueeze(2)).log()).sum(1)
        loss_values = -(self.p_t[None] * entropy).sum(-1)  # [1, M^2 + 1] * [C, M^2 + 1] -> [C, M^2 + 1] -> [C]

        if self.training:
            if self.update_weights:
                self.__update(x)
                weights = self.weights
            else:
                weights = x.max(-1)[0].max(-1)[0].mean(-1)  # [C, B, M, M] -> [C, B, M] -> [C, B] -> [C]
            loss_values = loss_values * self.weight_coef * weights

        if self.reduction == self.MEAN:
            loss_values = loss_values.mean()
        elif self.reduction == self.SUM:
            loss_values = loss_values.sum()

        return loss_values

    # we expect that input with dimensions [C, B, M, M]
    def __update(self, x: torch.Tensor) -> None:
        if self.weights is None:
            self.weights = torch.zeros(x.shape[0], device=x.device)
        else:
            self.weights = self.weights.detach()

        max_activations = x.max(-1)[0].max(-1)[0]  # [C, B, M, M] -> [C, B, M] -> [C, B]
        delta = max_activations - self.weights[:, None]  # [C, B] - [C, 1] -> [C, B]
        delta_total = delta.sum(-1)  # [C, B] -> [C]

        self.__cnt += x.shape[1]
        self.weights += delta_total / self.__cnt  # [C]

    def cuda(self, device_id: Optional[Union[int, torch.device]] = None) -> nn.Module:
        self.all_templates = self.all_templates.cuda(device_id)
        self.p_t = self.p_t.cuda(device_id)
        return super(FilterLoss, self).cuda(device_id)

    def cpu(self) -> nn.Module:
        self.all_templates = self.all_templates.cpu()
        self.p_t = self.p_t.cpu()
        return super(FilterLoss, self).cpu()

