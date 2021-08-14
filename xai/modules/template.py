# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from xai.hyperparams import Hyperparams

__all__ = ["Template"]


class Template(nn.Module):

    def __init__(self, map_size: int, hparams: Hyperparams):
        super().__init__()

        self.hparams = hparams
        self.map_size = map_size

        self.beta = nn.Parameter(torch.tensor(hparams.beta), requires_grad=True)

        idx_diff = torch.arange(map_size)[None].repeat(map_size, 1)  # [M] -> [1, M] -> [M, M], where M is map_size
        idx_diff = torch.stack([idx_diff.T, idx_diff], dim=-1).float()  # [M, M, 2]
        shape = idx_diff.shape
        # [M, M, 2] -> [M, M, 2M^2] -> [M^2, M, M, 2]
        idx_same = idx_diff.repeat(1, 1, map_size**2).reshape(map_size**2, *shape)
        self.pos_norms = (idx_same - idx_diff).norm(p=1, dim=-1)  # [M^2, M, M, 2] -> [M^2, M, M]

        self.neg_template = -torch.ones(1, map_size, map_size)  # [1, M, M]

    @property
    def pos_templates(self) -> torch.Tensor:
        return torch.clamp(1 - self.beta.data * self.pos_norms / self.map_size, min=-1)

    # we expect that x with dimensions [B, C, map_size, map_size], where map_size == self.template.map_size
    def get_masked_output(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        assert shape[2] == shape[3] == self.map_size

        # choose template that maximize activation and return x_masked
        indices = F.max_pool2d(x, shape[2], return_indices=True)[1].squeeze(-1).squeeze(-1)  # [B, 512]
        selected = self.pos_norms[indices]
        pos_templates = torch.clamp(1 - self.beta * selected / self.map_size, min=-1)
        x_masked = F.relu(x * pos_templates)
        return x_masked

    def cuda(self) -> "Template":
        self.pos_norms = self.pos_norms.cuda()
        self.neg_template = self.neg_template.cuda()
        return self

    def cpu(self) -> "Template":
        self.pos_norms = self.pos_norms.cpu()
        self.neg_template = self.neg_template.cpu()
        return self
