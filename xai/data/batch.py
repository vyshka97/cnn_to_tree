# -*- coding: utf-8 -*-

import torch

from typing import Optional, List

__all__ = ["Batch"]


class Batch:

    def __init__(self, x: torch.Tensor, y: torch.Tensor, paths: Optional[List[str]] = None):
        self.x = x
        self.y = y
        self.paths = paths

    def cuda(self, non_blocking: bool = False) -> "Batch":
        self.x = self.x.to(device='cuda', non_blocking=non_blocking)
        self.y = self.y.to(device='cuda', non_blocking=non_blocking)

        return self

    def cpu(self) -> "Batch":
        self.x = self.x.cpu()
        self.y = self.y.cpu()

        return self

    @property
    def size(self) -> int:
        return self.x.shape[0]

    def __repr__(self) -> str:
        lines = []
        for attr, value in self.__dict__.items():
            if value is not None:
                lines.append(f"Attr: {attr}:")

                if isinstance(value, torch.Tensor):
                    lines.append("Shape: {}".format(value.shape))
                lines.append(repr(value))
                lines.append("\n")

        return "\n".join(lines)
