# -*- coding: utf-8 -*-

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ["ConstLRScheduler"]


class ConstLRScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, initial_lr: float, last_epoch: int = 0):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, -1)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        return self.initial_lr
