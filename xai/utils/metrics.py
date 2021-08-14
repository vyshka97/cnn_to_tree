# -*- coding: utf-8 -*-

import logging
import attr
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from typing import Dict, Tuple

from xai.data import Batch
from xai.hyperparams import Hyperparams

logger = logging.getLogger(__name__)
__all__ = ["TaskMetrics", "LossMetrics", "EpochMetrics", "IterationMetrics", "Timestamp",
           "TaskMetricsCalculator", "IterationMetricsCalculator", "MetricsWriter"]


@attr.s
class TaskMetrics:
    THRESHOLDS = [0.5]

    tag = ""

    accuracy_at_threshold: Dict[float, float] = attr.ib(default={})


@attr.s
class LossMetrics:
    tag = ""

    losses: Dict[str, float] = attr.ib(default={})

    def __add__(self, other: "LossMetrics") -> "LossMetrics":
        res = deepcopy(self)
        for k, v in other.losses.items():
            if k not in res.losses:
                res.losses[k] = 0.
            res.losses[k] += v
        return res

    def __truediv__(self, other: int) -> "LossMetrics":
        res = deepcopy(self)
        for k, v in res.losses.items():
            res.losses[k] = v / other
        return res


@attr.s
class EpochMetrics:
    epoch_time = attr.ib(type=float)

    task_metrics = attr.ib(default=TaskMetrics(), type=TaskMetrics)
    loss_metrics = attr.ib(default=LossMetrics(), type=LossMetrics)


@attr.s
class IterationMetrics:
    epoch_num = attr.ib(type=int)
    lr = attr.ib(type=float)
    iter_time = attr.ib(type=float)

    grad_total_l2_norm = attr.ib(default=0., type=float)
    weights_total_l2_norm = attr.ib(default=0., type=float)

    grad_l2_norm_per_layer = attr.ib(default={}, type=dict)
    weights_l2_norm_per_layer = attr.ib(default={}, type=dict)

    train_loss_metrics = attr.ib(default=LossMetrics(), type=LossMetrics)


@attr.s
class Timestamp:
    epoch = attr.ib(type=int)
    global_iteration = attr.ib(type=int)


class TaskMetricsCalculator:

    def __init__(self, hparams: Hyperparams):
        self.hparams = hparams
        self._scores = None
        self._labels = None
        self.reset()

    def calc(self) -> TaskMetrics:
        self._scores = self._scores.cpu()
        self._labels = self._labels.cpu()

        indices = self._scores.argsort()
        self._scores = self._scores[indices]
        self._labels = self._labels[indices]

        thr_indices = torch.searchsorted(self._scores, torch.tensor(TaskMetrics.THRESHOLDS))
        accuracy = dict()
        for idx, thr in zip(thr_indices, TaskMetrics.THRESHOLDS):
            predictions = torch.zeros_like(self._labels)
            predictions[idx:] = 1
            accuracy[thr] = (self._labels == predictions).float().mean().item()

        metrics = TaskMetrics(accuracy_at_threshold=accuracy)
        self.reset()
        return metrics

    def update(self, predictions: torch.Tensor, batch: Batch) -> None:
        self._scores = torch.cat([self._scores, predictions[:, 1]])
        self._labels = torch.cat([self._labels, batch.y])

    def reset(self) -> None:
        self._scores = torch.empty(0, dtype=torch.float32)
        self._labels = torch.empty(0, dtype=torch.long)
        if self.hparams.use_cuda:
            self._scores = self._scores.cuda()
            self._labels = self._labels.cuda()


class IterationMetricsCalculator:
    def __init__(self, model: nn.Module):
        self.model = model

    def calc(self, epoch_num: int, lr: float, iter_time: float, train_loss_metrics: LossMetrics) -> IterationMetrics:
        grad_norm_per_layer, total_grad_norm = self._calc_grad_norm()
        weights_norm_per_layer, total_weights_norm = self._calc_weights_norm()
        return IterationMetrics(
            epoch_num=epoch_num, lr=lr, iter_time=iter_time, grad_total_l2_norm=total_grad_norm,
            grad_l2_norm_per_layer=grad_norm_per_layer, weights_l2_norm_per_layer=weights_norm_per_layer,
            weights_total_l2_norm=total_weights_norm, train_loss_metrics=train_loss_metrics
        )

    def _calc_grad_norm(self, norm_type: int = 2) -> Tuple[Dict[str, float], float]:
        norm_per_layer = {}
        total_norm = 0

        parameters = list(filter(lambda p: p[1].grad is not None, self.model.named_parameters()))
        for param_name, param_value in parameters:
            param_norm = param_value.grad.data.norm(norm_type)
            norm_per_layer[param_name] = param_norm
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        return norm_per_layer, total_norm

    def _calc_weights_norm(self, norm_type: int = 2) -> Tuple[Dict[str, float], float]:
        norm_per_layer = {}
        total_norm = 0
        parameters = list(self.model.named_parameters())

        for param_name, param_value in parameters:
            param_norm = param_value.data.norm(norm_type)
            norm_per_layer[param_name] = param_norm
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)

        return norm_per_layer, total_norm


class MetricsWriter:

    def __init__(self, summary_writer: SummaryWriter):
        self.summary_writer = summary_writer

    def _log_metrics_dict(self, global_step: int, metrics_dict: Dict[str, float], tag: str) -> None:
        for metric_name, metric_value in metrics_dict.items():
            self.summary_writer.add_scalar(f"{tag}/{metric_name}", metric_value, global_step)

    def log_loss_metrics(self, epoch: int, metrics: LossMetrics) -> None:
        self._log_metrics_dict(epoch, metrics.losses, f"{metrics.tag}/Losses")

    def _log_task_metrics(self, epoch: int, metrics: TaskMetrics) -> None:
        for metric_name, metric_values_dct in attr.asdict(metrics).items():
            metrics_dct = {f"{metric_name}_{at}": value for at, value in metric_values_dct.items()}
            self._log_metrics_dict(epoch, metrics_dct, f"{metrics.tag}/Metrics")

    def log_epoch_metrics(self, timestamp: Timestamp, epoch_metrics: EpochMetrics) -> None:
        if epoch_metrics is None:
            return

        global_step = timestamp.epoch

        self.summary_writer.add_scalar('Timer/epoch_time', epoch_metrics.epoch_time, global_step)

        self._log_task_metrics(global_step, epoch_metrics.task_metrics)
        self.log_loss_metrics(global_step, epoch_metrics.loss_metrics)

    def log_iteration_metrics(self, timestamp: Timestamp, iter_metrics: IterationMetrics) -> None:
        if iter_metrics is None:
            return

        global_step = timestamp.global_iteration

        self.summary_writer.add_scalar('Timer/iter_time', iter_metrics.iter_time, global_step)
        self.summary_writer.add_scalar('epoch_number', iter_metrics.epoch_num, global_step)
        self.summary_writer.add_scalar('lr', iter_metrics.lr, global_step)

        self.summary_writer.add_scalar('grads_total_norm', iter_metrics.grad_total_l2_norm, global_step)
        self._log_metrics_dict(global_step, iter_metrics.grad_l2_norm_per_layer, "grads_norm")

        self.summary_writer.add_scalar('weights_total_norm', iter_metrics.weights_total_l2_norm, global_step)
        self._log_metrics_dict(global_step, iter_metrics.weights_l2_norm_per_layer, "weights_norm")

        self.log_loss_metrics(global_step, iter_metrics.train_loss_metrics)
