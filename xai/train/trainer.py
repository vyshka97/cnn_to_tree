# -*- coding: utf-8 -*-

import logging
import time
import torch
import torch.nn.functional as F

from typing import Tuple, Dict, Callable, Optional
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models import LossCalculator, Forwarder, OutputType, XaiCnn
from xai.data import Batch
from xai.hyperparams import Hyperparams
from xai.losses import LossType
from xai.utils import (
    TrainerCheckpointParams, TaskMetricsCalculator, TaskMetrics, LossMetrics, EpochMetrics, MetricsWriter,
    IterationMetrics, IterationMetricsCalculator, Timestamp, CheckpointManager, get_writer
)

logger = logging.getLogger(__name__)

__all__ = ["Trainer"]


class Trainer:

    def __init__(self, hparams: Hyperparams, train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 model: XaiCnn, loss_calculator: LossCalculator, forwarder: Forwarder, optimizer: Optimizer,
                 scheduler: _LRScheduler, checkpoint_manager: CheckpointManager,
                 metrics_writer: Optional[MetricsWriter] = None):

        self.checkpoint_manager = checkpoint_manager
        self.hparams = hparams
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.forwarder = forwarder
        self.loss_calculator = loss_calculator
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Обертка для удобного логирования метрик.
        self.metrics_writer = metrics_writer if metrics_writer else MetricsWriter(get_writer())
        self.task_metrics_calc = TaskMetricsCalculator(hparams)
        self.iter_metrics_calc = IterationMetricsCalculator(model)

        # Список внешних callback-ов, вызываемых после завершения эпохи.
        self.post_epoch_callbacks = []

        # Последняя глобальная итерация последнего запуска train.
        self.train_stop_global_iteration = 0

    def train(self, start_epoch: int, stop_epoch: int, start_global_iteration: int = 0) -> float:
        if self.hparams.use_cuda:
            self.model = self.model.cuda()
            self.loss_calculator = self.loss_calculator.cuda()

        train_loss_metrics = LossMetrics()
        train_loss_metrics.tag = "Train"
        global_iter = start_global_iteration

        stop_train = False

        for epoch in range(start_epoch, stop_epoch):
            self.__on_pre_epoch(epoch)

            epoch_start_time = time.time()

            # Обучение.
            self.model.train()
            iters_num = 0

            for batch in tqdm(self.train_dataloader):
                iter_start_time = time.time()

                if self.hparams.use_cuda:
                    batch = batch.cuda()

                self.model.zero_grad()

                model_outputs = self.forwarder.forward_train(self.model, batch)
                loss, loss_metrics = self.loss_calculator.calc_train(model_outputs, batch, epoch)

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning("Loss has NaN or INF. Stop Training")
                    train_loss_metrics = train_loss_metrics + LossMetrics(loss_metrics)
                    stop_train = True
                    break

                loss.backward()

                if self.__on_post_backward(batch):
                    self.optimizer.step()

                    lr = self.scheduler.get_lr()
                    train_loss_metrics = train_loss_metrics + LossMetrics(loss_metrics)
                    iter_time = time.time() - iter_start_time

                    iter_metrics = self.iter_metrics_calc.calc(epoch, lr, iter_time, train_loss_metrics)

                    timestamp = Timestamp(epoch=epoch, global_iteration=global_iter)
                    self.__on_post_iteration(timestamp, iter_metrics)

                    if global_iter % self.hparams.log_iter_metrics_every == self.hparams.log_iter_metrics_every - 1:
                        # Обнуляем все лоссы в train, начинаем заново отсчет
                        train_loss_metrics = LossMetrics()
                        train_loss_metrics.tag = "Train"

                iters_num += 1
                global_iter += 1

            timestamp = Timestamp(epoch=epoch, global_iteration=global_iter)

            if epoch % self.hparams.log_epoch_metrics_every == self.hparams.log_epoch_metrics_every - 1:
                # Валидация.
                valid_loss_metrics, valid_task_metrics = self._validate(self.valid_dataloader, epoch)
                epoch_time = time.time() - epoch_start_time
                epoch_metrics = EpochMetrics(epoch_time=epoch_time,
                                             task_metrics=valid_task_metrics,
                                             loss_metrics=valid_loss_metrics)

                self._scheduler_step(epoch)

                ckpt_score = valid_loss_metrics.losses[LossType.CROSS_ENTROPY.value]

                self.__save_checkpoint(timestamp, score=ckpt_score)
                self.metrics_writer.log_epoch_metrics(timestamp, epoch_metrics)

            self.__on_post_epoch(epoch)

            if stop_train:
                return train_loss_metrics.losses[LossType.CROSS_ENTROPY.value]

        return train_loss_metrics.losses[LossType.CROSS_ENTROPY.value]

    def register_post_epoch_callback(self, callback: Callable) -> None:
        self.post_epoch_callbacks.append(callback)

    def _validate(self, dataloader: DataLoader, epoch: int) -> Tuple[LossMetrics, TaskMetrics]:
        self.model.eval()

        loss_metrics = LossMetrics()
        loss_metrics.tag = "Valid"

        iters_num = 0

        with torch.no_grad():
            for batch in tqdm(dataloader):

                if self.hparams.use_cuda:
                    batch = batch.cuda()

                model_outputs = self.forwarder.forward_val(self.model, batch)

                _, loss_metrics_ = self.loss_calculator.calc_val(model_outputs, batch, epoch)
                loss_metrics = loss_metrics + LossMetrics(loss_metrics_)
                self._update_task_metrics_and_figures(model_outputs, batch)

                iters_num += 1

        loss_metrics = loss_metrics / max(iters_num, 1)
        task_metrics = self.task_metrics_calc.calc()
        task_metrics.tag = "Valid"

        return loss_metrics, task_metrics

    def _update_task_metrics_and_figures(self, model_outputs: Dict[OutputType, torch.Tensor], batch: Batch) -> None:
        predictions = F.softmax(model_outputs[OutputType.MODEL_OUTPUT], dim=-1)
        self.task_metrics_calc.update(predictions, batch)

    def _scheduler_step(self, epoch: int) -> None:
        self.scheduler.step(epoch)

    def __on_pre_epoch(self, epoch: int) -> None:
        logger.info(f'epoch {epoch}')

    def __on_post_epoch(self, epoch: int) -> None:
        for callback in self.post_epoch_callbacks:
            callback()

        # Контрольный лог суммы весов сети для моделей.
        params_sum = 0.
        for p in self.model.parameters():
            params_sum += p.sum().item()

        logger.info(f'post epoch {epoch} model params sum: {params_sum}')

    def __on_post_iteration(self, timestamp: Timestamp, iter_metrics: IterationMetrics) -> None:
        if timestamp.global_iteration % self.hparams.log_iter_metrics_every == self.hparams.log_iter_metrics_every - 1:
            iter_metrics.train_loss_metrics = iter_metrics.train_loss_metrics / self.hparams.log_iter_metrics_every

            self.metrics_writer.log_iteration_metrics(timestamp, iter_metrics)

    def __on_post_backward(self, batch: Batch) -> bool:
        if self.hparams.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hparams.clip_grad)
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if torch.sum(torch.isnan(p.grad.data)).item() != 0 or torch.sum(torch.isinf(p.grad.data)).item() != 0:
                logger.warning(f'oops we have nan batch in {name}!')
                batch_paths_str = ''
                for path in batch.paths:
                    batch_paths_str += f'{path}\n'
                logger.warning(f'nan batch files: {batch_paths_str}')
                return False
        return True

    def __save_checkpoint(self, timestamp: Timestamp, score: float) -> None:
        data = TrainerCheckpointParams(
            hparams=self.hparams,
            model=self.model,
            optimizer=self.optimizer,
            epoch=timestamp.epoch,
            global_iteration=timestamp.global_iteration,
            score=score
        )

        self.checkpoint_manager.save_checkpoint(data)
