# -*- coding: utf-8 -*-

import os
import logging
import pprint
import sys
import attr
import torch
import torchvision.models as models

from typing import Optional
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

experiment_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(experiment_dir)

from src.common import (
    get_experiment_dir, get_hyperparams, get_train_dataloader, init_env, init_logging, ExperimentHyperparams,
    WorkerIniter, get_valid_dataloader, load_checkpoint
)

from xai.losses import CompositeLoss, LossType, FilterLoss
from xai.models import VGG16, VGG16Forwarder, VGG16LossCalculator
from xai.optim import ConstLRScheduler
from xai.train import Trainer
from xai.utils import CheckpointManager, CheckpointXaiCnn

logger = logging.getLogger()


def _train(exp_run_code: str, checkpoint_manager: CheckpointManager, checkpoint: Optional[CheckpointXaiCnn] = None,
           hparams: Optional[ExperimentHyperparams] = None, model: Optional[VGG16] = None,
           stop_epoch: int = 1000) -> None:

    start_epoch = 0
    start_global_iteration = 0

    if checkpoint is not None:
        if checkpoint.data.model:
            model = checkpoint.data.model

        if checkpoint.data.hparams:
            hparams = checkpoint.data.hparams

        if checkpoint.data.epoch:
            start_epoch = checkpoint.data.epoch + 1

        if checkpoint.data.global_iteration:
            start_global_iteration = checkpoint.data.global_iteration

    logger.info(f"PID: {os.getpid()}")
    logger.info(f"model params count: {model.count_parameters()}")

    logger.info(pprint.pformat(attr.asdict(hparams), width=1))

    # Создаем даталоадеры.
    worker_initer = WorkerIniter(exp_run_code)

    train_dataloader, post_epoch_callbacks = get_train_dataloader(hparams, worker_initer)
    logger.info(f'train dataloader len: {len(train_dataloader)}')
    logger.info(f"train dataset len: {len(train_dataloader.dataset)}")

    valid_dataloader = get_valid_dataloader(hparams, worker_initer)
    logger.info(f'valid dataloader len: {len(valid_dataloader)}')
    logger.info(f"valid dataset len: {len(valid_dataloader.dataset)}")

    logger.info(f'start_epoch: {start_epoch}')
    logger.info(f'stop_epoch: {stop_epoch}')
    assert stop_epoch > start_epoch

    logger.info(f'start_global_iteration: {start_global_iteration}')

    if hparams.use_cuda:
        model = model.cuda()

    # Если требуется "чистый" оптимайзер (например, меняем lr), вызывающий код должен
    # присвоить None в checkpoint.data.optimizer.
    if checkpoint and checkpoint.data.optimizer:
        optimizer = checkpoint.data.optimizer
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    else:
        optimizer = SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)

    train_criterion = CompositeLoss()
    train_criterion.add_loss(LossType.CROSS_ENTROPY, CrossEntropyLoss(), 1.0)

    if hparams.train_interpretable:
        base_filter_loss = FilterLoss(model.base_template, reduction="sum", update_weights=True, weight_coef=5e-5)
        train_criterion.add_loss(LossType.BASE_FILTER, base_filter_loss, 1.0)

        new_filter_loss = FilterLoss(model.new_template, reduction="sum", update_weights=True, weight_coef=5e-5)
        train_criterion.add_loss(LossType.NEW_FILTER, new_filter_loss, 1.0)

    val_criterion = CompositeLoss()
    val_criterion.add_loss(LossType.CROSS_ENTROPY, CrossEntropyLoss(), 1.0)
    val_criterion.add_loss(LossType.BASE_FILTER, FilterLoss(model.base_template, reduction="sum"), 1.0)
    val_criterion.add_loss(LossType.NEW_FILTER, FilterLoss(model.new_template, reduction="sum"), 1.0)

    loss_calculator = VGG16LossCalculator(hparams, train_loss=train_criterion, val_loss=val_criterion)

    forwarder = VGG16Forwarder(hparams)

    scheduler = ConstLRScheduler(optimizer, hparams.lr)

    trainer = Trainer(hparams, train_dataloader, valid_dataloader, model, loss_calculator, forwarder, optimizer,
                      scheduler, checkpoint_manager)

    if post_epoch_callbacks is not None:
        for callback in post_epoch_callbacks:
            trainer.register_post_epoch_callback(callback)

    loss = trainer.train(start_epoch, stop_epoch, start_global_iteration)
    logger.info(f'loss: {loss}')


def _from_scratch() -> None:
    init_env(device_id=0)

    hparams = get_hyperparams()
    hparams.use_flipping = False
    hparams.use_pretrained = True
    hparams.train_interpretable = True
    hparams.batch_size = 32
    hparams.lr = 1e-4

    exp_run_code = 'stage_0.from_paper'

    checkpoint_manager = init_logging(exp_run_code)

    model = VGG16(hparams)
    if hparams.use_pretrained:
        vgg16 = models.vgg16(pretrained=True)
        state_dict = vgg16.state_dict()
        keys_to_remove = []
        for name in state_dict:
            if name.startswith("classifier"):
                keys_to_remove.append(name)
        for key in keys_to_remove:
            state_dict.pop(key)
        model.load_state_dict(state_dict)

    logger.info(f'exp_run_code: {exp_run_code}')

    _train(exp_run_code, checkpoint_manager, model=model, hparams=hparams)


def _from_checkpoint() -> None:
    init_env(device_id=0)

    # CHANGEME: Формируем код эксперимента, который продолжаем.
    exp_run_code = 'stage_0.from_paper'
    start_checkpoint = 'checkpoint000000.best.pth'

    # CHANGEME: Указываем нужный чекпоинт.
    checkpoint_path = os.path.join(get_experiment_dir(), 'models', exp_run_code, start_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is not None and checkpoint.data.optimizer is not None:
        for param_group in checkpoint.data.optimizer.param_groups:
            param_group['lr'] = checkpoint.data.hparams.lr

    # Инициализируем логирование и сохранение чекпоинтов.
    checkpoint_manager = init_logging(exp_run_code)

    logger.info(f'continue train from {checkpoint_path}')
    logger.info(f'exp_run_code: {exp_run_code}')

    _train(exp_run_code, checkpoint_manager, checkpoint=checkpoint)


def _tune() -> None:
    init_env(device_id=0)

    prev_exp_code = 'stage_0.from_paper'
    exp_run_code = 'stage_1.from_paper.tune'
    start_checkpoint = 'checkpoint000000.pth'

    checkpoint_path = os.path.join(get_experiment_dir(), 'models', prev_exp_code, start_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    hparams = checkpoint.data.hparams
    model = checkpoint.data.model

    checkpoint.data.optimizer = None

    checkpoint_manager = init_logging(exp_run_code)

    logger.info(f'tune {checkpoint_path}')
    logger.info(f'exp_run_code: {exp_run_code}')

    _train(exp_run_code, checkpoint_manager, checkpoint=checkpoint, model=model, hparams=hparams)


def main():
    # CHANGEME: выбираем тип запуска
    _from_scratch()
    # _from_checkpoint()
    # _tune()


if __name__ == '__main__':
    main()
