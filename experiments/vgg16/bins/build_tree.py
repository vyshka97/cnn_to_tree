# -*- coding: utf-8 -*-

import os
import sys
import logging

experiment_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
sys.path.append(experiment_dir)

from src.common import get_experiment_dir, init_env, WorkerIniter, get_tree_building_dataloader, load_checkpoint
from xai.tree import TreeBuilder, Tree
from xai.models import VGG16Forwarder
from xai.utils import init_logger

logger = logging.getLogger()


def build() -> None:
    init_env(device_id=0)

    experiment_dir = get_experiment_dir()

    # CHANGEME: Формируем код эксперимента, который продолжаем.
    exp_run_code = 'stage_0.from_paper'
    start_checkpoint = 'checkpoint000050.best.pth'

    save_dir = os.path.join(experiment_dir, "tree_models")

    init_logger(logging.INFO, log_console=True)

    # CHANGEME: Указываем нужный чекпоинт.
    checkpoint_path = os.path.join(experiment_dir, 'models', exp_run_code, start_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    logger.info(f'build tree from {checkpoint_path}')
    logger.info(f'exp_run_code: {exp_run_code}')

    hparams = checkpoint.data.hparams
    model = checkpoint.data.model

    if hparams.use_cuda:
        model = model.cuda()
    forwarder = VGG16Forwarder(hparams)

    worker_initer = WorkerIniter(exp_run_code)

    train_dataloader = get_tree_building_dataloader(hparams, worker_initer)
    logger.info(f'train dataloader len: {len(train_dataloader)}')
    logger.info(f"train dataset len: {len(train_dataloader.dataset)}")

    builder = TreeBuilder(hparams, model, forwarder, train_dataloader, save_dir)
    builder.build()


def from_snapshot() -> None:
    init_env(device_id=0)

    experiment_dir = get_experiment_dir()

    # CHANGEME: Формируем код эксперимента, который продолжаем.
    exp_run_code = 'stage_0.from_paper'
    start_checkpoint = 'checkpoint000050.best.pth'

    save_dir = os.path.join(experiment_dir, "tree_models")

    init_logger(logging.INFO, log_console=True)

    # CHANGEME: Указываем нужный чекпоинт.
    checkpoint_path = os.path.join(experiment_dir, 'models', exp_run_code, start_checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    logger.info(f'continue tree building from {checkpoint_path}')
    logger.info(f'exp_run_code: {exp_run_code}')

    hparams = checkpoint.data.hparams

    worker_initer = WorkerIniter(exp_run_code)

    train_dataloader = get_tree_building_dataloader(hparams, worker_initer)
    logger.info(f'train dataloader len: {len(train_dataloader)}')
    logger.info(f"train dataset len: {len(train_dataloader.dataset)}")

    tree = Tree(hparams, save_dir)
    tree.train(n_iters=50, start_iter=50)


if __name__ == "__main__":
    from_snapshot()
