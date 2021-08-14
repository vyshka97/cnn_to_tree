# -*- coding: utf-8 -*-

import logging
import os
import attr
import torch

from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Callable, Dict, Optional

from xai.data import (
    ResizedDataset, VerticalFlip, HorizontalFlip, Collator, BalancedDataset, SequenceDataset, RandomApplier,
    ChainProcessor, Processor, Normalize, RestrictedLenDataset
)

from xai.hyperparams import Hyperparams
from xai.utils import (
    CheckpointManager, CheckpointDeleteParams, CheckpointXaiCnn, set_random_seed, init_logger
)
from xai.tree import Analyzer, Tree
from xai.models import VGG16Forwarder

logger = logging.getLogger()

# CHANGEME: Указываем числовой код эксперимента.
_EXPERIMENT_CODE = 'vgg16'


@attr.s
class ExperimentHyperparams(Hyperparams):
    pass


def get_root_path() -> str:
    root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
    return root_path


def get_experiment_dir() -> str:
    root_path = get_root_path()
    exp_dir = os.path.abspath(os.path.join(root_path, f'experiments/{_EXPERIMENT_CODE}'))
    return exp_dir


def get_data_dir() -> str:
    data_dir = '/data/'
    return data_dir


def get_hyperparams() -> ExperimentHyperparams:
    # CHANGEME: Настраиваем параметры, общие для всех запусков эксперимента.
    hparams = ExperimentHyperparams()
    hparams.lr = 1e-4
    hparams.model_out_size = 2
    hparams.log_iter_metrics_every = 1

    return hparams


def get_train_index_paths() -> Dict[int, str]:
    """
    Указываем индексы до train данных
    """
    data_dir = get_data_dir()

    return {
        1: os.path.join(data_dir, "VOC2010/index_files/train/bird.lst"),
        0: os.path.join(data_dir, "neg/index_files/train/all.lst")
    }


def get_validation_index_paths() -> Dict[int, str]:
    """
    Используем всю нашу валидацию
    """
    data_dir = get_data_dir()

    return {
        1: os.path.join(data_dir, "VOC2010/index_files/validation/bird.lst"),
        0: os.path.join(data_dir, "neg/index_files/validation/all.lst")
    }


def get_train_processor(hparams: ExperimentHyperparams) -> Processor:
    processors = [
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    if hparams.use_flipping:
        processors.append(RandomApplier(VerticalFlip(), 0.3))
        processors.append(RandomApplier(HorizontalFlip(), 0.3))

    return ChainProcessor(processors)


def get_valid_processor(hparams: ExperimentHyperparams) -> Processor:
    return ChainProcessor([
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_train_dataloader(hparams: ExperimentHyperparams, worker_init_fn: Callable) -> Tuple[DataLoader, List[Callable]]:
    processor = get_train_processor(hparams)

    datasets_with_probs = []
    dataset_lens = []
    idx_paths = get_train_index_paths()
    prob = 1. / len(idx_paths)
    for label, path in idx_paths.items():
        ds = ResizedDataset(path, hparams, label, processor=processor)
        datasets_with_probs.append((ds, prob))
        dataset_lens.append(len(ds))

    wrapper_dataset = BalancedDataset(datasets_with_probs, 2 * max(dataset_lens))

    restricted_len = hparams.n_iters * hparams.batch_size
    wrapper_dataset = RestrictedLenDataset(wrapper_dataset, restricted_len)

    collator = Collator()

    dataloader = DataLoader(wrapper_dataset, batch_size=hparams.batch_size, collate_fn=collator, shuffle=True,
                            num_workers=hparams.num_workers, pin_memory=hparams.use_cuda, worker_init_fn=worker_init_fn)
    return dataloader, [wrapper_dataset.on_post_epoch_callback]


def get_valid_dataloader(hparams: ExperimentHyperparams, worker_init_fn: Callable) -> DataLoader:
    processor = get_valid_processor(hparams)

    datasets = []
    for label, path in get_validation_index_paths().items():
        datasets.append(ResizedDataset(path, hparams, label, processor=processor))

    wrapper_dataset = SequenceDataset(datasets)

    collator = Collator()
    dataloader = DataLoader(wrapper_dataset, batch_size=hparams.batch_size, collate_fn=collator, shuffle=True,
                            num_workers=hparams.num_workers, pin_memory=hparams.use_cuda, worker_init_fn=worker_init_fn)
    return dataloader


def get_tree_building_dataloader(hparams: ExperimentHyperparams, worker_init_fn: Callable) -> DataLoader:
    processor = get_valid_processor(hparams)

    pos_train_index = get_train_index_paths()[1]
    dataset = ResizedDataset(pos_train_index, hparams, 1, processor=processor)
    dataset = RestrictedLenDataset(dataset, 100)

    collator = Collator()
    dataloader = DataLoader(dataset, batch_size=hparams.batch_size, collate_fn=collator, shuffle=False,
                            num_workers=hparams.num_workers, pin_memory=hparams.use_cuda, worker_init_fn=worker_init_fn)
    return dataloader


def get_tree_stuff(model_ckpt_path: str, tree_ckpt_path: str) -> Tuple[Dataset, Analyzer]:
    checkpoint = load_checkpoint(model_ckpt_path)
    hparams = checkpoint.data.hparams

    processor = get_valid_processor(hparams)

    pos_valid_idx = get_validation_index_paths()[1]
    dataset = ResizedDataset(pos_valid_idx, hparams, 1, processor=processor)

    forwarder = VGG16Forwarder(hparams)

    tree_ckpt_dir = os.path.dirname(tree_ckpt_path)
    iteration = int(os.path.basename(tree_ckpt_path).split(".")[0].split("_")[1])

    tree = Tree(hparams, tree_ckpt_dir)
    tree.load(iteration)
    analyzer = Analyzer(checkpoint.data.model, forwarder, tree, level=1)

    return dataset, analyzer


def init_env(device_id: Optional[int] = None) -> None:
    set_random_seed(1378)

    if device_id is not None:
        torch.cuda.set_device(device_id)


def get_log_path(exp_run_code: str) -> str:
    log_path = os.path.join(get_experiment_dir(), 'logs', exp_run_code, 'train.log')
    return log_path


def init_logging(exp_run_code: str) -> CheckpointManager:
    """
    Инициализируем логирование в файлы и tb, создаем менеджер чекпоинтов.
    """
    log_path = get_log_path(exp_run_code)
    log_dir = os.path.dirname(log_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    init_logger(logging.INFO, log_console=True, log_path=log_path)
    checkpoint_manager = CheckpointManager(get_experiment_dir(), exp_run_code,
                                           CheckpointDeleteParams(20, 50),
                                           CheckpointDeleteParams(20, 50))

    return checkpoint_manager


def load_checkpoint(checkpoint_path: str) -> CheckpointXaiCnn:
    checkpoint = CheckpointXaiCnn()
    checkpoint.load(checkpoint_path)
    return checkpoint


class WorkerIniter:
    def __init__(self, exp_run_code: str):
        self.exp_run_code = exp_run_code

    def __call__(self, worker_id: int):
        log_path = get_log_path(self.exp_run_code)
        init_logger(logging.INFO, log_console=True, log_path=log_path)
