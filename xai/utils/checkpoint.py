# -*- coding: utf-8 -*-

import datetime
import hashlib
import logging
import os
import sys
import attr
import math
import torch

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from torch import nn
from torch.optim.optimizer import Optimizer

from xai.hyperparams import Hyperparams

logger = logging.getLogger(__name__)
__all__ = ["TrainerCheckpointParams", "CheckpointDeleteParams", "CheckpointManager", "Checkpoint", "CheckpointXaiCnn"]


@attr.s
class TrainerCheckpointParams:
    hparams: Hyperparams = attr.ib()
    model: nn.Module = attr.ib()
    optimizer: Optimizer = attr.ib(default=None)
    epoch: int = attr.ib(default=None)
    global_iteration: int = attr.ib(default=None)
    score: float = attr.ib(default=None)
    best_score: float = attr.ib(default=None)
    run_name: str = attr.ib(default=None)


@attr.s
class CheckpointDeleteParams:
    # save checkpoints all last n checkpoints
    n: int = attr.ib(default=10)
    # save every k-th for all before
    k: int = attr.ib(default=10)


class CheckpointManager:
    def __init__(self, experiment_dir: str, run_name: str,
                 best_checkpoint_delete_params: CheckpointDeleteParams = CheckpointDeleteParams(30, 2),
                 other_checkpoint_delete_params: CheckpointDeleteParams = CheckpointDeleteParams()):

        self.experiment_dir = experiment_dir
        self.run_name = run_name
        self.best_score = math.inf

        self.best_checkpoints_paths = []
        self.best_checkpoint_delete_params = best_checkpoint_delete_params

        self.other_checkpoints_paths = []
        self.other_checkpoint_delete_params = other_checkpoint_delete_params

        self.checkpoint_dir = os.path.join(self.experiment_dir, 'models', self.run_name)
        logger.info(f'checkpoint_dir: {self.checkpoint_dir}')

    def save_checkpoint(self, data: TrainerCheckpointParams) -> None:
        is_best = False
        score = data.score

        if score < self.best_score:
            is_best = True
            self.best_score = score

        data.best_score = self.best_score
        data.run_name = self.run_name

        checkpoint_name = 'checkpoint' + '%06d' % data.epoch + ('.best' if is_best else '') + '.pth'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        checkpoint = CheckpointXaiCnn(data)
        checkpoint.save(checkpoint_path)

        if is_best:
            self.best_checkpoints_paths.append([checkpoint_path, True])
        else:
            self.other_checkpoints_paths.append([checkpoint_path, True])

        self.best_checkpoints_paths = self.delete_old_checkpoints(self.best_checkpoints_paths,
                                                                  self.best_checkpoint_delete_params)

        self.other_checkpoints_paths = self.delete_old_checkpoints(self.other_checkpoints_paths,
                                                                   self.other_checkpoint_delete_params)

    @staticmethod
    def delete_old_checkpoints(checkpoint_paths: List[list], delete_params: CheckpointDeleteParams) -> List[list]:
        for i in range(len(checkpoint_paths) - delete_params.n):
            if i % delete_params.k != 0 and checkpoint_paths[i][1]:
                path: str = checkpoint_paths[i][0]
                os.remove(path)
                checkpoint_paths[i][1] = False
        return checkpoint_paths


class Checkpoint:
    def __init__(self, data: Optional[TrainerCheckpointParams] = None):
        self.data = data
        self.timestamp = datetime.datetime.now().timestamp()
        self.checkpoint_path = None

    def save(self, checkpoint_path: str) -> None:
        snapshot: Dict[str, Any] = self._prepare_snapshot()

        snapshot['run_name'] = self.data.run_name
        snapshot['timestamp'] = self.timestamp

        p = Path(checkpoint_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('wb') as __fout:
            torch.save(snapshot, __fout)

    def _prepare_snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load(self, checkpoint_path: str) -> None:
        snapshot = torch.load(checkpoint_path, map_location="cpu")

        self.data = self._load_data(snapshot)

        self.data.run_name = snapshot['run_name']
        self.timestamp = snapshot['timestamp']

    def _load_data(self, snapshot: Dict[str, Any]) -> TrainerCheckpointParams:
        raise NotImplementedError()


class CheckpointXaiCnn(Checkpoint):
    def _prepare_snapshot(self) -> Dict[str, Any]:

        snapshot = {
            'hparams_dict': attr.asdict(self.data.hparams),
            'hparams_type': self._get_class_type(self.data.hparams),
            'model_type': self._get_class_type(self.data.model),
            'state_dict': self.data.model.state_dict(),
            'optimizer_type': self._get_class_type(self.data.optimizer),
            'optimizer_state_dict': self.data.optimizer.state_dict(),
            'epoch': self.data.epoch,
            'global_iteration': self.data.global_iteration,
        }

        return snapshot

    def on_post_load_hparams(self, hparams: Hyperparams) -> Hyperparams:
        return hparams

    def _load_data(self, snapshot: Dict[str, Any]) -> TrainerCheckpointParams:
        hparams = self._load_hparams(snapshot)
        hparams = self.on_post_load_hparams(hparams)

        if hparams is None:
            logger.error('hparams is None')
            raise Exception('checkpoint loading error')

        model = self._load_model(snapshot, hparams)
        optimizer = self._load_optimizer(snapshot, model_parameters=model.parameters())

        data = TrainerCheckpointParams(hparams=hparams, model=model, optimizer=optimizer,
                                       global_iteration=snapshot['global_iteration'], epoch=snapshot['epoch'])
        return data

    def _load_hparams(self, snapshot: Dict[str, Any]) -> Hyperparams:
        try:
            module_name, class_name = snapshot['hparams_type']
            hparams_dict: dict = snapshot['hparams_dict']
            hparams: Hyperparams = self._load_class(module_name, class_name)(**hparams_dict)
        except Exception as e:
            logger.warning("can't load hparams from checkpoint: %s" % e)
            hparams = None
        return hparams

    def _load_model(self, snapshot: Dict[str, Any], hparams: Hyperparams) -> nn.Module:
        try:
            module_name, class_name = snapshot['model_type']
            model: nn.Module = self._load_class(module_name, class_name)(hparams)
            model: nn.Module = self._load_xaicnn_state_dict(model, snapshot['state_dict'])
        except Exception as e:
            logger.warning("can't load model from checkpoint: %s" % e)
            model = None
        return model

    @staticmethod
    def _load_xaicnn_state_dict(model: nn.Module, state_dict: dict) -> nn.Module:
        parallel_model_state_loaded = False
        for k in state_dict.keys():
            if k.startswith('module.'):
                parallel_model_state_loaded = True
            break
        if parallel_model_state_loaded and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        elif not parallel_model_state_loaded and isinstance(model, torch.nn.DataParallel):
            model = model.module

        model_keys = model.state_dict().keys()
        remove_keys = []
        for k in state_dict.keys():
            if k not in model_keys:
                remove_keys.append(k)
        for k in remove_keys:
            state_dict.pop(k, None)

        model.load_state_dict(state_dict)

        return model

    def _load_optimizer(self, snapshot: Dict[str, Any], model_parameters) -> Optimizer:
        try:
            module_name, class_name = snapshot['optimizer_type']
            optimizer = self._load_class(module_name, class_name)(model_parameters, lr=0.9)
            optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        except Exception as e:
            logger.warning("can't load optimizer from checkpoint: %s" % e)
            optimizer = None
        return optimizer

    @staticmethod
    def _get_file_hash(path):
        def file_as_bytes(file):
            with file:
                return file.read()

        return hashlib.md5(file_as_bytes(open(path, 'rb'))).hexdigest()

    @staticmethod
    def _get_class_type(class_instance) -> Optional[Tuple[str, str]]:
        if class_instance is None:
            return

        module_name: str = class_instance.__module__
        full_class_name: str = str(type(class_instance)).replace("<class '", "").replace("'>", "")
        class_name: str = full_class_name.replace(module_name + '.', '')
        return module_name, class_name

    @staticmethod
    def _load_class(module_name: str, class_name: str) -> type:
        __import__(module_name)
        module = sys.modules[module_name]
        class_exemplar = module.__getattribute__(class_name)
        return class_exemplar
