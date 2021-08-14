# -*- coding: utf-8 -*-

import torch

from typing import Dict, Tuple, List, Union, Any, Optional
from torch import nn

from .xai_cnn import XaiCnn, OutputType, Forwarder, ForwardType, LossCalculator
from xai.hyperparams import Hyperparams
from xai.data import Batch
from xai.losses import CompositeLoss, LossType
from xai.modules import Template

__all__ = ["VGG16", "VGG16Forwarder", "VGG16LossCalculator"]


def make_encoder(cfg: List[Union[int, str]]) -> nn.Sequential:
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)


def make_classifier(in_size: int, hidden_size: int, out_size: int, dropout: float = 0.) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, out_size)
    )


class VGG16(XaiCnn):
    EXPECTED_IMAGE_SIZE = 224
    ENCODER_CONFIG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    CLF_HIDDEN_SIZE = 4096

    def __init__(self, hparams: Hyperparams):
        super().__init__(hparams)
        assert hparams.image_size[0] == hparams.image_size[1] == self.EXPECTED_IMAGE_SIZE

        self.features = make_encoder(self.ENCODER_CONFIG)

        n_channels = self.ENCODER_CONFIG[-1]

        self.new_conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pooling = nn.MaxPool2d(2, stride=2)

        features_map_size = self.get_feature_map_size
        clf_in_size = n_channels * ((features_map_size - 2) // 2 + 1) ** 2

        self.classifier = make_classifier(clf_in_size, self.CLF_HIDDEN_SIZE, hparams.model_out_size,
                                          dropout=hparams.dropout)

        self.base_template = Template(features_map_size, hparams)
        self.new_template = Template(features_map_size, hparams)

    @property
    def get_feature_map_size(self) -> int:
        size = self.EXPECTED_IMAGE_SIZE
        for layer in self.features:
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.Conv2d):
                if isinstance(layer.kernel_size, tuple):
                    kernel_size = layer.kernel_size[0]
                else:
                    kernel_size = layer.kernel_size
                if isinstance(layer.padding, tuple):
                    pad_size = layer.padding[0] + layer.padding[1]
                else:
                    pad_size = layer.padding * 2
                if isinstance(layer.stride, tuple):
                    stride = layer.stride[0]
                else:
                    stride = layer.stride
                size = (size - kernel_size + pad_size) // stride + 1
        return size

    # we expect that x with dimensions [B, 3, 224, 224]
    def default_forward(self, x: torch.Tensor) -> Dict[OutputType, Any]:
        result = dict()
        shape = x.shape
        assert len(shape) == 4 and shape[1] == 3
        assert shape[2] == self.hparams.image_size[0] and shape[3] == self.hparams.image_size[1]

        base_map = self.features(x)  # [B, 3, 224, 224] -> [B, 512, 14, 14]
        result[OutputType.BASE_INTER_MAP] = base_map
        if self.hparams.train_interpretable:
            x = self.base_template.get_masked_output(base_map)  # [B, 512, 14, 14] -> [B, 512, 14, 14]
        else:
            x = base_map

        new_map = self.new_conv(x)  # [B, 512, 14, 14] -> [B, 512, 14, 14]
        result[OutputType.NEW_INTER_MAP] = new_map

        if self.hparams.train_interpretable:
            x = self.new_template.get_masked_output(new_map)  # [B, 512, 14, 14] -> [B, 512, 14, 14]
        else:
            x = new_map

        x = self.pooling(x)  # [B, 512, 14, 14] -> [B, 512, 7, 7]

        x = x.flatten(start_dim=1)  # [B, 512, 7, 7] -> [B, 25088]

        logit = self.classifier(x)  # [B, 25088] -> [B, number of classes]
        result[OutputType.MODEL_OUTPUT] = logit

        return result

    def cuda(self, device: Optional[Union[int, torch.device]] = None) -> "VGG16":
        super().cuda(device=device)
        self.base_template = self.base_template.cuda()
        self.new_template = self.new_template.cuda()
        return self


class VGG16Forwarder(Forwarder):

    def forward_train(self, model: VGG16, batch: Batch) -> Dict[OutputType, Any]:
        return model.forward(ForwardType.DEFAULT_FORWARD, batch.x)

    def forward_val(self, model: VGG16, batch: Batch) -> Dict[OutputType, Any]:
        return self.forward_train(model, batch)


class VGG16LossCalculator(LossCalculator):

    def __init__(self, hparams: Hyperparams, train_loss: CompositeLoss, val_loss: CompositeLoss):
        super().__init__(hparams)
        self.train_loss = train_loss
        self.val_loss = val_loss

    def calc_train(self, fwd_res: Dict[OutputType, Any], b: Batch, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_input = {
            LossType.CROSS_ENTROPY: {
                'input': fwd_res[OutputType.MODEL_OUTPUT],
                'target': b.y,
            },
        }
        if self.hparams.train_interpretable:
            loss_input[LossType.BASE_FILTER] = {"input": fwd_res[OutputType.BASE_INTER_MAP]}
            loss_input[LossType.NEW_FILTER] = {"input": fwd_res[OutputType.NEW_INTER_MAP]}

        return self.train_loss.calculate(loss_input, epoch)

    def calc_val(self, fwd_res: Dict[OutputType, Any], b: Batch, epoch: int) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_input = {
            LossType.CROSS_ENTROPY: {
                'input': fwd_res[OutputType.MODEL_OUTPUT],
                'target': b.y,
            },
            LossType.BASE_FILTER: {"input": fwd_res[OutputType.BASE_INTER_MAP]},
            LossType.NEW_FILTER: {"input": fwd_res[OutputType.NEW_INTER_MAP]},
        }
        return self.val_loss.calculate(loss_input, epoch)

    def cpu(self) -> "VGG16LossCalculator":
        self.train_loss = self.train_loss.cpu()
        self.val_loss = self.val_loss.cpu()
        return self

    def cuda(self) -> "VGG16LossCalculator":
        self.train_loss = self.train_loss.cuda()
        self.val_loss = self.val_loss.cuda()
        return self
