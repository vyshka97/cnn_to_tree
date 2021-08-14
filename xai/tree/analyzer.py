# -*- coding: utf-8 -*-

import logging
import numpy as np

from typing import Any, Dict

from xai.models import XaiCnn, Forwarder, OutputType
from xai.data import Batch
from .node import Tree

logger = logging.getLogger(__name__)

__all__ = ["Analyzer"]


class Analyzer:

    def __init__(self, model: XaiCnn, forwarder: Forwarder, tree: Tree, level: int = 1):
        self.model = model.cpu()
        self.forwarder = forwarder
        self.tree = tree
        self.level = level

    def inference(self, instance: Dict[str, Any]) -> np.ndarray:
        batch = Batch(x=instance["x"].unsqueeze(0), y=instance["y"])

        model_out = self.forwarder.forward_val(self.model, batch)
        logit = model_out[OutputType.MODEL_OUTPUT][:, 1]  # [1]
        x = model_out[OutputType.NEW_INTER_MAP]  # [1, 512, 14, 14]

        x.retain_grad()
        logit.backward()

        g = x.grad.data  # [1, 512, 14, 14]
        filter_contributions = self.tree.inference(x.squeeze(0).detach(), g.squeeze(0), level=self.level)  # [512]

        return filter_contributions
