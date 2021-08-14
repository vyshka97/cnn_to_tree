# -*- coding: utf-8 -*-

import logging
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm

from xai.models import XaiCnn, Forwarder, OutputType
from xai.hyperparams import Hyperparams
from .node import Leaf, Tree

logger = logging.getLogger(__name__)

__all__ = ["TreeBuilder"]


class TreeBuilder:

    def __init__(self, hparams: Hyperparams, model: XaiCnn, forwarder: Forwarder, dataloader: DataLoader,
                 save_dir: str):
        self.hparams = hparams
        self.model = model
        self.dataloader = dataloader
        self.forwarder = forwarder
        self.save_dir = save_dir

    def build(self) -> Tree:
        logger.info("Compute activation magnitude norms")
        activation_norms = None
        n = len(self.dataloader.dataset)
        with torch.no_grad():
            for batch in tqdm(self.dataloader):
                if self.hparams.use_cuda:
                    batch = batch.cuda()

                x = self.forwarder.forward_val(self.model, batch)[OutputType.NEW_INTER_MAP]  # [B, 512, 14, 14]
                mean = x.mean((2, 3))  # [B, 512]
                if activation_norms is None:
                    activation_norms = mean.sum(0)
                else:
                    activation_norms += mean.sum(0)

        activation_norms /= n
        tree = Tree(self.hparams, self.save_dir, activation_norms=activation_norms.cpu().numpy())

        logger.info("Add leaves to tree")
        for batch in tqdm(self.dataloader):
            if self.hparams.use_cuda:
                batch = batch.cuda()

            self.model.zero_grad()

            model_out = self.forwarder.forward_val(self.model, batch)
            logit = model_out[OutputType.MODEL_OUTPUT][:, 1]  # [B]
            logit_sum = logit.sum()
            x = model_out[OutputType.NEW_INTER_MAP]  # [B, 512, 14, 14]
            x.retain_grad()

            logit_sum.backward()

            g = x.grad.data  # [B, 512, 14, 14]

            x_v = (x.detach().sum((2, 3)) / activation_norms).cpu().numpy()  # [B, 512]
            g_v = (g.sum((2, 3)) * activation_norms) / x.shape[2] ** 2  # [B, 512]
            norms = g_v.norm(dim=1, keepdim=True)  # [B, 1]
            g_v = (g_v / norms).cpu().numpy()

            logit = (logit.detach() / norms.squeeze()).cpu().numpy()

            for idx in range(batch.size):
                if batch.y[idx].item() == 1:
                    leaf = Leaf(x_v[idx], g_v[idx], logit[idx], path=batch.paths[idx])
                    tree.add_node(leaf)

        logger.info("Train tree")
        tree.train()
        return tree
