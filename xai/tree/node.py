# -*- coding: utf-8 -*-

import os
import torch
import shutil
import logging
import numpy as np
import pickle

from typing import Union, Optional
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm

from xai.hyperparams import Hyperparams

logger = logging.getLogger(__name__)

__all__ = ["Leaf", "Root", "Tree"]


class Node:
    def __init__(self):
        self.rationales = None  # [512]
        self.bias = None  # scalar


class Leaf(Node):
    def __init__(self, activations: np.ndarray, rationales: np.ndarray, logit: Union[float, np.ndarray],
                 path: Optional[str] = None):
        super().__init__()

        self.path = path

        if isinstance(logit, float):
            logit = np.array(logit)

        self.activations = activations  # [512]
        self.rationales = rationales  # [512]
        self.logit = logit  # [1]
        self.bias = (logit - activations @ rationales).item()  # scalar


class Root(Node):
    def __init__(self, beta: float, lambd: float, n_channels: int):
        super().__init__()

        self.beta = beta
        self.lambd = lambd
        self.n_channels = n_channels

        # initial random rationales
        self.rationales = np.random.randn(n_channels)  # [512]
        self.rationales /= np.linalg.norm(self.rationales, ord=2, keepdims=True)

        self.log_likelihood = None  # log likelihood

        self.gamma = None  # gamma

        self.n_children = 0  # number of children
        self.children = []  # list of children
        self.children_rationales = np.empty((n_channels, 0))  # [512, len(children)]
        self.children_biases = np.empty(0)  # [len(children)]

        self.n_leaves = 0  # number of leaves
        self.leaves_rationales = np.empty((n_channels, 0))  # [512, len(leaves)]
        self.leaves_activations = np.empty((n_channels, 0))  # [512, len(leaves)]
        self.leaves_logits = np.empty(0)  # [len(leaves)]

        self.criterion = None  # last criterion value

    def add_node(self, node: Node) -> None:
        self.children.append(node)
        self.children_rationales = np.append(self.children_rationales, node.rationales[:, None], axis=1)
        self.children_biases = np.append(self.children_biases, node.bias)
        self.n_children += 1

        # leaf or subroot
        if isinstance(node, Leaf):
            self.n_leaves += 1
            self.leaves_rationales = np.append(self.leaves_rationales, node.rationales[:, None].copy(), axis=1)
            self.leaves_activations = np.append(self.leaves_activations, node.activations[:, None].copy(), axis=1)
            self.leaves_logits = np.append(self.leaves_logits, node.logit.copy())
        else:
            self.leaves_rationales = np.append(self.leaves_rationales, node.leaves_rationales.copy(), axis=1)
            self.leaves_activations = np.append(self.leaves_activations, node.leaves_activations.copy(), axis=1)
            self.leaves_logits = np.append(self.leaves_logits, node.leaves_logits.copy())
            self.n_leaves += node.n_leaves

    def compute_rationales(self, compute_opt_params: bool = False) -> None:
        g_sum = self.leaves_rationales.sum(1)  # [512, len(leaves)] -> [512]
        objective = lambda g: -(g @ g_sum)
        constraint = {"type": "eq", "fun": lambda g: (g ** 2).sum() - 1}
        self.rationales = minimize(objective, self.rationales, method="SLSQP", constraints=constraint).x  # [512]

        # TODO: optimize alphas with bias
        alphas = np.ones(self.n_channels, dtype=np.int8)  # [512]
        weights = alphas * self.rationales
        bias_error = weights @ self.leaves_activations - self.leaves_logits  # [len(leaves)]
        objective = lambda b: ((bias_error + b) ** 2).mean()
        self.bias = minimize(objective, 0, method="SLSQP").x.item()

        if compute_opt_params:
            self.criterion = -self.beta * self.n_children
            logger.info(f"Criterion: {self.criterion}")

            self.gamma = 1 / self.leaves_logits.mean().item()  # [len(leaves)] -> scalar
            self.log_likelihood = np.log(softmax(self.leaves_logits * self.gamma)).sum().item()

    def compute_log_likelihood(self) -> float:
        # [512, n_children] -> [1, n_children]
        child_norms = np.linalg.norm(self.children_rationales[:, :self.n_children], ord=2, axis=0, keepdims=True)
        children_rationales = self.children_rationales[:, :self.n_children] / child_norms  # [512, n_children]

        # [len(leaves), 512] @ [512, n_children] -> [len(leaves), n_children] -> [len(leaves)]
        argmax = (self.leaves_rationales.T @ children_rationales).argmax(1)

        best_rationales = self.children_rationales[:, argmax]  # [512, len(leaves)]
        best_biases = self.children_biases[argmax]

        # [len(leaves), 512] @ [512, len(leaves)] -> [len(leaves), len(leaves)] -> [len(leaves)]
        predictions = np.diag(best_rationales.T @ self.leaves_activations) + best_biases
        log_likelihood = np.log(softmax(predictions * self.gamma)).sum().item()

        return log_likelihood

    def get_best_child(self, rationales: np.ndarray) -> Node:
        # [512] @ [512, n_children] -> [n_children] -> scalar
        argmax = (rationales @ self.children_rationales).argmax()
        return self.children[argmax.item()]

    def optimize(self) -> bool:
        max_diff = None
        best_pair = (None, None)
        best_weights = (None, None)
        best_criterion = None

        n = self.n_children
        pbar = tqdm(total=n * (n - 1) // 2)

        for i in range(n):
            for j in range(n):
                if i >= j:
                    continue

                pbar.update(1)

                # merging
                root = Root(self.beta, self.lambd, self.n_channels)
                first, second = self.children[i], self.children[j]
                root.add_node(first)
                root.add_node(second)
                root.compute_rationales(compute_opt_params=False)
                factor = root.n_leaves

                self.children[i] = root
                self.children_rationales[:, i] = root.rationales.copy()
                self.children_biases[i] = root.bias
                self.children[j] = self.children[-1]
                self.children_rationales[:, j] = self.children_rationales[:, -1]
                self.children_biases[j] = self.children_biases[-1]
                self.n_children -= 1

                # log likelihood computation
                log_likelihood = self.compute_log_likelihood()

                # unmerging: reset to previous state
                self.children[i] = first
                self.children_rationales[:, i] = first.rationales
                self.children_biases[i] = first.bias
                self.children[j] = second
                self.children_rationales[:, j] = second.rationales
                self.children_biases[j] = second.bias
                self.n_children += 1

                new_criterion = log_likelihood - self.log_likelihood - self.beta * (n - 1)
                diff = new_criterion - self.criterion
                if diff <= 0:
                    continue
                diff /= factor

                if max_diff is None or max_diff < diff:
                    max_diff = diff
                    best_pair = (i, j)
                    best_weights = (root.rationales, root.bias)
                    best_criterion = new_criterion

        logger.info(f"Criterion: {best_criterion}")
        if max_diff is None:
            return False

        self.criterion = best_criterion
        logger.info(f"Merge {best_pair[0]} and {best_pair[1]} to {best_pair[0]}")

        # merging
        root = Root(self.beta, self.lambd, self.n_channels)
        first, second = self.children[best_pair[0]], self.children[best_pair[1]]
        root.add_node(first)
        root.add_node(second)
        root.rationales = best_weights[0]
        root.bias = best_weights[1]

        self.children[best_pair[0]] = root
        self.children_rationales[:, best_pair[0]] = root.rationales.copy()
        self.children_biases[best_pair[0]] = root.bias
        self.children[best_pair[1]] = self.children[-1]
        self.children_rationales[:, best_pair[1]] = self.children_rationales[:, -1]
        self.children_biases[best_pair[1]] = self.children_biases[-1]
        self.n_children -= 1
        self.children = self.children[:-1]
        self.children_rationales = self.children_rationales[:, :-1]
        self.children_biases = self.children_biases[:-1]

        return True


class Tree:
    def __init__(self, hparams: Hyperparams, snapshot_dir: str, activation_norms: Optional[np.ndarray] = None):
        self.hparams = hparams
        self.snapshot_dir = snapshot_dir
        self.activation_norms = activation_norms  # [512]  # for inference
        if activation_norms is not None:
            self.root = Root(hparams.tree_beta, hparams.lambd, activation_norms.shape[0])

    def add_node(self, node: Node) -> None:
        self.root.add_node(node)

    def drop_snapshots(self) -> None:
        if os.path.isdir(self.snapshot_dir):
            shutil.rmtree(self.snapshot_dir)

        os.makedirs(self.snapshot_dir, exist_ok=True)

    def save(self, iteration: int) -> None:
        dct_to_save = {
            "activation_norms": self.activation_norms,
            "root": self.root,
        }
        path = os.path.join(self.snapshot_dir, f"tree_{iteration}.pkl")
        with open(path, 'wb') as __fout:
            pickle.dump(dct_to_save, __fout)

    def load(self, iteration: int = -1) -> None:
        filenames = []
        for filename in os.listdir(self.snapshot_dir):
            it = int(filename.split("_")[1].split(".")[0])
            filenames.append((filename, it))

        filenames.sort(key=lambda item: item[1])
        load_path = os.path.join(self.snapshot_dir, filenames[iteration][0])

        with open(load_path, 'rb') as __fib:
            dct = pickle.load(__fib)
            self.root = dct["root"]
            self.activation_norms = dct["activation_norms"]

    def train(self, n_iters: int = 30, start_iter: int = 0) -> None:
        if start_iter == 0:
            self.drop_snapshots()
            self.root.compute_rationales(compute_opt_params=True)
            self.save(0)
        else:
            self.load(start_iter - 1)

        iter_num = 1
        while iter_num <= n_iters:
            logger.info(f"Train iteration: {iter_num + start_iter - 1}")
            success = self.root.optimize()
            if not success:
                logger.info("The tree cannot be optimize. Stop training")
                return
            self.save(iter_num + start_iter - 1)
            iter_num += 1

        logger.info("Number of iterations exceeds limit. Stop training")

    def inference(self, activations: torch.Tensor, rationales: torch.Tensor, level: int = -1) -> np.ndarray:
        if self.root is None:
            raise ValueError("Load tree before inference")

        # [512, 14, 14] -> [512] / [512] -> [512]
        activations = activations.sum((1, 2)).numpy() / self.activation_norms
        # [512, 14, 14] -> [512] * [512] -> [512]
        rationales = rationales.sum((1, 2)).numpy() * self.activation_norms

        node = self.root
        level_ = 1
        while isinstance(node, Root) and (level_ == -1 or level_ <= level):
            node = node.get_best_child(rationales)
            level_ += 1

        filter_contributions = node.rationales * activations

        return filter_contributions
