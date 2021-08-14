# -*- coding: utf-8 -*-

import logging
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = ["set_random_seed"]


def set_random_seed(seed: int = 987) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    logger.info('deterministic mode, seed=%d', seed)
