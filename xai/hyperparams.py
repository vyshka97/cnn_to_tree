# -*- coding: utf-8 -*-

import attr


@attr.s
class Hyperparams:
    # model params
    use_pretrained = attr.ib(True, type=bool)
    train_interpretable = attr.ib(True, type=bool)
    beta = attr.ib(0.5, type=float)
    model_out_size = attr.ib(2, type=int)
    image_size = attr.ib((224, 224), type=tuple)
    dropout = attr.ib(0., type=float)

    # training params
    n_iters = attr.ib(50, type=int)  # number of training iterations
    use_cuda = attr.ib(True, type=bool)
    clip_grad = attr.ib(0., type=float)
    batch_size = attr.ib(32, type=int)
    num_workers = attr.ib(4, type=int)
    lr = attr.ib(1e-3, type=float)
    momentum = attr.ib(0.9, type=float)
    weight_decay = attr.ib(5e-4, type=float)
    use_flipping = attr.ib(False, type=bool)

    # filter loss params
    f_loss_weight = attr.ib(5e-6, type=float)

    # log params
    log_iter_metrics_every = attr.ib(10, type=int)
    log_epoch_metrics_every = attr.ib(1, type=int)

    # decision tree building params
    lambd = attr.ib(1e-6, type=float)
    tree_beta = attr.ib(1, type=float)
