# -*- coding: utf-8 -*-

import logging

__all__ = ["init_logger"]


def init_logger(log_level: int = logging.WARNING, log_console: bool = True, log_path: str = None):
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s:\t%(message)s', datefmt='%d.%m.%Y %H:%M:%S')
    handlers = []
    if log_console:
        handlers.append(logging.StreamHandler())
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path))
    root_logger = logging.getLogger()

    root_logger.handlers = []
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        root_logger.addHandler(handler)

    root_logger.setLevel(log_level)
