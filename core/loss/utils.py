import os
import importlib
import inspect
from loguru import logger
import torch.nn as nn


def get_loss_cls(name):
    cur_package = os.path.dirname(__file__)
    items = os.listdir(cur_package)
    for item in items:
        mod = os.path.splitext(item)[0]
        if mod == "loss":
            continue
        module = importlib.import_module('core.loss.{}'.format(mod))
        for _, member in inspect.getmembers(module):
            if inspect.isclass(member) and issubclass(member, nn.Module) and member.__name__ == name:
                return member
    return None

