import os
import importlib
import inspect
from loguru import logger
import torch.nn as nn


def get_model_cls(name):
    cur_package = os.path.dirname(__file__)
    items = os.listdir(cur_package)
    for item in items:
        module = importlib.import_module('core.model.{}'.format(os.path.splitext(item)[0]))
        for _, member in inspect.getmembers(module):
            if inspect.isclass(member) and issubclass(member, nn.Module) and member.__name__ == name:
                return member
    return None


def create_model(opt: dict):
    cls = get_model_cls(opt.name)
    if cls == None:
        logger.error("can not find model class: {}", opt.name)
        return None
    return cls(opt)
