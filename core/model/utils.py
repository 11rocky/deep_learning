from loguru import logger
import torch.nn as nn
from core.utils import get_cls_in_package


def create_model(opt: dict):
    cls = get_cls_in_package("core.model", opt.name, nn.Module)
    if cls == None:
        logger.error("can not find model class: {}", opt.name)
        return None
    return cls(opt)
