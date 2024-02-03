import os
import torch
from core.utils import get_cls_in_module, get_cls_in_package


def get_loss_cls(name):
    base_cls = torch.nn.modules.loss._Loss
    cls = get_cls_in_module("torch.nn.modules.loss", name, base_cls)
    if cls is not None:
        return cls
    return get_cls_in_package("core.loss", name, base_cls)
