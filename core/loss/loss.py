import os
import re
import torch.nn as nn
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .utils import get_loss_cls


class Loss(nn.Module):
    from core.data.custom_data import OutputData, InputData
    def __init__(self, opt: str, path: str) -> None:
        super(Loss, self).__init__()
        loss = opt.formula
        self._funcs = []
        self._losses = {}
        self._train_losses = {}
        self._total_name = "Total"
        funcs = loss.split("+")
        for func in funcs:
            parts = func.strip().split("*")
            if len(parts) == 2:
                scale = float(parts[0].strip())
                name = parts[1].strip()
            else:
                scale = 1.0
                name = parts[0].strip()
            pattern = re.compile(r"^(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\)$")
            res = pattern.search(name)
            if res is None:
                inputs = None
            else:
                name = res[1]
                inputs = [res[2], res[3]]
            args = getattr(opt, name, {})
            cls = get_loss_cls(name)
            logger.info("loss config: scale={}, name={}, inputs={}", scale, name, inputs)
            assert cls is not None, "can not get loss name={}".format(name)
            self._funcs.append((name, scale, cls(**args), inputs))
            self._losses[name] = []
            self._train_losses[name] = []
        self._losses[self._total_name] = []
        self._train_losses[self._total_name] = []
        self._cal_times = 0
        self._save_path = path
        self._in_train = True
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)
    
    def load_state(self, state):
        self._train_losses = state

    def state(self):
        return self._train_losses

    def start_log(self, train=True):
        self._in_train = train
        for loss in self._losses.values():
            loss.clear()
        self._cal_times = 0

    def end_log(self):
        if self._in_train:
            logger.info("training loss:")
        else:
            logger.info("validate loss:")
        for name, losses in self._losses.items():
            loss = sum(losses) / self._cal_times
            logger.info("  {}: {}", name, loss)
            if self._in_train:
                self._train_losses[name].append(loss)
                plt.clf()
                plt.xlabel("epoch")
                plt.ylabel("loss")
                epoch = len(self._train_losses[name])
                plt.xlim(1, epoch + 1)
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.plot([x + 1 for x in range(0, epoch)], self._train_losses[name])
                plt.savefig(os.path.join(self._save_path, f"{name}.jpg"))

    def forward(self, output: OutputData, input: InputData):
        total_loss = 0
        for (name, scale, loss_func, inputs) in self._funcs:
            if inputs is None:
                loss = scale * loss_func(output, input)
            else:
                args = []
                for i in inputs:
                    args.append(eval(i))
                loss = scale * loss_func(*args)
            self._losses[name].append(loss.item())
            total_loss += loss
        self._losses[self._total_name].append(total_loss.item())
        self._cal_times += 1
        return total_loss


def create_loss_function(opt, save_path):
    return Loss(opt, save_path)
