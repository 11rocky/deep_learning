import os
import re
import torch
import torch.nn as nn
import torch.distributed as dist
from loguru import logger
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .utils import get_loss_cls


class Loss(nn.Module):
    from core.data.custom_data import OutputData, InputData
    def __init__(self, opt, path: str, nprocs: int, train_mode: bool) -> None:
        super(Loss, self).__init__()
        self.nprocs = nprocs
        self.train_mode = train_mode
        loss = opt.formula
        self._funcs = []
        self._losses = {}
        self._losses_history = {}
        self._total_name = "Total"
        funcs = loss.split("+")
        for idx, func in enumerate(funcs):
            parts = func.strip().split("*")
            if len(parts) == 2:
                scale = float(parts[0].strip())
                name = parts[1].strip()
            else:
                scale = 1.0
                name = parts[0].strip()
            only_train = name.endswith("?")
            name = name.replace("?", "")
            pattern = re.compile(r"^(.+?)\s*\(\s*(.+?)\s*,\s*(.+?)\)$")
            res = pattern.search(name)
            if res is None:
                inputs = None
            else:
                name = res[1]
                inputs = [res[2], res[3]]
            args = getattr(opt, name, {})
            cls = get_loss_cls(name)
            logger.info("loss config: scale={}, name={}, inputs={}, only_train={}", scale, name, inputs, only_train)
            assert cls is not None, "can not get loss name={}".format(name)
            if only_train and not train_mode:
                continue
            self._funcs.append((name, scale, cls(**args), inputs))
            self._losses[f"{name}_{idx}"] = []
            self._losses_history[f"{name}_{idx}"] = []
        self._losses[self._total_name] = []
        self._losses_history[self._total_name] = []
        self._save_path = os.path.join(path, "train" if train_mode else "validate")
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    def load_losses(self, history):
        self._losses_history = history
    
    def get_losses(self):
        return self._losses_history

    def reset(self):
        for name in self._losses.keys():
            self._losses[name] = []

    def record(self):
        for name, losses in self._losses.items():
            loss = sum(losses) / len(losses)
            logger.info("  {}: {}", name, loss)
            self._losses_history[name].append(loss)
            plt.clf()
            plt.xlabel("epoch")
            plt.ylabel("loss")
            epoch = len(self._losses_history[name])
            plt.xlim(1, epoch + 1)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.plot([x + 1 for x in range(0, epoch)], self._losses_history[name])
            plt.savefig(os.path.join(self._save_path, f"{name}.jpg"))

    def summary(self):
        def reduce_mean(loss: torch.Tensor):
            rt = loss.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= self.nprocs
            return rt

        mean_losses = {}
        for name, loss in self._losses.items():
            mean_loss = reduce_mean(loss[-1])
            mean_losses[name] = mean_loss
        for name, loss in mean_losses.items():
            self._losses[name][-1] = mean_loss.item()

    def forward(self, output: OutputData, input: InputData):
        total_loss = 0
        for idx, (name, scale, loss_func, inputs) in enumerate(self._funcs):
            if inputs is None:
                loss = scale * loss_func(output, input)
            else:
                args = []
                for i in inputs:
                    args.append(eval(i))
                    assert args[-1] is not None, f"{i} is none"
                loss = scale * loss_func(*args)
            self._losses[f"{name}_{idx}"].append(loss)
            total_loss += loss
        self._losses[self._total_name].append(total_loss)
        return total_loss


def create_loss_function(opt, save_path, nproc, train_mode):
    return Loss(opt, save_path, nproc, train_mode)
