import os
import importlib
import inspect
from loguru import logger
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils import data
from .custom_data import InputData


def get_dataset_cls(name):
    cur_package = os.path.dirname(__file__)
    items = os.listdir(cur_package)
    for item in items:
        module = importlib.import_module("core.data.{}".format(os.path.splitext(item)[0]))
        for _, member in inspect.getmembers(module):
            if inspect.isclass(member) and issubclass(member, data.Dataset) and member.__name__ == name:
                return member
    return None


def create_dataloader(descs, opt:dict, train):
    datasets = []
    cls = get_dataset_cls(opt.dataset)
    if cls == None:
        logger.error("can not find dataset class: {}", opt.dataset)
        return None
    for idx, desc in enumerate(descs):
        datasets.append(cls(idx, desc, opt, train))
    dataset = ConcatDataset(datasets)
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler,
                      num_workers=opt.workers, collate_fn=InputData.collate_fn)
