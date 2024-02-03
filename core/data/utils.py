from loguru import logger
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils import data
from core.utils import get_cls_in_package
from .custom_data import InputData


def create_dataloader(descs, opt:dict, train):
    datasets = []
    cls = get_cls_in_package("core.data", opt.dataset, data.Dataset)
    if cls == None:
        logger.error("can not find dataset class: {}", opt.dataset)
        return None
    for idx, desc in enumerate(descs):
        datasets.append(cls(idx, desc, opt, train))
    dataset = ConcatDataset(datasets)
    sampler = DistributedSampler(dataset)
    return DataLoader(dataset, batch_size=opt.batch_size, sampler=sampler,
                      num_workers=opt.workers, collate_fn=InputData.collate_fn)
