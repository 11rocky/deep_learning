import os
import yaml
from easydict import EasyDict
from tqdm import tqdm
import torch
import torch.distributed as dist


def init_dist(local_rank, cfg):
    rank = local_rank + cfg.node_rank * cfg.local_size
    cfg.local_rank = local_rank
    cfg.rank = rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=cfg.world_size)


def init_cfg(cfg_file):
    with open(cfg_file, "r") as f:
        cfg = yaml.load(f, yaml.FullLoader)
    cfg = EasyDict(cfg)
    cfg.name = os.path.splitext(os.path.basename(cfg_file))[0]
    cfg.output_train = os.path.join("output", cfg.name, "train")
    cfg.output_test = os.path.join("output", cfg.name, "test")
    return cfg


def validate_model(model, data_loader, loss_func, device, save_fn=None):
    model.eval()
    loss_func.reset()
    rank = dist.get_rank()
    with torch.no_grad():
        with tqdm(total=len(data_loader), colour="green") as batch_bar:
            if rank == 0:
                batch_bar.set_description("validate batch: ")
            for _, input_data in enumerate(data_loader):
                input_data.to(device)
                pred_data = model(input_data)
                loss_func(pred_data, input_data)
                dist.barrier()
                loss_func.summary()
                if save_fn is not None:
                    save_fn(input_data, pred_data)
                if rank == 0:
                    batch_bar.update(1)
        if rank == 0:
            loss_func.record()
