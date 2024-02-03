import os
import datetime
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
from .storage import save_checkpoint, load_checkpoint, checkpoint_dir
from .utils import setup, val_model


def create_optimizer(type, params, **kwargs):
    if type == "Adam":
        return optim.Adam(params, **kwargs)
    elif type == "SGD":
        return optim.SGD(params, **kwargs)
    return None


def create_scheduler(type, optimizer, **kwargs):
    if type == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif type == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    return None


def resume_train(model_cfg, model, optimizer, scheduler):
    ckp_path = getattr(model_cfg, "resume", "")
    try:
        return load_checkpoint(ckp_path, model, optimizer, scheduler) + 1
    except:
        return 0


def train(local_rank, cfg):
    logger.add(os.path.join(cfg.output_base,
        "train_rank_{}_{}.log".format(local_rank, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    from core.model import create_model
    from core.loss import create_loss_function
    from core.data import create_dataloader

    device = setup(local_rank, cfg.world_size, cfg.port)
    ckp_dir = checkpoint_dir(cfg)

    model_cfg = cfg.model
    data_cfg = cfg.data
    learn_cfg = cfg.learn
    train_loader = create_dataloader(data_cfg.train, data_cfg, True)
    if train_loader is None:
        logger.error("invalid train_loader")
        return
    val_loader = create_dataloader(data_cfg.train, data_cfg, False)
    if val_loader is None:
        logger.error("invalid val_loader")
        return
    model = create_model(model_cfg)
    if model is None:
        logger.error("invalid model: {}", model_cfg.name)
        return
    optimizer = create_optimizer(learn_cfg.optimizer.type, model.parameters(),
                                 **getattr(learn_cfg.optimizer, "args", {}))
    if optimizer is None:
        logger.error("invalid optimizer")
        return
    lr_scheduler_cfg = getattr(learn_cfg, "lr_scheduler", None)
    scheduler = None
    if lr_scheduler_cfg is not None:
        scheduler = create_scheduler(learn_cfg.lr_scheduler.type, optimizer,
                                    **getattr(learn_cfg.lr_scheduler, "args", {}))
        if scheduler is None:
            logger.error("invalid scheduler")
            return
    model = model.to(device)
    model = DDP(model)
    start_epoch = resume_train(model_cfg, model, optimizer, scheduler)
    loss_func = create_loss_function(learn_cfg.loss, os.path.join(cfg.output_base, "loss"))
    if loss_func is None:
        logger.error("invalid loss_func: {}", learn_cfg.loss)
        return
    for epoch in range(start_epoch, learn_cfg.epochs):
        epoch += 1
        logger.info("------------------- epoch: {}/{}, learning rate: {}", epoch, learn_cfg.epochs, scheduler.get_last_lr())
        with tqdm(total=len(train_loader)) as batch_bar:
            batch_bar.set_description("training batch: ")
            model.train()
            if local_rank == 0:
                loss_func.start_log()
            for _, input_data in enumerate(train_loader):
                input_data.to(device)
                optimizer.zero_grad()
                output_data = model(input_data)
                loss = loss_func(output_data, input_data)
                loss.backward()
                optimizer.step()
                batch_bar.update(1)
            if scheduler is not None:
                scheduler.step()

        if local_rank == 0:
            loss_func.end_log()
            if epoch % learn_cfg.validate_model_period == 0:
                val_model(model, val_loader, loss_func, device)
            if epoch % learn_cfg.save_model_period == 0:
                save_checkpoint(epoch, model, optimizer, scheduler,
                                os.path.join(ckp_dir, "epoch_{:04}.pth".format(epoch)))
        torch.cuda.empty_cache()
        dist.barrier()
    dist.destroy_process_group()
