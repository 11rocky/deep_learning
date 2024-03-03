import os
import datetime
from loguru import logger
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from core.model import create_model
from core.loss import create_loss_function
from core.data import create_dataloader
from .storage import load_model, checkpoint_dir
from .utils import init_dist, validate_model


def test(local_rank, cfg):
    init_dist(local_rank, cfg)
    logger.add(os.path.join(cfg.output_test,
        "test_rank_{}_{}.log".format(cfg.rank, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    ckp_dir = checkpoint_dir(cfg)
    model_cfg = cfg.model
    learn_cfg = cfg.learn
    test_cfg = cfg.test
    data_cfg = test_cfg.data

    test_loader = create_dataloader(data_cfg.test, data_cfg, False)
    if test_loader is None:
        logger.error("invalid test_loader")
        return
    model = create_model(model_cfg)
    if model is None:
        logger.error("invalid model: {}", model_cfg.name)
        return
    model = model.to(device)
    model = DDP(model)
    loss_func = create_loss_function(learn_cfg.loss, os.path.join(cfg.output_test, "loss"),
                                     cfg.world_size, False)
    if loss_func is None:
        logger.error("invalid loss_func: {}", learn_cfg.loss)
        return
    load_model(os.path.join(ckp_dir, test_cfg.checkpoint), model)
    validate_model(model, test_loader, loss_func, device)
