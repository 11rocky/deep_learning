import os
import math
import datetime
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from .storage import load_model, checkpoint_dir
from .utils import setup, val_model


def test(local_rank, cfg):
    logger.add(os.path.join(cfg.output_base,
        "train_rank_{}_{}.log".format(local_rank, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))))
    from core.model import create_model
    from core.loss import create_loss_function
    from core.data import create_dataloader
    model_cfg = cfg.model
    learn_cfg = cfg.learn
    test_cfg = cfg.test
    data_cfg = test_cfg.data
    ckp_dir = checkpoint_dir(cfg)
    device = setup(local_rank, cfg.world_size, cfg.port)
    descs_num = len(data_cfg.test)
    each_num = int(math.ceil(descs_num / cfg.world_size))
    desc_offset = local_rank * each_num
    descs = data_cfg.test[desc_offset:desc_offset + each_num]
    test_loader = create_dataloader(descs, data_cfg, False)
    if test_loader is None:
        logger.error("invalid test_loader")
        return
    model = create_model(model_cfg)
    if model is None:
        logger.error("invalid model: {}", model_cfg.name)
        return
    model = model.to(device)
    model = DDP(model)
    load_model(os.path.join(ckp_dir, test_cfg.checkpoint), model)
    loss_func = create_loss_function(learn_cfg.loss, os.path.join(cfg.output_base, "loss"))
    if loss_func is None:
        logger.error("invalid loss_func: {}", learn_cfg.loss)
        return
    val_model(model, test_loader, loss_func, device)
