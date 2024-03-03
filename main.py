import os
import shutil
import argparse
from loguru import logger
import torch
import torch.multiprocessing as mp
from core.api.utils import init_cfg


def print_env(cfg):
    logger.info("current node: {}", cfg.node_rank)
    logger.info("node count: {}", cfg.nodes)
    logger.info("gpus per node: {}", cfg.local_size)
    logger.info("world size: {}", cfg.world_size)
    logger.info("port: {}:{}", cfg.addr, cfg.port)
    logger.info("torch version: {}", torch.__version__)
    logger.info("cuda device: {}", torch.cuda.get_device_name())
    logger.info("cuda version: {}", torch.version.cuda)
    logger.info("cudnn version: {}", torch.backends.cudnn.version())
    logger.info("cuda capability: {}.{}", torch.cuda.get_device_capability()[0], torch.cuda.get_device_capability()[1])
    logger.info("name: {}", cfg.name)


def main():
    opt = argparse.ArgumentParser("main")
    opt.add_argument("--nodes", type=int, default=1, help="node count")
    opt.add_argument("--gpus", type=int, default=1, help="gpu count one node")
    opt.add_argument("--addr", type=str, default="127.0.0.1", help="the address of master")
    opt.add_argument("--port", type=int, default=4455, help="master port")
    opt.add_argument("--node", type=int, default=0, help="current node id")
    opt.add_argument("--mode", type=str, default="train", help="run mode", choices=["train", "test"])
    opt.add_argument("--config", type=str, default="config/demo.yml", help="config file")
    opt.add_argument("--sync_bn", type=bool, default=True, help="sync batchnorm in all gpus")
    args = opt.parse_args()
    cfg = init_cfg(args.config)
    cfg.nodes = args.nodes
    cfg.local_size = args.gpus
    cfg.node_rank = args.node
    cfg.world_size = args.nodes * args.gpus
    cfg.addr = args.addr
    cfg.port = args.port
    cfg.sync_bn = args.sync_bn
    print_env(cfg)
    os.environ["MASTER_ADDR"] = args.addr
    os.environ["MASTER_PORT"] = str(args.port)
    if args.mode == "train":
        from core.api.train import train
        if os.path.exists(cfg.output_train):
            shutil.rmtree(cfg.output_train)
        os.makedirs(cfg.output_train)
        mp.spawn(train, (cfg,), nprocs=cfg.world_size, join=True)
    else:
        from core.api.test import test
        if not os.path.exists(cfg.output_test):
            os.makedirs(cfg.output_test)
        mp.spawn(test, (cfg,), nprocs=cfg.world_size, join=True)

if __name__ == "__main__":
    main()
