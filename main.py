import os
import argparse
import yaml
from loguru import logger
from easydict import EasyDict
import torch
import torch.multiprocessing as mp


def print_env(cfg):
    logger.info("gpu id list: {}", os.environ["CUDA_VISIBLE_DEVICES"])
    logger.info("world_size: {}", cfg.world_size)
    logger.info("port: {}", cfg.port)
    logger.info("torch version: {}", torch.__version__)
    logger.info("cuda device: {}", torch.cuda.get_device_name())
    logger.info("cuda version: {}", torch.version.cuda)
    logger.info("cudnn version: {}", torch.backends.cudnn.version())
    logger.info("cuda capability: {}.{}", torch.cuda.get_device_capability()[0], torch.cuda.get_device_capability()[1])
    logger.info("cuda archs: {}", torch.cuda.get_arch_list())
    logger.info("name: {}", cfg.name)


def main():
    opt = argparse.ArgumentParser("main")
    opt.add_argument("--config", type=str, default="config/demo.yml", help="config file")
    opt.add_argument("--mode", type=str, default="train", help="run mode", choices=["train", "test"])
    opt.add_argument("--gpus", type=str, default="0", help="gpu ids, such as: 0,1,2,3")
    opt.add_argument("--port", type=int, default=4455, help="master port")
    args = opt.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg = EasyDict(cfg)
    cfg.world_size = len(args.gpus.strip().split(","))
    cfg.port = args.port
    cfg.name = os.path.splitext(os.path.basename(args.config))[0]
    cfg.output_base = os.path.join("output", cfg.name, "train")
    if not os.path.exists(cfg.output_base):
        os.makedirs(cfg.output_base)
    print_env(cfg)
    if args.mode == "train":
        from core.utils.train import train
        mp.spawn(train, (cfg,), nprocs=cfg.world_size, join=True)
    else:
        from core.utils.test import test
        mp.spawn(test, (cfg,), nprocs=cfg.world_size, join=True)

if __name__ == "__main__":
    main()
