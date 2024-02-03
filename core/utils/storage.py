import os
import torch


def checkpoint_dir(cfg):
    return os.path.join(cfg.output_base, "checkpoints")


def save_checkpoint(epoch, model, optimizer, scheduler, path):
    ckp_dir = os.path.dirname(path)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["epoch"]


def load_model(path, model: torch.nn.Module):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
