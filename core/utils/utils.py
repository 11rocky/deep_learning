import os
import torch
import torch.distributed as dist


def setup(local_rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="nccl", init_method="env://", rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def val_model(model, data_loader, loss_func, device, save_fn=None):
    model.eval()
    loss_func.start_log(False)
    with torch.no_grad():
        for _, input_data in enumerate(data_loader):
            input_data.to(device)
            pred_data = model(input_data)
            loss_func(pred_data, input_data)
            if save_fn is not None:
                save_fn(input_data, pred_data)
    loss_func.end_log()
