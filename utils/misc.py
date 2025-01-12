import logging
import os.path as osp
import time
import datetime
import torch
import yaml


def init_logger():
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[console_handler])


def save_params(model, args, epoch_idx=None, iter_idx=None, optimizer=None, scheduler=None):
    torch.save(
        {
            'model': model.state_dict(),
            'args': yaml.safe_dump(args.__dict__, default_flow_style=False),
            'epoch': epoch_idx if epoch_idx is not None else None,
            'iter': iter_idx if iter_idx is not None else None,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None
        },
        f'{args.log_data_path}/checkpoints/{args.dataset}/{args.task}/{args.model["name"]}/{args.model["name"]}_s{args.seed}_{args.start_time}.pth'
    )


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def str_to_datetime(time: str, format: str):
    return datetime.datetime.strptime(time, format)