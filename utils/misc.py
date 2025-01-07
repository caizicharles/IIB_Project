import logging
import tensorboardX
import os.path as osp
import time
import datetime


def init_logger(args):

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=logging.INFO, handlers=[console_handler])

    writer = tensorboardX.SummaryWriter(log_dir=osp.join(args.log_path, 'tensorboard', args.start_time))

    return writer


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())
