import argparse
import yaml
from .misc import get_time_str


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)

    # Training Setting
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--val_freq', type=int, default=5)

    # Data Paths
    parser.add_argument('--raw_data_path', type=str, default='')
    parser.add_argument('--log_path', type=str, default='')

    # Config File
    config_parser = argparse.ArgumentParser(description='Algorithm Configs', add_help=False)
    config_parser.add_argument('-c', '--config', default=None, type=str, help='YAML config file')

    args_config, remaining = config_parser.parse_known_args()
    assert args_config.config is not None, 'Config file must be specified'

    with open(args_config.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)

    args.start_time = get_time_str()

    return args


if __name__ == '__main__':
    args = get_args()
    print(args)
