import time
import os.path as osp
import csv
import pickle


def read_csv_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.csv'):
        file_name += '.csv'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "r") as f:
        file = csv.DictReader(f)
        file = list(file)

    return file


def save_with_pickle(file: object, save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(file, f)

    return


def read_pickle_file(save_path: str, file_name: str):

    if not file_name.lower().endswith('.pickle'):
        file_name += '.pickle'

    file_path = osp.join(save_path, file_name)
    assert osp.isfile(file_path), f'Dataset save path: {file_path} is invalid.'

    with open(file_path, "rb") as f:
        file = pickle.load(f)

    return file


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())
