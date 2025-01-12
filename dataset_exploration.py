import os
import os.path as osp
import random
import logging
import time
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.dataset import BaseDataset

from utils.misc import init_logger, save_params
from utils.args import get_args
from utils.utils import *

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_everything(processed_data_path, dataset_name):
    folder_path = osp.join(processed_data_path, dataset_name)

    train_data = read_pickle_file(folder_path, 'train_data.pickle')
    val_data = read_pickle_file(folder_path, 'val_data.pickle')
    test_data = read_pickle_file(folder_path, 'test_data.pickle')

    return {'train_data': train_data, 'val_data': val_data, 'test_data': test_data}


def main(args):
    init_logger()
    logger.info('Process begins...')

    seed_everything(args.seed)

    file_lib = load_everything(args.processed_data_path, args.dataset)

    full_dataset = file_lib['train_data'] + file_lib['val_data'] + file_lib['test_data']

    positive_dataset = []
    negative_dataset = []
    for data in full_dataset:
        if data['label'] == 1:
            positive_dataset.append(data['ecg'])
        else:
            negative_dataset.append(data['ecg'])

    # for data in negative_dataset:
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #     axes[0].plot(data)
    #     axes[1].plot(data[:500])
    #     plt.tight_layout()
    #     plt.show()
    #     plt.clf()
    # exit()


    N = 200
    positive_subset = random.sample(positive_dataset, N)
    negaive_subset = random.sample(negative_dataset, N)

    positive_results = []
    negative_results = []
    cross_results = []

    for i, x in enumerate(tqdm(positive_subset)):
        for j, y in enumerate(positive_subset):
            if i > j:
                distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
                positive_results.append(distance)

    for i, x in enumerate(tqdm(negaive_subset)):
        for j, y in enumerate(negaive_subset):
            if i > j:
                distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
                negative_results.append(distance)

    for i, x in enumerate(tqdm(positive_subset)):
        for j, y in enumerate(negaive_subset):
            if i > j:
                distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
                cross_results.append(distance)

    save_with_pickle(positive_results, r'C:\Users\caizi\Documents\GitHub\IIB_Project\logs', 'dtw_positive_label.pickle')
    save_with_pickle(negative_results, r'C:\Users\caizi\Documents\GitHub\IIB_Project\logs', 'dtw_negative.pickle')
    save_with_pickle(cross_results, r'C:\Users\caizi\Documents\GitHub\IIB_Project\logs', 'dtw_cross_label.pickle')

    print(np.mean(positive_results), np.mean(negative_results), np.mean(cross_results))
    print(np.std(positive_results), np.std(negative_results), np.std(cross_results))
    exit()

    subset = positive_subset + negaive_subset

    results = np.zeros((2*N, 2*N))
    for i, x in enumerate(tqdm(subset)):
        for j, y in enumerate(subset):
            if i > j:
                distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
                results[i, j] = distance

    results = results + results.T
    np.fill_diagonal(results, np.min(results))
    results /= np.max(results)

    plt.imshow(results, cmap='plasma')
    plt.axvline(x=N - 0.5, color='white', linestyle='--', linewidth=2)  # Vertical line
    plt.axhline(y=N - 0.5, color='white', linestyle='--', linewidth=2)  # Horizontal line
    plt.colorbar(label='Distance')
    plt.title('Distance Heatmap')
    plt.xlabel('Time Series Index')
    plt.ylabel('Time Series Index')
    plt.show()
    exit()

    


if __name__ == '__main__':
    args = get_args()
    main(args=args)
