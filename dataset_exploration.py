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
import scipy.signal as sg
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

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

    N = 50
    positive_subset = random.sample(positive_dataset, N)
    negaive_subset = random.sample(negative_dataset, N)

    def compute_psd(ecg, fs=250):
        """
        Compute the Power Spectral Density (PSD) of an ECG signal using Welch's method.
        Returns frequency array and power spectral density array.
        """
        freqs, psd = sg.welch(ecg, fs=fs, nperseg=1024)
        return freqs, psd


    def find_max_min_frequency(freqs, psd):
        """
        Find the frequencies where the PSD is maximum and minimum.
        In practice, the 'min' can be trivial (could be zero freq),
        so interpret accordingly.
        """
        max_idx = np.argmax(psd)
        min_idx = np.argmin(psd)

        max_freq = freqs[max_idx]
        min_freq = freqs[min_idx]
        return max_freq, min_freq

    positive_psd = []
    negative_psd = []
    positive_max_freq = []
    positive_min_freq = []
    negative_max_freq = []
    negative_min_freq = []

    for idx, (p, n) in enumerate(zip(positive_dataset, negative_dataset)):
        p_freqs, p_psd = compute_psd(p, fs=200)
        max_p_freq, min_p_freq = find_max_min_frequency(p_freqs, p_psd)

        positive_psd.append(p_psd)
        positive_max_freq.append(max_p_freq)
        positive_min_freq.append(min_p_freq)

        n_freqs, n_psd = compute_psd(n, fs=200)
        max_n_freq, min_n_freq = find_max_min_frequency(n_freqs, n_psd)

        negative_psd.append(n_psd)
        negative_max_freq.append(max_n_freq)
        negative_min_freq.append(min_n_freq)

    positive_psd = np.stack(positive_psd)
    negative_psd = np.stack(negative_psd)
    X = np.concatenate((positive_psd, negative_psd), axis=0)
    Y = np.concatenate((np.zeros(len(positive_psd)), np.ones(len(negative_psd))))

    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_tsne = tsne.fit_transform(X)

    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap="bwr", s=60, alpha=0.8)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(*scatter.legend_elements(), title="Label (0=Good Quality, 1=Poor Quality)", fontsize=14)
    # plt.show()
    # exit()
    
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)

    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y, cmap="coolwarm", s=40, alpha=0.3)
    # plt.xlabel("PC 1", fontsize=14)
    # plt.ylabel("PC 2", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(*scatter.legend_elements(), title="Label (0=Good Quality, 1=Poor Quality)", fontsize=14)
    # plt.show()
    # plt.clf()

    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    meanprops = dict(linestyle='-', linewidth=2.5, color='blue')
    # median_line = Line2D([0], [0], color='red', label='Median')
    # mean_line   = Line2D([0], [0], color='blue', label='Mean')

    # plt.figure(figsize=(6, 6))
    # plt.boxplot([positive_max_freq, negative_max_freq], tick_labels=["Good Quality", "Poor Quality"], showmeans=True, meanline=True, showfliers=False, medianprops=medianprops, meanprops=meanprops)
    # plt.ylabel("Frequency (Hz)", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(handles=[median_line, mean_line], loc='best')
    # plt.show()
    # plt.clf()

    # plt.figure(figsize=(6, 6))
    # plt.boxplot([positive_min_freq, negative_min_freq], tick_labels=["Good Quality", "Poor Quality"], showmeans=True, meanline=True, showfliers=False, medianprops=medianprops, meanprops=meanprops)
    # plt.ylabel("Frequency (Hz)", fontsize=14)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(handles=[median_line, mean_line], loc='best')
    # plt.show()
    # plt.clf()
    # exit()

    # for data in negative_dataset:
    #     fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #     axes[0].plot(data)
    #     axes[1].plot(data[:500])
    #     plt.tight_layout()
    #     plt.show()
    #     plt.clf()
    # exit()

    

    positive_results = []
    negative_results = []
    cross_results = []

    # for i, x in enumerate(tqdm(positive_subset)):
    #     for j, y in enumerate(positive_subset):
    #         if i > j:
    #             distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
    #             positive_results.append(distance)

    # for i, x in enumerate(tqdm(negaive_subset)):
    #     for j, y in enumerate(negaive_subset):
    #         if i > j:
    #             distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
    #             negative_results.append(distance)

    # for i, x in enumerate(tqdm(positive_subset)):
    #     for j, y in enumerate(negaive_subset):
    #         if i > j:
    #             distance, path = fastdtw(x.flatten(), y.flatten(), dist=lambda x, y: euclidean([x], [y]))
    #             cross_results.append(distance)

    # save_with_pickle(positive_results, r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_positive_label.pickle')
    # save_with_pickle(negative_results, r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_negative.pickle')
    # save_with_pickle(cross_results, r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_cross_label.pickle')

    positive_results = read_pickle_file(r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_positive_label.pickle')
    negative_results = read_pickle_file(r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_negative.pickle')
    cross_results = read_pickle_file(r'C:\Users\ZC\Documents\GitHub\IIB_Project\logs', 'dtw_cross_label.pickle')

    print(np.mean(positive_results), np.mean(negative_results), np.mean(cross_results))
    print(np.std(positive_results), np.std(negative_results), np.std(cross_results))

    plt.figure(figsize=(8, 6))
    plt.boxplot([positive_results, negative_results, cross_results], tick_labels=["Poor Quality", "Good Quality", 'Cross'], showmeans=True, meanline=True, showfliers=False, medianprops=medianprops, meanprops=meanprops)
    plt.ylabel("DTW Distance", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    plt.clf()
    exit()

    subset = positive_subset + negaive_subset

    results = np.zeros((2 * N, 2 * N))
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
