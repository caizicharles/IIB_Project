import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm
import wfdb
import torch
import scipy
import scipy.signal as scs
from scipy.signal import butter, filtfilt, resample
from skimage.transform import resize
from copy import deepcopy
import matplotlib.pyplot as plt
from PIL import Image
import io
import neurokit2 as nk

from utils.args import get_args
from utils.utils import *

PATIENT_TEMPLATE = {
    'ecg': None,
    'ecg_for_spectrogram': [],
    'features': [],
    'spectrogram': [],
    'sampling_freq': None,
    'rec_idx': [],
    'labels': []
}

RECORD_TEMPLATE = {'patient_id': None, 'ecg': None, 'spectrogram': None, 'sampling_freq': None, 'label': None}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(raw_data_path: str, dataset_name: str, save: bool = False, save_path: str = None):

    raw_data = {}

    data_path = osp.join(raw_data_path, dataset_name)
    ecg_path = osp.join(data_path, 'ptECGs')
    rec_data_anon = read_csv_file(data_path, 'rec_data_anon.csv')  # list of dicts

    for anon in rec_data_anon:
        patient_id = str(anon['ptID']).zfill(6)
        raw_data[patient_id] = deepcopy(PATIENT_TEMPLATE)

    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0

    for anon in rec_data_anon:
        patient_id = str(anon['ptID']).zfill(6)
        diag_key = anon['measDiag']

        if dataset_name == 'Feas2':
            accepted_keys = ['1', '2', '3', '4']
            label_key = '4'
        elif dataset_name == 'Trial':
            accepted_keys = ['1', '2', '3', '4', '5']
            label_key = '5'

        if diag_key in accepted_keys:
            if diag_key == label_key:
                label = 1

                # if anon['tag_orig_Poor_Quality'] == '1':
                #   TP += 1
                # elif anon['tag_orig_Poor_Quality'] == '0':
                #   FN += 1
            else:
                label = 0

                # if anon['tag_orig_Poor_Quality'] == '1':
                #   FP += 1
                # elif anon['tag_orig_Poor_Quality'] == '0':
                #   TN += 1

            rec_idx = int(anon['measNo']) - 1
            assert rec_idx >= 0
            raw_data[patient_id]['rec_idx'].append(rec_idx)
            raw_data[patient_id]['labels'].append(label)

    # print(TP, FP, TN, FN, TP+FP+TN+FN)
    # exit()

    updated_raw_data = deepcopy(raw_data)

    for patient_id, info in raw_data.items():
        if len(info['rec_idx']) == 0:
            del updated_raw_data[patient_id]

    folders = os.listdir(ecg_path)
    selected_patient_ids = list(updated_raw_data.keys())

    for folder_id in tqdm(folders):
        folder_path = osp.join(ecg_path, folder_id)
        file_names = os.listdir(folder_path)

        for idx, name in enumerate(file_names):
            file_name, extension = osp.splitext(name)
            file_names[idx] = file_name

        file_names = set(file_names)
        file_names = sorted(file_names)

        for name in file_names:
            _, patient_id = name.split('_')
            patient_id = patient_id[2:]

            if patient_id not in selected_patient_ids:
                continue

            record_path = osp.join(folder_path, name)
            record = wfdb.rdrecord(record_path)

            rec_idx = updated_raw_data[patient_id]['rec_idx']
            selected_signals = record.p_signal[:, rec_idx]
            selected_signals = selected_signals.T

            updated_raw_data[patient_id]['ecg'] = selected_signals
            updated_raw_data[patient_id]['sampling_freq'] = record.fs

    if save:
        save_with_pickle(updated_raw_data, save_path, 'raw_data.pickle')

    return updated_raw_data


def filter_data(data, filtering_args, save: bool = False, save_path: str = None):

    def highpass_filter(signal, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, signal)

    def lowpass_filter(signal, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def bandpass_filter(signal, lowcutoff, highcutoff, fs, order=5):
        nyquist = 0.5 * fs
        low = lowcutoff / nyquist
        high = highcutoff / nyquist
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def downsample(signal, target_freq, fs):
        duration = len(signal) / fs
        target_samples = int(duration * target_freq)
        downsampled_signal = resample(signal, target_samples)
        return downsampled_signal

    def minmax_normalize(signal, val_range=(-1, 1)):
        min_val = np.min(signal)
        max_val = np.max(signal)
        a, b = val_range

        if max_val - min_val == 0:
            return np.full_like(signal, a)
        normalized_signal = a + (signal - min_val) * (b - a) / (max_val - min_val)
        return normalized_signal

    highcutoff_freq = filtering_args['highcutoff_freq']
    lowcutoff_freq = filtering_args['lowcutoff_freq']
    downsample_freq = filtering_args['downsample_freq']
    normalize = filtering_args['normalize']

    filtered_data = deepcopy(data)
    for patient_id, info in filtered_data.items():
        filtered_data[patient_id]['ecg'] = []

    for patient_id, info in tqdm(data.items(), desc='Filtering data'):
        signals = info['ecg']
        sampling_freq = info['sampling_freq']

        for ecg in signals:
            if highcutoff_freq is not None and lowcutoff_freq is not None:
                ecg = bandpass_filter(ecg, lowcutoff_freq, highcutoff_freq, sampling_freq)
            elif highcutoff_freq is not None and lowcutoff_freq is None:
                ecg = lowpass_filter(ecg, highcutoff_freq, sampling_freq)
            elif highcutoff_freq is None and lowcutoff_freq is not None:
                ecg = highpass_filter(ecg, lowcutoff_freq, sampling_freq)

            filtered_data[patient_id]['ecg_for_spectrogram'].append(ecg)

            if downsample_freq is not None:
                ecg = downsample(ecg, downsample_freq, sampling_freq)
                filtered_data[patient_id]['sampling_freq'] = downsample_freq

            if normalize:
                ecg = minmax_normalize(ecg, (-1, 1))

            filtered_data[patient_id]['ecg'].append(ecg)

        filtered_data[patient_id]['ecg'] = np.stack(filtered_data[patient_id]['ecg'])
        filtered_data[patient_id]['ecg_for_spectrogram'] = np.stack(filtered_data[patient_id]['ecg_for_spectrogram'])

    if save:
        save_with_pickle(filtered_data, save_path, 'filtered_data.pickle')

    return filtered_data


def construct_data(data, save: bool = False, save_path: str = None):

    def ecg_to_spectrogram(signal, sampling_freq, resolution=(512, 512), nperseg=256):
        frequencies, times, Sxx = scs.spectrogram(signal, fs=sampling_freq, nperseg=nperseg)

        Sxx_log = 10 * np.log10(Sxx + 1e-10)
        Sxx_normalized = (Sxx_log - np.min(Sxx_log)) / (np.max(Sxx_log) - np.min(Sxx_log))
        Sxx_image = (Sxx_normalized * 255).astype(np.uint8)
        Sxx_resized = resize(Sxx_image, resolution, anti_aliasing=True)
        spectrogram = (Sxx_resized * 255).astype(np.uint8)

        # fig = plt.figure(figsize=(6, 6), frameon=False)
        # ax = plt.Axes(fig, [0, 0, 1, 1])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        # ax.imshow(Sxx_resized, aspect='auto', cmap='gray', origin='lower')
        # buf = io.BytesIO()
        # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        # plt.close()
        # buf.seek(0)
        # spectrogram = np.array(Image.open(buf).convert('L'))

        return spectrogram

    def get_features(signal, sampling_freq):
        
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=sampling_freq, correct_artifacts=True, show=True)
        r_features = np.zeros((1, len(signal)))
        r_features[:, rpeaks['ECG_R_Peaks']] = 1

        wave_features = np.zeros((10, len(signal)))
        if len(rpeaks['ECG_R_Peaks']) > 5:
            quality = nk.ecg_quality(signal, sampling_rate=sampling_freq)
            _, waves = nk.ecg_delineate(signal, rpeaks, sampling_rate=sampling_freq)
            
            for idx, feat_indices in enumerate(waves.values()):
                feat_indices = np.array(feat_indices)
                feat_indices = feat_indices[~np.isnan(feat_indices)].astype(int)
                wave_features[idx, feat_indices] = 1
        else:
            quality = np.zeros(len(signal))

        info = nk.signal_findpeaks(signal)
        rate = nk.signal_rate(peaks=info["Peaks"], desired_length=len(signal), interpolation_method="monotone_cubic")

        rate = np.expand_dims(rate, axis=0)
        quality = np.expand_dims(quality, axis=0)

        features = np.concatenate((quality, r_features, wave_features, rate), axis=0)

        assert not np.isnan(features).any() and not np.isinf(features).any()

        return features

    for idx, (patient_id, info) in enumerate(tqdm(data.items(), desc='Constructing data')):
        # _signals = info['ecg_for_spectrogram']
        signals = info['ecg']
        sampling_freq = info['sampling_freq']

        for ecg in signals:
            features = get_features(ecg, sampling_freq)
            # spectrogram = ecg_to_spectrogram(ecg, sampling_freq)
            # data[patient_id]['spectrogram'].append(spectrogram)
            data[patient_id]['features'].append(features)

    if save:
        save_with_pickle(data, save_path, 'constructed_data.pickle')

    return data


def split_data(data, split_ratio=(0.8, 0.1, 0.1), split_mode='record', save: bool = False, save_path: str = None):

    assert split_mode in ['record', 'patient']

    if split_mode == 'patient':
        patient_num = len(data)
        patient_keys = np.array(list(data.keys()))
        np.random.shuffle(patient_keys)

        train_num = int(split_ratio[0] * patient_num)
        val_num = int(split_ratio[1] * patient_num)

        train_keys = patient_keys[:train_num]
        val_keys = patient_keys[train_keys:val_keys]
        test_keys = patient_keys[train_num + val_num:]

        train_data = {key: data[key] for key in train_keys}
        val_data = {key: data[key] for key in val_keys}
        test_data = {key: data[key] for key in test_keys}

    elif split_mode == 'record':
        record_num = 0
        full_data = []

        for patient_id, info in data.items():
            sampling_frq = info['sampling_freq']

            for ecg, feats, label in zip(info['ecg'], info['features'], info['labels']):
                # for ecg, spectrogram, label in zip(info['ecg'], info['spectrogram'], info['labels']):
                record = deepcopy(RECORD_TEMPLATE)
                record['patient_id'] = patient_id
                record['ecg'] = ecg
                record['features'] = feats
                # record['spectrogram'] = spectrogram
                record['label'] = label

                full_data.append(record)
                record_num += 1

        np.random.shuffle(full_data)

        train_num = int(split_ratio[0] * record_num)
        val_num = int(split_ratio[1] * record_num)

        train_data = full_data[:train_num]
        val_data = full_data[train_num:train_num + val_num]
        test_data = full_data[train_num + val_num:]

    if save:
        save_with_pickle(train_data, save_path, 'train_data.pickle')
        save_with_pickle(val_data, save_path, 'val_data.pickle')
        save_with_pickle(test_data, save_path, 'test_data.pickle')

    return {'train_data': train_data, 'val_data': val_data, 'test_data': test_data}


def run(args):

    seed_everything(args.seed)

    save_path = osp.join(args.processed_data_path, args.dataset)
    raw_data_path = osp.join(args.processed_data_path, args.dataset, 'raw_data.pickle')
    filtered_data_path = osp.join(args.processed_data_path, args.dataset, 'filtered_data.pickle')
    constructed_data_path = osp.join(args.processed_data_path, args.dataset, 'constructed_data.pickle')
    train_data_path = osp.join(args.processed_data_path, args.dataset, 'train_data.pickle')
    val_data_path = osp.join(args.processed_data_path, args.dataset, 'val_data.pickle')
    test_data_path = osp.join(args.processed_data_path, args.dataset, 'test_data.pickle')

    if not osp.exists(raw_data_path):
        raw_data = load_data(raw_data_path=args.raw_data_path,
                             dataset_name=args.dataset,
                             save=True,
                             save_path=save_path)
    else:
        raw_data = read_pickle_file(save_path, 'raw_data.pickle')
    print('load_data complete')

    if not osp.exists(filtered_data_path):
        filtered_data = filter_data(raw_data, filtering_args=args.filtering['args'], save=True, save_path=save_path)
    else:
        filtered_data = read_pickle_file(save_path, 'filtered_data.pickle')
    print('filter_data complete')

    if not osp.exists(constructed_data_path):
        constructed_data = construct_data(filtered_data, save=True, save_path=save_path)
    else:
        constructed_data = read_pickle_file(save_path, 'constructed_data.pickle')
    print('construct_data complete')

    if not (osp.exists(train_data_path) and osp.exists(val_data_path) and osp.exists(test_data_path)):
        split_ratio = (args.train_proportion, args.val_proportion, args.test_proportion)
        splitted_data = split_data(constructed_data,
                                   split_ratio=split_ratio,
                                   split_mode='record',
                                   save=False,
                                   save_path=save_path)

        train_data, val_data, test_data = splitted_data['train_data'], splitted_data['val_data'], splitted_data[
            'test_data']
    else:
        train_data = read_pickle_file(save_path, 'train_data.pickle')
        val_data = read_pickle_file(save_path, 'val_data.pickle')
        test_data = read_pickle_file(save_path, 'test_data.pickle')
    print(len(train_data), len(val_data), len(test_data))
    print('split_data completed')


if __name__ == '__main__':
    args = get_args()
    run(args=args)
