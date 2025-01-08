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
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset.dataset import BaseDataset
from dataloader.dataloader import custom_collate_fn

from trainer.optimizer import OPTIMIZERS
from trainer.scheduler import SCHEDULERS
from trainer.criterion import CRITERIA
from metrics.metrics import METRICS
from model.model import MODELS

from utils.misc import init_logger
from utils.args import get_args
from utils.utils import *

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
    init_logger(args)
    logger.info('Process begins...')

    seed_everything(args.seed)

    file_lib = load_everything(args.processed_data_path, args.dataset)

    train_dataset = BaseDataset(file_lib['train_data'])
    val_dataset = BaseDataset(file_lib['val_data'])
    test_dataset = BaseDataset(file_lib['test_data'])
    logger.info('Dataset ready')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_batch_size,
                            shuffle=False,
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.test_batch_size,
                             shuffle=False,
                             collate_fn=custom_collate_fn)
    logger.info('DataLoader ready')

    # for idx, data in enumerate(train_loader):
    #     ecg = data['ecg']
    #     spectrogram = data['spectrogram']
    #     labels = data['label']

    #     plt.plot(ecg[10])
    #     plt.savefig('/content/drive/MyDrive/IIB_Project/logs/im5.png')
    #     plt.clf()
    #     plt.imshow(spectrogram[10], cmap='grey')
    #     plt.savefig('/content/drive/MyDrive/IIB_Project/logs/im6.png')
    #     plt.clf()
    #     exit()

    model_configs = args.model['args'] | \
        {'ecg_length': args.ecg_length,
         'spectrogram_height': args.spectrogram_height,
         'spectrogram_width': args.spectrogram_width}

    model = MODELS[args.model['name']](model_configs)
    model.to(device)
    logger.info('Model ready')

    optimizer = OPTIMIZERS[args.optimizer['name']](filter(lambda p: p.requires_grad, model.parameters()),
                                                   **args.optimizer['args'])

    scheduler = None
    if args.scheduler is not None:
        scheduler = SCHEDULERS[args.scheduler['name']](optimizer, **args.scheduler['args'])

    global_iter_idx = [0]

    experiment_name = f'Experiments'
    mlflow.set_tracking_uri(osp.join(args.log_path, 'mlflow'))
    client = MlflowClient()
    try:
        EXP_ID = client.create_experiment(experiment_name)
    except:
        experiments = client.get_experiment_by_name(experiment_name)
        EXP_ID = experiments.experiment_id

    with mlflow.start_run(experiment_id=EXP_ID,
                          run_name=f'{args.dataset}_{args.model["name"]}_{args.model["mode"]}_{args.start_time}'):
        mlflow.log_params(vars(args))

        if args.model['mode'] == 'train':
            early_stopping_counter = 0
            best_score = 0.

            for epoch_idx in range(args.max_epoch):
                # Train
                single_train(
                    model,
                    train_loader,
                    epoch_idx,
                    global_iter_idx,
                    optimizer,
                    scheduler=scheduler,
                    criteria=[CRITERIA[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                    metrics=[METRICS[metric](**args.val_metrics[metric]) for metric in args.val_metrics],
                    logging_freq=args.logging_freq)

                # Validate
                if epoch_idx % args.val_freq == 0:
                    val_results = single_validate(
                        model,
                        val_loader,
                        epoch_idx,
                        criteria=[CRITERIA[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                        metrics=[METRICS[metric](**args.val_metrics[metric]) for metric in args.val_metrics])

                    test_results = single_test(
                        model,
                        test_loader,
                        epoch_idx,
                        criteria=[CRITERIA[criterion](**args.criterion[criterion]) for criterion in args.criterion],
                        metrics=[METRICS[metric](**args.test_metrics[metric]) for metric in args.test_metrics])

                    score = val_results[args.early_stopping_metric]

                    if score >= best_score:
                        best_model = deepcopy(model)
                        best_optimizer = deepcopy(optimizer)
                        best_scheduler = deepcopy(scheduler)
                        best_score = score
                        best_val_results = val_results
                        best_test_results = test_results
                        best_epoch = epoch_idx
                        best_iter = global_iter_idx[0]
                        early_stopping_counter = 0

                    if epoch_idx == args.max_epoch - 1:
                        early_stopping_counter = args.early_stopping_epoch
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= args.early_stopping_epoch:
                        logger.info(f'Early stopping triggered, best epoch: {best_epoch}')
                        log_best_val_results = {}
                        log_best_test_results = {}

                        for k, v in best_val_results.items():
                            _k = 'Best val ' + k
                            log_best_val_results[_k] = v

                            if isinstance(v, list):
                                for element in v:
                                    logger.info(f'Best val {k}: {element:.4f}')
                            else:
                                logger.info(f'Best val {k}: {v:.4f}')

                        for k, v in best_test_results.items():
                            _k = 'Best test ' + k
                            log_best_test_results[_k] = v

                            if isinstance(v, list):
                                for element in v:
                                    logger.info(f'Best test {k}: {element:.4f}')
                            else:
                                logger.info(f'Best test {k}: {v:.4f}')

                        mlflow.log_params(log_best_val_results)
                        mlflow.log_params(log_best_test_results)

                        RESULTS = f'\n------------------\nTEST RESULTS:\n\n{args.model["args"]}\n{args.optimizer["args"]}\n\n'
                        for k, v in log_best_test_results.items():
                          RESULTS += f'{k}: {v}\n'
                        RESULTS += '------------------'

                        print(RESULTS)

                        with open(osp.join(args.log_path, 'hyp_search_results.txt'), 'a') as file:
                            file.write(RESULTS)

                        break


def single_train(model,
                 data_loader,
                 epoch_idx,
                 global_iter_idx,
                 optimizer,
                 criteria,
                 metrics,
                 scheduler=None,
                 logging_freq=1):

    train_start_time = time.time()
    model.train()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, data in enumerate(data_loader):
        ecg = data['ecg'].to(device)
        spectrogram = data['spectrogram'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()

        output = model(ecg=ecg, spectrogram=spectrogram)

        loss = 0.
        for criterion in criteria:
            loss += criterion(output, labels)

        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        probability = torch.sigmoid(output)

        prob_all.append(probability.cpu().detach())
        target_all.append(labels.cpu().detach())

        if idx % logging_freq == 0:
            logger.info(
                f"Epoch: {epoch_idx:4d}, Iteration: {idx:4d} / {len(data_loader):4d} [{global_iter_idx[0]:5d}], Loss: {loss.item()}"
            )
        mlflow.log_metric(key='train_batch_loss', value=loss.item(), step=global_iter_idx[0])

        global_iter_idx[0] += 1

    if scheduler is not None:
        scheduler.step()

    epoch_loss_avg = np.mean(epoch_loss)
    logger.info(f"Epoch: {epoch_idx:4d},  [{global_iter_idx[0]:5d}], Epoch Loss: {epoch_loss_avg}")
    mlflow.log_metrics({
        'train_time': time.time() - train_start_time,
        'train_epoch_loss': epoch_loss_avg
    },
                       step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        mlflow.log_metric(key=f'train_{metric.NAME}', value=score, step=epoch_idx)


def single_validate(model, data_loader, epoch_idx, criteria, metrics):

    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, data in enumerate(tqdm(data_loader, desc='Validating')):
        ecg = data['ecg'].to(device)
        spectrogram = data['spectrogram'].to(device)
        labels = data['label'].to(device)

        with torch.no_grad():
            output = model(ecg=ecg, spectrogram=spectrogram)

            loss = 0.
            for criterion in criteria:
                loss += criterion(output, labels)

            epoch_loss.append(loss.item())

            probability = torch.sigmoid(output)

            prob_all.append(probability.cpu().detach())
            target_all.append(labels.cpu().detach())

    epoch_loss_avg = np.mean(epoch_loss)
    mlflow.log_metrics({'val_epoch_loss': epoch_loss_avg}, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    results = {}
    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        results[metric.NAME] = score
        mlflow.log_metric(key=f'val_{metric.NAME}', value=score, step=epoch_idx)

    return results


def single_test(model, data_loader, epoch_idx, criteria, metrics):

    model.eval()
    epoch_loss = []
    prob_all = []
    target_all = []

    for idx, data in enumerate(tqdm(data_loader, desc='Testing')):
        ecg = data['ecg'].to(device)
        spectrogram = data['spectrogram'].to(device)
        labels = data['label'].to(device)

        with torch.no_grad():
            output = model(ecg=ecg, spectrogram=spectrogram)

            loss = 0.
            for criterion in criteria:
                loss += criterion(output, labels)

            epoch_loss.append(loss.item())

            probability = torch.sigmoid(output)

            prob_all.append(probability.cpu().detach())
            target_all.append(labels.cpu().detach())

    epoch_loss_avg = np.mean(epoch_loss)
    mlflow.log_metrics({'test_epoch_loss': epoch_loss_avg}, step=epoch_idx)

    prob_all = np.concatenate(prob_all, axis=0)
    target_all = np.concatenate(target_all, axis=0)

    results = {}
    for metric in metrics:
        score = metric.calculate(prob_all, target_all)
        results[metric.NAME] = score
        mlflow.log_metric(key=f'test_{metric.NAME}', value=score, step=epoch_idx)

    return results


if __name__ == '__main__':
    args = get_args()
    main(args=args)