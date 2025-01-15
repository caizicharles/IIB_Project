from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import RNNLayer, TransformerLayer, CNNLayer
from model.modules import Encoder

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'selu': nn.SELU()
}


class GRU(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.features_num = args['features_num']
        self.hidden_dim = args['hidden_dim']
        self.layer_num = args['layer_num']
        self.out_dim = args['out_dim']
        self.dropout = args['dropout']

        self.gru = RNNLayer(input_size=1,#self.features_num + 1,
                            hidden_size=self.hidden_dim,
                            rnn_type='GRU',
                            num_layers=self.layer_num,
                            dropout=self.dropout)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, ecg, features, spectrogram):

        ecg = ecg.unsqueeze(dim=-1).to(torch.float32)
        # x = torch.cat((ecg, features), dim=1).to(torch.float32)
        _, x = self.gru(ecg)
        out = self.head(x)

        return out


class LSTM(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args['hidden_dim']
        self.layer_num = args['layer_num']
        self.out_dim = args['out_dim']
        self.dropout = args['dropout']

        self.lstm = RNNLayer(input_size=1,#self.features_num + 1,
                             hidden_size=self.hidden_dim,
                             rnn_type='LSTM',
                             num_layers=self.layer_num,
                             dropout=self.dropout)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, ecg, features, spectrogram):

        ecg = ecg.unsqueeze(dim=-1).to(torch.float32)
        # x = torch.cat((ecg, features), dim=1).to(torch.float32)
        _, x = self.lstm(ecg)
        out = self.head(x)

        return out


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.hidden_dim = args['hidden_dim']
        self.ff_dim = args['ff_dim']
        self.head_num = args['head_num']
        self.encoder_depth = args['encoder_depth']
        self.out_dim = args['out_dim']
        self.dropout = args['dropout']

        self.transformer = TransformerLayer(feature_size=1,#self.features_num + 1,
                                            heads=self.head_num,
                                            dropout=self.dropout,
                                            num_layers=self.encoder_depth)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, ecg, features, spectrogram):

        ecg = ecg.unsqueeze(dim=-1).to(torch.float32)
        # x = torch.cat((ecg, features), dim=1).to(torch.float32)
        _, x = self.transformer(ecg)
        out = self.head(x)

        return out
    

class CNN(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.features_num = args['features_num']
        self.hidden_dim = args['hidden_dim']
        self.layer_num = args['layer_num']
        self.out_dim = args['out_dim']

        self.cnn = CNNLayer(input_size=1,#self.features_num + 1,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layer_num)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, ecg, features, spectrogram):

        ecg = ecg.unsqueeze(dim=-1).to(torch.float32)
        # x = torch.cat((ecg, features), dim=1).to(torch.float32)
        _, x = self.cnn(ecg)
        out = self.head(x)

        return out


class BaseModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.ecg_length = args['ecg_length']
        self.features_num = args['features_num']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = args['out_dim']
        self.layer_num = args['layer_num']
        self.joint_head_hidden_sizes = args['joint_head_hidden_sizes']
        self.act_fn = ACTIVATION_FUNCTIONS[args['act_fn']]
        self.dropout_prob = args['dropout']

        self.cnn = CNNLayer(input_size=13,#self.features_num + 1,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layer_num)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, ecg, features, spectrogram):
        B, L = ecg.shape

        quality = features[:, 0].unsqueeze(dim=-1)
        rpeaks = features[:, 1].unsqueeze(dim=-1)
        waves = features[:, 2:-1].transpose(-1, -2)
        # rate = features[:, -1].unsqueeze(dim=-1)

        ecg = ecg.unsqueeze(dim=-1)
        x = torch.cat((ecg, quality, rpeaks, waves), dim=-1).to(torch.float32)

        _, h = self.cnn(x)
        h = self.dropout(h)

        out = self.head(h)

        return out


MODELS = {"BaseModel": BaseModel,
          'GRU': GRU,
          'LSTM': LSTM,
          'Transformer': Transformer,
          'CNN': CNN}
