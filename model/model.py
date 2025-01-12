from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyhealth.models import RNNLayer, TransformerLayer
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

    def __init__(self, configs):
        super().__init__()

        self.hidden_dim = configs['hidden_dim']
        self.layer_num = configs['layer_num']
        self.out_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.gru = RNNLayer(input_size=1,
                            hidden_size=self.hidden_dim,
                            rnn_type='GRU',
                            num_layers=self.layer_num,
                            dropout=self.dropout)
        
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        
    def forward(self, ecg, spectrogram):

        x = ecg.unsqueeze(dim=1).to(torch.float32)
        _, x = self.gru(x)
        out = self.head(x)

        return out
    

class LSTM(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.hidden_dim = configs['hidden_dim']
        self.layer_num = configs['layer_num']
        self.out_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.lstm = RNNLayer(input_size=1,
                            hidden_size=self.hidden_dim,
                            rnn_type='LSTM',
                            num_layers=self.layer_num,
                            dropout=self.dropout)
        
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        
    def forward(self, ecg, spectrogram):

        x = ecg.unsqueeze(dim=1).to(torch.float32)
        _, x = self.lstm(x)
        out = self.head(x)

        return out
    

class Transformer(nn.Module):

    def __init__(self, configs):
        super().__init__()

        self.hidden_dim = configs['hidden_dim']
        self.ff_dim = configs['ff_dim']
        self.head_num = configs['head_num']
        self.encoder_depth = configs['encoder_depth']
        self.out_dim = configs['out_dim']
        self.dropout = configs['dropout']

        self.transformer = TransformerLayer(feature_size=1,
                                            heads=self.head_num,
                                            dropout=self.dropout,
                                            num_layers=self.encoder_depth)

        self.head = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, ecg, spectrogram):

        x = ecg.unsqueeze(dim=1).to(torch.float32)
        _, x = self.transformer(x)
        out = self.head(x)

        return out


class BaseModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.ecg_length = args['ecg_length']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = args['out_dim']
        self.conv_kernel_size = args['conv_kernel_size']
        self.conv_stride = args['conv_stride']
        self.pool_kernel_size = args['pool_kernel_size']
        self.pool_stride = args['pool_stride']
        self.padding = args['padding']
        self.freq_encoder_hidden_sizes = args['freq_encoder_hidden_sizes']
        self.joint_head_hidden_sizes = args['joint_head_hidden_sizes']
        self.act_fn = ACTIVATION_FUNCTIONS[args['act_fn']]
        self.dropout_prob = args['dropout']

        # self.freq_encoder = Encoder(input_dim=2 * self.ecg_length,
        #                             out_dim=self.hidden_dim,
        #                             hidden_sizes=self.freq_encoder_hidden_sizes,
        #                             act_fn=self.act_fn,
        #                             dropout=self.dropout_prob)

        self.conv_time_1 = nn.Conv1d(in_channels=1,
                               out_channels=self.hidden_dim,
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.conv_time_2 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=2 * self.hidden_dim,
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.pool_time = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        conv_pool_out_dim_time = self._calculate_dim(self.ecg_length)
        self.fc = nn.Linear(conv_pool_out_dim_time, self.hidden_dim)

        self.joint_head = Encoder(input_dim=1 * self.hidden_dim,
                                    out_dim=self.out_dim,
                                    hidden_sizes=self.joint_head_hidden_sizes,
                                    act_fn=self.act_fn,
                                    dropout=self.dropout_prob)
        self.dropout = nn.Dropout(self.dropout_prob)

    def _calculate_dim(self, input_length):

        x = torch.zeros(1, 1, input_length)
        x = self.pool_time(self.conv_time_1(x))
        x = self.pool_time(self.conv_time_2(x))

        return x.numel()

    def forward(self, ecg, spectrogram):
        B, L = ecg.shape

        x_time = ecg.unsqueeze(dim=1).to(torch.float32)
        h_time = self.pool_time(F.relu(self.conv_time_1(x_time)))
        h_time = self.pool_time(F.relu(self.conv_time_2(h_time)))
        h_time = h_time.reshape(B, -1)
        h_time = F.relu(self.fc(h_time))
        h_time = self.dropout(h_time)

        # fourier_coeffs = torch.fft.fft(ecg)
        # fourier_coeffs = torch.view_as_real(fourier_coeffs).reshape(B, -1)
        # x_freq = fourier_coeffs.to(torch.float32)
        # h_freq = self.freq_encoder(x_freq)

        # h = torch.cat((h_time, h_freq), dim=-1)
        h = h_time
        out = self.joint_head(h)

        return out


MODELS = {"BaseModel": BaseModel}
