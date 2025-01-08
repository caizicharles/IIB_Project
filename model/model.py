from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeCNN(nn.Module):

    def __init__(self, args):
        super(TimeCNN, self).__init__()

        self.ecg_length = args['ecg_length']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = args['out_dim']
        self.conv_kernel_size = args['conv_kernel_size']
        self.conv_stride = args['conv_stride']
        self.pool_kernel_size = args['pool_kernel_size']
        self.pool_stride = args['pool_stride']
        self.padding = args['padding']
        self.dropout_prob = args['dropout']

        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=self.hidden_dim,
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim,
                               out_channels=2 * self.hidden_dim,
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.pool = nn.MaxPool1d(kernel_size=self.pool_kernel_size, stride=self.pool_stride)

        conv_pool_out_dim = self._calculate_dim(self.ecg_length)

        self.fc = nn.Linear(conv_pool_out_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def _calculate_dim(self, input_length):

        x = torch.zeros(1, 1, input_length)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        return x.numel()

    def forward(self, ecg, spectrogram):
        B, L = ecg.shape

        x = ecg.unsqueeze(dim=1).to(torch.float32)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.reshape(B, -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        out = self.head(x)

        return out


class FreqCNN(nn.Module):
    def __init__(self, args):
        super(FreqCNN, self).__init__()

        self.spectrogram_height = args['spectrogram_height']
        self.spectrogram_width = args['spectrogram_width']
        self.hidden_dim = args['hidden_dim']
        self.out_dim = args['out_dim']
        self.conv_kernel_size = args['conv_kernel_size']
        self.conv_stride = args['conv_stride']
        self.pool_kernel_size = args['pool_kernel_size']
        self.pool_stride = args['pool_stride']
        self.padding = args['padding']
        self.dropout_prob = args['dropout']
        
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.hidden_dim, 
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(in_channels=self.hidden_dim,
                               out_channels=2 * self.hidden_dim,
                               kernel_size=self.conv_kernel_size,
                               stride=self.conv_stride,
                               padding=self.padding)
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size,
                                 stride=self.pool_stride)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)

        conv_pool_out_dim = self._calculate_dim(self.spectrogram_height, self.spectrogram_width)
        
        self.fc = nn.Linear(conv_pool_out_dim, self.hidden_dim)
        self.head = nn.Linear(self.hidden_dim, self.out_dim)
        self.dropout = nn.Dropout(self.dropout_prob)

    def _calculate_dim(self, spectrogram_height, spectrogram_width):

        x = torch.zeros(1, 1, spectrogram_height, spectrogram_width)
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        return x.numel()
        
    def forward(self, ecg, spectrogram):
        B, H, W = spectrogram.shape

        x = spectrogram.unsqueeze(dim=1).to(torch.float32)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.reshape(B, -1)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        out = self.head(x)
        
        return out


class JointCNN(nn.Module):

    def __init__(self, args):
        super(JointCNN, self).__init__()

        # self.fc = nn.Linear(time_cnn.fc.out_features + freq_cnn.fc.out_features, num_classes)

    def forward(self, ecg, spectrogram):

        x_time = self.time_cnn(ecg)
        x_freq = self.freq_cnn(spectrogram)
        # combined = torch.cat((time_out, freq_out), dim=1)
        # output = self.fc(combined)

        return  #output


MODELS = {'TimeCNN': TimeCNN, 'FreqCNN': FreqCNN, "JointCNN": JointCNN}
