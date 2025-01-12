import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_dim, out_dim, hidden_sizes, act_fn=nn.ReLU(), dropout=0.1):
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.hidden_sizes = hidden_sizes
        self.act_fn = act_fn
        self.dropout = dropout

        assert len(self.hidden_sizes) > 0

        self.layers = []
        self.layers.append(torch.nn.Linear(self.input_dim, self.hidden_sizes[0]))
        self.layers.append(self.act_fn)
        self.layers.append(torch.nn.Dropout(self.dropout))

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))
            self.layers.append(self.act_fn)
            self.layers.append(nn.Dropout(self.dropout))

        self.layers.append(nn.Linear(self.hidden_sizes[-1], self.out_dim))

        self.nn = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nn(x)