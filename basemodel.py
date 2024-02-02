import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class LSTM(nn.Module):

    def __init__(self, n_features, sequence_len, batch_size=64, n_hidden=10, n_layers=2):
        super(LSTM, self).__init__()

        self.n_hidden = n_hidden
        self.sequence_len = sequence_len
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=False,
                            dropout=0)
        self.fcn = nn.Linear(in_features=n_hidden, out_features=1)
    def forward(self, input_tensor, hidden,cell):
        #self.reset_hidden_state()
        lstm_out, (hidden,cell) = self.lstm(input_tensor,(hidden,cell))  # lstm_out (batch_size, seq_len, hidden_size*2)
        out = lstm_out[:, -1, :]  # getting only the last time step's hidden state of the last layer
        # print("hidden states mean, std, min, max: ", lstm_out[:,:,:].mean().item(), lstm_out[:,:,:].std().item(), lstm_out[:,:,:].min().item(), lstm_out[:,:,:].max().item()) # lstm_out.shape -> out.shape: 64,16,100 -> 64,16. Batch size: 64, input_seq_len:  16, n_hidden*2 = 50*2 = 100 // *2 for bidirectional lstm
        out_regression = self.fcn(out)  # feeding lstm output to a fully connected network which outputs 3 nodes: mu, sigma, xi
        return out_regression, hidden,cell




class LSTM_v2(nn.Module):

    def __init__(self, n_features, sequence_len, batch_size=64, n_hidden=10, n_layers=2):
        super(LSTM_v2, self).__init__()

        self.n_hidden = n_hidden
        self.sequence_len = sequence_len
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=n_features,
                            hidden_size=n_hidden,
                            num_layers=n_layers,
                            batch_first=False,
                            bidirectional=False,
                            dropout=0)
        self.distribution_mu = nn.Linear(self.n_hidden * self.n_layers, 1)
        self.distribution_presigma = nn.Linear(self.n_hidden * self.n_layers, 1)
        self.distribution_sigma = nn.Softplus()

    def forward(self, input_tensor, hidden,cell):
        #self.reset_hidden_state()

        output, (hidden, cell) = self.lstm(input_tensor, (hidden, cell))
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        mu = self.distribution_mu(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden, cell








