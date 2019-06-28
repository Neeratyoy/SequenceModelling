import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from matplotlib import pyplot as plt

from lstm import LSTM

from sklearn.metrics import confusion_matrix, f1_score, classification_report

class LSTMSeq2SeqDifferent(nn.Module):
    """ LSTM Class for Sequence Labelling (many-to-many-different)

    The class creates the LSTM architecture as specified by the parameters.
    A fully connected layer is added to reduce the last hidden state to output_dim.

    Parameters
    ==========
    vocab_len: int from imdb dataset
    embed_dim: dimensions of the embeddings
    hidden_dim: number of hidden nodes required
    output_dim: numer of output nodes required (1 for sentiment analysis)
    pretrained_vec: weights from imdb object
    layers: number of LSTM cells to be stacked for depth
    bidirectional: boolean
    layernorm: boolean

    """
    def __init__(self, input_dim, hidden_dim, output_dim, layers=1,
                 bidirectional=False, layernorm=False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.layernorm = layernorm

        self.encoder = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers,
                         bidirectional=bidirectional, layernorm=layernorm)
        if self.bidirectional:
            self.decoder = LSTM(input_dim=output_dim, hidden_dim=2 * hidden_dim, layers=layers,
                                bidirectional=False, layernorm=layernorm)
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.decoder = LSTM(input_dim=output_dim, hidden_dim=hidden_dim, layers=layers,
                                bidirectional=False, layernorm=layernorm)
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden_state, cell_state):
        # encoding
        _, (hidden_state, _) = self.encoder(x, hidden_state, cell_state)
        batch_size = x.shape[1]
        timesteps = x.shape[0]
        x = torch.zeros(1, batch_size, self.output_dim)
        output = torch.tensor([])
        if self.bidirectional:
            hidden_state = torch.cat((hidden_state[-1,0,:,:].unsqueeze(0),
                                      hidden_state[-1,1,:,:].unsqueeze(0)), dim=2)
            cell_state = torch.zeros(1, batch_size, 2 * self.hidden_dim)
        # decoding
        for t in range(timesteps):
            x, (hidden_state, cell_state) = self.decoder(x, hidden_state, cell_state)
            x = self.softmax(self.fc(x))
            output = torch.cat((output, x), dim=0)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum
