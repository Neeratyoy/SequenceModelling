import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


## Sentiment Analysis network
class SentimentNetwork(nn.Module):

    def __init__(self, n_input, n_embed, n_hidden, n_output, pretrained_vec=None):
        super().__init__()

        self.hidden_dim = n_hidden

        self.embedding = nn.Embedding(n_input, n_embed)
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(pretrained_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = LSTM(ncell=1, input_dim=n_embed, hidden_dim=n_hidden,
                         output_dim=None, bidirectional=False, xavier_init=True)
        self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, h, c):
        embed = self.embedding(x)
        output, c = self.lstm(embed, h, c)
        # getting latest hidden layer (for next iteration)
        h = output[:, :, -1:]
        y = self.fc(h)
        return y, h, c


## Sentiment Analysis network - using PyTorch LSTM module
class SentimentNetworkBaseline(nn.Module):

    def __init__(self, n_input, n_embed, n_hidden, n_output, layers=1, bidirectional=False, pretrained_vec=None):
        super().__init__()

        self.hidden_dim = n_hidden
        self.bidirectional = bidirectional
        self.layers = layers

        self.embedding = nn.Embedding(n_input, n_embed)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(pretrained_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(n_embed, n_hidden, bidirectional=self.bidirectional, num_layers=layers)
        if self.bidirectional:
            self.fc = nn.Linear(2 * n_hidden, n_output)
        else:
            self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, h, c):
        embed = self.embedding(x)
        _, (h, _) = self.lstm(embed, (h, c))
        if self.bidirectional:
            h = h.view(self.layers, 2, embed.shape[1], self.hidden_dim)
            # Flattening hidden state for the 2 directions in bidirectional
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
        y = self.fc(h) #h.squeeze(0))
        return y