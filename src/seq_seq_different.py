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

    def forward(self, x, target, hidden_state, cell_state, teacher_forcing=0.5):
        device = 'cpu'
        if x.is_cuda:
            device = 'cuda'
        # encoding
        _, (hidden_state, cell_state) = self.encoder(x, hidden_state, cell_state)
        batch_size = x.shape[1]
        timesteps = target.shape[0]
        x = torch.zeros(1, batch_size, self.output_dim).to(device)
        output = torch.tensor([]).to(device)
        if self.bidirectional:
            # concatenating hidden states from two directions
            hidden_state = torch.cat((hidden_state[:self.layers,:,:],
                                      hidden_state[self.layers:,:,:]), dim=2)
            cell_state = torch.cat((cell_state[:self.layers,:,:],
                                    cell_state[self.layers:,:,:]), dim=2)
        # decoding
        for t in range(timesteps):
            x, (hidden_state, cell_state) = self.decoder(x, hidden_state, cell_state)
            x = self.softmax(self.fc(x))
            output = torch.cat((output, x), dim=0)
            choice = random.random()
            if choice < teacher_forcing:
                x = target[t].float().to(device)
                x = x.unsqueeze(0)
            else:
                # converting x to a one-hot encoding
                x = torch.zeros(x.shape).to(device).scatter_(2, torch.argmax(x, -1, keepdim=True), 1)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum

class PyTorchBaseline(nn.Module):
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

        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
                         bidirectional=bidirectional) #, layernorm=layernorm)
        if self.bidirectional:
            self.decoder = nn.LSTM(input_size=output_dim, hidden_size=2 * hidden_dim, num_layers=layers,
                                bidirectional=False) #, layernorm=layernorm)
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.decoder = nn.LSTM(input_size=output_dim, hidden_size=hidden_dim, num_layers=layers,
                                bidirectional=False) #, layernorm=layernorm)
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, target, hidden_state, cell_state, teacher_forcing=0.5):
        device = 'cpu'
        if x.is_cuda:
            device = 'cuda'
        # encoding
        _, (hidden_state, cell_state) = self.encoder(x, (hidden_state, cell_state))
        batch_size = x.shape[1]
        timesteps = target.shape[0]
        x = torch.zeros(1, batch_size, self.output_dim).to(device)
        output = torch.tensor([]).to(device)
        if self.bidirectional:
            # concatenating hidden states from two directions
            hidden_state = torch.cat((hidden_state[:self.layers,:,:],
                                      hidden_state[self.layers:,:,:]), dim=2)
            cell_state = torch.cat((cell_state[:self.layers,:,:],
                                    cell_state[self.layers:,:,:]), dim=2)
        # decoding
        for t in range(timesteps):
            x, (hidden_state, cell_state) = self.decoder(x, (hidden_state, cell_state))
            x = self.softmax(self.fc(x))
            output = torch.cat((output, x), dim=0)
            choice = random.random()
            if choice < teacher_forcing:
                x = target[t].float().to(device)
                x = x.unsqueeze(0)
            else:
                # converting x to a one-hot encoding
                x = torch.zeros(x.shape).to(device).scatter_(2, torch.argmax(x, -1, keepdim=True), 1)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


# def train(model, train_x, train_y, test_x, test_y, epochs, loss_fn, optimizer, teacher_forcing=0.5):
#     train_size = train_x.shape[1]
#     device = torch.device("cpu")
#     if train_x.is_cuda:
#         device = torch.device("cuda")
#     layers = model.layers
#     hidden_dim = model.hidden_dim
#     for i in range(1, epochs + 1):
#         model.train()
#         loss_tracker = []
#         ordering = torch.randperm(train_size)
#         train_x = train_x[:, ordering, :]
#         train_y = train_y[:, ordering, :]
#         for j in range(int(float(train_size) / batch_size) + 1):
#             optimizer.zero_grad()
#             start = j * batch_size
#             end = min((j + 1) * batch_size, train_size)
#             batch = end - start
#             if batch is 0:
#                 continue
#             if model.bidirectional:
#                 hidden_state = torch.zeros(2 * layers, batch, hidden_dim).to(device)
#                 cell_state = torch.zeros(2 * layers, batch, hidden_dim).to(device)
#             else:
#                 hidden_state = torch.zeros(layers, batch, hidden_dim).to(device)
#                 cell_state = torch.zeros(layers, batch, hidden_dim).to(device)
#             o = model(train_x[:, start:end, :], train_y[:, start:end, :], hidden_state,
#                       cell_state, teacher_forcing)
#             gt = torch.argmax(train_y[:, start:end, :], 2, keepdim=True).view(-1)
#             loss = loss_fn(o.view(-1, 2), gt)
#             loss_tracker.append(loss.item())
#             loss.backward()
#             optimizer.step()
#             print("Epoch #{:<3d}: Batch {:>3d}/{:<3d} -- "
#                   "Loss: {:2.5}".format(i, j + 1, int(train_size / batch_size),
#                                         loss_tracker[-1]), end='\r')
#         print()
#         f1_train = evaluate(model, train_x, train_y)
#         f1_test = evaluate(model, test_x, test_y)
#         print("Average Loss: {:2.6}".format(np.mean(loss_tracker)))
#         print("Training F1: {:3.4}".format(f1_train))
#         print("Test F1: {:3.4}".format(f1_test))
#         print("=" * 50)
#
#     return model
#
#
# def evaluate(model, x, y):
#     model.eval()
#     test_size = x.shape[1]
#     device = torch.device("cpu")
#     if x.is_cuda:
#         device = torch.device("cuda")
#     layers = model.layers
#     hidden_dim = model.hidden_dim
#     labels = []
#     preds = []
#     for j in range(int(test_size / batch_size) + 1):
#         optimizer.zero_grad()
#         start = j * batch_size
#         end = min((j + 1) * batch_size, test_size)
#         batch = end - start
#         if batch == 0:
#             continue
#         if model.bidirectional:
#             hidden_state = torch.zeros(2 * layers, batch, hidden_dim).to(device)
#             cell_state = torch.zeros(2 * layers, batch, hidden_dim).to(device)
#         else:
#             hidden_state = torch.zeros(layers, batch, hidden_dim).to(device)
#             cell_state = torch.zeros(layers, batch, hidden_dim).to(device)
#         with torch.no_grad():
#             o = model(x[:, start:end, :], y[:, start:end, :], hidden_state, cell_state, teacher_forcing=0)
#         pred = torch.argmax(o, 2, keepdim=True).view(-1).cpu().detach().numpy()
#         preds.extend(pred)
#         label = torch.argmax(y[:, start:end, :], 2,
#                              keepdim=True).view(-1).cpu().detach().numpy()
#         labels.extend(label)
#     return f1_score(labels, preds)