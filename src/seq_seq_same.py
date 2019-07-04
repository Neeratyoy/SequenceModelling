import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from matplotlib import pyplot as plt

from lstm import LSTM

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


class LSTMSeq2SeqSame(nn.Module):
    """ LSTM Class for Sequence to Sequence (many-to-many same)

    The class creates the LSTM architecture as specified by the parameters.
    A fully connected layer is added to reduce the last hidden state to output_dim.

    Parameters
    ==========
    input_dim: input dimensions
    hidden_dim: number of hidden nodes required
    output_dim: numer of output nodes required (1 for sentiment analysis)
    layers: number of LSTM cells to be stacked for depth
    bidirectional: boolean
    layernorm: boolean

    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.layernorm = layernorm

        self.lstm = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, layers=layers,
                         bidirectional=bidirectional, layernorm=layernorm)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden_state, cell_state):
        output, (_, _) = self.lstm(x, hidden_state, cell_state)
        orig_dims = output.shape
        # fc computation for each element
        output = self.fc(output.view(-1, output.shape[-1]))
        # reshaping to have (seq_len, batch, output)
        output = output.view(orig_dims[0], orig_dims[1], output.shape[1])
        output = self.softmax(output)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


class PyTorchBaseline(nn.Module):
    """ LSTM Class for Sequence to Sequence (many-to-many same)

    The class creates the LSTM architecture as specified by the parameters.
    A fully connected layer is added to reduce the last hidden state to output_dim.

    Parameters
    ==========
    input_dim: input dimensions
    hidden_dim: number of hidden nodes required
    output_dim: numer of output nodes required (1 for sentiment analysis)
    layers: number of LSTM cells to be stacked for depth
    bidirectional: boolean
    layernorm: boolean

    """

    def __init__(self, input_dim, hidden_dim, output_dim,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.layernorm = layernorm

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
                         bidirectional=bidirectional) #, layernorm=layernorm)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden_state, cell_state):
        output, (_, _) = self.lstm(x, (hidden_state, cell_state))
        orig_dims = output.shape
        # fc computation for each element
        output = self.fc(output.view(-1, output.shape[-1]))
        # reshaping to have (seq_len, batch, output)
        output = output.view(orig_dims[0], orig_dims[1], output.shape[1])
        output = self.softmax(output)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


class Seq2SeqSame():
    """ Class to carry out Sequence 2 Sequence (many-to-many same)

    The class is designed to provide training, evaluation and plotting modules.

    Parameters
    ==========
    model: a nn.Module object such that its forward() can be called implicitly
        model's forward() should take 3 arguments - input, hidden_state, cell_state
    optimizer: an initialized nn.optim functional
    loss_fn: an initialized nn loss functional
    device: A torch device object {"cpu", "cuda"}

    """

    def __init__(self, model, optimizer, loss_fn, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_criterion = loss_fn
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

    def plot_history(self, train, valid=[], epochs=None, file_path='./plot.png', stats='loss'):
        if epochs is None:
            epochs = np.arange(1, len(train) + 1)
        plt.clf()
        plt.plot(epochs, train, label="Training")
        if valid:  # valid is empty
            plt.plot(epochs, valid, label="Validation")
        plt.title("{} comparison".format(stats))
        plt.xlabel("epochs")
        plt.ylabel(stats)
        plt.legend()
        plt.grid()
        plt.savefig(file_path, dpi=300)

    def plot_stats(self, stats=None, freq=None, out_dir='./'):
        if stats is None:
            stats = self.stats
        if freq is None:
            freq = self.freq
        x = np.arange(start=freq, stop=freq * (len(stats['epoch']) + 1),
                      step=freq)
        self.plot_history(stats['train_score'], stats['valid_score'], x,
                          os.path.join(out_dir, 'accuracy.png'), 'accuracy')
        self.plot_history(stats['train_loss'], stats['valid_loss'], x,
                          os.path.join(out_dir, 'loss.png'), 'loss')

    def save_stats(self, stats, file_path='./stats.json'):
        with open(file_path, 'w') as f:
            json.dump(stats, f)

    def load_stats(self, file_path):
        with open(file_path, 'r') as f:
            self.stats = json.load(f)
        return self.stats

    def train(self, epochs, train_loader, valid_loader=None, freq=10, out_dir='./', create_dir=True):
        """ Function to train the model and save statistics

        Parameters
        ==========
        epochs: Number of epochs
        train_loader: The generator for training data containing
            Expects a torchtext BucketIterator
        valid_loader: The generator for validation data containing
            Expects a torchtext BucketIterator
        freq: The number of epoch intervals to validate and save models
        out_dir: Directory where models and plots are to be saved

        Returns
        =======
        model: The trained model

        """
        print("Beginning training model with {} parameters".format(self.model.count_parameters()))
        print("Files will be saved in: {}".format(out_dir))
        print()
        self.stats = {'loss': [], 'train_score': [], 'valid_score': [], 'epoch': [],
                      'train_loss': [], 'valid_loss': [], 'wallclock': []}
        self.freq = freq
        start_training = time.time()

        # create output directory if it does not exist
        if create_dir:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for i in range(1, epochs + 1):
            loss_tracker = []
            start_epoch = time.time()
            self.model.train()
            for j, (text, label) in enumerate(train_loader, start=1):
                # generate initial hidden & cell states
                batch = label.shape[0]
                start = time.time()
                if self.model.bidirectional:
                    hidden_state = torch.zeros(2 * self.model.layers, batch,
                                               self.model.hidden_dim).to(self.device)
                    cell_state = torch.zeros(2 * self.model.layers, batch,
                                             self.model.hidden_dim).to(self.device)
                else:
                    hidden_state = torch.zeros(self.model.layers, batch,
                                               self.model.hidden_dim).to(self.device)
                    cell_state = torch.zeros(self.model.layers, batch,
                                             self.model.hidden_dim).to(self.device)

                # input to have (seq_len, batch, input_dim)
                text = text.transpose(0, 1).to(self.device)
                label = label.transpose(0, 1).to(self.device)

                # forward pass
                output = self.model(text, hidden_state, cell_state)
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                # reshape to have (N, classes)
                loss = self.loss_criterion(output.contiguous().view(-1, output.shape[-1]),
                                           label.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()

                # print(".", end='')  # for colab (comment below print)
                print("Epoch #{}: Batch {}/{} -- Loss = {}; "
                      "Time taken: {}s".format(i, j, len(train_loader),
                                               loss.item(), time.time() - start), end='\r')
                loss_tracker.append(loss.item())

            self.stats['loss'].append(np.mean(loss_tracker))
            print()
            print("Epoch #{}: Average loss is {}".format(i, self.stats['loss'][-1]))
            if i % freq == 0 or i == 1:
                accuracy, train_loss = self.evaluate(train_loader, verbose=False)
                self.stats['train_score'].append(accuracy)
                self.stats['train_loss'].append(train_loss)
                self.stats['epoch'].append(i)
                self.stats['wallclock'].append(time.time() - start_training)

                if valid_loader is not None:
                    accuracy, val_loss = self.evaluate(valid_loader, verbose=False)
                    self.stats['valid_score'].append(accuracy)
                    self.stats['valid_loss'].append(val_loss)
                    print("Epoch #{}: Validation Accuracy is {}".format(i, self.stats['valid_score'][-1]))

                print("Epoch #{}: Train Accuracy is {}".format(i, self.stats['train_score'][-1]))
                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i)))
                self.save_stats(self.stats, os.path.join(out_dir, "stats.json"))

            print("Time taken for epoch: {}s".format(time.time() - start_epoch))
            print()

        self.plot_stats(out_dir=out_dir)
        print("Training completed in {}s".format(time.time() - start_training))
        return self.model, self.stats

    def evaluate(self, test_loader, verbose=True):
        """ Function to evaluate the model and return F-score

        epochs: Number of epochs
        train_loader: The generator for training data containing
            Expects a torchtext BucketIterator
        valid_loader: The generator for validation data containing
            Expects a torchtext BucketIterator
        freq: The number of epoch intervals to validate and save models
        out_dir: Directory where models and plots are to be saved

        """
        self.model.eval()

        preds = []
        labels = []
        losses = []

        with torch.no_grad():
            for text, label in test_loader:
                batch = label.shape[0]
                if self.model.bidirectional:
                    hidden_state = torch.zeros(2 * self.model.layers, batch,
                                               self.model.hidden_dim).to(self.device)
                    cell_state = torch.zeros(2 * self.model.layers, batch,
                                             self.model.hidden_dim).to(self.device)
                else:
                    hidden_state = torch.zeros(self.model.layers, batch,
                                               self.model.hidden_dim).to(self.device)
                    cell_state = torch.zeros(self.model.layers, batch,
                                             self.model.hidden_dim).to(self.device)

                # input to have (seq_len, batch, input_dim)
                text = text.transpose(0, 1).to(self.device)
                label = label.transpose(0, 1).to(self.device)

                output = self.model(text, hidden_state, cell_state)
                # reshape to have (N, classes)
                output = output.contiguous().view(-1, output.shape[-1])
                label = label.contiguous().view(-1)
                loss = self.loss_criterion(output, label)
                # get label predictions - since we only care about the predictions from
                # positions which has a label (i.e., non '_')
                label_ids = label.nonzero().contiguous().view(-1).cpu().detach().numpy()
                label = label[label_ids].cpu().detach().numpy()
                pred = torch.argmax(output, dim=1).cpu().detach().numpy()
                pred = pred[label_ids]

                preds.extend(pred)
                labels.extend(label)
                losses.append(loss.item())

        if verbose:
            print('Confusion Matrix: \n', confusion_matrix(labels, preds))
            print('Classification Report: \n', classification_report(labels, preds))

        return accuracy_score(labels, preds), np.mean(losses)