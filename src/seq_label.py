import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from matplotlib import pyplot as plt

from lstm import LSTM

from sklearn.metrics import confusion_matrix, f1_score, classification_report


class LSTMSeqLabel(nn.Module):
    """ LSTM Class for Sequence Labelling (many-to-one)

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
    def __init__(self, vocab_len, embed_dim, hidden_dim, output_dim, pretrained_vec,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.layernorm = layernorm

        self.embedding = nn.Embedding(vocab_len, embed_dim)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(pretrained_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = LSTM(input_dim=embed_dim, hidden_dim=hidden_dim, layers=layers,
                         bidirectional=bidirectional, layernorm=layernorm)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state, cell_state):
        embed = self.embedding(x)
        output, (_, _) = self.lstm(embed, hidden_state, cell_state)
        if self.bidirectional:
            ### Flattening output for the 2 directions in bidirectional
            # Taking the last output for Left-to-Right (t=T)
            # Taking the last output for Right-to-left (t=1)
            output = torch.cat((output[-1,:,0,:], output[0,:,1,:]), dim=1)
            output = output[-1].unsqueeze(0)
        else:
            output = output[-1].unsqueeze(0)
        output = self.fc(hidden_state)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


## Sentiment Analysis network - using PyTorch LSTM module
class SentimentNetworkBaseline(nn.Module):

    def __init__(self, n_input, n_embed, n_hidden, n_output, pretrained_vec=None,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()

        self.hidden_dim = n_hidden
        self.bidirectional = bidirectional
        self.layers = layers
        self.layernorm = layernorm

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
        if self.layernorm and self.bidirectional:
            self.ln = LayerNorm(2 * self.hidden_dim)
        elif self.layernorm:
            self.ln = LayerNorm(self.hidden_dim)

    def forward(self, x, h, c):
        embed = self.embedding(x)
        _, (h, _) = self.lstm(embed, (h, c))
        h = h[-1].unsqueeze(0)
        if self.bidirectional:
            h = h.view(self.layers, 2, embed.shape[1], self.hidden_dim)
            # Flattening hidden state for the 2 directions in bidirectional
            h = torch.cat((h[:,0,:,:], h[:,1,:,:]), dim=2)
        if self.layernorm:
            output = self.fc(self.ln(h))
        else:
            output = self.fc(h)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


class SeqLabel():
    """ Class to carry out Sequence Labelling (many-to-one)

    The class is designed to provide training, evaluation and plotting modules.

    Parameters
    ==========
    model: a nn.Module object such that its forward() can be called implicitly
        model's forward() should take 3 arguments - input, hidden_state, cell_state
    optimizer: an initialized nn.optim functional
    loss_fn: an initialized nn loss functional
    device: {"cpu", "cuda"}

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
            epochs = np.arange(1, len(train)+1)
        plt.clf()
        plt.plot(epochs, train, label="Training")
        if valid: # valid is empty
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
                          os.path.join(out_dir, 'f1_score.png'), 'f1 score')
        self.plot_history(stats['train_loss'], stats['valid_loss'], x,
                          os.path.join(out_dir, 'loss.png'), 'loss')

    def save_stats(self, stats, file_path='./stats.json'):
        with open(file_path, 'w') as f:
            json.dump(stats, f)

    def load_stats(self, file_path):
        with open(file_path, 'r') as f:
            self.stats = json.load(f)
        return self.stats

    def train(self, epochs, train_loader, valid_loader=None, freq=10, out_dir='./'):
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
        print("Beginning training model with {} parameters\n".format(self.model.count_parameters()))
        self.stats = {'loss': [], 'train_score': [], 'valid_score': [], 'epoch': [],
                      'train_loss': [], 'valid_loss': [], 'wallclock': []}
        self.freq = freq
        start_training = time.time()
        for i in range(1, epochs+1):
            loss_tracker = []
            start_epoch = time.time()
            self.model.train()
            for j, batch in enumerate(train_loader, start=1):
                # generate initial hidden & cell states
                start = time.time()
                hidden_state = torch.zeros(1, batch.label.shape[0],
                                           self.model.hidden_dim, requires_grad=True).to(self.device)
                cell_state = torch.zeros(1, batch.label.shape[0],
                                         self.model.hidden_dim, requires_grad=True).to(self.device)

                # forward pass
                output = self.model(batch.text, hidden_state, cell_state)
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                loss = self.loss_criterion(output.view(-1), batch.label)
                loss.backward()
                self.optimizer.step()

                # print(".", end='') # for colab (comment below print)
                print("Epoch #{}: Batch {}/{} -- Loss = {}; "
                      "Time taken: {}s".format(i, j, len(train_loader),
                                               loss.item(), time.time() - start), end='\r')
                loss_tracker.append(loss.item())

            self.stats['loss'].append(np.mean(loss_tracker))

            print()
            print("Epoch #{}: Average loss is {}".format(i, self.stats['loss'][-1]))
            if i % freq == 0:
                f1, train_loss = self.evaluate(train_loader, verbose=False)
                self.stats['train_score'].append(f1)
                self.stats['train_loss'].append(train_loss)
                self.stats['epoch'].append(i)
                self.stats['wallclock'].append(time.time() - start_training)
                print("Epoch #{}: Train F1-score is {}".format(i, self.stats['train_score'][-1]))
                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i+1)))
                self.save_stats(self.stats, os.path.join(out_dir, "stats.json"))

                if valid_loader is not None:
                    f1, val_loss = self.evaluate(valid_loader, verbose=False)
                    self.stats['valid_score'].append(f1)
                    self.stats['valid_loss'].append(val_loss)
                    print("Epoch #{}: Validation F1-score is {}".format(i, self.stats['valid_score'][-1]))
            print("Time taken for epoch: {}s".format(time.time() - start_epoch))
            print()

        self.plot_history(self.stats['train_score'], self.stats['valid_score'], stats='f1',
                          file_path=os.path.join(out_dir, "f1score_{}.png".format(i)))
        self.plot_history(self.stats['train_loss'], self.stats['valid_loss'], stats='loss',
                          file_path=os.path.join(out_dir, "loss_{}.png".format(i)))
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
            for batch in test_loader:
                hidden_state = torch.zeros(1, batch.label.shape[0],
                                           self.model.hidden_dim).to(self.device)
                cell_state = torch.zeros(1, batch.label.shape[0],
                                         self.model.hidden_dim).to(self.device)
                output = self.model(batch.text, hidden_state, cell_state)
                output = output.view(output.shape[1])
                loss = self.loss_criterion(output, batch.label)
                # get label predictions - since we predict only probabilities for label 1
                pred = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
                preds.extend(pred)
                labels.extend(batch.label.cpu().detach().numpy())
                losses.append(loss.item())

        if verbose:
            print('Confusion Matrix: \n', confusion_matrix(labels, preds))
            print('Classification Report: \n', classification_report(labels, preds))
        return f1_score(labels, preds), np.mean(losses)
