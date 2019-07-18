import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score

from lstm import LSTM
from transformer import PositionalEncoding, Encoder, get_pad_mask_n_dim


## ----------------------------------------------------------------------------
## LSTM MODELS
## ----------------------------------------------------------------------------

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
        self.name = 'lstm'

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

    def forward(self, x, hidden_state, cell_state):
        output, (_, _) = self.lstm(x, hidden_state, cell_state)
        orig_dims = output.shape
        # fc computation for each element
        output = self.fc(output.view(-1, output.shape[-1]))
        # reshaping to have (seq_len, batch, output)
        output = output.view(orig_dims[0], orig_dims[1], output.shape[1])
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
        self.name = 'lstm'

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

    def forward(self, x, hidden_state, cell_state):
        output, (_, _) = self.lstm(x, (hidden_state, cell_state))
        orig_dims = output.shape
        # fc computation for each element
        output = self.fc(output.view(-1, output.shape[-1]))
        # reshaping to have (seq_len, batch, output)
        output = output.view(orig_dims[0], orig_dims[1], output.shape[1])
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


## ----------------------------------------------------------------------------
## TRANSFORMER MODELS
## ----------------------------------------------------------------------------

class TransformerSeq2SeqSame(nn.Module):
    """ Transformer Class for Sequence to Sequence (many-to-many same)

    The class creates the Transformer encoder architecture as specified by the parameters.
    A fully connected layer is added to reduce the attention to output_dim.

    Parameters
    ==========
    in_dim: input vocab size from bAbi dataset
    out_dim: output dimensions of the model
    N: number of encoder & decoder layers
    model_dim: embedding dimension, also the dimensionality at which the transformer operates
    key_dim: dimensions for query & key in attention calculation
    value_dim: dimensions for value in attention calculation
    ff_dim: dimensions for Positionwise feed-forward sublayer
    max_len: max length to generate positional encodings (default=10000)
    batch_first: if batch is the 1st input dimension [seq_len, batch, dim] (default=False)
    """
    def __init__(self, in_dim, out_dim, N, heads, model_dim, key_dim, value_dim, ff_dim, max_len=10000, batch_first=True):
        
        super().__init__()
        self.name = 'transformer'
        
        self.batch_first = batch_first
        self.model_dim = model_dim
        
        # define layers
        self.embed = nn.Linear(in_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len)
        self.encoder = Encoder(N, heads, model_dim, key_dim, value_dim, ff_dim)
        # final output layer
        self.fc = nn.Linear(model_dim, out_dim)

        # xavier initialization
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        # transpose to use [batch, seq_len, dim]
        if not self.batch_first:
            x = x.transpose(0, 1)
            
        x = self.embed(x)
        x = self.pos_enc(x)
        x = self.encoder(x, mask)
        x = self.fc(x)
        
        # transpose back to original [seq_len, batch, dim]
        if not self.batch_first:
            x = x.transpose(0, 1)
        return x
        
    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tot_sum

        
## ----------------------------------------------------------------------------
## TASK SPECIFIC METHODS
## ----------------------------------------------------------------------------

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
        
        # get forward call for model
        if self.model.name == 'transformer':
            self.forward = self.transformer_forward
        else:
            self.forward = self.lstm_forward

    def plot_history(self, train, valid=[], epochs=None, file_path='./plot.png', stats='loss'):
        if epochs is None:
            epochs = np.arange(1, len(train) + 1)
        plt.clf()
        if train:  # train is not empty
            plt.plot(epochs, train, label="Training")
        if valid:  # valid is not empty
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
        x = np.arange(start=0, stop=freq * (len(stats['epoch'])), step=freq)
        x[0] = 1
        self.plot_history(stats['train_score'], stats['valid_score'], x,
                          os.path.join(out_dir, 'wer.png'), 'wer')
        self.plot_history(stats['train_loss'], stats['valid_loss'], x,
                          os.path.join(out_dir, 'loss.png'), 'loss')

    def save_stats(self, stats, file_path='./stats.json'):
        with open(file_path, 'w') as f:
            json.dump(stats, f)

    def load_stats(self, file_path):
        with open(file_path, 'r') as f:
            self.stats = json.load(f)
        return self.stats
      
    def lstm_forward(self, x):
        ''' calls forward pass for LSTM '''
        if self.model.bidirectional:
            hidden_state = torch.zeros(2 * self.model.layers, x.shape[1],
                                       self.model.hidden_dim).to(self.device)
            cell_state = torch.zeros(2 * self.model.layers, x.shape[1],
                                     self.model.hidden_dim).to(self.device)
        else:
            hidden_state = torch.zeros(self.model.layers, x.shape[1],
                                       self.model.hidden_dim).to(self.device)
            cell_state = torch.zeros(self.model.layers, x.shape[1],
                                     self.model.hidden_dim).to(self.device)
        # forward pass
        output = self.model(x, hidden_state, cell_state)
        return output
      
    def transformer_forward(self, x):
        ''' calls forward pass for Transformer '''
        mask = get_pad_mask_n_dim(x.transpose(0,1), x.transpose(0,1), pad_pos=0)
        output = self.model(x, mask)
        return output

    def train(self, epochs, train_loader, valid_loader=None, freq=10, out_dir='./',
              vocab=None, wer_dict=None, create_dir=True, train_eval=True):
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
        
        # create output directory if it does not exist
        if create_dir:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        
        start_training = time.time()

        for i in range(1, epochs + 1):
            loss_tracker = []
            start_epoch = time.time()
            self.model.train()
            for j, (text, label) in enumerate(train_loader, start=1):
                # generate initial hidden & cell states
                batch = label.shape[0]
                start = time.time()
                # input to have (seq_len, batch, input_dim)
                text = text.transpose(0, 1).to(self.device)
                label = label.transpose(0, 1).to(self.device)

                # forward pass
                output = self.forward(text)
                
                # reshape to have (N, classes)
                output = output.contiguous().view(-1, output.shape[-1])
                label = label.contiguous().view(-1)
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                loss = self.loss_criterion(output, label.contiguous().view(-1))
                loss.backward()
                self.optimizer.step()

                # print(".", end='')  # for colab (comment below print)
                # print("Epoch #{}: Batch {}/{} -- Loss = {}; "
                #       "Time taken: {}s".format(i, j, len(train_loader),
                #                                loss.item(), time.time() - start), end='\r')
                loss_tracker.append(loss.item())

            self.stats['loss'].append(np.mean(loss_tracker))
            print()
            print("Epoch #{}: Average loss is {}".format(i, self.stats['loss'][-1]))
            if i % freq == 0 or i == 1:
                self.stats['epoch'].append(i)
                self.stats['wallclock'].append(time.time() - start_training)
                if train_eval:
                    accuracy, train_loss = self.evaluate(train_loader, vocab,
                                                         wer_dict, verbose=False)
                    self.stats['train_score'].append(accuracy)
                    self.stats['train_loss'].append(train_loss)
                    print("Epoch #{}: Train WER is {}".format(i, self.stats['train_score'][-1]))
                if valid_loader is not None:
                    accuracy, val_loss = self.evaluate(valid_loader, vocab,
                                                       wer_dict, verbose=False)
                    self.stats['valid_score'].append(accuracy)
                    self.stats['valid_loss'].append(val_loss)
                    print("Epoch #{}: Validation WER is {}".format(i, self.stats['valid_score'][-1]))

                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i)))
                self.save_stats(self.stats, os.path.join(out_dir, "stats.json"))

            print("Time taken for epoch: {}s".format(time.time() - start_epoch))
            print()

        self.plot_stats(freq=freq, out_dir=out_dir)
        print("Training completed in {}s".format(time.time() - start_training))
        return self.model, self.stats

    def evaluate(self, test_loader, vocab=None, wer_dict=None, verbose=True):
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
                # input to have (seq_len, batch, input_dim)
                text = text.transpose(0, 1).to(self.device)
                label = label.transpose(0, 1).to(self.device)

                output = self.forward(text)
                
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

        vocab = np.array(vocab)
        scores = []
        scores = [wer_dict[str(vocab[labels[i]])][str(vocab[preds[i]])] for i in range(len(preds))]
        wer_score = np.mean(scores)

        if verbose:
            print('Confusion Matrix: \n', confusion_matrix(labels, preds))
            print()
            print('Classification Report: \n', classification_report(labels, preds))

        return wer_score, np.mean(losses)