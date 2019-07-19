import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from lstm import LSTM
from transformer import PositionalEncoding, Encoder, get_pad_mask


## ----------------------------------------------------------------------------
## LSTM MODELS
## ----------------------------------------------------------------------------

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
    pretrained_vec: weights from embedding model (GloVe)
    layers: number of LSTM cells to be stacked for depth
    bidirectional: boolean
    layernorm: boolean

    """
    def __init__(self, vocab_len, embed_dim, hidden_dim, output_dim, pretrained_vec,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()
        self.name = 'lstm'

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.layernorm = layernorm
        self.layers = layers

        self.embedding = nn.Embedding(vocab_len, embed_dim)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding = self.embedding.from_pretrained(
            pretrained_vec, freeze=True)

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
            output = output.unsqueeze(0)
        else:
            output = output[-1].unsqueeze(0)
        output = self.fc(output)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        # tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


## Sentiment Analysis network - using PyTorch LSTM module
class PyTorchBaseline(nn.Module):

    def __init__(self, n_input, n_embed, n_hidden, n_output, pretrained_vec=None,
                 layers=1, bidirectional=False, layernorm=False):
        super().__init__()
        self.name = 'lstm'

        self.hidden_dim = n_hidden
        self.bidirectional = bidirectional
        self.layers = layers
        self.layernorm = layernorm

        self.embedding = nn.Embedding(n_input, n_embed)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding = self.embedding.from_pretrained(
            pretrained_vec, freeze=True)

        self.lstm = nn.LSTM(n_embed, n_hidden, bidirectional=self.bidirectional, num_layers=layers)
        if self.bidirectional:
            self.fc = nn.Linear(2 * n_hidden, n_output)
        else:
            self.fc = nn.Linear(n_hidden, n_output)

    def forward(self, x, h, c):
        embed = self.embedding(x)
        output, (_, _) = self.lstm(embed, (h, c))
        output = output[-1].unsqueeze(0)
        output = self.fc(output)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

    def load(self, file_path):
        self.load_state_dict(torch.load(file_path))

    def count_parameters(self):
        tot_sum = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # tot_sum = sum(p.numel() for p in self.lstm.parameters() if p.requires_grad)
        # tot_sum += sum(p.numel() for p in self.fc.parameters() if p.requires_grad)
        return tot_sum


## ----------------------------------------------------------------------------
## TRANSFORMER MODELS
## ----------------------------------------------------------------------------

from transformer import PositionalEncoding, Encoder, get_pad_mask

class TransformerSeqLabel(nn.Module):
    """ Transformer Class for Sequence Labelling (many-to-one)

    The class creates the Transformer encoder architecture as specified by the parameters.
    A fully connected layer is added to reduce the attention to output_dim.
    The final prediction is averaged over sequence length to get final score

    Parameters
    ==========
    in_dim: input vocab size from imdb dataset
    out_dim: output dimensions of the model
    N: number of encoder & decoder layers
    model_dim: embedding dimension, also the dimensionality at which the transformer operates
    key_dim: dimensions for query & key in attention calculation
    value_dim: dimensions for value in attention calculation
    ff_dim: dimensions for Positionwise feed-forward sublayer
    max_len: max length to generate positional encodings (default=10000)
    batch_first: if batch is the 1st input dimension [seq_len, batch, dim] (default=False)
    pretrained_vec: weights from embedding model (GloVe)
    """
    def __init__(self, in_dim, out_dim, N, heads, embed_dim, model_dim, key_dim, value_dim, ff_dim,
                 dropout=0.1, max_len=10000, batch_first=True, pretrained_vec=None):
        
        super().__init__()
        self.name = 'transformer'
        
        self.batch_first = batch_first
        self.model_dim = model_dim
        self.embed_dim = embed_dim
        
        # define layers
        self.embedding = nn.Embedding(in_dim, embed_dim)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding = self.embedding.from_pretrained(
            pretrained_vec, freeze=True)
        if embed_dim != model_dim:
            self.fc_in = nn.Linear(embed_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len)
        self.encoder = Encoder(N, heads, model_dim, key_dim, value_dim, ff_dim, dropout=dropout)
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
            
        x = self.embedding(x)
        if self.embed_dim != self.model_dim:
            x = self.fc_in(x)
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
        
        # get forward call for model
        if self.model.name == 'transformer':
            self.forward = self.transformer_forward
        else:
            self.forward = self.lstm_forward

    def plot_history(self, train, valid=[], epochs=None, file_path='./plot.png', stats='loss'):
        if epochs is None:
            epochs = np.arange(1, len(train)+1)
        plt.clf()
        if train:  # train is not empty
            plt.plot(epochs, train, label="Training")
        if valid: # valid is not empty
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
                          os.path.join(out_dir, 'f1.png'), 'f1')
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
        ''' calls forward pass for Transformer'''
        # generating padding mask
        # mask = None
        mask = get_pad_mask(x.transpose(0,1), x.transpose(0,1), pad=1)
        output = self.model(x, mask)
        # mean over all attention output (seq_len) for a given sequence
        output = output.mean(dim=0)
        return output

    def train(self, epochs, train_loader, valid_loader=None, freq=10, out_dir=None, create_dir=True, train_eval=True):
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
        freq_train_time = 0.0

        # create output directory if it does not exist
        if create_dir and out_dir:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        
        for i in range(1, epochs+1):
            loss_tracker = []
            start_epoch = time.time()
            self.model.train()
            for j, batch in enumerate(train_loader, start=1):
                # generate initial hidden & cell states
                start = time.time()
                
                # forward pass
                output = self.forward(batch.text)
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                loss = self.loss_criterion(output.view(-1), batch.label)
                loss.backward()
                self.optimizer.step()

                # print(".", end='') # for colab (comment below print)
                # print("Epoch #{}: Batch {}/{} -- Loss = {}; "
                #       "Time taken: {}s".format(i, j, len(train_loader),
                #                                loss.item(), time.time() - start), end='\r')
                loss_tracker.append(loss.item())

            # Save stats and models if out_dir is given
            self.stats['loss'].append(np.mean(loss_tracker))
            freq_train_time += time.time() - start_epoch  # to include only train time and not eval
            print()
            print("Epoch #{}: Average loss is {}".format(i, self.stats['loss'][-1]))
            if i % freq == 0 or i == 1:
                self.stats['epoch'].append(i)
                self.stats['wallclock'].append(freq_train_time)
                freq_train_time = 0.0
                # self.stats['wallclock'].append(time.time() - start_training)
                if train_eval:
                    f1, train_loss = self.evaluate(train_loader, verbose=False)
                    self.stats['train_score'].append(f1)
                    self.stats['train_loss'].append(train_loss)
                    print("Epoch #{}: Train F1 is {}".format(i, self.stats['train_score'][-1]))
                if valid_loader is not None:
                    f1, val_loss = self.evaluate(valid_loader, verbose=False)
                    self.stats['valid_score'].append(f1)
                    self.stats['valid_loss'].append(val_loss)
                    print("Epoch #{}: Validation F1 is {}".format(i, self.stats['valid_score'][-1]))

                if out_dir:
	                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i)))
	                self.save_stats(self.stats, os.path.join(out_dir, "stats.json"))

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
                output = self.forward(batch.text)
                loss = self.loss_criterion(output.view(-1), batch.label)
                # get label predictions - since we predict only probabilities for label 1
                pred = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
                preds.extend(pred)
                labels.extend(batch.label.cpu().detach().numpy())
                losses.append(loss.item())

        if verbose:
            print('Confusion Matrix: \n', confusion_matrix(labels, preds))
            print('Classification Report: \n', classification_report(labels, preds))
        return f1_score(labels, preds), np.mean(losses)
