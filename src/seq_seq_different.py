import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
import random
from matplotlib import pyplot as plt

from lstm import LSTM
from transformer import PositionalEncoding, Encoder, Decoder, get_subsequent_mask

from sklearn.metrics import confusion_matrix, f1_score, classification_report


## ----------------------------------------------------------------------------
## LSTM MODELS
## ----------------------------------------------------------------------------

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


## ----------------------------------------------------------------------------
## TRANSFORMER MODELS
## ----------------------------------------------------------------------------

class TransformerSeq2SeqDifferent(nn.Module):
    
    def __init__(self, in_dim, out_dim, N, heads, model_dim, key_dim, value_dim, ff_dim, 
                 max_len=10000, batch_first=True):
        
        super().__init__()
        self.name = 'transformer'
        
        self.batch_first = batch_first
        self.model_dim = model_dim
        
        # define layers
        # embedding layers
        self.src_embed = nn.Linear(in_dim, model_dim)
        self.tgt_embed = nn.Linear(in_dim, model_dim)
        self.pos_enc = PositionalEncoding(model_dim, max_len)
        # encoder-decoder
        self.encoder = Encoder(N, heads, model_dim, key_dim, value_dim, ff_dim)
        self.decoder = Decoder(N, heads, model_dim, key_dim, value_dim, ff_dim)
        # final output layer
        self.fc = nn.Linear(model_dim, out_dim)
    
        # xavier initialization
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):

        # transpose to use [batch, seq_len, dim]
        if not self.batch_first:
            src = src.transpose(0, 1)
            tgt = tgt.transpose(0, 1)
        
        # get subsequent mask for target sequence
        tgt_subseq_mask = get_subsequent_mask(tgt)
        # combine with tgt mask if provided
        if tgt_mask is not None:
            tgt_subseq_mask = (tgt_mask + tgt_subseq_mask).gt(0)
        
        ## get encoder attention from source
        src = self.src_embed(src)
        src = self.pos_enc(src)
        src_attn = self.encoder(src, src_mask)
        
        ## get decoder attention from target & source attention
        tgt = self.tgt_embed(tgt)
        tgt = self.pos_enc(tgt)
        x = self.decoder(src_attn, tgt, src_mask, tgt_subseq_mask)
        
        x = self.fc(x)
        # transpose to use [batch, seq_len, dim]
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

    def generate(self, src, start_token, src_mask=None, max_len=1):
        
        # transpose to use [batch, seq_len, dim]
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        ## get encoder attention from source
        src = self.src_embed(src)
        src = self.pos_enc(src)
        src_attn = self.encoder(src, src_mask)
        
        # initialize target with start symbol - 1 x b x dim
        tgt = torch.tensor(start_token).view(1,1,-1).repeat(src.shape[0],1,1).float()
        
        for i in range(max_len-1):
            # generate subsequent mask for target sequence
            tgt_subseq_mask = get_subsequent_mask(tgt)
            ## get decoder attention from target & source attention
            tgt_embed = self.tgt_embed(tgt)
            tgt_embed = self.pos_enc(tgt_embed)
            x = self.decoder(src_attn, tgt_embed, src_mask, tgt_subseq_mask)
            # get last predictions and combine with target for next iteration
            x = self.fc(x[:,-1:])
            x = torch.zeros_like(x).scatter_(2, torch.argmax(x, -1, keepdim=True), 1)            
            tgt = torch.cat((tgt, x), dim=1)
            
        # transpose to use [batch, seq_len, dim]
        if not self.batch_first:
            tgt = tgt.transpose(0, 1)
        return tgt


## ----------------------------------------------------------------------------
## TASK SPECIFIC METHODS
## ----------------------------------------------------------------------------

def create_ncopy_task(sequence_length, n_copy, batch_size, start_token, train_test_ratio, train_valid_ratio=None):
    # Generating data
    state_size = sequence_length  # sequence length
    # sequence_length = state_size
    data_size = min(pow(2, state_size), 5000)

    if state_size > 15:
        data_x = np.random.randint(0, 2, (data_size, sequence_length))
    else:
        data_x = []
        # generating all possible binary numbers in the range of [0, 2^state_size]
        for i in range(data_size):
            data_x.append([int(x) for x in list(np.binary_repr(i, width=state_size))])
        data_x = np.array(data_x)

    data_y = data_x.copy()
    for i in range(n_copy - 1):
        data_y = np.concatenate((data_y, data_x), axis=1)

    # Reshaping for tensors
    data_x = np.transpose(data_x).reshape(sequence_length, data_size, 1)
    data_x = torch.from_numpy(data_x).float()
    data_x = torch.zeros(data_x.shape[0], data_x.shape[1], 2).scatter_(2, data_x.long(), 1)

    data_y = np.transpose(data_y).reshape(n_copy * sequence_length, data_size, 1)
    data_y = torch.from_numpy(data_y).float()
    data_y = torch.zeros(data_y.shape[0], data_y.shape[1], 2).scatter_(2, data_y.long(), 1)

    # adding start token
    data_x = torch.nn.functional.pad(data_x, (0,0,0,0,1,0), 'constant', start_token)
    data_y = torch.nn.functional.pad(data_y, (0,0,0,0,1,0), 'constant', start_token)

    # Creating training and test sets
    train_size = train_test_ratio
    ordering = torch.randperm(data_size)
    data_x = data_x[:, ordering, :]
    data_y = data_y[:, ordering, :]
    train_x = data_x[:, :int(train_size * len(ordering)), :]
    train_y = data_y[:, :int(train_size * len(ordering)), :]
    test_x = data_x[:, int(train_size * len(ordering)):, :]
    test_y = data_y[:, int(train_size * len(ordering)):, :]

    # Creating training and validation sets
    if train_valid_ratio is not None:
        train_size = train_valid_ratio
        ordering = torch.randperm(train_x.shape[1])
        train_x = train_x[:, ordering, :]
        train_y = train_y[:, ordering, :]
        valid_x = train_x[:, int(train_size * len(ordering)):, :]
        valid_y = train_y[:, int(train_size * len(ordering)):, :]
        train_x = train_x[:, :int(train_size * len(ordering)), :]
        train_y = train_y[:, :int(train_size * len(ordering)), :]

    # Creating iterators
    train_loader = []
    for j in range(int(train_x.shape[1] / batch_size) + 1):
        start = j * batch_size
        end = min((j + 1) * batch_size, train_x.shape[1])
        batch = end - start
        if batch == 0:
            continue
        train_loader.append((train_x[:, start:end:, ], train_y[:, start:end, :]))
    test_loader = []
    for j in range(int(test_x.shape[1] / batch_size) + 1):
        start = j * batch_size
        end = min((j + 1) * batch_size, test_x.shape[1])
        batch = end - start
        if batch == 0:
            continue
        test_loader.append((test_x[:, start:end:, ], test_y[:, start:end, :]))
    valid_loader = None
    if train_valid_ratio is not None:
        valid_loader = []
        for j in range(int(valid_x.shape[1] / batch_size) + 1):
            start = j * batch_size
            end = min((j + 1) * batch_size, valid_x.shape[1])
            batch = end - start
            if batch == 0:
                continue
            valid_loader.append((valid_x[:, start:end:, ], valid_y[:, start:end, :]))

    return train_loader, test_loader, valid_loader


class Seq2SeqDifferent():
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

    def __init__(self, model, optimizer, loss_fn, start_token=1, teacher_forcing=0.5, device=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_criterion = loss_fn
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.start_token = start_token
        self.teacher_forcing = teacher_forcing
        
        # get forward call for model
        if self.model.name == 'transformer':
            self.forward = self.transformer_forward
        else:
            self.forward = self.lstm_forward

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
    
    def lstm_forward(self, x, y, eval=False):
        ''' forward call for LSTM network '''
        if self.model.bidirectional:
            # shape -> layers x batch x hidden_dim
            hidden_state = torch.zeros(2 * self.model.layers, x.shape[1],
                                       self.model.hidden_dim).to(self.device)
            cell_state = torch.zeros(2 * self.model.layers, x.shape[1],
                                     self.model.hidden_dim).to(self.device)
        else:
            hidden_state = torch.zeros(self.model.layers, x.shape[1],
                                       self.model.hidden_dim).to(self.device)
            cell_state = torch.zeros(self.model.layers, x.shape[1],
                                     self.model.hidden_dim).to(self.device)
        teacher_forcing = 0 if eval else self.teacher_forcing
        
        # forward pass
        o = self.model(x, y, hidden_state, cell_state, teacher_forcing)
        # return the predicted responses, without the start token if training
        o = o[1:] if not eval else o
        return o
    
    def transformer_forward(self, x, y, eval=False):
        ''' forward call for transformer network '''
        if eval:
            o = self.model.generate(x, start_token=self.start_token, max_len=y.shape[0])
        else:
            o = self.model(x, y[:-1])
        return o

    def train(self, epochs, train_loader, valid_loader=None, freq=10,
              out_dir='./', create_dir=True):
        """ Function to train the model and save statistics

        Parameters
        ==========
        epochs: Number of epochs
        train_loader: The generator for training data containing
            Expects a torchtext BucketIterator
        valid_loader: The generator for validation data containing
            Expects a torchtext BucketIterator
        teacher_forcing: Probability of teacher forcing
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
            for j, (train_x, train_y) in enumerate(train_loader, start=1):
                # generate initial hidden & cell states
                train_x = train_x.to(self.device)
                train_y = train_y.to(self.device)
                batch = train_y.shape[1]
                start = time.time()
                
                o = self.forward(train_x, train_y)
                
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                # ignoring start token for ground truth
                gt = torch.argmax(train_y[1:], 2, keepdim=True).contiguous().view(-1)
                loss = self.loss_criterion(o.contiguous().view(-1, 2), gt)
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
                f1, train_loss = self.evaluate(train_loader, verbose=False)
                self.stats['train_score'].append(f1)
                self.stats['train_loss'].append(train_loss)
                self.stats['epoch'].append(i)
                self.stats['wallclock'].append(time.time() - start_training)
                print("Epoch #{}: Train F1 is {}".format(i, self.stats['train_score'][-1]))

                if valid_loader is not None:
                    f1, val_loss = self.evaluate(valid_loader, verbose=False)
                    self.stats['valid_score'].append(f1)
                    self.stats['valid_loss'].append(val_loss)
                    print("Epoch #{}: Validation F1 is {}".format(i, self.stats['valid_score'][-1]))

                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i)))
                self.save_stats(self.stats, os.path.join(out_dir, "stats.json"))

            print("Time taken for epoch: {}s".format(time.time() - start_epoch))
            print()

        self.plot_stats(freq=freq, out_dir=out_dir)
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
            for i, (x, y) in enumerate(test_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                batch = y.shape[1]
                
                print('.', end='')
                
                o = self.forward(x, y, eval=True)
                gt = torch.argmax(y, 2, keepdim=True).view(-1)
                loss = self.loss_criterion(o.contiguous().view(-1, 2), gt)

                pred = torch.argmax(o, 2, keepdim=True).view(-1).cpu().detach().numpy()
                preds.extend(pred)
                label = torch.argmax(y, 2, keepdim=True).view(-1).cpu().detach().numpy()
                labels.extend(label)
                losses.append(loss.item())
                
        print()
        loss = np.mean(losses)
        if verbose:
            print('Confusion Matrix: \n', confusion_matrix(labels, preds))
            print()
            print('Classification Report: \n', classification_report(labels, preds))

        return f1_score(labels, preds), loss
