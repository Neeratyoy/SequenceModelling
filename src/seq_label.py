import torch
import torch.nn as nn
import numpy as np
import os
from matplotlib import pyplot as plt

from lstm import LSTM

from sklearn.metrics import confusion_matrix, f1_score, classification_report


class LSTM_SeqLabel(nn.Module):
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

    """
    def __init__(self, vocab_len, embed_dim, hidden_dim, output_dim, pretrained_vec,
                 layers=1, bidirectional=False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_len, embed_dim)
        # not training embedding layer if pretrained embedding is provided
        if pretrained_vec is not None:
            self.embedding.weight.data.copy_(pretrained_vec)
            self.embedding.weight.requires_grad = False

        self.lstm = LSTM(input_dim=embed_dim, hidden_dim=hidden_dim, layers=layers,
                         bidirectional=bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state, cell_state):
        embed = self.embedding(x)
        _, (hidden_state, _) = self.lstm(embed, hidden_state, cell_state)
        if self.bidirectional:
            # Flattening hidden state for the 2 directions in bidirectional
            hidden_state = torch.cat((hidden_state[:,0,:,:], hidden_state[:,1,:,:]), dim=2)
        # hidden_state = hidden_state[-1]
        output = self.fc(hidden_state)
        return output

    def save(self, file_path='./model.pkl'):
        torch.save(self.state_dict(), file_path)

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

    def plot_history(self, train, valid=None, epochs=None, file_path='./plot.png', stats='loss'):
        if epochs is None:
            epochs = np.arange(1, len(train)+1)
        plt.clf()
        plt.plot(epochs, train, label="Training")
        if valid is not None:
            plt.plot(epochs, valid, label="Validation")
        plt.title("{} comparison".format(stats))
        plt.xlabel("epochs")
        plt.ylabel(stats)
        plt.legend()
        plt.grid()
        plt.savefig(file_path, dpi=300)

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
                      'train_loss': [], 'valid_loss': []}
        for i in range(epochs):
            loss_tracker = []

            self.model.train()
            for j, batch in enumerate(train_loader):
                # generate initial hidden & cell states
                hidden_state = torch.zeros(1, batch.label.shape[0],
                                           self.model.hidden_dim, requires_grad=True).to(self.device)
                cell_state = torch.zeros(1, batch.label.shape[0],
                                         self.model.hidden_dim, requires_grad=True).to(self.device)

                # forward pass
                output = self.model(batch.text, hidden_state, cell_state)
                # backward pass for the batch (+ weight updates)
                self.optimizer.zero_grad()
                loss = self.loss_criterion(output.view(output.shape[1]), batch.label)
                loss.backward()
                self.optimizer.step()

                # print(".", end='') # for colab (comment below print)
                print("Epoch #{}: Batch {}/{} -- Loss = {}".format(i + 1, j + 1, len(train_loader),
                                                                   loss.item()), end='\r')
                loss_tracker.append(loss.item())

            self.stats['loss'].append(np.mean(loss_tracker))
            loss_tracker = []
            print()
            print("Epoch #{}: Average loss is {}".format(i + 1, self.stats['loss'][-1]))
            if i % freq == 0:
                f1, train_loss = self.evaluate(train_loader, verbose=False)
                self.stats['train_score'].append(f1)
                self.stats['train_loss'].append(train_loss)
                self.stats['epoch'].append(i+1)
                print("Epoch #{}: Train F1-score is {}".format(i + 1, self.stats['train_score'][-1]))
                self.model.save(os.path.join(out_dir, "model_epoch_{}.pkl".format(i+1)))
                self.plot_history(self.stats['train_score'], stats='f1',
                                  file_path=os.path.join(out_dir, "f1score_{}.png".format(i+1)))
                self.plot_history(self.stats['train_loss'], stats='loss',
                                  file_path=os.path.join(out_dir, "loss_{}.png".format(i+1)))
            if i % freq == 0 and valid_loader is not None:
                f1, val_loss = self.evaluate(valid_loader, verbose=False)
                self.stats['valid_score'].append(f1)
                self.stats['valid_loss'].append(val_loss)
                print("Epoch #{}: Validation F1-score is {}".format(i + 1, self.stats['valid_score'][-1]))
                self.plot_history(self.stats['train_loss'], self.stats['valid_loss'],
                                  self.stats['epoch'], stats='loss',
                                  file_path=os.path.join(out_dir, "loss_{}.png".format(i+1)))
            print()
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





# def train_sentiment(model, train_loader, device, epochs, optimizer, loss_criterion, valid_loader=None):
#     """
#     Trains the model of class LSTM using data from data_loader passed as argument
#
#     Parameters
#     ==========
#     model: object of class torch.nn.Module
#     train_loader: an object of class BucketIterator containing training data
#         The next() function of the object returns a batch object, which has 2 members
#         text   - which is a 3D tensor of [sequence, batch, dim]
#             Example-A batch of 10 sentences of 5 words each where each word has
#             an embedding vector of size 256, the 3D tensor shape will be [5, 10, 256]
#         labels - a 1D tensor of length batch_size containing the classes as integers (torch.long())
#     valid_loader: Optional parameter, an object of class BucketIterator containing validation data.
#         Similar to train_loader
#         If passed, then it validates with the given data
#     Returns
#     =======
#     Trained model object of class LSTM, list containing loss progress
#     (, and list containing validation F1 score)
#
#     """
#
#     stats = {'loss': [], 'train_score': [], 'valid_score': []}
#     for i in range(epochs):
#         loss_tracker = []
#
#         model.train()
#         for j, batch in enumerate(train_loader):
#             # generate initial hidden & cell states
#             hidden_state = torch.zeros(1, batch.label.shape[0],
#                                        model.hidden_dim, requires_grad=True).to(device)
#             cell_state = torch.zeros(1, batch.label.shape[0],
#                                      model.hidden_dim, requires_grad=True).to(device)
#
#             # forward pass
#             output = model(batch.text, hidden_state, cell_state)
#             # backward pass for the batch (+ weight updates)
#             optimizer.zero_grad()
#             loss = loss_criterion(output.squeeze(1), batch.label)
#             loss.backward()
#             nn.utils.clip_grad_value_(model.parameters(), 10)
#             optimizer.step()
#
#             # colab_output.clear('batch_stats')
#             # with colab_output.use_tags('batch_stats'):
#             #     print("Epoch #{}: Batch {}/{} -- Loss = {}".format(i + 1, j + 1, len(train_loader),
#             #                                                        loss.item()))
#
#             print("Epoch #{}: Batch {}/{} -- Loss = {}".format(i + 1, j + 1, len(train_loader),
#                                                                loss.item()), end='\r')
#             loss_tracker.append(loss.item())
#
#         stats['loss'].append(np.mean(loss_tracker))
#         loss_tracker = []
#         print()
#         print("Epoch #{}: Average loss is {}".format(i + 1, stats['loss'][-1]))
#         if i % 2 == 0:
#             f1 = evaluate_sentiment(model, train_loader, device, verbose=False)
#             stats['train_score'].append(f1)
#             print("Epoch #{}: Train F1-score is {}".format(i + 1, stats['train_score'][-1]))
#         if i % 2 == 0 and valid_loader is not None:
#             f1 = evaluate_sentiment(model, valid_loader, device, verbose=False)
#             stats['valid_score'].append(f1)
#             print("Epoch #{}: Validation F1-score is {}".format(i + 1, stats['valid_score'][-1]))
#         print()
#     return model, stats
#
#
# def evaluate_sentiment(model, test_loader, device, verbose=True):
#     """
#     Evaluates the model of class LSTM using test data passed
#
#     Parameters
#     ==========
#     model: object of class LSTM
#     test_data: a tuple containing (x, y)
#         x - a 3D tensor of [sequence, len_test_data, dim]
#             Example-A batch of 10 sentences of 5 words each where each word has
#             an embedding vector of size 256, the 3D tensor shape will be [5, 10, 256]
#         y - a 1D tensor containing the classes as integers (torch.long())
#     verbose: prints the confusion matrix and F-score
#
#     Returns
#     =======
#     F-score (float)
#
#     """
#     model.eval()
#
#     preds = []
#     labels = []
#
#     with torch.no_grad():
#         for batch in test_loader:
#             hidden_state = torch.zeros(1, batch.label.shape[0],
#                                        model.hidden_dim, requires_grad=True).to(device)
#             cell_state = torch.zeros(1, batch.label.shape[0],
#                                      model.hidden_dim, requires_grad=True).to(device)
#             output = model(batch.text, hidden_state, cell_state)
#             output = output.view(output.shape[1])
#             # get label predictions - since we predict only probabilities for label 1
#             pred = torch.round(torch.sigmoid(output)).cpu().detach().numpy()
#             preds.extend(pred)
#             labels.extend(batch.label.cpu().detach().numpy())
#
#     if verbose:
#         print('Confusion Matrix: \n', confusion_matrix(labels, preds))
#         print('Classification Report: \n', classification_report(labels, preds))
#     return f1_score(labels, preds)
