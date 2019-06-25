import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext import data
from torchtext.vocab import GloVe

import os
import json
import time
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report

from IPython.display import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = None


# Embedding dimension/dimension for LSTM cell inputs
embed_dim = 300
# Number of hidden nodes
hidden_dim = 256
# Number of output nodes
output_dim = 1
# Number of LSTMs cells to be stacked
layers = 1
# Boolean value for bidirectioanl or not
bidirectional = False
# Boolean value to use LayerNorm or not
layernorm = False


batch_size = 8
# Percentage of training data
split_ratio = 0.8
learning_rate = 0.001
epochs = 100


from imdb import IMDB_dataset

imdb = IMDB_dataset(split_ratio, seed)
imdb.load(verbose = True)
imdb.build_vocab(embed_dim)
train_loader, valid_loader, test_loader = imdb.create_data_loader(batch_size, 
                                                                  device)
vocab_len = len(imdb.TEXT.vocab)


# Our implementation

from seq_label import LSTMSeqLabel, SeqLabel

# Initializing model
model = LSTMSeqLabel(vocab_len, embed_dim, hidden_dim, output_dim, 
                      imdb.pretrained_weights, layers, bidirectional,
                      layernorm)
model.to(device)

# Initializing optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_criterion = nn.BCEWithLogitsLoss()

# Initializing task
task = SeqLabel(model, optimizer, loss_criterion, device)

# Training
freq = 5    # epoch interval to calculate F1 score and save models
out_dir = "../results/SeqLabel/"
# out_dir = "/content/drive/My Drive/colab/seq_label/"
model, stats = task.train(epochs, train_loader, valid_loader, freq, out_dir)

print("=" * 50)

test_f1, test_loss = task.evaluate(test_loader, verbose=True)
print("Test F1: {}\nTest Loss: {}\n".format(test_f1, test_loss))