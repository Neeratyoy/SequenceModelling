import torch
from torchtext.datasets import IMDB
from torchtext import data
from torchtext.vocab import GloVe

import random


class IMDB_dataset():
    def __init__(self, split_ratio=0.8, seed=None, device=None):
        self.split_ratio = split_ratio
        self.seed = seed
        random.seed(self.seed)
        self.random_state = random.getstate()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device


    def load(self, split_ratio=None, random_state=None, verbose=False):
        if split_ratio is None:
            split_ratio = self.split_ratio
        assert(split_ratio <= 1)
        if random_state is None:
            random_state = self.random_state

        ## create field - tokenize text & create label classes
        self.TEXT = data.Field(tokenize = 'spacy')
        self.LABEL = data.LabelField(dtype = torch.float)

        # load dataset
        self.train_data, self.test_data = IMDB.splits(self.TEXT, self.LABEL)

        # split training into train & validation
        self.train_data, self.valid_data = self.train_data.split(split_ratio=split_ratio,
                                                                 random_state=random_state)
        if verbose:
            print('Training data size:   ', len(self.train_data))
            print('Validation data size: ', len(self.valid_data))
            print('Test data size:       ', len(self.test_data))

    def build_vocab(self, dim=300, max_vocab_size=25000):
        self.max_vocab_size = max_vocab_size
        self.dim = dim
        if not hasattr(self, "train_data") or not hasattr(self, "TEXT") or not hasattr(self, "LABEL"):
            self.load()
        self.TEXT.build_vocab(self.train_data, max_size=self.max_vocab_size,
                              vectors=GloVe(name='6B', dim=self.dim))
        self.LABEL.build_vocab(self.train_data)
        self.pretrained_weights = self.train_data.fields['text'].vocab.vectors

    def create_data_loader(self, batch_size = 64, device=None):
        if not hasattr(self, "train_data") or not hasattr(self, "TEXT") or not hasattr(self, "LABEL"):
            self.load()
        if not hasattr(self, "max_vocab_size") or not hasattr(self, "dim"):
            self.build_vocab()
        # creating loader
        ## data loader equivalent in torchtext batch iterator - buckets similar lengths together
        self.batch_size = batch_size

        if device is None:
            device = self.device

        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=batch_size, device=device)
        return train_iterator, valid_iterator, test_iterator
