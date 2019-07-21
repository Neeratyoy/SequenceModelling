import os
import math
import spacy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext import data
from torchtext.vocab import GloVe
from torchtext.datasets import IWSLT
from torchtext.data import Field, BucketIterator

class IWSLTDataset():
    def __init__(self, load_embeddings=True, seed=None):
        self.seed = seed
        self.load_embeddings = load_embeddings
        random.seed(self.seed)
        self.random_state = random.getstate()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _download_embeddings(self, dim=300, dest='.data/'):
        # check if destination exists, else create
        self.dest = dest # os.path.join(dest, 'iwslt{}'.format(dim))
        file = 'embed_tweets_de_{}D_fasttext'.format(dim)
        file_zip = '{}.zip'.format(file)
        if not os.path.exists(self.dest):
            os.makedirs(self.dest)
        print(os.path.join(self.dest, file))
        print(not os.path.isfile(os.path.join(self.dest, file)))
        if not os.path.isfile(os.path.join(self.dest, file)):
            os.system('wget http://4530.hostserv.eu/resources/{} '
                      '-O {}'.format(file_zip, os.path.join(self.dest, file_zip)))
            os.system('unzip ' + os.path.join(self.dest, file_zip))

    def load(self):
        # for tokenizing the english sentences
        spacy_en = spacy.load('en')
        # for tokenizing the german sentences
        spacy_de = spacy.load('de')

        def tokenize_de(text):
            # tokenizes the german text into a list of strings(tokens) and reverse it
            # we are reversing the input sentences, as it is observed
            # by reversing the inputs we will get better results
            return [tok.text for tok in spacy_de.tokenizer(text)] ## [::-1]     # list[::-1] used to reverse the list

        def tokenize_en(text):
            # tokenizes the english text into a list of strings(tokens)
            return [tok.text for tok in spacy_en.tokenizer(text)]

        self.SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True)
        self.TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True)
        self.train_data, self.valid_data, self.test_data = IWSLT.splits(exts=('.de', '.en'),
                                                                        fields=(self.SRC, self.TRG))
        print("Number of training samples: {}".format(len(self.train_data.examples)))
        print("Number of validation samples: {}".format(len(self.valid_data.examples)))
        print("Number of testing samples: {}".format(len(self.test_data.examples)))

    def build_vocab(self, dim=100, dest='.data/'):
        vec_en = None
        vec_de = None
        if self.load_embeddings:
            assert(dim == 100 or dim == 300)
            self.dim = dim
            file = 'embed_tweets_de_{}D_fasttext'.format(dim)
            if not os.path.isfile(os.path.join(dest, file)) and \
               not os.path.isfile(os.path.join(dest, file+'.pt')):
                print('Cannot find dataset in path {}.\nDownloading dataset...'.format(dest))
                self._download_embeddings(dim, dest)
            vec_en = GloVe(name='6B', dim=300)
            vec_de = torchtext.vocab.Vectors(file, cache=dest)

        self.SRC.build_vocab(self.train_data, min_freq=2, vectors=vec_de)
        self.TRG.build_vocab(self.train_data, min_freq=2, vectors=vec_en)
        self.src_vocab_size = len(self.SRC.vocab)
        self.trg_vocab_size = len(self.TRG.vocab)
        self.src_pretrained_weights = self.train_data.fields['src'].vocab.vectors
        self.trg_pretrained_weights = self.train_data.fields['trg'].vocab.vectors

    def create_data_loader(self, batch_size=64, device=None):
        if not hasattr(self, "train_data") or not hasattr(self, "SRC") or not hasattr(self, "TRG"):
            self.load()
        if not hasattr(self, "src_vocab_size") or not hasattr(self, "trg_vocab_size"):
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


if __name__ == "__main__":
    iwslt = IWSLTDataset(load_embeddings=True, seed=42)
    iwslt.load()
    # dim=100 for a smaller, quick test
    # dim=300 provides a richer embedding
    iwslt.build_vocab(dim=100, dest='.data/')
    # preparing data loaders (BucketIterator)
    train, valid, test = iwslt.create_data_loader(batch_size=64, device='cuda')
    for batch in train:
        pass
    # x: batch.src; y: batch.trg
    # shape should be of [sequence_length, batch_size]
    print(batch.src.shape, batch.trg.shape)
