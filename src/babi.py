import random
import re
import os
from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import DataLoader, TensorDataset, random_split

class bAbi_dataset():
    def __init__(self, split_ratio=0.8, seed=None, device=None):
        self.split_ratio = split_ratio
        self.seed = seed
        random.seed(self.seed)
        self.random_state = random.getstate()
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        assert(split_ratio <= 1)
        
        # url to download dataset from
        self.url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
        
    def _download(self, dest='.data'):
        # check if destination exists, else create
        self.dest = os.path.join(dest, 'babi')
        if not os.path.exists(os.path.join(self.dest, 'tasks_1-20_v1-2.tar.gz')):
            print('Cannot find dataset in path %s. Downloading dataset...' % dest)
            if not os.path.exists(self.dest):
                os.makedirs(self.dest)
            os.system('wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz -O '+self.dest+'/tasks_1-20_v1-2.tar.gz')
            os.system('tar xzf '+self.dest+'/tasks_1-20_v1-2.tar.gz -C '+self.dest)

        self.dest = os.path.join(self.dest, 'tasks_1-20_v1-2')

    def load(self, verbose=False):
        
        # download dataset if file not present locally
        self._download()
        
        with open(os.path.join(self.dest, 'en-10k/qa1_single-supporting-fact_train.txt'), 'r') as f:
            train_lines = f.readlines()
        with open(os.path.join(self.dest, 'en-10k/qa1_single-supporting-fact_test.txt'), 'r') as f:
            test_lines = f.readlines()

        # parse train & test data into text and labels
        train_text, train_label, dictionary = self.parse_stories(train_lines, dictionary={'_': 0, '.': 1, '?': 2})
        test_text, test_label, _ = self.parse_stories(test_lines, dictionary=None)
        
        self.vocab = list(dictionary.keys())
        self.max_vocab_size = len(dictionary.keys())
        
        # convert text into tensors
        train_text, train_label = self.text_to_tensor(train_text, train_label, self.vocab)
        test_text, test_label = self.text_to_tensor(test_text, test_label, self.vocab)

        # split into train & validation
        self.train_data = TensorDataset(train_text, train_label)
        self.test_data = TensorDataset(test_text, test_label)

        # split training data
        split_ratio = 0.8
        train_size = int(train_text.shape[0] * split_ratio)
        self.train_data, self.valid_data = random_split(self.train_data, [train_size, train_text.shape[0]-train_size])
        
        if verbose:
            print('Training data size:   ', len(self.train_data.indices))
            print('Validation data size: ', len(self.valid_data.indices))
            print('Test data size:       ', len(self.test_data.indices))

    def build_vocab(self, dim=300, max_vocab_size=25000):
        pass

    def create_data_loader(self, batch_size = 64):
        if not hasattr(self, "train_data") or not hasattr(self, "TEXT") or not hasattr(self, "LABEL"):
            self.load()
        if not hasattr(self, "max_vocab_size") or not hasattr(self, "dim"):
            self.build_vocab()
        # creating loader
        ## data loader equivalent in torchtext batch iterator - buckets similar lengths together
        self.batch_size = batch_size

        train_iterator = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        valid_iterator = DataLoader(self.valid_data, batch_size=self.batch_size)
        test_iterator = DataLoader(self.test_data, batch_size=self.batch_size)
        return train_iterator, valid_iterator, test_iterator

    def parse_stories(self, lines, dictionary=None):
        '''
        Parse stories provided in the bAbi tasks format

        Parameters
        ----------
        lines: list of lines read from the bAbi task files
        dictionary: initial dictionary of words in the babi task 
            default=None i.e., it will be created on the go

        Returns
        -------
            a list of stories, labels and dictionary
        '''
        text = []
        label = []
        TEXT = []
        LABEL = []

        # initialize dictionary as counter object
        if dictionary is None:
          dictionary = {}
        dictionary = OrderedDict(dictionary)

        for line in lines:
            nid, line = line.strip().split(' ', 1)
            nid = int(nid)
            if nid == 1:
                text = []
                label = []

            if '\t' in line:
                q, a, supporting = line.split('\t')
                # adding an placeholder for answer
                q = q.replace('? ', ' ?')
                # creating label to hold place for answer
                l = re.sub(r'(\w|\n)+', '_', q)
                l = l.replace('?', a)

                q = q.split(' ')
                l = l.split(' ')
                # update dictionary
                for i in q:
                  dictionary[i] = 1
                for i in l:
                  dictionary[i] = 1
                text.extend(q)
                label.extend(l)
                # append entire story so far to data
                TEXT.append(text.copy())
                LABEL.append(label.copy())

            else:
                line = line.replace('.', ' .')
                s = line.split(' ')
                l = ['_'] * len(s)
                # update dictionary
                for i in s:
                  dictionary[i] = 1
                text.extend(s)
                label.extend(l)

        return TEXT, LABEL, dictionary
      
    def text_to_tensor(self, text, label, vocab):
        # pad to make them all the same lengthMb
        max_len = np.max([len(t) for t in text])
        text = [['_'] * (max_len - len(t)) + t for t in text]
        label = [['_'] * (max_len - len(t)) + t for t in label]

        # convert to numpy array
        text = np.array(text).reshape(-1, 1).astype(object)
        label = np.array(label).reshape(-1, 1).astype(object)
                
        # encode labels 
        onehot = OneHotEncoder(categories=[vocab], handle_unknown='error')
        text = onehot.fit_transform(text).todense()
        label = np.array([[vocab.index(t[0])] for t in label])
        # reshape to original dim
        text = np.array(text).reshape(-1, max_len, len(vocab))
        label = np.array(label).reshape(-1, max_len)
        
        # convert to tensor
        text = torch.from_numpy(text).float()
        label = torch.from_numpy(label).long()
        return text, label