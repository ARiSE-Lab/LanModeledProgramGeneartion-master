import json, os, string, random, time, pickle, gc
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pdb

class LSTMData(data.Dataset):

    def __init__(self, corpus, split = 'train'):
        if split == "train":
            self.data = corpus.train
        elif split == "val":
            self.data = corpus.val
        elif split == 'test':
            self.data = corpus.test
        
        print('%s set size = %d' % (split, len(self.data)))

    def __getitem__(self, index):
        item = self.data[index]
        context = item[:-1]
        target = item[1:]
        return torch.LongTensor(context), torch.LongTensor([target])

    def __len__(self):
        return len(self.data)