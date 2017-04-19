import json, os, string, random, time, pickle, gc
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pdb

class NgramData(data.Dataset):

    def __init__(self, corpus, split = 'train'):
        if split == "train":
            data = corpus.train
        elif split == "val":
            data = corpus.val
        elif split == 'test':
            data = corpus.test
        
        print('%s set size = %d' % (split, len(data)))
    

        self.trigrams = [([data[i], data[i + 1]], data[i + 2]) for i in range(len(data) - 2)]
        print(" %s corpus and length of trigrams....: %d" %(split, len (self.trigrams)))

    def __getitem__(self, index):
        item = self.trigrams[index]
        context = item[0]
        target = item[1]
        return torch.LongTensor(context), torch.LongTensor([target])

    def __len__(self):
        return len(self.trigrams)