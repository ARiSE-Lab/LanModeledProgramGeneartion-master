import json, os, string, random, time, pickle, gc
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pdb

class LSTMData(data.Dataset):

    def __init__(self, args, split = 'train'):
        self.data = pickle.load(open(os.path.join(args.data_path, split+".data.sentences")))
        print('%s set size = %d' % (split, len(self.data)))

    def __getitem__(self, index):
        item = self.data[index]
        context = item[:-1]
        target = item[1:]
        return torch.LongTensor(context), torch.LongTensor([target])

    def __len__(self):
        return len(self.data)