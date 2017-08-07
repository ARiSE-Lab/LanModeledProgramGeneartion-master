import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import pdb
import pickle

class NgramData(data.Dataset):

    def __init__(self, args, split = 'train'):
        self.tokens = {}
        data = pickle.load(open(args.data_path))
        print('%s set size = %d' % (split, len(data)))

        self.trigrams = list()
        for sentence in data:
            if(len(sentence)<=200):
                for word in sentence:
                    self.tokens[word] = 1
                self.trigrams+=[([sentence[i], sentence[i + 1]], sentence[i + 2]) for i in range(len(sentence) - 2)]    
        print(" %s corpus and length of trigrams....: %d" %(split, len (self.trigrams)))


    def __getitem__(self, index):
        item = self.trigrams[index]
        context = item[0]
        target = item[1]
        return torch.LongTensor(context), torch.LongTensor([target])

    def __len__(self):
        return len(self.trigrams)
   
    def num_tokens():
        return len(self.tokens)