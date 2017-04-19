from nltk.tokenize import word_tokenize
import numpy as np
import json, os, torch
import util

"""I feel it's better to do data preprocessing on Jupyter"""

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        #word = word.lower() 
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, data_path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(data_path, 'train.data'))
        self.val = self.tokenize(os.path.join(data_path, 'val.data'))
        self.test = self.tokenize(os.path.join(data_path, 'test.data'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<SOS>'] + line.split() + ['<EOS>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        self.dictionary.add_word('<PAD>')

        with open(path, 'r') as f:
            ids = list()
            maxSequenceLength = 0
            for line in f:
                words = ['<SOS>'] + line.split() + ['<EOS>']
                id_sen = list()
                token = 0
                for word in words:
                    id_sen.append(self.dictionary.word2idx[word])
                    token += 1
                if token > maxSequenceLength: maxSequenceLength = token
                ids.append(id_sen)

        for i in xrange(len(ids)):
            sentence = ids[i]
            if len(sentence) < maxSequenceLength:
                for j in xrange(maxSequenceLength - len(sentence)):
                    sentence.append(self.dictionary.word2idx['<PAD>'])
            ids[i] = sentence
        return ids
