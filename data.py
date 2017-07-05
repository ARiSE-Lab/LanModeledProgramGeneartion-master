###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/1/2017
# Many codes are from Wasi Ahmad data.py
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################
from nltk.tokenize import word_tokenize
import numpy as np
import json, os, torch
import util



class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Create and store three special tokens
        self.start_token = '<SOS>'
        self.end_token = '<EOS>'
        self.unknown_token = '<UNKNOWN>'
        self.pad_token = '<PAD>'
        self.idx2word.append(self.start_token)
        self.word2idx[self.start_token] = len(self.idx2word) - 1
        self.idx2word.append(self.end_token)
        self.word2idx[self.end_token] = len(self.idx2word) - 1
        self.idx2word.append(self.unknown_token)
        self.word2idx[self.unknown_token] = len(self.idx2word) - 1
        self.idx2word.append(self.pad_token)
        self.word2idx[self.pad_token] = len(self.idx2word) - 1

    def add_word(self, word):
        #word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def contains(self, word):
        word = word.lower()
        return True if word in self.word2idx else False

    def __len__(self):
        return len(self.idx2word)


class Instance(object):
    def __init__(self):
        self.sentence1 = []
        self.target = []

    def add_sentence(self, sentence, dictionary, is_test_instance=False):
#### fix this
        words = [dictionary.start_token] + word_tokenize(util.sepearte_operator(sentence.lower())) + [dictionary.end_token]
        #removes <, >
        words.pop(1)
        words.pop(2)
        words.pop(len(words)-2)
        words.pop(len(words)-3)

        if is_test_instance:
            for i in range(len(words)):
                if dictionary.contains(words[i].lower()) == False:
                    words[i] = dictionary.unknown_token
        else:
            for word in words:
                dictionary.add_word(word.lower())

        self.sentence1 = words[:-1]
        self.target = words[1:]


class Corpus(object):
    def __init__(self, args):
        path = args.data_path
        self.dictionary = Dictionary()
        self.max_sent_length = 0
#### fix this
        self.train = self.parse(os.path.join(path, args.train_data))
        self.valid = self.parse(os.path.join(path, args.valid_data))
        self.test = self.parse(os.path.join(path, args.test_data), True)

    def parse(self, path, is_test_instance=False):
        """Parses the content of a file."""
        assert os.path.exists(path)

        samples = []
        with open(path, 'r') as f:
            for line in f:
                instance = Instance()
                if is_test_instance:
                    instance.add_sentence(line, self.dictionary, is_test_instance)
                else:
                    instance.add_sentence(line, self.dictionary)
                    if self.max_sent_length < len(instance.sentence1):
                        self.max_sent_length = len(instance.sentence1)
                samples.append(instance)
        return samples


    ### from example
    def __init2__(self, args):
        path = args.data_path
        self.dictionary = Dictionary()
        self.max_sent_length = 0
        self.dictionary = Dictionary()
        self.train_c = self.tokenize(os.path.join(path, 'trainS.txt'))
        self.valid_c = self.tokenize(os.path.join(path, 'trainS.txt'))
        self.test_c = self.tokenize(os.path.join(path, 'trainS.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = ['<start>'] + line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = ['<start>'] + line.split() + ['<eos>']
                sentence_len = len(words)
                if(self.max_sent_length<sentence_len): self.max_sent_length = sentence_len
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
