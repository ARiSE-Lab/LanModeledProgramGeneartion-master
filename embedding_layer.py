
###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 3/27/2017
# Codes are adopted from WAsi Ahmad nn_layer.py
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

import  torch
import torch.nn as nn
import numpy as np
import util

class Embedding_Drop_Layer(nn.Module):
    """Embedding class which includes only an embedding layer."""

    def __init__(self, vocab_size, embedding_dim, dropout=0.25):
        """"Constructor of the class"""
        super(Embedding_Drop_Layer, self).__init__()
        ## transform index within vovab_size to emb_dim vec
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)


    def forward(self, input_variable):
        """"Defines the forward computation of the embedding layer."""
        embedded = self.embedding(input_variable)
        embedded = self.drop(embedded)
        return embedded

    def init_embedding_weights(self, dictionary, embeddings_index, embedding_dim):
        """Initialize weight parameters for the embedding layer."""
        pretrained_weight = np.empty([len(dictionary), embedding_dim], dtype=float)
        for i in range(len(dictionary)):
            if dictionary.idx2word[i] in embeddings_index:
                pretrained_weight[i] = embeddings_index[dictionary.idx2word[i]]
            else:
                pretrained_weight[i] = util.initialize_out_of_vocab_words(embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))

    def init_weight(self, initrange):
        self.embedding.weight.data.uniform_(-initrange, initrange)