###############################################################################
# Author: Md Rizwan Parvez
# Project: Quora Duplicate Question Detection
# Date Created: 3/27/2017
# many codes are adopted from Wasi Ahmad QuestionClassifier
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

import torch, helper
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import util
#### fix this
from embedding_layer import Embedding_Drop_Layer
# from encoder import EncoderRNN


class LanguageModel(nn.Module):

    def __init__(self, dictionary, embeddings_index, args):
        """"Constructor of the class."""
        super(LanguageModel, self).__init__()
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = args



        self.embedding_drop = Embedding_Drop_Layer(len(self.dictionary), self.config.emsize, self.config.dropout)

        # if rnn_type in ['LSTM', 'GRU']:
        #     self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        # else:
        #     try:
        #         nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
        #     except KeyError:
        #         raise ValueError( """An invalid option for `--model` was supplied,
        #                          options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
        #     self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        # self.decoder = nn.Linear(nhid, ntoken)
        #
        # # Optionally tie weights as in:
        # # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # # https://arxiv.org/abs/1608.05859
        # # and
        # # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # # https://arxiv.org/abs/1611.01462
        # if tie_weights:
        #     if nhid != ninp:
        #         raise ValueError('When using the tied flag, nhid must be equal to emsize')
        #     self.decoder.weight = self.encoder.weight

        self.init_weights()

        # self.rnn_type = rnn_type
        # self.nhid = nhid
        # self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
        self.embedding_drop.init_embedding_weights(self.dictionary, self.embeddings_index, self.config.emsize)
        # self.decoder.bias.data.fill_(0)
        # self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        ## accepts tensor or variable and make it variable
        input = util.getVariable(input)
        hidden = util.getVariable(hidden)

        if(self.config.cuda):
            input = input.cuda()
            hidden = hidden.cuda()

        emb_drop = self.embedding_drop(input)

        return emb_drop
        # output, hidden = self.rnn(emb_drop, hidden)
        # output = self.drop(output)
        # decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        # return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     if self.rnn_type == 'LSTM':
    #         return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
    #                 Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
    #     else:
    #         return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
