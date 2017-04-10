###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
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
from embedding_layer import Embedding_Drop_Layer
from encoder import EncoderDropoutRNN
from decoder import DecoderLinear

class LanguageModel(nn.Module):

    def __init__(self, dictionary, embeddings_index, args):
        """"Constructor of the class."""
        super(LanguageModel, self).__init__()
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = args
        self.vocab_size = len(self.dictionary)

        self.embedding_drop = Embedding_Drop_Layer(self.vocab_size, self.config.emsize, self.config.dropout)
        self.encoder_drop = EncoderDropoutRNN(args)
        self.decoder = DecoderLinear(self.config.nhid, self.vocab_size)

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
        #self.embedding_drop.init_embedding_weights(self.dictionary, self.embeddings_index, self.config.emsize) ## it initializes with glove embeddings
        self.embedding_drop.init_weight(initrange)#don't know why for encoder only need to init hidden n cell states not weights
        self.decoder.init_weights(initrange)


    def forward(self, input, hidden):
        ## accepts tensor or variable and make it variable
        #input = util.getVariable(input)
        #hidden = util.getVariable(hidden)

        input = util.repackage_hidden(input, self.config.cuda)
        hidden = util.repackage_hidden(hidden, self.config.cuda)

        emb_drop = self.embedding_drop(input)
        output, hidden = self.encoder_drop(emb_drop, hidden)
        decoded = self.decoder(output) #process evrything and returns in batch x seq_len x vocab_size or seq x bsz x vocab
        return decoded, hidden

    def init_hidden(self, bsz):
        return self.encoder_drop.init_weights(bsz)
        # weight = next(self.parameters()).data
        # if self.rnn_type == 'LSTM':
        #     return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
        #             Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        # else:
        #     return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
