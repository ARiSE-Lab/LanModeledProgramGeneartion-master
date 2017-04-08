###################################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 3/27/2017
# many codes are adopted from Wasi Ahmad encoder
# File Description: This files encodes the instances by a rnn like 'lstm' , 'gru'
##################################################################################

import util, torch
import torch.nn as nn
from torch.autograd import Variable



class EncoderDropoutRNN(nn.Module):
    """A stacked RNN encoder that encodes a given sequence."""

    def __init__(self, config):
        """"Constructor of the class"""
        super(EncoderDropoutRNN, self).__init__()
        self.n_layers = config.nlayers
        self.em_size = config.emsize
        self.hidden_size = config.nhid
        self.dropout = config.dropout
        self.rnn_type = config.model
        self.bidirectional = config.bidirection
        self.drop = nn.Dropout(self.dropout)

        if self.rnn_type in ['LSTM', 'GRU']:
            if not config.cell:
                print('='*90,"\nWARNING: If you want LSTMCell, so you should probably run with --cell\n", '='*89)
                self.rnn = getattr(nn, self.rnn_type)(self.em_size, self.hidden_size, self.n_layers,
                                                      batch_first=False, dropout=self.dropout,
                                                      bidirectional=self.bidirectional)
                if(not self.bidirectional):
                    print('='*90, "\nWARNING: If you want bidirectional, so you should probably run with --bidirection\n", '='*89)
            else:
                print("WARNING: in encoder!!! Only encodes no dropout!!!!!!!!!!")
                self.rnn = getattr(nn, self.rnn_type + 'Cell')(self.em_size, self.hidden_size)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNNCell(self.em_size, self.hidden_size, nonlinearity=nonlinearity)

    def forward(self, input_variable, hidden):
        #Assumes that receves Variables
        output, hidden = self.rnn(input_variable, hidden)
        output = self.drop(output)
        return output, hidden

    def init_weights(self, bsz):
        """Initialize weight parameters for the encoder."""
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_()), Variable(
                weight.new(self.n_layers, bsz, self.hidden_size).zero_())
        else:
            return Variable(weight.new(self.n_layers, bsz, self.hidden_size).zero_())
