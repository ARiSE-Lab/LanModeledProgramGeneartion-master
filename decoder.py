###################################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 3/27/2017
# File Description: This files decodes the encoder hidden states
# to a dictionary  word
##################################################################################

import util, torch
import torch.nn as nn
from torch.autograd import Variable



class DecoderLinear(nn.Module):
    """A stacked RNN encoder that encodes a given sequence."""

    def __init__(self, nhid, vocab_size):
        """"Constructor of the class"""
        super(DecoderLinear, self).__init__()
        self.decoder = nn.Linear(nhid, vocab_size)

    def init_weights(self, initrange):
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, output):
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

