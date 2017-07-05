import util, data #, helper, train
import torch, random
from torch import optim
import torch.nn as nn
# from encoder import EncoderRNN
from embedding_layer import Embedding_Drop_Layer
from torch.autograd import Variable
import sys
args = util.get_args()

###############################################################################
# Author: Md Rizwan Parvez
# Project: Quora Duplicate Question Detection
# Date Created: 3/27/2017
# many codes are adopted from Wasi Ahmad QuestionClassifier
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################
a = torch.LongTensor([[1,23,45,0,0,0], [23,23,23,23,2,3]])
print(a)
print(a.view(-1, 6))



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



#### fix this
# from nn_layer import EmbeddingLayer
# from encoder import EncoderRNN


class LanguageModel(nn.Module):
    def __init__(self, dictionary, embeddings_index, args):
        """"Constructor of the class."""
        super(LanguageModel, self).__init__()
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = args

        #### fix this
        #         self.num_directions = 2 if args.bidirection else 1

        self.embedding = Embedding_Drop_Layer(len(dictionary), self.config.emsize, self.config.dropout)
        #self.embedding = Embedding_Drop_Layer(len(dictionary), 300, 0.25)
        #         self.forward_encoder = EncoderRNN(args)
        #         if self.num_directions == 2:
        #             self.backward_encoder = EncoderRNN(args)

        #         self.dropout = nn.Dropout(args.dropout)
        #         self.relu = nn.ReLU()
        #         self.linear = nn.Linear(args.nhid * 2, 2)

        #         self.out = nn.Sequential(
        #             self.relu,
        #             self.dropout,
        #             self.relu,
        #             self.dropout,
        #             self.relu,
        #             self.dropout,
        #             self.linear)

        # Initializing the weight parameters for the embedding layer and the encoder.

        #### fix this
        #         self.embedding.init_embedding_weights(self.dictionary, self.embeddings_index, self.config.emsize)
        self.embedding.init_embedding_weights(self.dictionary, self.embeddings_index, 300)

    def forward(self, batch_sentence1):
        """"Defines the forward computation of the question classifier."""
        batch_variable = Variable(batch_sentence1)
        #### fix this
        #### make everything cuda
        if (self.config.cuda == True): batch_variable = batch_variable.cuda()
        return self.embedding(batch_variable)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
###############################################################################
# Load data
###############################################################################

#### fix this
# corpus = data.Corpus(args.data)
corpus = data.Corpus(args)
print('Train set size = ', len(corpus.train))
print('Development set size = ', len(corpus.dev))
# print('Test set size = ', len(corpus.test))
print('Vocabulary size = ', len(corpus.dictionary))

###############################################################################
# load_emb
###############################################################################
#### fix this
file_name = 'train_corpus_3' + 'embeddings_index.p'
embeddings_index = util.get_initial_embeddings(file_name, '/if1/kc2wc/data/glove/', 'glove.6B.300d_w_header.txt',corpus.dictionary)
print('Number of OOV words = ', len(corpus.dictionary) - len(embeddings_index))

###############################################################################
# batchify
###############################################################################
#### fix this
train_batches = util.batchify(corpus.train, args.batch_size)
# #### fix this
dev_batches = util.batchify(corpus.dev, args.batch_size)
# print (batchify([2,3,4,3,4,355,4,342,90], 2))
print('num_batches: ', len(train_batches))
print(len(train_batches[0]), train_batches[0][0].sentence1)
# ###############################################################################
# # Build the model
# ###############################################################################

model = LanguageModel(corpus.dictionary, embeddings_index, args)
model.cuda()
# print('==========================just after model initialization')
list = [[4,14], [14,4]]
l_t = torch.LongTensor(list)
# list_var = Variable(l_t)
# print('========================== before calling model forward', file = sys.stderr)
print(model(l_t)[0][1])

# model = LanguageModel(corpus.dictionary, embeddings_index, args)
# model.cuda()
# list = [[4,14], [14,4]]
# l_t = torch.LongTensor(list)
# # list_var = Variable(l_t)
# print(model(l_t)[0][1])
