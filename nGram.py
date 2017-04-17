###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/1/2017
# Some codes are from Wasi Ahmad main.py
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################
import torch.nn as nn
import util, data #, helper, train
import torch, random
from torch import optim
# import model_rnd
import torch.nn.functional as F
import time, math
# from encoder import EncoderRNN
# from embedding_layer import Embedding_Drop_Layer
from torch.autograd import Variable
import train
args = util.get_args()
# args = get_args()
# Set the random seed manually for reproducibility.

print ('='*90, '\nWARNING:::: please fix  pickle_file_name, nepochs, dictionary lower case, batchify, data path, batch size, trim_data for non divisible by batch size,  Glove embedding initialization (model_rnd.py 45)!!!!!\n', '='*89)
print ('='*90, '\nWARNING:::: if you have insatnce based lstm you need to init model for each batch!!!!!\n', '='*89)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print('='*90, "\nWARNING: You have a CUDA device, so you should probably run with --cuda\n", '='*89)
    else:
        torch.cuda.manual_seed(args.seed)
###############################################################################
# Load data
###############################################################################

#### fix this
# corpus = data.Corpus(args.data)
corpus = data.Corpus(args)
print('Train set size = ', len(corpus.train_c))
# print('Development set size = ', len(corpus.dev))
# print('Test set size = ', len(corpus.test))
print('Vocabulary size = ', len(corpus.dictionary))

###############################################################################
# load_emb
###############################################################################
#### fix this
#file_name = 'train_corpus_3' + 'embeddings_index.p'
file_name = 'soft_data.p'
embeddings_index = util.get_initial_embeddings(file_name, args.data_path, args.word_vectors_directory, args.Glove_filename, corpus.dictionary)
#print('Number of OOV words = ', len(corpus.dictionary) - len(embeddings_index))

###############################################################################
# batchify
###############################################################################
#### fix this
# train_batches = util.batchify(corpus.train, args.batch_size, args.cuda)


trigrams = [([corpus.train_c[i], corpus.train_c[i + 1]], corpus.train_c[i + 2])
            for i in range(len(corpus.train_c) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = len(corpus.dictionary)
word_to_ix = corpus.dictionary.word2idx
EMBEDDING_DIM = 300
CONTEXT_SIZE = 2
class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler((vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.001)
print(len (trigrams))
for epoch in range(10):
    total_loss = torch.Tensor([0])
    c= 0;
    for context, target in trigrams:
        c+=1;
        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        # context_idxs = [word_to_ix[w] for w in context]
        context_idxs = context
        context_var = Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, Variable(
            torch.LongTensor([target])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
        if(c/5000==0):print('epoch: ', epoch, ' step: ', c,  " loss: ", loss.data, ' total so far: ', total_loss)
    losses.append(total_loss)
    print('epoch: ', epoch, " loss: ", total_loss)
print(losses)  # The loss decreased every iteration over the training data!