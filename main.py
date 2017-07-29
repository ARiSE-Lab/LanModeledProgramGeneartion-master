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
import model_rnd
import time, math
# from encoder import EncoderRNN
from embedding_layer import Embedding_Drop_Layer
from torch.autograd import Variable
import train
args = util.get_args()
# args = get_args()
# Set the random seed manually for reproducibility.

print ('='*90, '\nWARNING:::: please fix nepochs, dictionary lower case, batchify, data path, batch size, trim_data for non divisible by batch size,  Glove embedding initialization (model_rnd.py 45), pickle_file_name!!!!!\n', '='*89)
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
file_name = 'PBT.p'
#embeddings_index = util.get_initial_embeddings(file_name, args.data_path, args.word_vectors_directory, args.Glove_filename, corpus.dictionary)
#print('Number of OOV words = ', len(corpus.dictionary) - len(embeddings_index))

###############################################################################
# batchify
###############################################################################
#### fix this
train_batches = util.batchify(corpus.train_c, args.batch_size, args.cuda)
##### fix this
valid_batches = util.batchify(corpus.valid_c, args.batch_size, args.cuda)
#### fix this
test_batches = util.batchify(corpus.test_c, args.batch_size, args.cuda)
# print (batchify([2,3,4,3,4,355,4,342,90], 2))
print('train_batches: ', train_batches.size(), ' valid batches: ', valid_batches.size(), ' test batches: ', test_batches.size())

# ###############################################################################
# # Build the model
# ###############################################################################

model = model_rnd.LanguageModel(corpus.dictionary, args)

if args.cuda:
    torch.cuda.set_device(args.gpu)
    model.cuda()

# ###############################################################################
# # Dummy use the model
####fix this
# ###############################################################################

# list = [[4,14], [14,4]]
# l_t = torch.LongTensor(list)
# # list_var = Variable(l_t)
# # print('========================== before calling model forward', file = sys.stderr)
# print(model(l_t, l_t)[0][1])

# ###############################################################################
# # Train the model
# ###############################################################################
## loss: CrossEntropyLoss :: Combines LogSoftMax and NLLoss in one single class
if args.debug_mode:
    train_batches = train_batches[:500]
train = train.Train(model, corpus.dictionary, 'CrossEntropyLoss')
train.train_epochs(train_batches, valid_batches)





###############################################################################
# fix this
###############################################################################
train_data = train_batches
val_data = valid_batches
test_data = test_batches
#ntokens = len(corpus.dictionary)
#model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
#if args.cuda:
#    model.cuda()

criterion = nn.CrossEntropyLoss()

###############################################################################
# testing code
###############################################################################



# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = util.evaluate(test_data, model, corpus.dictionary, args.bptt,  criterion)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
