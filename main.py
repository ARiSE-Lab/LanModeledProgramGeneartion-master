###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/1/2017
# Some codes are from Wasi Ahmad main.py
# File Description: This script provides a definition of the corpus, each
# example in the corpus and the dictionary.
###############################################################################

import util, data #, helper, train
import torch, random
from torch import optim
import model
# from encoder import EncoderRNN
from embedding_layer import Embedding_Drop_Layer
from torch.autograd import Variable

args = util.get_args()
# args = get_args()
# Set the random seed manually for reproducibility.


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
# print('Development set size = ', len(corpus.dev))
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

# ###############################################################################
# # Build the model
# ###############################################################################

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = model.LanguageModel(corpus.dictionary, embeddings_index, args)

if args.cuda:
    torch.cuda.set_device(args.gpu)
    model.cuda()

list = [[4,14], [14,4]]
l_t = torch.LongTensor(list)
# list_var = Variable(l_t)
# print('========================== before calling model forward', file = sys.stderr)
print(model(l_t, l_t)[0][1])
# ###############################################################################
# # Train the model
# ###############################################################################

# train = train.Train(question_classifier, dictionary, embeddings_index)
# train.train_epochs(train_batches, dev_batches, 1)
