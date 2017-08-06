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
import time, math, os
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
    	#torch.cuda.set_device(2)
    	torch.cuda.manual_seed(args.seed)
###############################################################################
# Load data
###############################################################################

#### fix this
# corpus = data.Corpus(args.data)
corpus = data.Corpus(args)
print('Train set size = ', len(corpus.train_data), len(corpus.train_label))
# print('Development set size = ', len(corpus.dev))
# print('Test set size = ', len(corpus.test))
print('Vocabulary size = ', len(corpus.dictionary))

###############################################################################
# load_emb
###############################################################################
#### fix this
#file_name = 'train_corpus_3' + 'embeddings_index.p'
#file_name = 'PBT.p'
#embeddings_index = util.get_initial_embeddings(file_name, args.data_path, args.word_vectors_directory, args.Glove_filename, corpus.dictionary)
#print('Number of OOV words = ', len(corpus.dictionary) - len(embeddings_index))

###############################################################################
# batchify
###############################################################################
#### fix this

train_data_trimed, train_label_trimed = util.batchify(corpus.train_data, corpus.train_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense
##### fix this
#valid_batches = util.batchify(corpus.valid, args.batch_size, args.cuda)
valid_data_trimed, valid_label_trimed = util.batchify(corpus.valid_data, corpus.valid_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense

#### fix this
#test_batches = util.batchify(corpus.test, args.batch_size, args.cuda) 
test_data_trimed, test_label_trimed = util.batchify(corpus.test_data, corpus.test_label, args.batch_size, args.cuda) #[82915, 20] batch size = 20, it's kinda seq len in gen sense

assert len(train_data_trimed) == len(train_label_trimed)
assert len(valid_data_trimed) == len(valid_label_trimed)
assert len(test_data_trimed) == len(test_label_trimed)
# print (batchify([2,3,4,3,4,355,4,342,90], 2))
print('train_batches: size: ', len(train_data_trimed) ) #, 'seq len: ', len(train_data_trimed[0]), '1st instance: ', train_data_trimed[0][:50], '1st label: ', train_label_trimed[0][:50] )# , train_batches[0][0].sentence1)


# ###############################################################################
# # Build the model
# ###############################################################################

model_f = model_rnd.LanguageModel(corpus.dictionary, args)
model_b = model_rnd.LanguageModel(corpus.dictionary, args)

if args.cuda:
    torch.cuda.set_device(args.gpu)
    model_f.cuda()
    model_b.cuda()

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
    train_data_trimed = train_data_trimed[:50]
    valid_data_trimed = valid_data_trimed[:50]
    test_data_trimed = test_data_trimed[:50]
    #print(len(train_data_trimed))
    #print(train_data_trimed[::-1][-1][::-1])
    #exit()
train = train.Train(model_f, model_b, corpus.dictionary, 'CrossEntropyLoss')
train.train_epochs(train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed)





###############################################################################
# fix this
###############################################################################
#train_data = train_batches
#val_data = valid_batches
#test_data = test_batches
#ntokens = len(corpus.dictionary)
#model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
#if args.cuda:
#    model.cuda()

#criterion = nn.CrossEntropyLoss(size_average=True)

###############################################################################
# testing code
###############################################################################



# Load the best saved model.
#with open(args.save, 'rb') as f:
  #  model = torch.load(f)
if os.path.isfile(os.path.join(args.log_dir, 'forward_model_best.pth.tar')) and os.path.isfile(os.path.join(args.log_dir, 'backward_model_best.pth.tar')):
    print("=> loading  best models for testing")
    checkpoint_forward= torch.load(os.path.join(args.log_dir, 'forward_model_best.pth.tar'))
    checkpoint_backward= torch.load(os.path.join(args.log_dir, 'backward_model_best.pth.tar'))
    args.start_epoch = checkpoint_forward['epoch']
    best_perplexity_forward = checkpoint_forward['perplexity']
    best_perplexity_backward = checkpoint_backward['perplexity']
    model_f.load_state_dict(checkpoint_forward['state_dict'])
    model_b.load_state_dict(checkpoint_backward['state_dict'])

# Run on test data.
test_loss = train.validate(test_data_trimed, test_label_trimed, model_f, model_b, args.nepochs, is_test = True)
print('=' * 89)
print('| End of training | test loss {:.2f} | test ppl {:5.2f} |'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
