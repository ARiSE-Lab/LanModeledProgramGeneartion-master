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

args = util.get_args()
# args = get_args()
# Set the random seed manually for reproducibility.

print ('='*90, '\nWARNING:::: please fix nepochs, dictionary lower case, batchify, data path, batch size, trim_data for non divisible by batch size,  Glove embedding initialization (model_rnd.py 45), pickle_file_name!!!!!\n', '='*89)

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
embeddings_index = util.get_initial_embeddings(file_name, args.data_path, args.word_vectors_directory, args.Glove_filename, corpus.dictionary)
print('Number of OOV words = ', len(corpus.dictionary) - len(embeddings_index))

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
print('train_batches: ', train_batches.size())

# ###############################################################################
# # Build the model
# ###############################################################################

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
else:
    model = model_rnd.LanguageModel(corpus.dictionary, embeddings_index, args)

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
#train = train.Train(model, corpus.dictionary, embeddings_index, 'CrossEntropyLoss')
#train.train_epochs(train_batches, valid_batches, 1)





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
# Training code
###############################################################################







def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = util.get_batch(train_data, i, args.bptt)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = util.repackage_hidden(hidden, args.cuda)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = util.evaluate(val_data, model, corpus.dictionary)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

# Run on test data.
test_loss = util.evaluate(test_data, model, corpus.dictionary)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
