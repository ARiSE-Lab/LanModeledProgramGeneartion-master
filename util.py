###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/1/2017
# Some codes are from Wasi Ahmad util.py
# File Description: This is the files where the args are parsed and
#all the necessary methods are
###############################################################################

from argparse import ArgumentParser
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
import pickle
from torch.autograd import Variable
import sys, os, time, math
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_args():
    parser = ArgumentParser(description='attend_analyze_aggregate_nli')
    parser.add_argument('--data', type=str, default='../data/snli_1.0/',
                        help='location of the training data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirectional recurrent unit')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=300,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=.25,
                        help='decay ratio for learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
#### fix this
    parser.add_argument('--instance', type=int, default = 0, help='sentence based model (1) or not (0)')
#### fix this
    parser.add_argument('--epochs', type=int, default=1,
                        help='upper limit of epoch')
#### fix this
    parser.add_argument('--train_data', type=str, default='train_corpus.txt',
                        help='train corpus path')
    parser.add_argument('--valid_data', type=str, default='train_corpus.txt',
                        help='valid corpus path')
    parser.add_argument('--test_data', type=str, default='train_corpus.txt',
                        help='test corpus path')
#### fix this
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--sos_token', type=int, default=0,
                        help='index of the start of a sentence token')
    parser.add_argument('--eos_token', type=int, default=1,
                        help='index of the end of a sentence token')
    parser.add_argument('--max_length', type=int, default=10,
                        help='maximum length of a query')
    parser.add_argument('--min_length', type=int, default=3,
                        help='minimum length of a query')
    parser.add_argument('--teacher_forcing_ratio', type=int, default=1.0,
                        help='use the real target outputs as each next input, instead of using '
                             'the decoder\'s guess as the next input')
    parser.add_argument('--reverse_seq', type=bool, default=False,
                        help='allow reverse sequence for seq2seq model')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--cell', action='store_true',
                        help='use CELL for computation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='number of gpu can be used for computation')
    parser.add_argument('--print_every', type=int, default=200, metavar='N',
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=200,
                        help='plotting interval')
    parser.add_argument('--dev_every', type=int, default=500,
                        help='development report interval')
    parser.add_argument('--save_every', type=int, default=500,
                        help='saving model interval')
    parser.add_argument('--resume_snapshot', action='store_true',
                        help='resume previous execution')
    parser.add_argument('--save_path', type=str, default='../output/',
                        help='path to save the final model')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding version')
    parser.add_argument('--word_vectors_directory', type=str, default='/if1/kc2wc/data/glove/',
                        help='Path of GloVe word embeddings directory')
    parser.add_argument('--Glove_filename', type=str, default='glove.6B.300d_w_header.txt',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--data_path', default='./soft_data/')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    args = parser.parse_args()
    return args

def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)

def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

def initialize_out_of_vocab_words(dimension):
    """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
    return np.random.normal(size=dimension)

def sepearte_operator(x):
    x = x.replace('++', ' ++')
    x = x.replace('--', ' --')
    return x

def repackage_hidden(h, cuda):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        var = Variable(h.data)
        if(cuda): var = var.cuda()
        return var
    else:
        return tuple(repackage_hidden(v, cuda) for v in h)

def getVariable(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return h
    else:
        return  Variable(h)

def normalize_word_embedding(v):
    return np.array(v) / norm(np.array(v))

def load_word_embeddings(directory, file, dic):
    #     print (os.path.join(directory, file))
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        try:
            values = line.split()
            word = values[0]
            #### fix this
            if (word in dic.word2idx):
                embeddings_index[word] = normalize_word_embedding([float(x) for x in values[1:]])
        except ValueError as e:
            print(e)
    f.close()
    return embeddings_index


def get_initial_embeddings(file_name, path, directory, file, dic):
    file_name= os.path.join(path,file_name)
    if (os.path.isfile(file_name)):
        # print('========================== loading input glove matrix for corpus dictionary', file = sys.stderr)
        embeddings_index = pickle.load(open(file_name, 'rb'))
        # print('========================== loading complete', file = sys.stderr)
    else:
        # print('========================== no cached file!!! starting to generate now', file = sys.stderr)
        embeddings_index = load_word_embeddings(directory, file, dic)
        # print('========================== Generation comple dumping now', file = sys.stderr)
        save_object(embeddings_index, file_name)
        # print('========================== Saved dictionary completed!!!', file = sys.stderr)
    return embeddings_index



def save_model_states(model, loss, epoch, tag):
    """Save a deep learning network's states in a file."""
    snapshot_prefix = os.path.join(args.save_path, tag)
    snapshot_path = snapshot_prefix + '_loss_{:.6f}_epoch_{}_model.pt'.format(loss, epoch)
    with open(snapshot_path, 'wb') as f:
        torch.save(model.state_dict(), f)


def load_model_states(model, filename):
    """Load a previously saved model states."""
    filepath = os.path.join(args.save_path, filename)
    with open(filepath, 'rb') as f:
        model.load_state_dict(torch.load(f))

def sentence_to_tensor(sentence, max_sent_length, dictionary):
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sentence)):
        word = sentence[i]
        if word in dictionary.word2idx:
            sen_rep[i] = dictionary.word2idx[word]
        else:
            sen_rep[i] = dictionary.word2idx[dictionary.unknown_token]
    return sen_rep


def instances_to_tensors(instances, dictionary, num_sentences=1):
    """Convert a list of sequences to a list of tensors."""
    max_sent_length = 0
    for item in instances:
        if max_sent_length < len(item.sentence1):
            max_sent_length = len(item.sentence1)
        if(num_sentences==2):
            if max_sent_length < len(item.sentence2):
                max_sent_length = len(item.sentence2)

    all_sentences1 = torch.LongTensor(len(instances), max_sent_length)
    all_sentences2 = torch.LongTensor(len(instances), max_sent_length)
    labels = torch.LongTensor(len(instances))
    for i in range(len(instances)):
        all_sentences1[i] = sentence_to_tensor(instances[i].sentence1, max_sent_length, dictionary)
        all_sentences2[i] = sentence_to_tensor(instances[i].sentence2, max_sent_length, dictionary)
        labels[i] = instances[i].label
    return Variable(all_sentences1), Variable(all_sentences2), Variable(labels)


def save_plot(points, filename):
    """Generate and save the plot."""
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(filename)
    plt.close(fig)  # close the figure


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))

def batchify(data, bsz, cuda):
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    #data = data[0:nbatch * bsz, cuda]
    data = data.narrow(0, 0, nbatch * bsz)

    #batched_data = [data[bsz * i: bsz * (i + 1)] for i in range(nbatch)]
    #if (bsz * nbatch != len(data)): batched_data.append(data[bsz * nbatch:])
    #     print (batched_data)
    #return batched_data  # num_batch x batch_size x instance


    batched_data = data.view(bsz, -1).t().contiguous()
    if cuda:
        batched_data = batched_data.cuda()
    return batched_data

def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source, model, dictionary, bptt, criterion):
    # Turn on evaluation mode which disables dropout.
    args = get_args()
    model.eval()
    total_loss = 0
    ntokens = len(dictionary)
    eval_batch_size = args.batch_size #// 2
    hidden = model.init_hidden(eval_batch_size)
    hidden = repackage_hidden(hidden, args.cuda)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden, args.cuda)
    return total_loss[0] / len(data_source)
