from argparse import ArgumentParser
from numpy.linalg import norm
from nltk.tokenize import word_tokenize
import pickle
import sys
import numpy as np

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
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=.95,
                        help='decay ratio for learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    #### fix this

    parser.add_argument('--epochs', type=int, default=1,
                        help='upper limit of epoch')
    parser.add_argument('--train_data', type=str, default='train_corpus.txt',
                        help='train corpus path')
    #### fix this

    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='batch size')
    parser.add_argument('-- bptt', type=int, default=10,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.25,
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
    parser.add_argument('--gpu', type=int, default=1,
                        help='number of gpu can be used for computation')
    parser.add_argument('--print_every', type=int, default=2000, metavar='N',
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=2000,
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
    parser.add_argument('--word_vectors_directory', type=str, default='../data/glove/',
                        help='Path of GloVe word embeddings')
    parser.add_argument('--data_path', default='./')

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





def get_initial_embeddings(file_name, directory, file, dic):
    if (os.path.isfile(file_name)):
        print('========================== loading input matrix', file = sys.stderr)
        embeddings_index = pickle.load(open(file_name, 'rb'))
        print('========================== loading complete', file = sys.stderr)
        else:
        print('========================== no cached file!!! starting to generate now', file = sys.stderr)
        embeddings_index = load_word_embeddings(directory, file, corpus.dictionary)
        print('========================== Generation comple dumping now', file = sys.stderr)
        save_object(embeddings_index, file_name)
        print('========================== Saved dictionary completed!!!', file = sys.stderr)
        return embeddings_index

    def batchify(data, bsz, cuda_true=True):
        nbatch = len(data) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        #     data = data[0:nbatch * bsz]
        # Evenly divide the data across the bsz batches.
        #     print (bsz)
        #     batched_data = [[data[bsz * i + j] for j in range(bsz)] for i in range(nbatch)]
        batched_data = [data[bsz * i: bsz * (i + 1)] for i in range(nbatch)]
        if (bsz * nbatch != len(data)): batched_data.append(data[bsz * nbatch:])
        #     print (batched_data)
        if cuda_true: batched_data = batched_data.cuda()
        return batched_data  # num_batch x batch_size x instance
