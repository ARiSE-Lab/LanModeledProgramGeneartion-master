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
