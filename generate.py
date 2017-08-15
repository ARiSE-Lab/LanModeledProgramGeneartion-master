###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse, sys, os
import model_rnd, util
import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data_path', type=str, default='./soft_data/',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='50',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
parser.add_argument('--train_data', type=str, default='train.data',
                        help='train corpus path')
parser.add_argument('--valid_data', type=str, default='valid.data',
                        help='valid corpus path')
parser.add_argument('--test_data', type=str, default='test.data',
                        help='test corpus path')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--log_dir', type=str, default='./log_adam')
parser.add_argument('--max_length', type=int, default=200,
                        help='maximum length of a line')
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
parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--lr_decay', type=float, default=.5,
                        help='decay ratio for learning rate')
parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cell', action='store_true',
                        help='use CELL for computation')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

corpus = data.Corpus(args)
forward_model = model_rnd.LanguageModel(corpus.dictionary, args)
backward_model = model_rnd.LanguageModel(corpus.dictionary, args)

if os.path.isfile(os.path.join(args.log_dir, 'forward_model_best.pth.tar')) and os.path.isfile(os.path.join(args.log_dir, 'backward_model_best.pth.tar')):
    print("=> loading  checkpoints")
    checkpoint_forward= torch.load(os.path.join(args.log_dir, 'forward_model_best.pth.tar'))
    checkpoint_backward= torch.load(os.path.join(args.log_dir, 'backward_model_best.pth.tar'))
    
    
    forward_model.load_state_dict(checkpoint_forward['state_dict'])
    backward_model.load_state_dict(checkpoint_backward['state_dict'])
else:
    print("=> no checkpoint found")

if args.cuda:
    forward_model.cuda()
    backward_model.cuda()

forward_model.eval()
backward_model.eval()



ntokens = len(corpus.dictionary)
#hidden = model.init_hidden(args.batch_size)
eval_batch_size = args.batch_size
input_b = Variable(torch.rand(1, 1).mul(ntokens).long().fill_(corpus.dictionary.word2idx['<eos>']), volatile=True)
input_f = Variable(torch.rand(1, 1).mul(ntokens).long().fill_(corpus.dictionary.word2idx['<start>']), volatile=True)
if args.cuda:
    input_f.data = input_f.data.cuda()
    input_b.data = input_b.data.cuda()

output_forward = torch.zeros(args.words - 1,ntokens) # #and assumed only one batch first item is not needed to predict
output_backward = torch.zeros(args.words -1,ntokens) ##assumed only one batch and first item (literraly last item) is not needed to predict


with open(args.outf, 'w') as outf:

    for i in range(args.words-1):
        if i==0: print('strat generation')
        hidden_f = forward_model.init_hidden(eval_batch_size) #for each sentence need to initialize
        hidden_f = util.repackage_hidden(hidden_f, args.cuda)
        hidden_b = backward_model.init_hidden(eval_batch_size) #for each sentence need to initialize
        hidden_b = util.repackage_hidden(hidden_b, args.cuda) 
        output_f, hidden_f = forward_model(input_f, hidden_f)
        output_b, hidden_b = backward_model(input_b, hidden_b)
        output_f = output_f.view(-1, ntokens)
        output_b = output_b.view(-1, ntokens)

        word_weights = output_f.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input_f.data.fill_(word_idx)

        word_weights = output_b.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        input_b.data.fill_(word_idx)
            
        #if i==0: print(output_f)
        output_forward [i] = output_f[0].data #assumed only one batch
        output_backward[i] = output_b[0].data

    idx = torch.range(output_backward.size(0)-1, 0, -1).long()
    #idx = torch.autograd.Variable(idx)
    #if args.cuda:
    #    idx = idx.cuda()
    output_b_flipped = output_backward.index_select(0, idx)
    
    output = output_forward + output_b_flipped
    for i in range(output.size(0)):
        output[i][corpus.dictionary.word2idx['<eos>']] = -1
        output[i][corpus.dictionary.word2idx['<start>']] = -1

    word_weights = [ output_elem.div(args.temperature).exp().cpu() for output_elem in output]
    word_idx = [ torch.multinomial(word_weight, 1)[0] for word_weight in word_weights]
    words = [ corpus.dictionary.idx2word[word_id] for word_id in word_idx]

    outf.write('[<start>] ') 
    for word in words:
        outf.write(word + ('\n' if 'eos' in word else ' '))
    outf.write('[<eos>] ') 
    print('| Generated {}/{} words'.format(i+2, args.words), file = sys.stderr)