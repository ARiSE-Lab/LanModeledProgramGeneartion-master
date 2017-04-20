import argparse
import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser()

# Model parameters.
parser.add_argument('--data_path', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
parser.add_argument('--sentences', type=int, default='10', help='number of sentences to generate')
parser.add_argument('--sentence_len', type=int, default='10', help='length of sentences to generate')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

model.eval()
model.cuda()

corpus = data.Corpus(args.data_path)
ntokens = len(corpus.dictionary)

with open(args.outf, 'w') as outf:
    for i in xrange(args.sentences):
        hidden = model.init_hidden(1)
        input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True) # only one word
        input.data = input.data.cuda()
        for j in xrange(sentence_len):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.data.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

        outf.write("\n")