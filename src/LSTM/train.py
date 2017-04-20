import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import data
import model
import data_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--save', type=str,  default='model.pt', help='path to save the final model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    corpus = data.Corpus(args.data_path)
    ntokens = len(corpus.dictionary)
    print('Vocabulary size = ', ntokens)

    train_data = data_loader.LSTMData(corpus, 'train')
    val_data = data_loader.LSTMData(corpus, 'val')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
        pin_memory = True, num_workers = 6)
    val_loader = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False,
        pin_memory = False, num_workers = 4)

    
    model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay = 1e-4)

    best_val_loss = None

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(args, epoch, model, criterion, train_loader, optimizer, ntokens)
        val_loss = evaluate(args, epoch, model, criterion, val_loader, optimizer, ntokens)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        
        # if not best_val_loss or val_loss < best_val_loss:
        #     with open(args.save, 'wb') as f:
        #         torch.save(model, f)
        #     best_val_loss = val_loss


def repackage_hidden(h): # I guess think function should only be used in a whole corpus
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def train(args, epoch, model, criterion, train_loader, optimizer, ntokens):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        hidden = model.init_hidden(args.batch_size)
        optimizer.zero_grad() #i think this is equal to model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), target)
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        total_loss += loss.data

        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:.0f}% batches | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, 100. * batch_idx / len(train_loader), lr,
                elapsed * 1000, loss.data[0], math.exp(loss.data[0])))

def evaluate(args, epoch, model, criterion, val_loader, optimizer, ntokens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data.cuda()), Variable(target.cuda())
        hidden = model.init_hidden(args.batch_size)
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), target)

        total_loss += len(data) * loss.data

        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:.0f}% batches| lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, 100. * batch_idx / len(val_loader), lr,
                elapsed * 1000, loss.data[0], math.exp(loss.data[0])))

        return total_loss[0]/len(val_loader)

if __name__=='__main__':
    main()
