import RNN_model
import data_loader
import torch.nn as nn
import data 
import torch, random
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys, math, pdb, string, shutil, pickle, time, pickle
import numpy as np
import argparse
from torch.utils.data import DataLoader
import pdb
import shutil

ntokens = 18540
maxSequenceLength = 50
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=512, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--nepochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='batch size')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', type=int, default=1)


    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.log_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.log_dir))
        return
    if not os.path.exists(args.log_dir):os.makedirs(args.log_dir)

    train_data = data_loader.LSTMData(args, 'train')
    val_data = data_loader.LSTMData(args, 'val')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
        pin_memory = True, num_workers = 6)
    val_loader = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False,
        pin_memory = False, num_workers = 4)

    
    model = RNN_model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
    model = model.cuda()
    print('Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    criterion = nn.CrossEntropyLoss(size_average=False).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum = 0.9, weight_decay = 1e-4)

    best_perplexity = 100
    if args.resume:
        trainF = open(os.path.join(args.log_dir, 'train.csv'), 'a')
        testF = open(os.path.join(args.log_dir, 'test.csv'), 'a')
        if os.path.isfile(os.path.join(args.log_dir, 'checkpoint.pth.tar')):
            print("=> loading checkpoint")
            checkpoint = torch.load(os.path.join(args.log_dir, 'checkpoint.pth.tar'))
            args.start_epoch = checkpoint['epoch']
            best_perplexity = checkpoint['perplexity']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")
    else:
        trainF = open(os.path.join(args.log_dir, 'train.csv'), 'w')
        testF = open(os.path.join(args.log_dir, 'test.csv'), 'w')

    print("===start training===")
    for epoch in range(args.start_epoch, args.nepochs + 1):
        train(args, epoch, model, criterion, train_loader, optimizer, ntokens, trainF)
        ppl = val(args, epoch, model, criterion, val_loader, optimizer, ntokens, testF)
        is_best = ppl < best_perplexity
        best_perplexity = min(ppl, best_perplexity)
        save_checkpoint(args, {'epoch':epoch + 1, 'state_dict': model.state_dict(), 'perplexity': ppl},
            is_best, os.path.join(args.log_dir, 'checkpoint.pth.tar'))
        os.system('python plot.py {} &'.format(args.log_dir))

def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))


def train(args, epoch, model, criterion, train_loader, optimizer, ntokens, trainF):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    counter = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        counter += data.size(1)
        data, target = Variable(data.cuda()), Variable(target.cuda())
        hidden = model.init_hidden(len(data))
        optimizer.zero_grad() #i think this is equal to model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), target.view(-1, 1).squeeze())
        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)

        total_loss += loss.data

        elapsed = time.time() - start_time
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:4.6f}'
                  'Time {:.3f} -- '.format(
                epoch, batch_idx, 100. * batch_idx / len(train_loader),
                loss.data[0], elapsed))
    ppl = torch.exp(total_loss / counter)[0]
    print("counter...size of training data: {}".format(counter))
    print('Training epoch: ', epoch, " loss: ", total_loss[0]/counter, " ppl:", ppl)
    trainF.write('{}, {}, {}\n'.format(epoch, total_loss[0]/counter, ppl))
    trainF.flush()

def val(args, epoch, model, criterion, val_loader, optimizer, ntokens, testF):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    counter = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        counter += data.size(1)
        data, target = Variable(data.cuda()), Variable(target.cuda())
        hidden = model.init_hidden(len(data))
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), target.view(-1, 1).squeeze())
        total_loss += loss.data

    ppl = torch.exp(total_loss / counter)[0]
    print('Validation epoch: ', epoch, " loss: ", total_loss[0]/counter, " ppl:", ppl)
    testF.write('{}, {}, {}\n'.format(epoch, total_loss[0]/counter, ppl))
    testF.flush()
    return ppl

if __name__=='__main__':
    main()
