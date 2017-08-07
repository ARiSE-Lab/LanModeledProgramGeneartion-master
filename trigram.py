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
import n_gram_data_loader
import pdb
import torch.backends.cudnn as cudnn
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type = int, default=10)
    parser.add_argument('--data_path', type = str)
    parser.add_argument('--batch_size', type = int, default = 100)
    parser.add_argument('--seed', type = int, default = 1)
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--log_dir', type=str, './log_ngram')
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--context_size', type=int, default=2)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.log_dir) and not args.resume:
        print('Path {} exists! and not resuming'.format(args.log_dir))
        #return
    if not os.path.exists(args.log_dir):os.makedirs(args.log_dir)

    train_data = n_gram_data_loader.NgramData(args, 'train')
    val_data = n_gram_data_loader.NgramData(args, 'val')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
        pin_memory = True, num_workers = 6)
    val_loader = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False,
        pin_memory = False, num_workers = 4)


    
    print(train_data.num_tokens())











def save_checkpoint(args, state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

def train(args, epoch, model, criterion, train_loader, optimizer, trainF):
    model.train()
    total_loss = 0
    counter = 0
    for batch_idx, (context, target) in enumerate(train_loader):
        end = time.time()
        counter += len(context)
        context, target = context.cuda(), target.cuda()
        context_var, target = Variable(context), Variable(target)
        model.zero_grad()
        log_probs = model(context_var)
        loss = criterion(log_probs, target.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.data
        pred = log_probs.data.max(1)[1]
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:4.6f}'
                  'Time {:.3f} -- '.format(
                epoch, batch_idx, 100. * batch_idx / len(train_loader),
                loss.data.cpu()[0], time.time() - end))
    ppl = torch.exp(total_loss / counter)[0]
    print("counter...size of training data: {}".format(counter))
    print('Training epoch: ', epoch, " loss: ", total_loss[0]/counter, " ppl:", ppl)
    trainF.write('{}, {}, {}\n'.format(epoch, total_loss[0]/counter, ppl))
    trainF.flush()

def val(args, epoch, model, criterion, val_loader, optimizer, testF):
    model.eval()
    total_loss = 0
    counter = 0
    for batch_idx, (context, target) in enumerate(val_loader):
        counter += len(context)
        context, target = context.cuda(), target.cuda()
        context_var, target = Variable(context), Variable(target)
        model.zero_grad()
        log_probs = model(context_var)
        loss = criterion(log_probs, target.squeeze())
        total_loss += loss.data
    ppl = torch.exp(total_loss / counter)[0]
    print('Validation epoch: ', epoch, " loss: ", total_loss[0]/counter, " ppl:", ppl)
    testF.write('{}, {}, {}\n'.format(epoch, total_loss[0]/counter, ppl))
    testF.flush()
    return ppl

if __name__=='__main__':
    main()