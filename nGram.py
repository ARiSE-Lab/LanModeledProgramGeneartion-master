import torch.nn as nn
import util, data 
import torch, random
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os, sys, math, pdb, string, shutil, pickle, time
import numpy as np
import argparse
from torch.utils.data import DataLoader
import n_gram_data_loader
import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type = int, default=10)
    parser.add_argument('--data_path', type = str)
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--seed', type = int, default = 1)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    corpus = data.Corpus(args)
    vocab = corpus.dictionary
    print('Vocabulary size = ', len(vocab))

    train_data = n_gram_data_loader.NgramData(corpus, 'train')
    val_data = n_gram_data_loader.NgramData(corpus, 'val')
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, 
        pin_memory = True, num_workers = 6)
    val_loader = DataLoader(val_data,  batch_size=args.batch_size, shuffle=False,
        pin_memory = False, num_workers = 4)


    EMBEDDING_DIM = 300
    CONTEXT_SIZE = 2
    class NGramLanguageModeler(nn.Module):

        def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs).view(len(inputs), -1)
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            return out

    losses = []
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.cuda()
    model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    for epoch in range(args.nepochs):
        total_loss = torch.Tensor([0])
        c= 0;
        for batch_idx, (context, target) in enumerate(train_loader):
            c+=args.batch_size
            context, target = context.cuda(), target.cuda()
            context_var, target = Variable(context), Variable(target)
            model.zero_grad()
            log_probs = model(context_var)
            loss = loss_function(log_probs, target.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.data.cpu()
        print('epoch: ', epoch, " loss: ", total_loss.tolist()[0], " ppl:", torch.exp(total_loss/c).tolist()[0])
    # print(losses)
if __name__=='__main__':
    main()