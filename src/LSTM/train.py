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
    # parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true',  help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt', help='path to save the final model')

    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
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
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    lr = args.lr
    best_val_loss = None

    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
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
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
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

if __name__=='__main__':
    main()
