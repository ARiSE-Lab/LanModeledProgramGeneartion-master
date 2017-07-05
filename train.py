###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/7/2017
# Many codes are from Wasi Ahmad train.py
# File Description: This script contains code to train the model.
###############################################################################

import time, util, torch, os
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import math

args = util.get_args()


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, max_len, dictionary, embeddings_index, loss_f):
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.model = model
        self.loss_f = loss_f
        self.criterion = getattr(nn, self.loss_f)() # nn.CrossEntropyLoss()  # Combines LogSoftMax and NLLoss in one single class
        self.num_directions = 2 if args.bidirection else 1
        self.lr = args.lr
        self.seq_len = max_len

        # Adam optimizer is used for stochastic optimization
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)

    def train_epochs(self, train_data, val_data): #train_batches, dev_batches
        """Trains model for n_epochs epochs"""
        # Loop over epochs.
        best_val_loss = None




            # At any point you can hit Ctrl + C to break out of training early.
        try:
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
            for epoch in range(args.start_epoch, args.epochs + 1):
                # train(args, epoch, model, criterion, train_loader, optimizer, ntokens, trainF)
                plot_losses = self.train(train_data, epoch, trainF)
                # ppl = val(args, epoch, model, criterion, val_loader, optimizer, ntokens, testF)
                ppl = util.evaluate(val_data, self.model, self.dictionary, args.bptt, self.criterion, testF, epoch)
                is_best = ppl < best_perplexity
                best_perplexity = min(ppl, best_perplexity)
                save_checkpoint(args, {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'perplexity': ppl},
                                is_best, os.path.join(args.log_dir, 'checkpoint.pth.tar'))
                os.system('python3 plot.py {} &'.format(args.log_dir))
                if not is_best:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    # self.lr /= 4.0
                    self.lr = self.lr * args.lr_decay
                    self.optimizer.param_groups[0]['lr'] = self.lr
                    print("Decaying learning rate to %g" % self.lr)

            # for epoch in range(1, args.epochs + 1):
            #     epoch_start_time = time.time()
            #
            #     plot_losses = self.train(train_data, epoch)
            #     #print('Saving losses')
            #     #util.save_plot(plot_losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch)))
            #
            #     val_loss = util.evaluate(val_data, self.model, self.dictionary, args.bptt, self.criterion)
            #     #util.save_plot(losses, self.config.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch + 1)))
            #
            #     print('-' * 89)
            #     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            #           'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
            #                                      val_loss, math.exp(val_loss)))
            #     print('-' * 89)
            #     # Save the model if the validation loss is the best we've seen so far.
            #     if not best_val_loss or val_loss < best_val_loss:
            #         with open(args.save, 'wb') as f:
            #             torch.save(self.model, f)
            #         best_val_loss = val_loss
            #     else:
            #         # Anneal the learning rate if no improvement has been seen in the validation dataset.
            #         #self.lr /= 4.0
            #         self.lr = self.lr * args.lr_decay
            #         self.optimizer.param_groups[0]['lr'] = self.lr
            #         print("Decaying learning rate to %g" % self.lr)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def save_checkpoint(args, state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    def train(self, train_batches, epoch, trainF):
        # Turn on training mode which enables dropout.
        self.model.train()

        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        best_dev_loss = -1
        last_best_dev_loss = -1
        epoch_loss = 0
        total_loss = 0
        start_time = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        ntokens = len(self.dictionary)
        hidden = self.model.init_hidden(args.batch_size)

        counter = 0
        for batch_no, batch in enumerate(train_batches):

            # self.optimizer.zero_grad()
            train_sentences1, train_labels = util.instances_to_tensors(batch,self.dictionary)
            #print("data: ", train_sentences1.size())
            #print("target: ", train_labels.size())

            if args.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_labels = train_labels.cuda()

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.

            if (args.instance): hidden = self.model.init_hidden(args.batch_size)  # for each sentence need to initialize
            hidden = util.repackage_hidden(hidden, args.cuda)

            self.optimizer.zero_grad()
            output, hidden = self.model(train_sentences1, hidden)
            loss = self.criterion(output.view(-1, ntokens), train_labels)

            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()

            total_loss += loss.data
            plot_loss_total += loss.data
            epoch_loss += loss.data

            counter+=args.batch_size

            if batch_no % args.print_every == 0 and batch_no > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | loss now {:8.2}| total loss {:8.2}'.format(
                    epoch, batch_no, len(train_batches), self.lr,
                                  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), total_loss, epoch_loss[0]))
                total_loss = 0
                start_time = time.time()

            if batch_no % args.plot_every == 0 and batch_no > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg[0])
                plot_loss_total = 0

        ppl = torch.exp(epoch_loss[0] / counter)
        print("counter...size of training data: {}".format(counter))
        print('Training epoch: ', epoch, " loss: ", epoch_loss[0] / counter, " ppl:", ppl)
        print("Time to complete epoch: ", time.time() - start_time)
        trainF.write('{}, {}, {}\n'.format(epoch, epoch_loss[0] / counter, ppl))
        trainF.flush()
        print('returning plot lossses: ', plot_losses)
        return plot_losses








    def train_(self, train_data, epoch):
        # Turn on training mode which enables dropout.
        self.model.train()
        total_loss = 0
        start_time = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0

        ntokens = len(self.dictionary)
        hidden = self.model.init_hidden(args.batch_size)

        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            data, targets = util.get_batch(train_data, i, args.bptt)
            print("data: ", data)
            print("target: ", targets)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.

            if(args.instance): hidden = self.model.init_hidden(args.batch_size)#for each sentence need to initialize
            hidden = util.repackage_hidden(hidden, args.cuda)

            self.optimizer.zero_grad()
            output, hidden = self.model(data, hidden)
            loss = self.criterion(output.view(-1, ntokens), targets)

            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()


            total_loss += loss.data
            plot_loss_total += loss.data

            if batch % args.print_every == 0 and batch > 0:
                cur_loss = total_loss[0] / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // args.bptt, self.lr,
                                  elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()


            if batch % args.plot_every == 0 and batch > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg[0])
                plot_loss_total = 0

        print('returning plot lossses: ', plot_losses)
        return plot_losses

    def validate(self, dev_batches):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        num_batches = len(dev_batches)
        avg_loss = 0
        for batch_no in range(num_batches):
            dev_sentences1, dev_sentences2, dev_labels = helper.instances_to_tensors(dev_batches[batch_no],
                                                                                     self.dictionary)
            if args.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            softmax_prob = self.model(dev_sentences1, dev_sentences2)
            loss = self.criterion(softmax_prob, dev_labels)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            avg_loss += loss.data[0]

        # Turn on training mode at the end of validation.
        self.model.train()

        return avg_loss / num_batches