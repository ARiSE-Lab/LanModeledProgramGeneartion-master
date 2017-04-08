###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/7/2017
# Many codes are from Wasi Ahmad train.py
# File Description: This script contains code to train the model.
###############################################################################

import time, util, helper, torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm

args = util.get_args()


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, dictionary, embeddings_index, loss):
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.model = model
        self.loss = loss
        self.criterion = getattr(nn, self.loss)() # nn.CrossEntropyLoss()  # Combines LogSoftMax and NLLoss in one single class
        self.num_directions = 2 if args.bidirection else 1
        self.lr = args.lr

        # Adam optimizer is used for stochastic optimization
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def train_epochs(self, train_batches, dev_batches, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(n_epochs):
            losses = self.train(train_batches, dev_batches, (epoch + 1))
            helper.save_plot(losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch + 1)))

    def train(self, train_batches, dev_batches, epoch_no):
        # Turn on training mode which enables dropout.
        self.model.train()

        start = time.time()
        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        best_dev_loss = -1
        last_best_dev_loss = -1

        num_batches = len(train_batches)
        print('epoch %d started' % epoch_no)

        for batch_no in range(num_batches):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, train_sentences2, train_labels = helper.instances_to_tensors(train_batches[batch_no],
                                                                                           self.dictionary)
            if args.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            softmax_prob = self.model(train_sentences1, train_sentences2)
            loss = self.criterion(softmax_prob, train_labels)

            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = torch.mean(loss)
            loss.backward()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs.
            clip_grad_norm(self.model.parameters(), args.clip)
            self.optimizer.step()

            if batch_no % args.print_every == 0 and batch_no > 0:
                print_loss_avg = print_loss_total / args.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % args.plot_every == 0 and batch_no > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

            if batch_no % args.dev_every == 0 and batch_no > 0:
                dev_loss = self.validate(dev_batches)
                print('validation loss = %.4f' % dev_loss)
                if best_dev_loss == -1 or best_dev_loss > dev_loss:
                    best_dev_loss = dev_loss
                else:
                    # no improvement in validation loss, so apply learning rate decay
                    self.lr = self.lr * args.lr_decay
                    self.optimizer.param_groups[0]['lr'] = self.lr
                    print("Decaying learning rate to %g" % self.lr)

            if batch_no % args.save_every == 0 and batch_no > 0:
                if last_best_dev_loss == -1 or last_best_dev_loss > best_dev_loss:
                    last_best_dev_loss = best_dev_loss
                    helper.save_model(self.model, last_best_dev_loss, epoch_no, 'model')

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