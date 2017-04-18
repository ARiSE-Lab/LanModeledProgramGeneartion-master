import time, util, torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import math

args = util.get_args()


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, dictionary, embeddings_index, loss_f):
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.model = model
        self.loss_f = loss_f
        self.criterion = getattr(nn, self.loss_f)() # nn.CrossEntropyLoss()  # Combines LogSoftMax and NLLoss in one single class
        self.num_directions = 2 if args.bidirection else 1
        self.lr = args.lr

        # Adam optimizer is used for stochastic optimization
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)

    def train_epochs(self, train_data, val_data): #train_batches, dev_batches
        """Trains model for n_epochs epochs"""
        # Loop over epochs.
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()

                plot_losses = self.train(train_data, epoch)
                #print(plot_losses)
                #util.save_plot(plot_losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch)))

                val_loss = util.evaluate(val_data, self.model, self.dictionary, args.bptt, self.criterion)


                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    with open(args.save, 'wb') as f:
                        torch.save(self.model, f)
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    #self.lr /= 4.0
                    self.lr = self.lr * args.lr_decay
                    self.optimizer.param_groups[0]['lr'] = self.lr
                    print("Decaying learning rate to %g" % self.lr)
        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')



    def train_(self, train_batches, dev_batches, epoch_no, ):
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
            train_sentences1, train_sentences2, train_labels = util.instances_to_tensors(train_batches[batch_no],
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
                    util.show_progress(start, batch_no / num_batches), batch_no,
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
                    util.save_model(self.model, last_best_dev_loss, epoch_no, 'model')

        return plot_losses

    def train(self, train_data, epoch):
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