###############################################################################
# Author: Md Rizwan Parvez
# Project: LanModeledProgramGeneration
# Date Created: 4/7/2017
# Many codes are from Wasi Ahmad train.py
# File Description: This script contains code to train the model.
###############################################################################

import time, util, torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm
import math, os

args = util.get_args()


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, dictionary, loss_f):
        self.dictionary = dictionary
        self.model = model
        self.loss_f = loss_f
        self.criterion = getattr(nn, self.loss_f)(size_average=True) # nn.CrossEntropyLoss()  # Combines LogSoftMax and NLLoss in one single class
        self.num_directions = 2 if args.bidirection else 1
        self.lr = args.lr

        # Adam optimizer is used for stochastic optimization
        self.optimizer = optim.SGD(self.model.parameters(), self.lr)

    def train_epochs(self, train_data_trimed, train_label_trimed , valid_data_trimed, valid_label_trimed): #train_batches, dev_batches
        """Trains model for n_epochs epochs"""
        # Loop over epochs.
        best_val_loss = None

        # At any point you can hit Ctrl + C to break out of training early.
        try:

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
            #for epoch in range(1, args.epochs + 1):
                epoch_start_time = time.time()

                #plot_losses = 
                self.train(train_data_trimed, train_label_trimed, epoch, trainF)
                #print(plot_losses)
                #util.save_plot(plot_losses, args.save_path + 'training_loss_plot_epoch_{}.png'.format((epoch)))

                val_loss = util.evaluate(valid_data_trimed, valid_label_trimed , self.model, self.dictionary, self.criterion, epoch, testF)


                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:.2f} | '
                      'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                 val_loss, math.exp(val_loss) ))
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

    def save_checkpoint(args, state, is_best, filename):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.log_dir, 'model_best.pth.tar'))

    def train_(self, train_batches, dev_batches, epoch_no ):
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

    def train(self, train_data_trimed, train_label_trimed, epoch, trainF):
        # Turn on training mode which enables dropout.
        self.model.train()
        batch_loss = 0
        start_time = time.time()

        plot_losses = []
        print_loss_total = 0
        plot_loss_total = 0
        epoch_mean_loss = 0
        brgin_epoch_time = time.time()
        ntokens = len(self.dictionary)
        hidden = self.model.init_hidden(args.batch_size)
        print ('Training Starts!! Epoch: ', epoch, '\n', '=='*39)

        for batch, i in enumerate(range(0, len(train_data_trimed) -1, args.batch_size)):
            data, targets = util.get_minibatch(train_data_trimed, train_label_trimed, i, args.batch_size, self.dictionary.padding_id )
            mask = targets.ne(self.dictionary.padding_id).data
            #print (targets, mask)
            #print (data.size(), data, targets.size(), targets)
            #print (' batch train data; ', len(train_data_trimed) ,'this batch: ', data.size(), ' target size: ', targets.size()) #data size 35 x 20 (here bptt x batch_size) {in gen: batch_size x seq_len}
            #continue
            #data = data.t().contiguous() # after permute data size 20 x 35 (here batch_size x bptt) {in gen: seq_len x batch_size so no need in gen mode }
            #targets = targets.t().contiguous() # same as data
            #if i == 0: print ('train data; AFTER PERMUTE ', train_data.size() ,'this batch: ', data.size(), ' target size: ', targets.size())
            #targets = targets.view(-1)
            #if i == 0: print (' btch first train data; ', train_data.size() ,'this batch: ', data.size(), ' target size: ', targets.size(), targets) #data size 35 x 20 (here bptt x batch_size) {in gen: batch_size x seq_len}
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            
            hidden = self.model.init_hidden(args.batch_size) #for each sentence need to initialize
            hidden = util.repackage_hidden(hidden, args.cuda)
            #print 
            self.optimizer.zero_grad()
            output, hidden = self.model(data, hidden)
            #if i == 0: print ('final output: ', output.size(), output,  '\n output.view(-1, ntokens): ', output.view(-1, ntokens))
            loss =  self.criterion(output.view(-1, ntokens), targets)


            # Important if we are using nn.DataParallel()
            #print(loss.data)
            #mean_loss = torch.mean(torch.masked_select(loss.data, mask))
            #assert np.count_nonzero(mask.numpy())*mean_loss.data[0] == loss.data[0] 
            loss.backward()
            self.optimizer.step()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            clip_grad_norm(self.model.parameters(), args.clip)
            


            batch_loss += loss.data
            plot_loss_total += loss.data
            epoch_mean_loss += loss.data

            if batch % args.print_every == 0 and batch > 0:
                cur_loss = batch_loss[0] / args.print_every
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data_trimed) // args.batch_size, self.lr,
                                  elapsed * 1000 / args.print_every, cur_loss, math.exp(cur_loss)))
                batch_loss = 0
                start_time = time.time()


            if batch % args.plot_every == 0 and batch > 0:
                plot_loss_avg = plot_loss_total / args.plot_every
                plot_losses.append(plot_loss_avg[0])
                plot_loss_total = 0
            #exit()

        ppl = torch.exp(epoch_mean_loss/batch)[0]
        #print("counter...size of training data: {}".format(counter))
        print('Training epoch: ', epoch, " avg loss: ",(epoch_mean_loss/batch)[0], " ppl:", ppl)
        print("Time to complete epoch: ", time.time() - brgin_epoch_time)
        trainF.write('{}, {}, {}\n'.format(epoch, (epoch_mean_loss/batch)[0], ppl))
        trainF.flush()
        #print('returning plot lossses: ', plot_losses)
        #return plot_losses

    def validate(self, dev_batches, testF):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

       
        args = get_args()
        model.eval()
        total_loss = 0
        ntokens = len(dictionary)
        eval_batch_size = args.batch_size #// 2
        
         
        for batch, i in enumerate(range(0, len(valid_data_trimed) - 1, eval_batch_size)):
            data, targets = get_minibatch(valid_data_trimed, valid_label_trimed, i, eval_batch_size, dictionary.padding_id, evaluation=True)
            hidden = model.init_hidden(eval_batch_size) #for each sentence need to initialize
            hidden = repackage_hidden(hidden, args.cuda)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss +=  criterion(output_flat, targets).data
        

        # Turn on training mode at the end of validation.
        self.model.train()
        ppl = torch.exp(total_loss / counter)[0]
        print('Validation epoch: ', epoch, " loss: ", total_loss[0]/counter, " ppl:", ppl)
        testF.write('{}, {}, {}\n'.format(epoch, total_loss[0]/counter, ppl))
        testF.flush()
        return ppl

        #return avg_loss / num_batches